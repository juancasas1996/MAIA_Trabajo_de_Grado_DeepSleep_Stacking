"""Modelo B: red CNN-Transformer sobre la señal cruda, con canales enmascarados.

La arquitectura y la predicción son las de experimentacion/modelo_b_red/2_entrenamiento.ipynb; el preprocesamiento,
el de 1_preprocesamiento.ipynb.
"""
import mne
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .lectura import EPOCH, FS, K, SPE

CNN_CHANNELS = (1, 32, 64, 128, 128)
CNN_KERNELS = (7, 5, 5, 3)
CNN_STRIDES = (2, 2, 4, 2)
CNN_PADDINGS = (3, 2, 2, 1)


# ---------------------------------------------------------------------------------------------------------------
# Preprocesamiento
# ---------------------------------------------------------------------------------------------------------------

def zscore(x):
    return ((x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)).astype(np.float32)


def preparar(noche):
    """Épocas que ve la red: señal filtrada 0.3-35 Hz y z-score por época (float16); EMG en log con z-score robusto.

    Los canales ausentes quedan en cero (la máscara los apaga).
    """
    keep = noche.epocas[noche.validas]
    n = noche.n_registro
    out = {}
    for k in ("fpz", "pz", "eog"):
        if k in noche.canales:
            sig = mne.filter.filter_data(noche.senales[k], FS, 0.3, 35.0, method="fir", fir_window="hamming", verbose="error")
            out[k] = zscore(sig[: n * SPE].reshape(n, SPE)[keep]).astype(np.float16)
        else:
            out[k] = np.zeros((len(keep), SPE), dtype=np.float16)
    if "emg" in noche.canales:
        # El EMG viene a 1 Hz y MNE lo interpola a 100 Hz: cada 100 muestras está el valor original
        emg = np.log1p(np.abs(noche.senales["emg"][: n * SPE: FS].reshape(n, EPOCH)[keep]) * 1e6)
        med = np.median(emg)
        iqr = np.subtract(*np.percentile(emg, [75, 25])) or 1.0
        out["emg"] = ((emg - med) / iqr).astype(np.float32)
    else:
        out["emg"] = np.zeros((len(keep), EPOCH), dtype=np.float32)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Arquitectura
# ---------------------------------------------------------------------------------------------------------------

def positional_encoding(n_tokens, d_model):
    pos = torch.arange(n_tokens).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    pe = torch.zeros(n_tokens, d_model)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class PreNormLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model), nn.Dropout(dropout))

    def forward(self, x, need_weights=False):
        h = self.norm1(x)
        a, w = self.attn(h, h, h, need_weights=need_weights, average_attn_weights=True)
        x = x + a                         # residual de la atención
        x = x + self.ff(self.norm2(x))    # residual de la red pequeña
        return x, w


class SleepEffFormer(nn.Module):
    """Clasificador de una época (reimplementación propia de SleepEffFormer, Paul et al., 2026)."""

    def __init__(self, n_classes=5, d_model=128, nhead=4, n_layers=2, d_ff=256,
                 cnn_dropout=0.25, tf_dropout=0.1, head_dropout=0.25):
        super().__init__()
        blocks = []
        for i in range(len(CNN_KERNELS)):
            blocks += [nn.Conv1d(CNN_CHANNELS[i], CNN_CHANNELS[i + 1], CNN_KERNELS[i], CNN_STRIDES[i],
                                 CNN_PADDINGS[i], bias=False),
                       nn.BatchNorm1d(CNN_CHANNELS[i + 1]), nn.GELU(), nn.Dropout(cnn_dropout)]
        self.cnn = nn.Sequential(*blocks)
        with torch.no_grad():
            self.n_tokens = self.cnn(torch.zeros(1, 1, SPE)).shape[-1]  # 94
        self.register_buffer("pe", positional_encoding(self.n_tokens, d_model).unsqueeze(0))
        self.pos_dropout = nn.Dropout(tf_dropout)
        self.layers = nn.ModuleList([PreNormLayer(d_model, nhead, d_ff, tf_dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(d_model, n_classes))

    def forward(self, x):
        h = self.cnn(x).transpose(1, 2)        # (B, 94, 128)
        h = self.pos_dropout(h + self.pe)
        for layer in self.layers:
            h, _ = layer(h)
        return self.head(self.norm(h).mean(1))  # promedio de los 94 tokens -> 5 logits


class EpochEncoder(nn.Module):
    """CNN + Transformer de SleepEffFormer, sin la cabeza y con GroupNorm: (N, 3000) -> (N, 128)."""

    def __init__(self):
        super().__init__()
        base = SleepEffFormer()
        for i, mod in enumerate(base.cnn):
            if isinstance(mod, nn.BatchNorm1d):
                base.cnn[i] = nn.GroupNorm(8, mod.num_features)
        self.cnn, self.pos_dropout, self.layers, self.norm = base.cnn, base.pos_dropout, base.layers, base.norm
        self.register_buffer("pe", base.pe)

    def forward(self, x):
        h = self.cnn(x.unsqueeze(1)).transpose(1, 2)
        h = self.pos_dropout(h + self.pe)
        for layer in self.layers:
            h, _ = layer(h)
        return self.norm(h).mean(1)


class CnnTransformer(nn.Module):
    """sig (B, L, 3, 3000), emg (B, L, 30), mask (B, 4) -> logits (B, L, 5)."""

    def __init__(self, d=128, n_classes=5, dropout=0.3, seq_layers=2, seq_heads=4, seq_ff=256, seq_dropout=0.1,
                 max_len=512):
        super().__init__()
        self.d = d
        self.encoder = EpochEncoder()
        self.channel_emb = nn.Parameter(torch.zeros(3, d))          # un vector por canal (fpz, pz, eog)
        self.emg = nn.Sequential(nn.Linear(30, 64), nn.GELU(), nn.Linear(64, d))
        self.fuse = nn.Linear(d + 4, d)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("seq_pe", positional_encoding(max_len, d))   # posición de la época en el trozo
        self.seq_layers = nn.ModuleList([PreNormLayer(d, seq_heads, seq_ff, seq_dropout) for _ in range(seq_layers)])
        self.seq_norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, n_classes)

    def forward(self, sig, emg, mask):
        b, length = sig.shape[:2]
        feats = sig.new_zeros(b, length, 4, self.d, dtype=torch.float32)
        for c in range(3):  # solo se codifican los canales presentes
            sel = mask[:, c]
            if sel.any():
                x = sig[sel][:, :, c].reshape(-1, sig.shape[-1])
                feats[sel, :, c] = (self.encoder(x).float() + self.channel_emb[c]).reshape(int(sel.sum()), length, self.d)
        sel = mask[:, 3]
        if sel.any():
            feats[sel, :, 3] = self.emg(emg[sel]).float()
        m = mask.float()[:, None, :, None]
        pooled = (feats * m).sum(2) / m.sum(2).clamp(min=1.0)              # promedio de los presentes
        h = self.fuse(torch.cat([pooled, mask.float()[:, None, :].expand(b, length, 4)], -1))
        h = self.dropout(F.gelu(h)) + self.seq_pe[:length]
        for layer in self.seq_layers:   # cada época mira a todas las del trozo
            h, _ = layer(h)
        return self.out(self.dropout(self.seq_norm(h)))


# ---------------------------------------------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------------------------------------------

def cargar(ruta, device="cpu"):
    """Red de modelo/modelo_b_red/modelo_final.pt, lista para predecir, y sus hiperparámetros."""
    ck = torch.load(ruta, map_location=device, weights_only=False)
    red = CnnTransformer(**ck["ARQUITECTURA"]).to(device)
    red.load_state_dict(ck["model_state_dict"])
    return red.eval(), ck["HP"]


@torch.no_grad()
def predecir(red, datos, mascara, hp, device="cpu", batch=16):
    """Probabilidades (n, 5) de las épocas de preparar(): ventanas de seq_len épocas que avanzan de a stride.

    Cada época queda en varias ventanas y se promedian sus probabilidades.
    """
    device = torch.device(device)
    sig = torch.from_numpy(np.stack([datos[k] for k in ("fpz", "pz", "eog")], 1)).to(device)
    emg = torch.from_numpy(datos["emg"]).to(device)
    n = len(emg)
    L = min(hp["seq_len"], n)
    starts = list(range(0, n - L + 1, hp["stride"]))
    if starts[-1] != n - L:
        starts.append(n - L)
    acc = torch.zeros(n, K, device=device)
    cnt = torch.zeros(n, 1, device=device)
    mask1 = torch.tensor(mascara, device=device)
    for j in range(0, len(starts), batch):
        st = torch.tensor(starts[j:j + batch], device=device)
        idx = st[:, None] + torch.arange(L, device=device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = red(sig[idx].float() if device.type != "cuda" else sig[idx], emg[idx], mask1.expand(len(st), 4))
        p = torch.softmax(logits.float(), -1)
        acc.index_add_(0, idx.reshape(-1), p.reshape(-1, K))
        cnt.index_add_(0, idx.reshape(-1), torch.ones(idx.numel(), 1, device=device))
    return (acc / cnt).cpu().numpy()
