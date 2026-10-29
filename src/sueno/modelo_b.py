"""Modelo B: red sobre la señal cruda, un solo modelo para las 12 combinaciones de canales.

CNN + Transformer por época y canal (codificador de SleepEffFormer con GroupNorm, compartido por Fpz-Cz, Pz-Oz y
EOG), fusión con máscara de las modalidades presentes y 2 capas de Transformer entre 80 épocas seguidas.
"""
import copy
import csv
import json
import time
from pathlib import Path

import mne
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .config import COMBINACIONES, EPOCH, ESTADIOS, FS, K, MODALIDADES, SEED, SPE, mascara
from .evaluacion import metricas

# ---------------------------------------------------------------------------------------------------------------
# Preprocesamiento
# ---------------------------------------------------------------------------------------------------------------


def zscore(x):
    return ((x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)).astype(np.float32)


def preparar(noche):
    """Épocas que ve la red: señal filtrada 0.3-35 Hz y z-score por época (float16); EMG en log con z-score robusto.

    Los canales ausentes quedan en cero (la máscara los apaga). Devuelve también y y pos (posición en el registro).
    """
    keep = noche.epocas[noche.validas]
    n = noche.n_registro
    out = {"y": noche.etiquetas[noche.validas].astype(np.int64), "pos": keep}
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

CNN_CHANNELS = (1, 32, 64, 128, 128)
CNN_KERNELS = (7, 5, 5, 3)
CNN_STRIDES = (2, 2, 4, 2)
CNN_PADDINGS = (3, 2, 2, 1)
ARQUITECTURA = dict(d=128, n_classes=5, dropout=0.3, seq_layers=2, seq_heads=4, seq_ff=256, seq_dropout=0.1, max_len=512)


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
        x = x + a
        x = x + self.ff(self.norm2(x))
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
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        h = self.cnn(x).transpose(1, 2)
        h = self.pos_dropout(h + self.pe)
        for layer in self.layers:
            h, _ = layer(h)
        return self.head(self.norm(h).mean(1))


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


class Red(nn.Module):
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
# Entrenamiento
# ---------------------------------------------------------------------------------------------------------------

HP = dict(seq_len=80, batch=2, lr=3e-4, weight_decay=1e-4, max_epochs=100, min_epochs=40, patience=15, clip=1.0,
          ema=0.999, alpha=0.44, stride=40)
SAMPLE_WEIGHT = {c: (2.0 if c in ("fpz", "pz") else 1.0) for c in COMBINACIONES}   # los casos más difíciles, el doble
VAL_CONFIGS = ("fpz", "pz", "eeg2_eog_emg")


class Nights:
    """Noches concatenadas en tensores del dispositivo, con el inicio y el largo de cada una."""

    def __init__(self, data, nights, device):
        self.nights = list(nights)
        lengths = [len(data[n]["y"]) for n in self.nights]
        self.start = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
        self.length = np.array(lengths)
        self.sig = torch.from_numpy(np.concatenate(
            [np.stack([data[n][k] for k in ("fpz", "pz", "eog")], 1) for n in self.nights])).to(device)
        self.emg = torch.from_numpy(np.concatenate([data[n]["emg"] for n in self.nights])).to(device)
        self.y = torch.from_numpy(np.concatenate([data[n]["y"] for n in self.nights])).to(device)


def class_weights(y, alpha):
    counts = np.bincount(y, minlength=K).astype(float)
    return (len(y) / (K * np.maximum(counts, 1))) ** alpha


def train_windows(nd, rng, L):
    """Inicios de los trozos de una pasada: un desfase al azar por noche, trozos sin traslape, barajados."""
    starts = []
    for s, n in zip(nd.start, nd.length):
        off = int(rng.integers(0, L)) if n > L else 0
        starts += [s + o for o in range(off, max(n - L, 0) + 1, L)]
    return rng.permutation(np.array(starts, dtype=np.int64))


class EMA:
    """Promedio exponencial de los pesos del modelo."""

    def __init__(self, model, decay):
        self.decay, self.steps = decay, 0
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.steps += 1
        w = 1 - min(self.decay, (1 + self.steps) / (10 + self.steps))   # al inicio sigue al modelo más de cerca
        for e, m in zip(self.model.parameters(), model.parameters()):
            e.lerp_(m.detach(), w)
        for e, m in zip(self.model.buffers(), model.buffers()):
            if e.dtype.is_floating_point:
                e.lerp_(m, w)
            else:
                e.copy_(m)


def _amp(device):
    return torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda")


@torch.no_grad()
def predict_night(model, nd, i, mask, device, L=None, stride=None, batch=16):
    """Probabilidades (n, 5) de la noche i: ventanas de 80 épocas que avanzan de 40 en 40, promediadas.

    mask: [fpz, pz, eog, emg] de los canales presentes.
    """
    model.eval()
    L, stride = L or HP["seq_len"], stride or HP["stride"]
    s, n = int(nd.start[i]), int(nd.length[i])
    L = min(L, n)
    starts = list(range(0, n - L + 1, stride))
    if starts[-1] != n - L:
        starts.append(n - L)
    acc = torch.zeros(n, K, device=device)
    cnt = torch.zeros(n, 1, device=device)
    mask1 = torch.tensor(mask, device=device)
    for j in range(0, len(starts), batch):
        st = torch.tensor(starts[j:j + batch], device=device)
        idx = s + st[:, None] + torch.arange(L, device=device)
        with _amp(device):
            logits = model(nd.sig[idx].float() if device.type != "cuda" else nd.sig[idx], nd.emg[idx],
                           mask1.expand(len(st), 4))
        p = torch.softmax(logits.float(), -1)
        loc = st[:, None] + torch.arange(L, device=device)
        acc.index_add_(0, loc.reshape(-1), p.reshape(-1, K))
        cnt.index_add_(0, loc.reshape(-1), torch.ones(loc.numel(), 1, device=device))
    return (acc / cnt).cpu().numpy()


def evaluate(model, nd, configs, device):
    y = nd.y.cpu().numpy()
    out = {}
    for cfg in configs:
        p = np.concatenate([predict_night(model, nd, i, mascara(cfg), device) for i in range(len(nd.nights))])
        out[cfg] = metricas(y, p.argmax(1))
    return out


def fit(train, val_nd, log_path, device, seed=SEED, max_epochs=None):
    """Entrena y devuelve (mejor pasada, pesos EMA de la mejor pasada)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    max_epochs = max_epochs or HP["max_epochs"]
    L, bs = HP["seq_len"], HP["batch"]

    model = Red(**ARQUITECTURA).to(device)
    ema = EMA(model, HP["ema"])
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=HP["max_epochs"], eta_min=1e-6)
    w = torch.tensor(class_weights(train.y.cpu().numpy(), HP["alpha"]), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    cfg_names = list(COMBINACIONES)
    p = np.array([SAMPLE_WEIGHT[c] for c in cfg_names])
    p = p / p.sum()
    masks = torch.tensor([mascara(c) for c in cfg_names], device=device)

    best, best_state = {"score": -1.0, "epoch": 0}, None
    with open(log_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["epoch", "train_loss", *[f"val_mf1_{c}" for c in VAL_CONFIGS], "val_f1_N1_fpz", "score", "lr", "seconds"])
        for epoch in range(1, max_epochs + 1):
            t0 = time.perf_counter()
            model.train()
            starts = train_windows(train, rng, L)
            total, nb = 0.0, 0
            for j in range(0, len(starts), bs):
                st = torch.from_numpy(starts[j:j + bs]).to(device)
                idx = st[:, None] + torch.arange(L, device=device)
                mk = masks[torch.from_numpy(rng.choice(len(cfg_names), size=len(st), p=p)).to(device)]
                sig = train.sig[idx] if device.type == "cuda" else train.sig[idx].float()
                with _amp(device):
                    logits = model(sig, train.emg[idx], mk)
                    loss = criterion(logits.float().reshape(-1, K), train.y[idx].reshape(-1))
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), HP["clip"])
                scaler.step(opt)
                scaler.update()
                ema.update(model)
                total += loss.item()
                nb += 1
            lr = opt.param_groups[0]["lr"]
            sched.step()

            val = evaluate(ema.model, val_nd, VAL_CONFIGS, device)
            score = float(np.mean([val[c]["mf1"] for c in VAL_CONFIGS]))
            wr.writerow([epoch, total / nb, *[val[c]["mf1"] for c in VAL_CONFIGS], val["fpz"]["f1_N1"], score, lr,
                         round(time.perf_counter() - t0, 1)])
            f.flush()
            if score > best["score"]:
                best = {"score": score, "epoch": epoch}
                best_state = copy.deepcopy(ema.model.state_dict())
            if epoch >= HP["min_epochs"] and epoch - best["epoch"] >= HP["patience"]:
                break
    return best, best_state


def entrenar_y_predecir(nombre, fold, noches_dir, salida, device, max_epochs=None, n_noches=None):
    """Un entrenamiento (fold_k o final): entrena, predice sus noches de test con las 12 combinaciones y guarda.

    Escribe en salida/: <nombre>.npz (predicciones), <nombre>_log.csv y <nombre>_resumen.json; el final además
    guarda modelo_final.pt. n_noches limita las noches (solo para pruebas rápidas).
    """
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    disponibles = sorted(p.stem for p in Path(noches_dir).glob("SC4*.npz"))
    de = lambda sujetos: [n for n in disponibles if int(n[3:5]) in set(sujetos)]  # noqa: E731
    tr, va, te = de(fold["train"]), de(fold["val"]), de(fold["test"])
    if n_noches:
        tr, va, te = tr[:n_noches], va[:2], te[:2]
    t0 = time.perf_counter()
    data = {n: dict(np.load(Path(noches_dir) / f"{n}.npz")) for n in tr + va}
    best, best_state = fit(Nights(data, tr, device), Nights(data, va, device), salida / f"{nombre}_log.csv", device,
                           max_epochs=max_epochs)
    del data

    model = Red(**ARQUITECTURA).to(device)
    model.load_state_dict(best_state)
    if nombre == "final":
        torch.save({"model_state_dict": {k: v.cpu() for k, v in best_state.items()}, "ARQUITECTURA": ARQUITECTURA,
                    "HP": HP, "CONFIGS": COMBINACIONES, "MODALIDADES": MODALIDADES, "ESTADIOS": ESTADIOS,
                    "mejor_pasada": best["epoch"], "puntaje_val": best["score"]}, salida / "modelo_final.pt")
    test_data = {n: dict(np.load(Path(noches_dir) / f"{n}.npz")) for n in te}
    test_nd = Nights(test_data, te, device)
    arr = {}
    for cfg in COMBINACIONES:
        for i, n in enumerate(te):
            arr[f"{cfg}__{n}"] = predict_night(model, test_nd, i, mascara(cfg), device).astype(np.float32)
    for n in te:
        arr[f"y__{n}"], arr[f"pos__{n}"] = test_data[n]["y"], test_data[n]["pos"]
    np.savez_compressed(salida / f"{nombre}.npz", **arr)
    (salida / f"{nombre}_resumen.json").write_text(json.dumps(
        {"mejor_pasada": best["epoch"], "puntaje_val": best["score"], "minutos": round((time.perf_counter() - t0) / 60, 1)}))


def cargar(ruta, device="cpu"):
    """Red del checkpoint, lista para predecir, y sus hiperparámetros."""
    ck = torch.load(ruta, map_location=device, weights_only=False)
    red = Red(**ck["ARQUITECTURA"]).to(device)
    red.load_state_dict(ck["model_state_dict"])
    return red.eval(), ck["HP"]


def predecir(red, noche_b, mask, hp=HP, device="cpu"):
    """Probabilidades (n, 5) de una noche ya preparada con preparar()."""
    device = torch.device(device)
    nd = Nights({"noche": noche_b}, ["noche"], device)
    return predict_night(red, nd, 0, mask, device, L=hp["seq_len"], stride=hp["stride"])
