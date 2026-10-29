"""Rasgos por época del modelo A: espectro, forma de onda, ondículas, pendiente 1/f y contexto de la noche.

El prefijo de cada columna dice de qué canal sale (fpz_, pz_, eog_, emg_, time_). Si falta un canal, sus columnas
quedan en NaN y XGBoost las trata como faltantes.
"""
import antropy as ant
import numpy as np
import pandas as pd
import pywt
from scipy import signal, stats

from .config import COMBINACIONES, EPOCH, FS, MODALIDADES, SPE

EEG_BANDS = {"sdelta": (0.5, 1), "delta": (1, 4), "theta": (4, 8), "alpha": (8, 12), "sigma": (12, 16), "beta": (16, 30)}
EPS = 1e-12
DWT_BANDS = ["cA4", "cD4", "cD3", "cD2", "cD1"]
# Rasgos a los que se les agregan medias móviles y épocas vecinas
KEY_SUFFIXES = [
    "abs_sdelta", "abs_delta", "abs_theta", "abs_alpha", "abs_sigma", "abs_beta",
    "rel_delta", "rel_sigma", "ratio_slow_fast", "spec_entropy", "perm_entropy", "sub_sigma_max",
    "abs_slow", "abs_rapid", "std",          # EOG (y std del EEG)
    "mean", "log_mean",                      # EMG
    "ap_slope", "dwt_cD2_logenergy", "dwt_cD3_logenergy", "dwt_cA4_logenergy",
]
CROSS = {"corr_fpz": "fpz"}  # rasgos con prefijo de un canal que también usan otro


def _band_power(f, psd, lo, hi):
    idx = (f >= lo) & (f < hi)
    return psd[:, idx].sum(1) * (f[1] - f[0])


def eeg_features(x, p):
    """x: (n_épocas, 3000) en µV. p: prefijo del canal."""
    out = {}
    f, psd = signal.welch(x, fs=FS, nperseg=4 * FS, noverlap=2 * FS, axis=-1)
    total = _band_power(f, psd, 0.5, 30) + EPS
    bp = {b: _band_power(f, psd, lo, hi) + EPS for b, (lo, hi) in EEG_BANDS.items()}
    for b, v in bp.items():
        out[f"{p}_abs_{b}"] = np.log10(v)
        out[f"{p}_rel_{b}"] = v / total
    out[f"{p}_abs_total"] = np.log10(total)
    out[f"{p}_ratio_delta_theta"] = np.log10(bp["delta"] / bp["theta"])
    out[f"{p}_ratio_delta_beta"] = np.log10(bp["delta"] / bp["beta"])
    out[f"{p}_ratio_theta_alpha"] = np.log10(bp["theta"] / bp["alpha"])
    out[f"{p}_ratio_slow_fast"] = np.log10((bp["delta"] + bp["theta"]) / (bp["alpha"] + bp["beta"]))

    band = (f >= 0.5) & (f < 30)
    fb, pb = f[band], psd[:, band] + EPS
    cum = np.cumsum(pb, 1) / pb.sum(1, keepdims=True)
    out[f"{p}_sef95"] = fb[np.argmax(cum >= 0.95, axis=1)]
    out[f"{p}_peak_freq"] = fb[np.argmax(pb, 1)]
    pn = pb / pb.sum(1, keepdims=True)
    out[f"{p}_spec_entropy"] = -(pn * np.log2(pn)).sum(1) / np.log2(pn.shape[1])

    out[f"{p}_std"] = x.std(1)
    q75, q25 = np.percentile(x, [75, 25], axis=1)
    out[f"{p}_iqr"] = q75 - q25
    out[f"{p}_skew"] = stats.skew(x, axis=1)
    out[f"{p}_kurt"] = stats.kurtosis(x, axis=1)
    mob, comp = ant.hjorth_params(x, axis=1)
    out[f"{p}_hjorth_mob"], out[f"{p}_hjorth_comp"] = mob, comp
    out[f"{p}_zcr"] = ant.num_zerocross(x, normalize=True, axis=1)
    out[f"{p}_line_length"] = np.abs(np.diff(x, axis=1)).mean(1)
    out[f"{p}_perm_entropy"] = np.array([ant.perm_entropy(v, order=3, normalize=True) for v in x])
    out[f"{p}_higuchi"] = np.array([ant.higuchi_fd(v.astype(np.float64)) for v in x])

    # 6 sub-ventanas de 5 s
    sub = x.reshape(len(x), 6, 5 * FS)
    fs_, psub = signal.welch(sub, fs=FS, nperseg=250, noverlap=125, axis=-1)
    for b, (lo, hi) in [("sigma", (12, 16)), ("delta", (0.5, 4))]:
        idx = (fs_ >= lo) & (fs_ < hi)
        lp = np.log10(psub[..., idx].sum(-1) * (fs_[1] - fs_[0]) + EPS)
        out[f"{p}_sub_{b}_max"] = lp.max(1)
        out[f"{p}_sub_{b}_std"] = lp.std(1)
    ptp = np.ptp(sub, axis=-1)
    out[f"{p}_sub_ptp_max"] = ptp.max(1)
    out[f"{p}_sub_ptp_std"] = ptp.std(1)
    return out


def eog_features(x, fpz=None):
    out = {}
    f, psd = signal.welch(x, fs=FS, nperseg=4 * FS, noverlap=2 * FS, axis=-1)
    out["eog_abs_slow"] = np.log10(_band_power(f, psd, 0.3, 2) + EPS)
    out["eog_abs_rapid"] = np.log10(_band_power(f, psd, 2, 5) + EPS)
    out["eog_abs_total"] = np.log10(_band_power(f, psd, 0.3, 30) + EPS)
    out["eog_std"] = x.std(1)
    out["eog_line_length"] = np.abs(np.diff(x, axis=1)).mean(1)
    out["eog_max_abs"] = np.abs(x).max(1)
    out["eog_kurt"] = stats.kurtosis(x, axis=1)
    if fpz is not None:
        xc = x - x.mean(1, keepdims=True)
        fc = fpz - fpz.mean(1, keepdims=True)
        out["eog_corr_fpz"] = (xc * fc).sum(1) / (np.sqrt((xc**2).sum(1) * (fc**2).sum(1)) + EPS)
    return out


def emg_features(v):
    """v: (n_épocas, 30) valores a 1 Hz de la envolvente del EMG, en µV."""
    return {"emg_mean": v.mean(1), "emg_median": np.median(v, 1), "emg_std": v.std(1),
            "emg_min": v.min(1), "emg_max": v.max(1), "emg_log_mean": np.log10(np.abs(v.mean(1)) + EPS)}


def dwt_features(x, p):
    """Ondícula biortogonal 2.2 en 4 niveles: 8 medidas por cada una de las 5 sub-bandas."""
    out = {}
    coeffs = pywt.wavedec(x, "bior2.2", level=4, axis=-1)
    for band, c in zip(DWT_BANDS, coeffs):
        q = f"{p}_dwt_{band}"
        out[f"{q}_logenergy"] = np.log10((c**2).mean(1) + EPS)
        out[f"{q}_teager"] = np.log10(np.abs(c[:, 1:-1] ** 2 - c[:, :-2] * c[:, 2:]).mean(1) + EPS)
        mob, comp = ant.hjorth_params(c, axis=1)
        out[f"{q}_hjorth_mob"], out[f"{q}_hjorth_comp"] = mob, comp
        out[f"{q}_kurt"] = stats.kurtosis(c, axis=1)
        out[f"{q}_skew"] = stats.skew(c, axis=1)
        out[f"{q}_katz"] = np.array([ant.katz_fd(v) for v in c])
        out[f"{q}_petrosian"] = np.array([ant.petrosian_fd(v) for v in c])
    return out


def aperiodic_features(x, p):
    """Pendiente 1/f: recta en log-log entre 2 y 30 Hz, sin 7.5-16 Hz (alfa y husos)."""
    f, psd = signal.welch(x, fs=FS, nperseg=4 * FS, noverlap=2 * FS, axis=-1)
    m = (f >= 2) & (f <= 30) & ~((f > 7.5) & (f < 16))
    lx = np.log10(f[m])
    ly = np.log10(psd[:, m] + EPS)
    A = np.vstack([lx, np.ones_like(lx)]).T
    coef, *_ = np.linalg.lstsq(A, ly.T, rcond=None)  # una recta por época
    pred = (A @ coef).T
    ss_res = ((ly - pred) ** 2).sum(1)
    ss_tot = ((ly - ly.mean(1, keepdims=True)) ** 2).sum(1) + EPS
    return {f"{p}_ap_slope": coef[0], f"{p}_ap_offset": coef[1], f"{p}_ap_r2": 1 - ss_res / ss_tot}


def add_context(df, rolls=(5, 15), lags=(-2, -1, 1, 2)):
    """z-score robusto de la noche, medias móviles de 5 y 15 épocas y épocas t±1, t±2 (una sola noche, en orden)."""
    base = [c for c in df.columns if not c.startswith("time_")]
    med = df[base].median()
    iqr = (df[base].quantile(0.75) - df[base].quantile(0.25)).replace(0, 1.0)
    z = ((df[base] - med) / iqr).astype(np.float32)
    z.columns = [f"{c}_z" for c in base]

    keys = [f"{c}_z" for c in base if any(c.endswith("_" + s) for s in KEY_SUFFIXES)]
    parts = [df, z]
    for w in rolls:
        r = z[keys].rolling(w, center=True, min_periods=1).mean()
        r.columns = [f"{c}_roll{w}" for c in keys]
        parts.append(r.astype(np.float32))
    for k in lags:
        s = z[keys].shift(-k)  # lag +1 = la época siguiente
        s.columns = [f"{c}_lag{'p' if k > 0 else 'm'}{abs(k)}" for c in keys]
        parts.append(s.astype(np.float32))
    return pd.concat(parts, axis=1)


def rasgos_noche(noche):
    """Tabla (épocas analizadas × rasgos) de los canales presentes, con contexto."""
    cut = lambda s: (s[: noche.n_registro * SPE] * 1e6).reshape(noche.n_registro, SPE)[noche.epocas]  # noqa: E731
    ep = {m: cut(noche.senales[m]) for m in noche.canales if m != "emg"}
    feats = {}
    for m in ("fpz", "pz"):
        if m in ep:
            feats.update(eeg_features(ep[m], m))
    if "eog" in ep:
        feats.update(eog_features(ep["eog"], ep.get("fpz")))
    if "emg" in noche.canales:
        # MNE interpola el EMG (1 Hz) a 100 Hz: cada 100 muestras está el valor original
        emg = (noche.senales["emg"] * 1e6)[: noche.n_registro * SPE: FS].reshape(noche.n_registro, EPOCH)[noche.epocas]
        feats.update(emg_features(emg))
    df = pd.DataFrame(feats).astype(np.float32)
    df.insert(0, "time_hours_since_lights_off", ((noche.epocas * EPOCH - noche.lights_off) / 3600).astype(np.float32))

    dwt = {}
    for m in ("fpz", "pz", "eog"):
        if m in ep:
            dwt.update(dwt_features(ep[m], m))
    for m in ("fpz", "pz"):
        if m in ep:
            dwt.update(aperiodic_features(ep[m], m))
    df = pd.concat([df, pd.DataFrame(dwt).astype(np.float32)], axis=1)
    return add_context(df)


def channel_index(cols):
    """canal -> posiciones de las columnas que dependen de él (eog_corr_fpz depende también de Fpz-Cz)."""
    ch = np.array([c.split("_")[0] for c in cols])
    idx = {c: set(np.where(ch == c)[0]) for c in MODALIDADES}
    for i, col in enumerate(cols):
        for key, other in CROSS.items():
            if key in col:
                idx[other].add(i)
    return {c: np.array(sorted(v), dtype=int) for c, v in idx.items()}


def apply_mask(X, configs_per_row, idx):
    """Pone en NaN las columnas de los canales ausentes de cada fila. Modifica X."""
    names = np.asarray(configs_per_row)
    for cfg, chans in COMBINACIONES.items():
        r = np.where(names == cfg)[0]
        if len(r) == 0:
            continue
        for ch in MODALIDADES:
            if ch not in chans:
                X[np.ix_(r, idx[ch])] = np.nan
    return X


def random_mask(X, idx, rng):
    """A cada fila le sortea una de las 12 combinaciones y apaga los canales que faltan. Modifica X."""
    return apply_mask(X, rng.choice(list(COMBINACIONES), size=len(X)), idx)
