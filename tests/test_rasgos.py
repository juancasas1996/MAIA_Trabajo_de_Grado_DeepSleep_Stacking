"""Rasgos del modelo A y enmascaramiento de canales."""
import numpy as np
import pandas as pd
from conftest import noche_falsa

from sueno.config import FS, SPE
from sueno.rasgos import add_context, apply_mask, channel_index, eeg_features, rasgos_noche


def test_alfa_domina_con_un_seno_de_10_hz():
    t = np.arange(SPE) / FS
    x = (50 * np.sin(2 * np.pi * 10 * t))[None]
    f = eeg_features(x, "fpz")
    bandas = {b: f[f"fpz_abs_{b}"][0] for b in ("sdelta", "delta", "theta", "alpha", "sigma", "beta")}
    assert max(bandas, key=bandas.get) == "alpha"
    assert abs(f["fpz_peak_freq"][0] - 10) < 0.5


def test_una_fila_por_epoca_y_sin_faltantes(noche):
    df = rasgos_noche(noche)
    assert len(df) == len(noche.epocas)
    sin_vecinas = [c for c in df.columns if "_lag" not in c]   # las vecinas faltan solo en los bordes
    assert not df[sin_vecinas].isna().any().any()


def test_canal_ausente_no_genera_columnas():
    """Sin Fpz-Cz no hay columnas fpz_, ni eog_corr_fpz (que lo necesita)."""
    cols = rasgos_noche(noche_falsa(("pz", "eog"))).columns
    assert not [c for c in cols if c.startswith(("fpz_", "emg_"))]
    assert not [c for c in cols if "corr_fpz" in c]
    assert [c for c in cols if c.startswith("pz_")] and [c for c in cols if c.startswith("eog_")]


def test_contexto_dentro_de_la_noche():
    """z-score robusto (mediana e IQR) y la época siguiente (lagp1); la última no tiene vecina."""
    df = pd.DataFrame({"fpz_abs_delta": [1.0, 2.0, 3.0]})
    out = add_context(df)
    assert np.allclose(out["fpz_abs_delta_z"], [-1, 0, 1])
    assert np.allclose(out["fpz_abs_delta_z_lagp1"][:2], [0, 1])
    assert np.isnan(out["fpz_abs_delta_z_lagp1"][2])


def test_mascara_apaga_canales_ausentes():
    """Con solo Pz-Oz, las columnas de Fpz-Cz, EOG y EMG quedan en NaN, también la del EOG que usa Fpz-Cz."""
    cols = ["fpz_abs_delta", "pz_abs_delta", "eog_rms", "eog_corr_fpz", "emg_rms"]
    idx = channel_index(cols)
    assert 3 in idx["fpz"]
    X = apply_mask(np.ones((2, len(cols)), dtype=np.float32), ["pz", "eeg2_eog_emg"], idx)
    assert np.isnan(X[0]).tolist() == [True, False, True, True, True]
    assert not np.isnan(X[1]).any()
