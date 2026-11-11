"""Meta-modelo: combina A y B con las épocas vecinas (experimentacion/meta_modelo/1_entrenamiento.ipynb).

54 entradas por época: log-probabilidades de A y B en t-2..t+2 (2 × 5 × 5 = 50) y la máscara de canales (4).
Adentro, regresión logística + XGBoost pequeño, promediados.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from .lectura import K, MODALIDADES

EPS = 1e-6
CONTEXT = (-2, -1, 0, 1, 2)


def fill_window(validas, p_valid):
    """Ubica las predicciones de B en todas las épocas analizadas: las que la red no vio toman la de la época anterior
    (o la de la primera vista, si van al inicio)."""
    idx = np.where(validas, np.arange(len(validas)), -1)
    idx = np.maximum.accumulate(idx)
    idx[idx < 0] = np.argmax(validas)
    full = np.empty((len(validas), p_valid.shape[1]), dtype=np.float32)
    full[validas] = p_valid
    full[~validas] = full[idx[~validas]]
    return full


def entradas(pa, pb, mask):
    """54 entradas de una noche. pa, pb: (épocas, 5) probabilidades de A y B; mask: [fpz, pz, eog, emg].

    En los bordes de la noche, las vecinas que no existen valen log(1/5).
    """
    cols = {}
    for nombre, prob in (("xgb", pa), ("b", pb)):
        lp = np.log(np.clip(prob, EPS, 1))
        for o in CONTEXT:
            shifted = np.full_like(lp, np.log(1 / K))
            if o == 0:
                shifted = lp
            elif o > 0:
                shifted[:-o] = lp[o:]
            else:
                shifted[-o:] = lp[:o]
            for k in range(K):
                cols[f"{nombre}_k{k}_o{o:+d}"] = shifted[:, k]
    df = pd.DataFrame(cols).astype(np.float32)
    for m, on in zip(MODALIDADES, mask):
        df[f"mask_{m}"] = np.float32(on)
    return df


def cargar(directorio):
    """Regresión logística, XGBoost y las 54 columnas de entrada de modelo/meta_modelo."""
    d = Path(directorio)
    lr = joblib.load(d / "meta_lr.joblib")
    m = xgb.XGBClassifier()
    m.load_model(d / "meta_xgb.json")   # predict_proba usa los árboles hasta la mejor ronda
    return lr, m, json.loads((d / "columnas.json").read_text())


def predecir(lr, m, columnas, pa, pb, mask):
    """Probabilidades (épocas, 5): promedio de la regresión logística y el XGBoost."""
    X = entradas(pa, pb, mask)[columnas]
    return np.mean([lr.predict_proba(X), m.predict_proba(X)], axis=0).astype(np.float32)
