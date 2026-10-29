"""Modelo A: rasgos por época + un solo XGBoost con canales enmascarados al azar (sirve para las 12 combinaciones)."""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.utils.class_weight import compute_sample_weight

from .config import COMBINACIONES, SEED, sujeto
from .lectura import leer_noche
from .rasgos import apply_mask, channel_index, random_mask, rasgos_noche

# Hiperparámetros de una búsqueda con optuna en una etapa anterior del proyecto (no se vuelven a buscar)
PARAMS = {
    "objective": "multi:softprob", "tree_method": "hist", "eval_metric": "mlogloss", "n_jobs": -1,
    "learning_rate": 0.03394046833983564, "max_depth": 10, "min_child_weight": 43.70990468130503,
    "subsample": 0.9041986740582306, "colsample_bytree": 0.4436910153386966,
    "reg_lambda": 0.019634341572933336, "gamma": 3.4211651325607844,
}
ALPHA = 0.4401524937396013   # pesos por clase = balanceado^α
N_MAX = 2000
EARLY_STOPPING = 100
META_COLS = ["night", "subject", "epoch_idx", "y"]


def _tabla_noche(job):
    nid, psg, hyp, hora = job
    warnings.filterwarnings("ignore")
    noche = leer_noche({"psg": psg, "hipnograma": hyp, "hora_apagado": hora})
    df = rasgos_noche(noche)
    df.insert(0, "y", noche.etiquetas)
    df.insert(0, "epoch_idx", noche.epocas)
    df.insert(0, "subject", sujeto(nid))
    df.insert(0, "night", nid)
    return df


def tabla(noches, horas, n_procesos=8):
    """Rasgos de todas las noches (ventana de sueño completa, también las épocas sin etiqueta).

    noches: [(noche, psg, hipnograma)]; horas: noche -> hora de apagado de la luz.
    """
    jobs = [(n, p, h, horas[n]) for n, p, h in noches]
    return pd.concat(Parallel(n_jobs=n_procesos)(delayed(_tabla_noche)(j) for j in jobs), ignore_index=True)


def columnas(df):
    return [c for c in df.columns if c not in META_COLS]


def pesos(y, alpha=ALPHA):
    return compute_sample_weight("balanced", y) ** alpha


def _filas(df, sujetos, solo_etiquetadas=True):
    m = df.subject.isin(list(sujetos))
    return df[m & (df.y >= 0)] if solo_etiquetadas else df[m]


def entrenar(df, cols, rng, sujetos_train, sujetos_val=None, n_estimators=N_MAX):
    """Con sujetos_val, early stopping sobre ellos (también enmascarados); sin ellos, n_estimators rondas fijas."""
    idx = channel_index(cols)
    tr = _filas(df, sujetos_train)
    p = {**PARAMS, "random_state": SEED}
    Xtr = random_mask(tr[cols].to_numpy(np.float32, copy=True), idx, rng)
    if sujetos_val is None:
        modelo = xgb.XGBClassifier(**p, n_estimators=n_estimators)
        modelo.fit(Xtr, tr.y, sample_weight=pesos(tr.y), verbose=False)
        return modelo
    va = _filas(df, sujetos_val)
    Xva = random_mask(va[cols].to_numpy(np.float32, copy=True), idx, rng)
    modelo = xgb.XGBClassifier(**p, n_estimators=n_estimators, early_stopping_rounds=EARLY_STOPPING)
    modelo.fit(Xtr, tr.y, sample_weight=pesos(tr.y), eval_set=[(Xva, va.y)],
               sample_weight_eval_set=[pesos(va.y)], verbose=False)
    return modelo


def predecir_combinaciones(modelo, df, cols, sujetos):
    """combinación -> noche -> (y, probabilidades), con todas las épocas de la ventana."""
    idx = channel_index(cols)
    d = _filas(df, sujetos, solo_etiquetadas=False)
    base_x = d[cols].to_numpy(np.float32)
    grupos = d.groupby("night").indices
    out = {}
    for cfg in COMBINACIONES:
        proba = modelo.predict_proba(apply_mask(base_x.copy(), [cfg] * len(base_x), idx))
        out[cfg] = {n: (d.y.values[i], proba[i].astype(np.float32)) for n, i in grupos.items()}
    return out


def guardar(modelo, cols, directorio):
    d = Path(directorio)
    d.mkdir(parents=True, exist_ok=True)
    modelo.save_model(d / "xgb.json")
    (d / "columnas.json").write_text(json.dumps(cols))
