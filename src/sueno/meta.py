"""Meta-modelo: combina A y B con las épocas vecinas. Uno solo para las 12 combinaciones (recibe la máscara).

54 entradas por época: log-probabilidades de A y B en t-2..t+2 (2 × 5 × 5 = 50) y la máscara de canales (4).
Adentro, regresión logística + XGBoost pequeño, promediados.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from .config import COMBINACIONES, K, MODALIDADES, SEED, mascara

EPS = 1e-6
CONTEXT = (-2, -1, 0, 1, 2)
ALPHA = 0.44
XGB_PARAMS = dict(objective="multi:softprob", tree_method="hist", eval_metric="mlogloss", learning_rate=0.05,
                  max_depth=4, subsample=0.8, colsample_bytree=0.8, min_child_weight=10, n_jobs=-1)


def fill_window(validas, p_valid):
    """Ubica las predicciones de B en toda la ventana: las épocas que la red no vio toman la de la época anterior
    (o la de la primera vista, si van al inicio)."""
    idx = np.where(validas, np.arange(len(validas)), -1)
    idx = np.maximum.accumulate(idx)
    idx[idx < 0] = np.argmax(validas)
    full = np.empty((len(validas), p_valid.shape[1]), dtype=np.float32)
    full[validas] = p_valid
    full[~validas] = full[idx[~validas]]
    return full


def entradas(pa, pb, mask):
    """54 entradas de una noche. pa, pb: (épocas, 5); mask: [fpz, pz, eog, emg]. En los bordes, log(1/5)."""
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


def tabla(pred_a, pred_b):
    """Entradas de todas las combinaciones y noches. pred_*: combinación -> noche -> (y, p) en la misma ventana."""
    frames = []
    for cfg in COMBINACIONES:
        for n in sorted(pred_a[cfg]):
            y, pa = pred_a[cfg][n]
            d = entradas(pa, pred_b[cfg][n][1], mascara(cfg))
            d.insert(0, "y", y)
            d.insert(0, "night", n)
            d.insert(0, "config", cfg)
            frames.append(d)
    return pd.concat(frames, ignore_index=True)


def columnas(df):
    return [c for c in df.columns if c not in ("config", "night", "y")]


def pesos(y, alpha=ALPHA):
    return compute_sample_weight("balanced", y) ** alpha


class MetaModelo:
    """Regresión logística + XGBoost, promediados."""

    def fit(self, X, y, X_val, y_val):
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, random_state=SEED))
        Xt, yt = pd.concat([X, X_val]), np.concatenate([y, y_val])   # la LR no necesita validación: usa todo
        lr.fit(Xt, yt, logisticregression__sample_weight=pesos(yt))
        m = xgb.XGBClassifier(**XGB_PARAMS, n_estimators=1000, early_stopping_rounds=50, random_state=SEED)
        m.fit(X, y, sample_weight=pesos(y), eval_set=[(X_val, y_val)], sample_weight_eval_set=[pesos(y_val)], verbose=False)
        self.lr, self.xgb = lr, m
        return self

    def predict_proba(self, X):
        return np.mean([self.lr.predict_proba(X), self.xgb.predict_proba(X)], axis=0)


def entrenar(df, cols, sujetos_train, sujetos_val):
    sujeto = df.night.str[3:5].astype(int)
    lab = df.y.values >= 0
    tr = lab & sujeto.isin(sujetos_train).values
    va = lab & sujeto.isin(sujetos_val).values
    return MetaModelo().fit(df.loc[tr, cols], df.y.values[tr], df.loc[va, cols], df.y.values[va])


def por_combinacion(df, proba):
    """Probabilidades de la tabla -> combinación -> noche -> (y, p)."""
    out = {}
    for (cfg, n), i in df.groupby(["config", "night"]).indices.items():
        out.setdefault(cfg, {})[n] = (df.y.values[i], proba[i].astype(np.float32))
    return out


def guardar(meta, cols, directorio):
    d = Path(directorio)
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(meta.lr, d / "meta_lr.joblib")
    meta.xgb.save_model(d / "meta_xgb.json")   # guarda también best_iteration
    (d / "columnas.json").write_text(json.dumps(cols))
