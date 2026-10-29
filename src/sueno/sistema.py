"""El sistema para predecir: una noche en EDF -> estadio de cada época de 30 s, con su confianza.

    sistema = make_pipeline(
        LeerNoche(),                                    EDF + canales presentes -> épocas
        make_union(ModeloA(), ModeloB(), Mascara()),    5 + 5 + 4 = 14 columnas por época
        MetaModelo(),                                   contexto t±2 -> 54 entradas -> LR + XGBoost
        Confianza(),                                    estadio, 2.ª opción y "revisar"
    )

Los modelos ya están entrenados (scripts/entrenamiento_completo.py): fit() no hace nada y cada paso carga su modelo
desde modelos/.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, make_union

from . import meta, modelo_b
from .config import EPOCH, ESTADIOS, K, MODALIDADES, MODELOS
from .lectura import leer_noche
from .rasgos import channel_index, rasgos_noche


class _Inferencia(BaseEstimator):
    """Los pasos ya vienen entrenados: fit no hace nada."""

    def fit(self, X, y=None):
        return self

    def __sklearn_is_fitted__(self):
        return True


class LeerNoche(_Inferencia, TransformerMixin):
    """dict de entrada (ver lectura.leer_noche) -> Noche."""

    def transform(self, X):
        return leer_noche(X)


class ModeloA(_Inferencia, TransformerMixin):
    """Rasgos + XGBoost enmascarado: (épocas, 5) probabilidades."""

    def __init__(self, directorio=MODELOS / "modelo_a_xgboost"):
        self.directorio = directorio

    def _cargar(self):
        if not hasattr(self, "booster_"):
            d = Path(self.directorio)
            self.columnas_ = json.loads((d / "columnas.json").read_text())
            self.idx_ = channel_index(self.columnas_)
            self.booster_ = xgb.Booster()
            self.booster_.load_model(d / "xgb.json")

    def transform(self, noche):
        self._cargar()
        X = rasgos_noche(noche).reindex(columns=self.columnas_).to_numpy(np.float32)  # canales ausentes -> NaN
        for m in MODALIDADES:
            if m not in noche.canales:
                X[:, self.idx_[m]] = np.nan
        return self.booster_.predict(xgb.DMatrix(X, missing=np.nan))


class ModeloB(_Inferencia, TransformerMixin):
    """Red sobre la señal cruda: (épocas, 5) probabilidades."""

    def __init__(self, ruta=MODELOS / "modelo_b" / "modelo_final.pt"):
        self.ruta = ruta

    def transform(self, noche):
        if not hasattr(self, "red_"):
            self.red_, self.hp_ = modelo_b.cargar(self.ruta)
        p = modelo_b.predecir(self.red_, modelo_b.preparar(noche), noche.mascara, self.hp_)
        return meta.fill_window(noche.validas, p)


class Mascara(_Inferencia, TransformerMixin):
    """Canales presentes [fpz, pz, eog, emg], repetidos en cada época: (épocas, 4)."""

    def transform(self, noche):
        return np.tile(np.array(noche.mascara, dtype=np.float32), (len(noche.epocas), 1))


class MetaModelo(_Inferencia, TransformerMixin):
    """Recibe las 14 columnas de la unión (A, B, máscara) y devuelve (épocas, 5): promedio de LR y XGBoost."""

    def __init__(self, directorio=MODELOS / "meta_modelo"):
        self.directorio = directorio

    def _cargar(self):
        if not hasattr(self, "lr_"):
            d = Path(self.directorio)
            self.columnas_ = json.loads((d / "columnas.json").read_text())
            self.lr_ = joblib.load(d / "meta_lr.joblib")
            self.xgb_ = xgb.Booster()
            self.xgb_.load_model(d / "meta_xgb.json")
            # Se entrenó con early stopping: predice con los árboles hasta la mejor ronda, como predict_proba
            self.rondas_ = int(self.xgb_.attr("best_iteration")) + 1

    def transform(self, Z):
        self._cargar()
        X = meta.entradas(Z[:, :K], Z[:, K:2 * K], Z[0, 2 * K:])[self.columnas_]
        p_lr = self.lr_.predict_proba(X)
        p_xgb = self.xgb_.predict(xgb.DMatrix(X), iteration_range=(0, self.rondas_))
        return np.mean([p_lr, p_xgb], axis=0).astype(np.float32)


class Confianza(_Inferencia):
    """Estadio de cada época, su confianza, la 2.ª opción y si hay que revisarla; aviso si la noche es dudosa."""

    def __init__(self, umbral=0.65, aviso=0.2732):
        self.umbral = umbral
        self.aviso = aviso

    def predict(self, P):
        o = np.argsort(-P, 1)
        i = np.arange(len(P))
        conf = P[i, o[:, 0]]
        revisar = conf < self.umbral
        return {"estadio": [ESTADIOS[k] for k in o[:, 0]], "confianza": conf.round(4).tolist(),
                "segunda_opcion": [ESTADIOS[k] for k in o[:, 1]], "prob_segunda": P[i, o[:, 1]].round(4).tolist(),
                "revisar": revisar.tolist(), "fraccion_revisar": float(revisar.mean()),
                "aviso_noche": bool(revisar.mean() > self.aviso), "probabilidades": P.round(4).tolist()}


def cargar_sistema(modelos=MODELOS):
    modelos = Path(modelos)
    cfg = json.loads((modelos / "sistema.json").read_text())
    return make_pipeline(
        LeerNoche(),
        make_union(ModeloA(modelos / "modelo_a_xgboost"), ModeloB(modelos / "modelo_b" / "modelo_final.pt"), Mascara()),
        MetaModelo(modelos / "meta_modelo"),
        Confianza(cfg["umbral_confianza"], cfg["aviso_noche_fraccion_revisar"]),
    )


def analizar(sistema, entrada):
    """Corre el pipeline y agrega qué épocas del registro se analizaron, con qué canales y cuánto tardó.

    segundos_proceso cronometra solo el procesamiento (EDF -> hipnograma), sin la red ni la cola del endpoint.
    """
    t0 = time.perf_counter()
    noche = sistema[0].transform(entrada)
    salida = sistema[1:].predict(noche)
    return {"noche": noche.nombre, "canales": list(noche.canales), "epoca": noche.epocas.tolist(),
            "segundos_desde_inicio": (noche.epocas * EPOCH).tolist(), **salida,
            "segundos_proceso": round(time.perf_counter() - t0, 3)}
