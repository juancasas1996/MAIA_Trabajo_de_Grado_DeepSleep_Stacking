"""El modelo de modelos/ (el candidato, en un PR) está completo, carga y es coherente con el código.

Corre antes del E2E: si algo falla aquí, no se despliega nada.
"""
import json

import joblib
import numpy as np
import pytest
import xgboost as xgb
from conftest import noche_falsa

from sueno.config import MODELOS
from sueno.modelo_a import mas_importantes
from sueno.rasgos import rasgos_noche
from sueno.sistema import cargar_sistema

ARCHIVOS = ["modelo_a_xgboost/xgb.json", "modelo_a_xgboost/columnas.json", "modelo_b/modelo_final.pt",
            "meta_modelo/meta_lr.joblib", "meta_modelo/meta_xgb.json", "meta_modelo/columnas.json", "sistema.json"]


@pytest.mark.parametrize("archivo", ARCHIVOS)
def test_estan_todas_las_piezas(archivo):
    assert (MODELOS / archivo).is_file()


def test_modelo_a_coincide_con_los_rasgos():
    """El XGBoost usa columnas que calcula rasgos_noche con los 4 canales (todas o una parte), en el mismo orden."""
    cols = json.loads((MODELOS / "modelo_a_xgboost/columnas.json").read_text())
    b = xgb.Booster()
    b.load_model(MODELOS / "modelo_a_xgboost/xgb.json")
    assert b.num_features() == len(cols)
    todas = list(rasgos_noche(noche_falsa()).columns)
    assert [c for c in todas if c in set(cols)] == cols


def test_elegir_las_columnas_mas_importantes():
    """Elegir n columnas da n, en el orden del modelo; elegir de nuevo sobre esas mismas no cambia nada."""
    cols = json.loads((MODELOS / "modelo_a_xgboost/columnas.json").read_text())
    n = len(cols) // 2
    elegidas = mas_importantes(MODELOS / "modelo_a_xgboost", n)
    assert len(elegidas) == n
    assert [c for c in cols if c in set(elegidas)] == elegidas
    assert mas_importantes(MODELOS / "modelo_a_xgboost", len(cols) + 10) == cols


def test_meta_modelo_54_entradas():
    cols = json.loads((MODELOS / "meta_modelo/columnas.json").read_text())
    assert len(cols) == 54
    assert joblib.load(MODELOS / "meta_modelo/meta_lr.joblib").n_features_in_ == 54
    b = xgb.Booster()
    b.load_model(MODELOS / "meta_modelo/meta_xgb.json")
    assert b.num_features() == 54
    assert b.attr("best_iteration") is not None   # se predice hasta la mejor ronda


def test_sistema_json():
    s = json.loads((MODELOS / "sistema.json").read_text())
    assert 0.5 <= s["umbral_confianza"] <= 0.95
    assert 0 < s["aviso_noche_fraccion_revisar"] < 1


@pytest.mark.parametrize("canales", [("fpz",), ("pz", "emg"), ("fpz", "pz", "eog", "emg")])
def test_el_sistema_completo_predice(canales):
    """Las 3 piezas juntas, con una noche inventada: un estadio y 5 probabilidades por época."""
    sistema = cargar_sistema()
    noche = noche_falsa(canales, n=100)
    salida = sistema[1:].predict(noche)
    p = np.array(salida["probabilidades"])
    assert p.shape == (100, 5)
    assert np.allclose(p.sum(1), 1, atol=1e-3)
    assert len(salida["estadio"]) == len(salida["revisar"]) == 100
