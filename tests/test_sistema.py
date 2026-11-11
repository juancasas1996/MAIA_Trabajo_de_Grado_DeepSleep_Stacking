"""Pruebas del sistema con los modelos de modelo/ (en un PR, los del candidato). No necesitan datos: usan una noche
inventada. Corren antes del E2E: si alguna falla, no se despliega nada.
"""
import json

import joblib
import mne
import numpy as np
import pytest
import xgboost as xgb
from conftest import noche_falsa

from sistema.lectura import hypnogram_labels, sleep_window
from sistema.modelo_a import rasgos_noche
from sistema.prediccion import MODELO, carpeta_modelo_a, probabilidades


def test_hipnograma_a_etiquetas(tmp_path):
    """Cada anotación pasa a su estadio; N3 y N4 se juntan en N3; lo que no es estadio queda en -1."""
    ann = mne.Annotations(onset=[0, 60, 90, 120, 150], duration=[60, 30, 30, 30, 30],
                          description=["Sleep stage W", "Sleep stage 3", "Sleep stage 4", "Sleep stage R", "Movement time"])
    ruta = tmp_path / "hipnograma.csv"
    ann.save(ruta)
    assert hypnogram_labels(ruta, 7).tolist() == [0, 0, 3, 3, 4, -1, -1]


def test_ventana_de_sueno():
    """30 min (60 épocas) de vigilia antes del primer sueño y después del último, sin salirse del registro."""
    etiquetas = np.zeros(400, dtype=int)
    etiquetas[100:200] = 2
    assert sleep_window(etiquetas) == (40, 259)
    assert sleep_window(etiquetas[80:220]) == (0, 139)


def test_estan_todas_las_piezas():
    """Los tres modelos y sistema.json están y son coherentes entre sí."""
    cols_a = json.loads((carpeta_modelo_a() / "columnas.json").read_text())
    a = xgb.Booster()
    a.load_model(carpeta_modelo_a() / "xgb.json")
    assert a.num_features() == len(cols_a)
    assert (MODELO / "modelo_b_red/modelo_final.pt").is_file()
    assert len(json.loads((MODELO / "meta_modelo/columnas.json").read_text())) == 54
    assert joblib.load(MODELO / "meta_modelo/meta_lr.joblib").n_features_in_ == 54
    s = json.loads((MODELO / "sistema.json").read_text())
    assert 0.5 <= s["umbral_confianza"] <= 0.95
    assert 0 < s["aviso_noche_fraccion_revisar"] < 1


def test_modelo_a_coincide_con_los_rasgos():
    """El XGBoost usa columnas que calcula el sistema con los 4 canales (todas o una parte), en el mismo orden, y
    calcular solo esas da los mismos valores que calcular todas."""
    cols = json.loads((carpeta_modelo_a() / "columnas.json").read_text())
    noche = noche_falsa()
    todas = rasgos_noche(noche)
    assert [c for c in todas.columns if c in set(cols)] == cols
    assert np.array_equal(rasgos_noche(noche, cols).to_numpy(), todas[cols].to_numpy(), equal_nan=True)


@pytest.mark.parametrize("canales", [("fpz",), ("pz", "emg"), ("fpz", "pz", "eog", "emg")])
def test_el_sistema_completo_predice(sistema, canales):
    """Las 3 piezas juntas: 5 probabilidades por época, que suman 1, en A, B y el meta-modelo."""
    for p in probabilidades(sistema, noche_falsa(canales, n=100)):
        assert p.shape == (100, 5)
        assert np.allclose(p.sum(1), 1, atol=1e-3)


def test_un_canal_ausente_no_cuenta(sistema):
    """Un canal que no se pidió no cambia nada, aunque su señal esté en la noche."""
    sola = noche_falsa(("fpz", "eog"), n=80, seed=1)
    con_otros = noche_falsa(("fpz", "eog"), n=80, seed=1)
    con_otros.senales |= {m: x for m, x in noche_falsa(n=80, seed=2).senales.items() if m in ("pz", "emg")}
    assert np.array_equal(probabilidades(sistema, sola)[2], probabilidades(sistema, con_otros)[2])
