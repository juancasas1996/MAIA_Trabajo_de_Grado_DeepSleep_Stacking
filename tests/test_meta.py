"""Entradas del meta-modelo, relleno de épocas y confianza."""
import numpy as np

from sueno.config import mascara
from sueno.evaluacion import elegir_umbral, metricas_confusion, resumen_confianza
from sueno.meta import entradas, fill_window


def test_rellenar_epocas_no_vistas():
    """Las épocas que la red no vio toman la predicción de la anterior, o la de la primera si van al inicio."""
    validas = np.array([False, True, False, False, True])
    p = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=np.float32)
    assert fill_window(validas, p).argmax(1).tolist() == [0, 0, 0, 0, 1]


def test_54_entradas_con_vecinas():
    """2 modelos × 5 épocas (t-2..t+2) × 5 estadios + 4 de máscara; fuera de la noche, probabilidad uniforme."""
    p = np.eye(5, dtype=np.float32)[[0, 1, 2]] * 0.96 + 0.01
    df = entradas(p, p, mascara("fpz"))
    assert df.shape == (3, 54)
    assert np.isclose(df["xgb_k1_o+1"][0], np.log(0.97))   # la época siguiente a la 0 es N1
    assert np.isclose(df["xgb_k0_o-1"][0], np.log(1 / 5))  # la 0 no tiene anterior
    assert df[["mask_fpz", "mask_pz", "mask_eog", "mask_emg"]].iloc[0].tolist() == [1, 0, 0, 0]


def test_metricas_de_una_confusion_perfecta():
    m = metricas_confusion(np.diag([10, 5, 20, 8, 7]))
    assert m["acc"] == m["mf1"] == m["kappa"] == 1


def test_confianza_manda_a_revisar_lo_dudoso():
    """Con umbral 0.65: la época con 0.9 se entrega y la de 0.5 va a revisar con sus 2 opciones."""
    y = np.array([0, 1])
    p = np.array([[0.9, 0.1, 0, 0, 0], [0.5, 0.4, 0.1, 0, 0]])
    r = resumen_confianza([(y, p)], 0.65)
    assert r["cobertura"] == 0.5 and r["acc_seguras"] == 1
    assert r["acc_dudosas"] == 0 and r["top2_dudosas"] == 1


def test_umbral_es_el_menor_que_cumple_el_90():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 2000)
    conf = rng.uniform(0.3, 1, 2000)
    acierta = rng.uniform(size=2000) < conf          # acierta con probabilidad = su confianza
    pred = np.where(acierta, y, (y + 1) % 5)
    p = np.full((2000, 5), 0.0)
    p[np.arange(2000), pred] = conf
    p[np.arange(2000), (pred + 2) % 5] = 1 - conf
    umbral, aviso = elegir_umbral({"fpz": {"SC4001": (y, p)}})
    assert resumen_confianza([(y, p)], umbral)["acc_seguras"] >= 0.90
    assert resumen_confianza([(y, p)], round(umbral - 0.05, 2))["acc_seguras"] < 0.90
    assert 0 <= aviso <= 1
