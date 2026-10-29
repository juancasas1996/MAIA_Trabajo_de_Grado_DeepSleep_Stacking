"""Preprocesamiento y red del modelo B."""
import numpy as np
import pytest
import torch
from conftest import noche_falsa

from sueno.config import EPOCH, SPE, mascara
from sueno.modelo_b import ARQUITECTURA, Red, predecir, preparar, zscore


@pytest.fixture(scope="module")
def red():
    torch.manual_seed(0)
    return Red(**ARQUITECTURA).eval()


def test_zscore_por_epoca():
    z = zscore(np.random.default_rng(0).normal(5, 3, (4, SPE)))
    assert np.allclose(z.mean(1), 0, atol=1e-5)
    assert np.allclose(z.std(1), 1, atol=1e-4)


def test_preparar_formas_y_tipos(noche):
    out = preparar(noche)
    for k in ("fpz", "pz", "eog"):
        assert out[k].shape == (60, SPE) and out[k].dtype == np.float16
    assert out["emg"].shape == (60, EPOCH) and out["emg"].dtype == np.float32
    assert all(np.isfinite(out[k].astype(np.float32)).all() for k in ("fpz", "pz", "eog", "emg"))


def test_preparar_canal_ausente_en_ceros():
    out = preparar(noche_falsa(("fpz",)))
    assert out["fpz"].any()
    assert not out["pz"].any() and not out["eog"].any() and not out["emg"].any()


def test_preparar_solo_epocas_con_estadio():
    """Con hipnograma, la red no ve las épocas sin estadio."""
    out = preparar(noche_falsa(validas=[True] * 50 + [False] * 10))
    assert len(out["fpz"]) == len(out["y"]) == 50


def test_tamano_de_la_red(red):
    assert sum(p.numel() for p in red.parameters() if p.requires_grad) == 660_069


def test_probabilidades_por_epoca(red):
    """Noches más cortas y más largas que una ventana de 80: todas las épocas reciben 5 probabilidades que suman 1."""
    for n in (50, 130):
        p = predecir(red, preparar(noche_falsa(n=n)), mascara("eeg2_eog_emg"))
        assert p.shape == (n, 5)
        assert np.allclose(p.sum(1), 1, atol=1e-5)


def test_canal_apagado_no_influye(red):
    """Si la máscara apaga Pz-Oz, cambiar su señal no cambia la predicción."""
    a = preparar(noche_falsa(n=40, seed=1))
    b = {**a, "pz": np.random.default_rng(9).normal(size=a["pz"].shape).astype(np.float16)}
    m = mascara("fpz_eog")
    assert np.allclose(predecir(red, a, m), predecir(red, b, m), atol=1e-6)
