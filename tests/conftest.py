import numpy as np
import pytest

from sueno.config import FS, SPE
from sueno.lectura import Noche


def noche_falsa(canales=("fpz", "pz", "eog", "emg"), n=60, seed=0, validas=None):
    """Una noche de n épocas con ruido en los canales dados (en V, como los lee MNE)."""
    rng = np.random.default_rng(seed)
    senales = {m: rng.normal(0, 20e-6, n * SPE) for m in canales}
    if "emg" in senales:   # el EMG viene a 1 Hz y MNE lo repite a 100 Hz
        senales["emg"] = np.repeat(np.abs(rng.normal(0, 5e-6, n * SPE // FS)), FS)
    validas = np.ones(n, dtype=bool) if validas is None else np.asarray(validas)
    etiquetas = np.where(validas, rng.integers(0, 5, n), -1)
    return Noche("SC4001", tuple(canales), senales, n, np.arange(n), etiquetas, validas, 0.0, None)


@pytest.fixture
def noche():
    return noche_falsa()
