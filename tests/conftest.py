import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sistema import cargar_sistema  # noqa: E402
from sistema.lectura import FS, SPE, Noche  # noqa: E402


def noche_falsa(canales=("fpz", "pz", "eog", "emg"), n=60, seed=0):
    """Una noche de n épocas con ruido en los canales dados (en V, como los lee MNE)."""
    rng = np.random.default_rng(seed)
    senales = {m: rng.normal(0, 20e-6, n * SPE) for m in canales}
    if "emg" in senales:   # el EMG viene a 1 Hz y MNE lo repite a 100 Hz
        senales["emg"] = np.repeat(np.abs(rng.normal(0, 5e-6, n * SPE // FS)), FS)
    validas = np.ones(n, dtype=bool)
    return Noche("SC4001", tuple(canales), senales, n, np.arange(n), rng.integers(0, 5, n), validas, 0.0)


@pytest.fixture(scope="session")
def sistema():
    return cargar_sistema()
