"""Constantes y rutas del proyecto.

Las rutas de los datos se pueden cambiar con variables de entorno (por ejemplo en Colab, donde los EDF se copian al
disco de la máquina):
  - SUENO_DATOS: carpeta de datos (por defecto, data/ en la raíz del repositorio);
  - SUENO_EDF: carpeta con los EDF de Sleep Cassette (por defecto, <datos>/sleep-cassette).
"""
import json
import os
from functools import lru_cache
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DATOS = Path(os.environ.get("SUENO_DATOS", RAIZ / "data"))
EDF = Path(os.environ.get("SUENO_EDF", DATOS / "sleep-cassette"))
SUJETOS_XLS = DATOS / "SC-subjects.xls"
PROCESADOS = DATOS / "procesados"
MODELOS = RAIZ / "modelos"
RESULTADOS = RAIZ / "resultados"

SEED = 42

# Señales
FS = 100       # Hz de EEG y EOG
EPOCH = 30     # segundos por época
SPE = FS * EPOCH
CANALES = {"fpz": "EEG Fpz-Cz", "pz": "EEG Pz-Oz", "eog": "EOG horizontal", "emg": "EMG submental"}
MODALIDADES = ("fpz", "pz", "eog", "emg")

# Estadios (norma AASM: S3 y S4 de R&K se juntan en N3)
ESTADIOS = ["W", "N1", "N2", "N3", "REM"]
K = len(ESTADIOS)
STAGE_MAP = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
             "Sleep stage 3": 3, "Sleep stage 4": 4, "Sleep stage R": 4}
UNKNOWN = -1

# Las 12 combinaciones de canales con al menos un EEG (las mismas de particiones.json, en el mismo orden)
COMBINACIONES = {
    "fpz": ("fpz",), "pz": ("pz",), "eeg2": ("fpz", "pz"),
    "fpz_eog": ("fpz", "eog"), "pz_eog": ("pz", "eog"), "eeg2_eog": ("fpz", "pz", "eog"),
    "fpz_emg": ("fpz", "emg"), "pz_emg": ("pz", "emg"), "eeg2_emg": ("fpz", "pz", "emg"),
    "fpz_eog_emg": ("fpz", "eog", "emg"), "pz_eog_emg": ("pz", "eog", "emg"), "eeg2_eog_emg": ("fpz", "pz", "eog", "emg"),
}


def mascara(cfg):
    """[fpz, pz, eog, emg] -> True si el canal está en la combinación."""
    return [m in COMBINACIONES[cfg] for m in MODALIDADES]


@lru_cache
def particion():
    """particiones.json: 16 sujetos de test, 62 de desarrollo y 5 folds."""
    return json.loads((RAIZ / "particiones.json").read_text())


def sujeto(noche):
    """SC4ssN -> ss"""
    return int(noche[3:5])
