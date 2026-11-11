"""Lectura de una noche: EDF -> épocas de 30 s.

Sin hipnograma se analiza todo el registro, o el rango de épocas que se pida. Con hipnograma se analiza la ventana de
sueño (30 min de vigilia a cada lado) y la red no ve las épocas sin estadio, como en el entrenamiento.
"""
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

mne.set_log_level("ERROR")

FS = 100            # Hz de EEG y EOG
EPOCH = 30          # segundos por época
SPE = FS * EPOCH    # muestras por época
ESTADIOS = ["W", "N1", "N2", "N3", "REM"]
K = len(ESTADIOS)
MODALIDADES = ("fpz", "pz", "eog", "emg")
CANALES = {"fpz": "EEG Fpz-Cz", "pz": "EEG Pz-Oz", "eog": "EOG horizontal", "emg": "EMG submental"}

STAGE_MAP = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
             "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4}
UNKNOWN = -1


@dataclass
class Noche:
    nombre: str
    canales: tuple          # modalidades presentes, en el orden de MODALIDADES
    senales: dict           # modalidad -> señal completa del registro, en V (EMG interpolado a 100 Hz)
    n_registro: int         # épocas de 30 s del registro
    epocas: np.ndarray      # épocas que se analizan (posición en el registro)
    etiquetas: np.ndarray   # estadio de cada época analizada (0..4, o -1 si no tiene o no hay hipnograma)
    validas: np.ndarray     # épocas que ve la red: todas, salvo con hipnograma (solo las que tienen estadio)
    lights_off: float       # segundos desde el inicio del registro hasta que se apagó la luz

    @property
    def mascara(self):
        """[fpz, pz, eog, emg] -> True si el canal está presente."""
        return [m in self.canales for m in MODALIDADES]


def hypnogram_labels(hyp_path, n_epochs, epoch_sec=EPOCH):
    """Etiqueta (0..4 o -1) de cada época del registro, contando desde su inicio."""
    ann = mne.read_annotations(hyp_path)
    labels = np.full(n_epochs, UNKNOWN, dtype=np.int64)
    for onset, duration, desc in zip(ann.onset, ann.duration, ann.description):
        start = int(round(onset / epoch_sec))
        stop = min(start + int(round(duration / epoch_sec)), n_epochs)
        labels[start:stop] = STAGE_MAP.get(desc, UNKNOWN)
    return labels


def sleep_window(labels, wake_margin_min=30, epoch_sec=EPOCH):
    """(primera, última) época del periodo de sueño, con 30 min de vigilia a cada lado."""
    sleep_idx = np.where(labels > 0)[0]
    margin = int(wake_margin_min * 60 / epoch_sec)
    return max(sleep_idx[0] - margin, 0), min(sleep_idx[-1] + margin, len(labels) - 1)


def lights_off_offset(meas_date, lights_off):
    """Segundos desde el inicio del registro hasta que se apagó la luz."""
    start = meas_date.hour * 3600 + meas_date.minute * 60 + meas_date.second
    lo = lights_off.hour * 3600 + lights_off.minute * 60 + lights_off.second
    delta = lo - start
    return delta + 86400 if delta < 0 else delta


def leer_noche(psg, canales=None, hipnograma=None, hora_apagado=None, epoca_inicio=None, epoca_fin=None):
    """Lee el EDF y decide qué épocas se analizan.

    - canales: modalidades a usar, entre fpz, pz, eog y emg (por defecto, las que tenga el EDF). Al menos un EEG.
    - hipnograma: si se da, se analiza la ventana de sueño, como en el entrenamiento.
    - hora_apagado: hora en que se apagó la luz ("HH:MM:SS" o datetime.time). Si no se da, el inicio de lo analizado.
    - epoca_inicio, epoca_fin: sin hipnograma, rango de épocas del registro a analizar, con los dos extremos incluidos
      (por defecto, todo el registro).
    """
    psg = Path(psg)
    info = mne.io.read_raw_edf(psg, preload=False, verbose="error")
    en_edf = [m for m in MODALIDADES if CANALES[m] in info.ch_names]
    desconocidos = set(canales or []) - set(MODALIDADES)
    if desconocidos:
        raise ValueError(f"canales desconocidos: {', '.join(sorted(desconocidos))}")
    canales = tuple(m for m in MODALIDADES if m in (canales or en_edf))
    faltan = [m for m in canales if m not in en_edf]
    if faltan:
        raise ValueError(f"el EDF no tiene {', '.join(CANALES[m] for m in faltan)}")
    if not {"fpz", "pz"} & set(canales):
        raise ValueError("hace falta al menos un EEG: Fpz-Cz o Pz-Oz")

    raw = mne.io.read_raw_edf(psg, include=[CANALES[m] for m in canales], preload=True, verbose="error")
    n = int(raw.n_times // SPE)
    if hipnograma:
        labels = hypnogram_labels(hipnograma, n)
        first, last = sleep_window(labels)
        epocas = np.arange(first, last + 1)
        etiquetas = labels[epocas]
        validas = etiquetas != UNKNOWN
    else:
        first = 0 if epoca_inicio is None else int(epoca_inicio)
        last = n - 1 if epoca_fin is None else int(epoca_fin)
        epocas = np.arange(max(first, 0), min(last, n - 1) + 1)
        etiquetas = np.full(len(epocas), UNKNOWN, dtype=np.int64)
        validas = np.ones(len(epocas), dtype=bool)
    if len(epocas) == 0:
        raise ValueError("no hay épocas para analizar")

    if hora_apagado:
        hora = dt.time.fromisoformat(hora_apagado) if isinstance(hora_apagado, str) else hora_apagado
        lights_off = lights_off_offset(raw.info["meas_date"], hora)
    else:
        lights_off = float(epocas[0] * EPOCH)
    senales = {m: raw.get_data(picks=[CANALES[m]])[0] for m in canales}
    return Noche(psg.name[:6], canales, senales, n, epocas, etiquetas, validas, lights_off)
