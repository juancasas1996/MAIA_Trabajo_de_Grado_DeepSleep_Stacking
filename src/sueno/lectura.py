"""Lectura de una noche: EDF + hipnograma -> épocas de 30 s.

Lo usan el entrenamiento (con hipnograma, para la ventana de sueño y las etiquetas) y el endpoint (sin hipnograma).
"""
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from .config import CANALES, EPOCH, MODALIDADES, SPE, STAGE_MAP, SUJETOS_XLS, UNKNOWN

mne.set_log_level("ERROR")


@dataclass
class Noche:
    nombre: str
    canales: tuple            # modalidades presentes, en el orden de MODALIDADES
    senales: dict             # modalidad -> señal completa del registro, en V (EMG interpolado a 100 Hz)
    n_registro: int           # épocas de 30 s del registro
    epocas: np.ndarray        # épocas que se analizan (posición en el registro)
    etiquetas: np.ndarray     # estadio de cada época analizada (0..4, o -1 si no tiene o no hay hipnograma)
    validas: np.ndarray       # bool por época analizada: las que ve el modelo B (todas, salvo con hipnograma)
    lights_off: float         # segundos desde el inicio del registro hasta que se apagó la luz
    meas_date: object

    @property
    def mascara(self):
        return [m in self.canales for m in MODALIDADES]


def list_nights(raw_dir):
    """[(noche, ruta_psg, ruta_hipnograma)] ordenado por noche."""
    psgs = {p.name[:6]: p for p in Path(raw_dir).glob("SC4*-PSG.edf")}
    hyps = {p.name[:6]: p for p in Path(raw_dir).glob("SC4*-Hypnogram.edf")}
    return [(k, psgs[k], hyps[k]) for k in sorted(psgs) if k in hyps]


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


def horas_apagado(xls=SUJETOS_XLS):
    """noche -> hora en que se apagó la luz, desde SC-subjects.xls."""
    meta = pd.read_excel(xls)
    return {f"SC4{int(r.subject):02d}{int(r.night)}": r.LightsOff for _, r in meta.iterrows()}


def leer_noche(entrada):
    """Lee el EDF y decide qué épocas se analizan.

    entrada es un dict con
      - "psg": ruta del EDF (obligatorio);
      - "canales": modalidades a usar, entre fpz, pz, eog y emg (por defecto, las que tenga el EDF). Al menos un EEG;
      - "hora_apagado": hora en que se apagó la luz ("HH:MM:SS" o datetime.time). Si no se da, el inicio de lo analizado;
      - "epoca_inicio" y "epoca_fin": rango de épocas del registro a analizar (por defecto, todo el registro);
      - "hipnograma": con él se analiza la ventana de sueño (30 min de vigilia a cada lado) y el modelo B no ve las
        épocas sin estadio, como en el entrenamiento.
    """
    psg = Path(entrada["psg"])
    info = mne.io.read_raw_edf(psg, preload=False, verbose="error")
    en_edf = [m for m in MODALIDADES if CANALES[m] in info.ch_names]
    canales = tuple(m for m in MODALIDADES if m in (entrada.get("canales") or en_edf))
    desconocidos = set(entrada.get("canales") or []) - set(MODALIDADES)
    if desconocidos:
        raise ValueError(f"canales desconocidos: {', '.join(sorted(desconocidos))}")
    faltan = [m for m in canales if m not in en_edf]
    if faltan:
        raise ValueError(f"el EDF no tiene {', '.join(CANALES[m] for m in faltan)}")
    if not {"fpz", "pz"} & set(canales):
        raise ValueError("hace falta al menos un EEG: Fpz-Cz o Pz-Oz")

    raw = mne.io.read_raw_edf(psg, include=[CANALES[m] for m in canales], preload=True, verbose="error")
    n = int(raw.n_times // SPE)
    if entrada.get("hipnograma"):
        labels = hypnogram_labels(entrada["hipnograma"], n, EPOCH)
        first, last = sleep_window(labels)
        epocas = np.arange(first, last + 1)
        etiquetas = labels[epocas]
        validas = etiquetas != UNKNOWN
    else:
        first, last = int(entrada.get("epoca_inicio", 0)), int(entrada.get("epoca_fin", n - 1))
        epocas = np.arange(max(first, 0), min(last, n - 1) + 1)
        etiquetas = np.full(len(epocas), UNKNOWN, dtype=np.int64)
        validas = np.ones(len(epocas), dtype=bool)
    if len(epocas) == 0:
        raise ValueError("no hay épocas para analizar")

    meas_date = raw.info["meas_date"]
    hora = entrada.get("hora_apagado")
    if hora:
        hora = dt.time.fromisoformat(hora) if isinstance(hora, str) else hora
        lights_off = lights_off_offset(meas_date, hora)
    else:
        lights_off = float(epocas[0] * EPOCH)
    senales = {m: raw.get_data(picks=[CANALES[m]])[0] for m in canales}
    return Noche(psg.name[:6], canales, senales, n, epocas, etiquetas, validas, lights_off, meas_date)
