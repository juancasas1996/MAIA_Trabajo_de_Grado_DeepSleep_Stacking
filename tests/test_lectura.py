"""Hipnograma, ventana de sueño y hora de apagado de la luz."""
import datetime as dt

import mne
import numpy as np

from sueno.lectura import hypnogram_labels, lights_off_offset, sleep_window


def test_hipnograma_a_etiquetas(tmp_path):
    """Cada anotación pasa a su estadio; N3 y N4 se juntan en N3; lo que no es estadio queda en -1."""
    ann = mne.Annotations(onset=[0, 60, 90, 120, 150], duration=[60, 30, 30, 30, 30],
                          description=["Sleep stage W", "Sleep stage 3", "Sleep stage 4", "Sleep stage R", "Movement time"])
    ruta = tmp_path / "hipnograma.csv"
    ann.save(ruta)
    assert hypnogram_labels(ruta, 7).tolist() == [0, 0, 3, 3, 4, -1, -1]


def test_ventana_de_sueno_deja_30_min_de_vigilia():
    """30 min = 60 épocas de vigilia antes del primer sueño y después del último."""
    etiquetas = np.zeros(400, dtype=int)
    etiquetas[100:200] = 2
    assert sleep_window(etiquetas) == (40, 259)


def test_ventana_de_sueno_no_se_sale_del_registro():
    etiquetas = np.zeros(100, dtype=int)
    etiquetas[10:95] = 2
    assert sleep_window(etiquetas) == (0, 99)


def test_apagado_de_luz():
    inicio = dt.datetime(2026, 1, 1, 22, 0, 0)
    assert lights_off_offset(inicio, dt.time(22, 30, 0)) == 1800
    # si se apaga después de medianoche, se cuenta al día siguiente
    assert lights_off_offset(inicio, dt.time(0, 15, 0)) == 8100
