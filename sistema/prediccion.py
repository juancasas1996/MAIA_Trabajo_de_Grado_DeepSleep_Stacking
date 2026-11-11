"""Predicción de una noche: EDF -> estadio de cada época de 30 s, con su confianza, la 2.ª opción y "revisar".

    noche = leer_noche(...)           épocas de 30 s de los canales presentes
    pa = modelo A (rasgos)            (épocas, 5)
    pb = modelo B (señal cruda)       (épocas, 5)
    p = meta-modelo(pa, pb, máscara)  (épocas, 5)
    confianza = max(p) -> "revisar" si es menor que el umbral de sistema.json

analizar() devuelve la respuesta del endpoint (un dict listo para JSON); predecir_noche(), la misma salida como tabla.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import meta, modelo_a, modelo_b
from .lectura import EPOCH, ESTADIOS, leer_noche

MODELO = Path(__file__).resolve().parents[1] / "modelo"


def cargar_sistema(modelo=MODELO, device="cpu"):
    """Las tres piezas de modelo/ y la configuración de sistema.json, listas para predecir."""
    modelo = Path(modelo)
    cfg = json.loads((modelo / "sistema.json").read_text())
    booster, columnas_a = modelo_a.cargar(modelo / "modelo_a_xgboost")
    red, hp = modelo_b.cargar(modelo / "modelo_b_red" / "modelo_final.pt", device)
    lr, meta_xgb, columnas_meta = meta.cargar(modelo / "meta_modelo")
    return {"a": (booster, columnas_a), "b": (red, hp), "meta": (lr, meta_xgb, columnas_meta), "device": device,
            "umbral": cfg["umbral_confianza"], "aviso": cfg["aviso_noche_fraccion_revisar"]}


def probabilidades(sistema, noche):
    """Probabilidades (épocas analizadas, 5) de A, de B y del meta-modelo."""
    pa = modelo_a.predecir(*sistema["a"], noche)
    red, hp = sistema["b"]
    pb = modelo_b.predecir(red, modelo_b.preparar(noche), noche.mascara, hp, sistema["device"])
    pb = meta.fill_window(noche.validas, pb)
    return pa, pb, meta.predecir(*sistema["meta"], pa, pb, noche.mascara)


def analizar(sistema, entrada):
    """Respuesta del endpoint para una noche.

    entrada: dict con "psg" (ruta local del EDF) y, opcionales, "canales", "hora_apagado", "epoca_inicio",
    "epoca_fin" e "hipnograma" (ver lectura.leer_noche).

    Devuelve, por época analizada, el estadio, la confianza, la 2.ª opción, su probabilidad, si hay que revisarla y
    las 5 probabilidades; y para la noche, la fracción a revisar y el aviso. segundos_proceso cronometra solo el
    procesamiento (EDF -> hipnograma), sin cargar los modelos ni la cola del endpoint.
    """
    t0 = time.perf_counter()
    noche = leer_noche(**entrada)
    p = probabilidades(sistema, noche)[2]
    o = np.argsort(-p, 1)
    i = np.arange(len(p))
    conf = p[i, o[:, 0]]
    revisar = conf < sistema["umbral"]
    return {"noche": noche.nombre, "canales": list(noche.canales), "epoca": noche.epocas.tolist(),
            "segundos_desde_inicio": (noche.epocas * EPOCH).tolist(),
            "estadio": [ESTADIOS[k] for k in o[:, 0]], "confianza": conf.round(4).tolist(),
            "segunda_opcion": [ESTADIOS[k] for k in o[:, 1]], "prob_segunda": p[i, o[:, 1]].round(4).tolist(),
            "revisar": revisar.tolist(), "fraccion_revisar": float(revisar.mean()),
            "aviso_noche": bool(revisar.mean() > sistema["aviso"]), "probabilidades": p.round(4).tolist(),
            "segundos_proceso": round(time.perf_counter() - t0, 3)}


def predecir_noche(psg, canales=None, hipnograma=None, hora_apagado=None, epoca_inicio=None, epoca_fin=None,
                   sistema=None):
    """La respuesta de analizar() como tabla: una fila por época analizada.

    En salida.attrs quedan la noche, los canales usados, la fracción a revisar, si se dispara el aviso por noche y
    los segundos de proceso. Los argumentos son los de lectura.leer_noche; sistema, el de cargar_sistema (si no se
    da, se carga).
    """
    entrada = {"psg": psg, "canales": canales, "hipnograma": hipnograma, "hora_apagado": hora_apagado,
               "epoca_inicio": epoca_inicio, "epoca_fin": epoca_fin}
    r = analizar(sistema or cargar_sistema(), entrada)
    p = np.array(r["probabilidades"])
    salida = pd.DataFrame({k: r[k] for k in ("epoca", "segundos_desde_inicio", "estadio", "confianza",
                                             "segunda_opcion", "prob_segunda", "revisar")})
    for k, s in enumerate(ESTADIOS):
        salida[f"p_{s}"] = p[:, k]
    salida.attrs = {k: r[k] for k in ("noche", "canales", "fraccion_revisar", "aviso_noche", "segundos_proceso")}
    return salida
