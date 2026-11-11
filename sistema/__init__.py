"""Sistema de estadificación del sueño para predecir una noche nueva.

Usa los modelos ya entrenados de modelo/ (los notebooks de experimentacion/ los entrenan):

    from sistema import predecir_noche

    salida = predecir_noche("SC4001E0-PSG.edf", canales=("fpz",))

Módulos:
  lectura     EDF (y, si hay, hipnograma) -> épocas de 30 s
  modelo_a    rasgos por época + XGBoost con canales enmascarados
  modelo_b    red CNN-Transformer sobre la señal cruda
  meta        meta-modelo que combina A y B
  prediccion  junta todo: estadio, confianza, 2.ª opción y "revisar" de cada época (analizar() es lo que
              responde el endpoint)
"""
# torch antes que xgboost: en macOS, si xgboost carga primero su OpenMP, las convoluciones de torch en CPU se caen
import torch  # noqa: F401

from .prediccion import analizar, cargar_sistema, predecir_noche  # noqa: E402

__all__ = ["analizar", "cargar_sistema", "predecir_noche"]
