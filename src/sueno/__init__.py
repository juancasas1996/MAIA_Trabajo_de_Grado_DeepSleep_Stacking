"""Clasificación automática de los estadios del sueño (W, N1, N2, N3, REM) con canales variables.

Módulos:
  config      constantes, rutas y las 12 combinaciones de canales
  lectura     EDF + hipnograma -> épocas de 30 s
  rasgos      rasgos por época del modelo A
  modelo_a    XGBoost con canales enmascarados
  modelo_b    red sobre la señal cruda (CNN + Transformer)
  meta        meta-modelo que combina A y B
  evaluacion  métricas, bootstrap por sujeto, confianza y matrices de confusión
  sistema     el pipeline para predecir una noche (lo usa el endpoint)
"""
# torch antes que xgboost: en macOS, si xgboost carga primero su OpenMP, las convoluciones de torch en CPU se caen
import torch  # noqa: F401
