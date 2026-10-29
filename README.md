# Estadificación del sueño con canales variables

Proyecto de grado — Maestría en Inteligencia Artificial (MAIA), Universidad de los Andes · Grupo 3.

Clasificación automática de los estadios del sueño (W, N1, N2, N3, REM) en épocas de 30 s a partir de polisomnografía, con **cualquier combinación de canales que incluya al menos un EEG**:

| EEG (al menos uno) | EOG | EMG | Combinaciones |
|---|---|---|---|
| Fpz-Cz, Pz-Oz o los dos | con o sin | con o sin | 3 × 2 × 2 = **12** |

El sistema tiene tres piezas, y cada una es un solo modelo para las 12 combinaciones:

```
                    noche completa (los canales que traiga)
             ┌───────────────────┴────────────────────┐
  Modelo A: 719 rasgos por época           Modelo B: señal cruda, 80 épocas a la vez
  + XGBoost con canales en NaN             CNN → Transformer → Transformer entre épocas
             └───────────────┬────────────────────────┘
          probabilidades por época de cada uno (t−2 … t+2) + qué canales hay
                             │
              Meta-modelo (regresión logística + XGBoost)
                             │
          estadio por época, confianza, 2.ª opción y "revisar"
```

## Contenido

```
├── src/sueno/                    el código del sistema
│   ├── config.py                 constantes, rutas y las 12 combinaciones
│   ├── lectura.py                EDF + hipnograma -> épocas de 30 s
│   ├── rasgos.py                 rasgos por época del modelo A
│   ├── modelo_a.py               XGBoost con canales enmascarados
│   ├── modelo_b.py               red sobre la señal cruda
│   ├── meta.py                   meta-modelo
│   ├── evaluacion.py             métricas, bootstrap por sujeto, confianza y matrices de confusión
│   └── sistema.py                el pipeline para predecir una noche (make_pipeline de scikit-learn)
├── scripts/
│   └── entrenamiento_completo.py entrena todo y deja los modelos en modelos/
├── tests/                        pruebas unitarias y del modelo en modelos/
├── modelos/                      los tres modelos en producción (ver modelos/README.md)
├── particiones.json              partición por sujeto: 16 de test + 5 folds sobre 62, y las 12 combinaciones
├── data/                         los datos (no se suben al repositorio)
└── paso_1_xgboost/, paso_2_deep_model/, paso_3_meta_model/
                                  notebooks de experimentación: explican cada paso, sección por sección
```

## Instalación

```bash
pip install -e ".[dev]"
```

Python 3.11. Las versiones de las librerías están fijas en `pyproject.toml`. Los notebooks usan `requirements.txt`.

## Datos

[Sleep-EDF Database Expanded v1.0.0](https://physionet.org/content/sleep-edfx/1.0.0/), subconjunto Sleep Cassette: 153 noches de 78 personas sanas. Se ponen a mano en `data/`:

```
data/
├── sleep-cassette/      los 306 EDF (SC4ssNE0-PSG.edf y SC4ssN*-Hypnogram.edf)
└── SC-subjects.xls
```

Los cálculos intermedios van en `data/procesados/`. Con las variables de entorno `SUENO_DATOS` y `SUENO_EDF` se pueden usar otras carpetas.

## Entrenamiento

```bash
python scripts/entrenamiento_completo.py                  # todo
python scripts/entrenamiento_completo.py --pasos a meta   # solo algunos pasos
python scripts/entrenamiento_completo.py --prueba         # 18 noches, 50 rondas y 2 pasadas: ver que todo corre (~25 min en CPU)
```

| Paso | Qué hace | Dónde | Tiempo aprox. |
|---|---|---|---|
| `rasgos` | rasgos por época de las 153 noches | CPU | 5 min |
| `noches` | señal preprocesada para el modelo B | CPU | 5 min |
| `a` | modelo A: 5 folds + modelo final | CPU | 30 min |
| `b` | modelo B: 5 folds + modelo final | GPU (A100), `--en-paralelo 6` | 2 h |
| `meta` | meta-modelo: 5 folds + modelo final | CPU | 15 min |
| `evaluar` | métricas, IC, umbral de confianza, matrices de confusión y `modelos/sistema.json` | CPU | 2 min |

Cada paso guarda lo que calcula y, si se corta, al volver a correr retoma donde iba. Al final deja los modelos en `modelos/` y las métricas en `resultados/`. Con semilla 42, los pasos `a` y `meta` son deterministas: el código reproduce byte a byte los archivos de `modelos/`.

**Modelo B en Colab:** se sube el repositorio y los datos a Drive, se instala con `pip install -e .` y se corre `python scripts/entrenamiento_completo.py --pasos noches b --en-paralelo 6` en una A100. Los 6 entrenamientos corren a la vez en la misma GPU.

## Pruebas

```bash
pytest
```

Revisan la lectura de las noches, los rasgos, el preprocesamiento y la red del modelo B, el meta-modelo, la confianza, la partición y que el modelo de `modelos/` esté completo, cargue y prediga.

## Resultados

Del informe de avance. **CV:** predicciones fuera de fold de los 62 sujetos de desarrollo. **Test:** 16 sujetos que no se usaron para ninguna decisión; entre corchetes, el IC 95 % por bootstrap agrupado por persona (2 000 remuestras).

| Canales | CV MF1 | Test MF1 [IC 95 %] | Test κ |
|---|---|---|---|
| Solo Fpz-Cz | 0.802 | 0.772 [0.736, 0.802] | 0.750 |
| Solo Pz-Oz | 0.762 | 0.737 [0.686, 0.773] | 0.704 |
| Fpz-Cz + EOG | 0.811 | 0.782 [0.749, 0.811] | 0.755 |
| Los 4 canales | 0.819 | 0.784 [0.753, 0.811] | 0.757 |
| **Combinación al azar por noche** | | **0.768 [0.730, 0.799]** | 0.740 |

Con un umbral de confianza de 0.65 (elegido con la CV), el sistema entrega el 80.5 % de la noche con 87.8 % de acierto y marca el resto para revisión con dos opciones; la correcta está entre ellas el 86.1 % de las veces.

## Créditos

El codificador de época del modelo B es una reimplementación propia de SleepEffFormer (Paul et al., 2026).
