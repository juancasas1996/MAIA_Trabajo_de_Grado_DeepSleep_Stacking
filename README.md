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
  Modelo A: 100 rasgos por época           Modelo B: señal cruda, 80 épocas a la vez
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
├── despliegue/                   endpoint de SageMaker, MLflow, E2E y despliegue (ver despliegue/README.md)
├── .github/workflows/            CI/CD: pruebas, E2E en cada PR y despliegue al hacer merge a main
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
python scripts/entrenamiento_completo.py --pasos a meta evaluar --columnas 100   # modelo A con sus 100 columnas más importantes
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

**Modelo A con menos columnas:** `--columnas N` entrena el modelo A solo con las N columnas que más ganancia total tienen en el modelo A que está en `modelos/`. El modelo B no cambia; el meta-modelo se vuelve a entrenar porque sus entradas son las probabilidades de A. Elegir de nuevo sobre un modelo ya recortado da las mismas columnas, así que el comando reproduce el modelo.

**Modelo B en Colab:** se sube el repositorio y los datos a Drive, se instala con `pip install -e .` y se corre `python scripts/entrenamiento_completo.py --pasos noches b --en-paralelo 6` en una A100. Los 6 entrenamientos corren a la vez en la misma GPU.

## Pruebas

```bash
pytest
```

Revisan la lectura de las noches, los rasgos, el preprocesamiento y la red del modelo B, el meta-modelo, la confianza, la partición y que el modelo de `modelos/` esté completo, cargue y prediga.

## CI/CD

| Cuándo | Workflow | Qué hace |
|---|---|---|
| A mano (Actions → Pruebas → *Run workflow*) | **Pruebas** | `pytest`, con una tabla de cada prueba en el resumen. El E2E y el Despliegue las corren primero |
| PR a `main` que cambia `modelos/`, `src/`, `despliegue/`, `tests/` o `pyproject.toml` | **E2E** | pruebas → despliega el candidato en un endpoint aparte → 32 noches de test al candidato y a producción → comenta en el PR MF1, κ, F1 por estadio, matriz de confusión y latencia → borra el candidato |
| Merge a `main` que cambia esas carpetas | **Despliegue** | pruebas → registra la versión en MLflow → actualiza el endpoint de producción → prueba de humo → tag `modelo-v<versión>` |

El modelo nunca llega a producción sin el merge de una persona. Detalles, secrets y uso local en [`despliegue/README.md`](despliegue/README.md).

## Resultados

**CV:** predicciones fuera de fold de los 62 sujetos de desarrollo. **Test:** 16 sujetos que no se usaron para ninguna decisión; entre corchetes, el IC 95 % por bootstrap agrupado por persona (2 000 remuestras).

### Versiones del modelo

| Versión | Modelo A | Meta-modelo | Test MF1, canales al azar | Estado |
|---|---|---|---|---|
| v1 (informe de avance) | 719 columnas, 612 rondas | 288 rondas | 0.768 [0.730, 0.799] | la del informe |
| **v2** | **100 columnas**, 542 rondas | 242 rondas | 0.767 [0.727, 0.797] | **la que está en `modelos/`** |

El modelo B es el mismo en las dos. La v2 usa las 100 columnas con más ganancia total en el modelo A de la v1 (que se entrenó solo con los sujetos de desarrollo). Esas 100 columnas explican el 70 % de la ganancia total: 51 de Fpz-Cz, 40 de Pz-Oz, 5 del EOG, 3 del EMG y la hora de la noche.

Al predecir, solo se calculan los 62 rasgos base que necesitan esas 100 columnas (de 213), con los mismos valores que la tabla completa. El cálculo de los rasgos baja de ~1.2 s a ~0.75 s por noche en un Mac M3, pero el tiempo total casi no cambia: lo que más tarda es la red B (~2.5 s por noche con un canal y ~7 s con los 4, de un total de ~3 s y ~8.5 s).

**v2 frente a v1 en el test**, diferencia pareada (mismos sujetos y mismas combinaciones sorteadas):

| | ΔMF1 [IC 95 %] | Δκ [IC 95 %] | ΔF1 N1 [IC 95 %] |
|---|---|---|---|
| Sistema, canales al azar | −0.001 [−0.005, +0.002] | −0.002 [−0.006, +0.002] | −0.005 [−0.011, +0.001] |
| Solo el modelo A | **−0.012** [−0.019, −0.005] | **−0.014** [−0.023, −0.007] | **−0.021** [−0.038, −0.005] |

El modelo A solo pierde algo de forma significativa, pero el meta-modelo lo compensa con la red B: el sistema queda empatado (los IC cruzan 0).

### Por combinación de canales

| Canales | CV MF1 v1 | CV MF1 v2 | Test MF1 v1 [IC 95 %] | Test MF1 v2 [IC 95 %] | Test κ v2 |
|---|---|---|---|---|---|
| Solo Fpz-Cz | 0.802 | 0.799 | 0.772 [0.736, 0.802] | 0.772 [0.737, 0.801] | 0.748 |
| Solo Pz-Oz | 0.762 | 0.760 | 0.737 [0.686, 0.773] | 0.735 [0.681, 0.772] | 0.703 |
| Fpz-Cz + EOG | 0.811 | 0.808 | 0.782 [0.749, 0.811] | 0.779 [0.746, 0.807] | 0.751 |
| Los 4 canales | 0.819 | 0.818 | 0.784 [0.753, 0.811] | 0.782 [0.752, 0.809] | 0.755 |
| **Combinación al azar por noche** | | | **0.768 [0.730, 0.799]** | **0.767 [0.727, 0.797]** | 0.738 |

Con un umbral de confianza de 0.65 (elegido con la CV), el sistema v2 entrega el 80.2 % de la noche con 87.8 % de acierto (v1: 80.5 % y 87.8 %) y marca el resto para revisión con dos opciones; la correcta está entre ellas el 86.1 % de las veces (v1: 86.1 %).

**Cautela:** las 100 columnas se eligieron con un modelo entrenado con los mismos sujetos de la CV, así que la CV de la v2 puede salir un poco optimista. El test no tiene ese problema.

## Créditos

El codificador de época del modelo B es una reimplementación propia de SleepEffFormer (Paul et al., 2026).
