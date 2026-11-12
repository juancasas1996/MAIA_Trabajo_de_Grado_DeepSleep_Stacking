# Estadificación del sueño con canales variables

Clasificación automática de los estadios del sueño (W, N1, N2, N3, REM) en épocas de 30 s a partir de polisomnografía, con **cualquier combinación de canales que incluya al menos un EEG**:

| EEG (al menos uno) | EOG | EMG | Combinaciones |
|---|---|---|---|
| Fpz-Cz, Pz-Oz o los dos | con o sin | con o sin | 3 × 2 × 2 = **12** |

## Modelos

| Modelo | Entrada | Idea |
|---|---|---|
| **A: XGBoost** | las 100 columnas más importantes de 719 rasgos por época (espectro, forma de la señal, ondículas, pendiente 1/f y contexto de las épocas vecinas) | un único XGBoost para las 12 combinaciones: en el entrenamiento se apagan canales al azar (sus columnas se ponen en NaN) |
| **B: CNN-Transformer** | la señal cruda, 80 épocas seguidas | una CNN + Transformer resume cada época y canal en un vector, se promedian los canales presentes y un segundo Transformer relaciona las 80 épocas del trozo; en el entrenamiento se sortea la combinación de canales de cada trozo |
| **Meta-modelo** | las probabilidades de A y B en la época y en sus 2 vecinas de cada lado, y los canales presentes (54 entradas) | regresión logística + XGBoost pequeño, promediados; uno solo para las 12 combinaciones |

El codificador de época de la CNN-Transformer es una reimplementación propia de SleepEffFormer (Paul et al., 2026). La CNN-Transformer reemplaza a la CNN-BiLSTM de la versión anterior, que miraba 20 épocas con una BiLSTM.

## Contenido

```
├── experimentacion/
│   ├── modelo_a_xgboost/
│   │   ├── 1_preprocesamiento.ipynb   lee las noches y calcula los 719 rasgos por época
│   │   └── 2_entrenamiento.ipynb      validación cruzada, modelo final y resultados, con los 719 rasgos
│   ├── modelo_a_xgboost_100_features/
│   │   └── 1_entrenamiento.ipynb      el mismo XGBoost con las 100 columnas más importantes (usa la tabla de rasgos de arriba)
│   ├── modelo_b_red/
│   │   ├── 1_preprocesamiento.ipynb   filtra y normaliza la señal de cada noche
│   │   └── 2_entrenamiento.ipynb      arquitectura, validación cruzada, modelo final y resultados
│   ├── meta_modelo/
│   │   ├── 1_entrenamiento.ipynb      combina A y B: validación cruzada, modelo final y resultados
│   │   └── 2_confianza.ipynb          canales al azar por noche, umbral de confianza y aviso por noche
│   └── semillas/                      el sistema completo con las semillas 1, 2, 3 y 4
│       ├── 1_entrenamiento_red.ipynb  la CNN-Transformer con cada semilla (GPU)
│       ├── 2_entrenamiento_xgboost.ipynb  el XGBoost de 100 columnas con cada semilla
│       ├── 3_meta_modelo.ipynb        el meta-modelo y el umbral de confianza de cada semilla
│       ├── 4_resumen.ipynb            comparación de las 5 semillas
│       └── modelos/semilla_S/         las piezas de cada semilla, con las mismas carpetas que modelo/
├── modelo/
│   ├── modelo_a_xgboost/
│   │   ├── xgb.json                   el XGBoost de 719 rasgos (612 rondas)
│   │   └── columnas.json              los 719 rasgos de entrada, en orden
│   ├── modelo_a_xgboost_100_features/ el XGBoost con las 100 columnas más importantes (el que usa el sistema)
│   │   ├── xgb.json                   542 rondas
│   │   └── columnas.json              las 100 columnas de entrada, en orden
│   ├── modelo_b_red/
│   │   └── modelo_final.pt            pesos e hiperparámetros de la red (660 069 parámetros)
│   ├── meta_modelo/
│   │   ├── meta_lr.joblib             la regresión logística, con su estandarización
│   │   ├── meta_xgb.json              el XGBoost (294 rondas)
│   │   └── columnas.json              las 54 entradas, en orden
│   └── sistema.json                   qué archivos forman el sistema, cómo se combinan, umbral de confianza y aviso por noche
├── sistema/                           código para predecir una noche nueva con los modelos de modelo/
│   ├── lectura.py                     EDF (y, si hay, hipnograma) -> épocas de 30 s
│   ├── modelo_a.py                    rasgos por época + XGBoost con canales enmascarados
│   ├── modelo_b.py                    red CNN-Transformer sobre la señal cruda
│   ├── meta.py                        meta-modelo que combina A y B
│   └── prediccion.py                  junta todo: estadio, confianza, 2.ª opción y "revisar"
├── despliegue/                        endpoint de SageMaker, registro en MLflow, E2E y prueba de humo (ver despliegue/README.md)
├── tests/                             pruebas del sistema con una noche inventada (pytest tests)
├── particiones.json                   partición por sujeto: 16 de test + 5 folds sobre 62
├── requirements.txt
└── data/                              los datos (no se suben al repositorio)
```

## Datos

[Sleep-EDF Database Expanded v1.0.0](https://physionet.org/content/sleep-edfx/1.0.0/), subconjunto Sleep Cassette: 153 noches de 78 personas sanas. Se ponen a mano en `data/`:

```
data/
├── sleep-cassette/      los 306 EDF (SC4ssNE0-PSG.edf y SC4ssN*-Hypnogram.edf)
└── SC-subjects.xls
```

Lo que calculan los notebooks queda en `data/procesados/<modelo>/`.

## Cómo correrlo

```bash
pip install -r requirements.txt
```

Con Python 3.11, los notebooks de cada modelo se corren en orden. El meta-modelo va al final, porque usa las predicciones de A y B:

| Notebook | Qué hace |
|---|---|
| `modelo_a_xgboost/1_preprocesamiento.ipynb` | rasgos de las 153 noches → `data/procesados/modelo_a_xgboost/rasgos.parquet` |
| `modelo_a_xgboost/2_entrenamiento.ipynb` | 5 folds + modelo final con los 719 rasgos → `modelo/modelo_a_xgboost/` |
| `modelo_a_xgboost_100_features/1_entrenamiento.ipynb` | lo mismo con las 100 columnas más importantes → `modelo/modelo_a_xgboost_100_features/` |
| `modelo_b_red/1_preprocesamiento.ipynb` | señal filtrada y normalizada → `data/procesados/modelo_b_red/noches/` |
| `modelo_b_red/2_entrenamiento.ipynb` | 5 folds + modelo final, en paralelo en una GPU → `modelo/modelo_b_red/` |
| `meta_modelo/1_entrenamiento.ipynb` | 5 folds + modelo final con las predicciones de A (100 columnas) y B → `modelo/meta_modelo/` y `modelo/sistema.json`, que anota qué XGBoost usa el sistema |
| `meta_modelo/2_confianza.ipynb` | umbral de confianza y aviso por noche, elegidos con la CV → `modelo/sistema.json` |
| `semillas/1_entrenamiento_red.ipynb` … `4_resumen.ipynb` | el sistema completo con las semillas 1–4 → `experimentacion/semillas/modelos/semilla_S/`; no toca `modelo/` |

El XGBoost y el meta-modelo corren en CPU. La CNN-Transformer necesita una GPU NVIDIA; se entrenó en una A100 de Google Colab. En Colab, el repositorio va en Google Drive, en `MyDrive/prueba_MAIA_Trabajo_de_Grado_DeepSleep_Stacking`, con los EDF en su `data/sleep-cassette/`.

**Reproducibilidad (semilla 42):**
- XGBoost: es determinista. El de 719 rasgos da 592, 656, 660, 508 y 644 rondas en los folds y 612 en el modelo final; el de 100 columnas, 703, 753, 477, 374 y 407, y 542. Los dos reproducen byte a byte sus archivos de `modelo/`.
- CNN-Transformer: en una A100, las mejores pasadas son 96, 74, 89, 75 y 35 en los folds y 88 en el modelo final. En otra GPU los decimales pueden cambiar un poco.
- Meta-modelo: su XGBoost da 465, 229, 206, 260 y 145 rondas en los folds y 294 en el modelo final. Como depende de las predicciones de la red, una diferencia de 10⁻⁷ en la GPU puede mover algunas rondas, pero las métricas cambian menos de 0.001.

## Predecir una noche

`sistema/` predice una noche nueva con los modelos ya entrenados de `modelo/`, sin volver a entrenar. Las funciones son las mismas de los notebooks:

```python
from sistema import cargar_sistema, predecir_noche

sistema = cargar_sistema()   # carga las tres piezas una vez
salida = predecir_noche("data/sleep-cassette/SC4001E0-PSG.edf", canales=("fpz",), sistema=sistema)
```

`salida` tiene una fila por época de 30 s con el estadio, la confianza, la 2.ª opción, si hay que revisarla y las probabilidades de los 5 estadios. En `salida.attrs` quedan la fracción de la noche a revisar y si se dispara el aviso por noche.

- **Canales:** cualquier combinación con al menos un EEG, entre `fpz`, `pz`, `eog` y `emg`. Por defecto, los que tenga el EDF.
- **Sin hipnograma** se analiza todo el registro, o el rango que se pida con `epoca_inicio=` y `epoca_fin=` (épocas de 30 s, con los dos extremos incluidos). Con `hipnograma=` se analiza la ventana de sueño, como en el entrenamiento.
- **Hora de apagado de la luz:** con `hora_apagado="HH:MM:SS"`. Si no se da, se cuenta desde la primera época analizada.

**Verificado contra los notebooks** (noches de test, 4 combinaciones de canales): el modelo A da las mismas probabilidades (diferencia 0). La red se entrenó y predijo en una GPU con media precisión, y en CPU sus probabilidades difieren hasta en 2·10⁻³; el estadio elegido cambia en 1 de cada 10 000 épocas en la red y en 3 de cada 10 000 en el meta-modelo.

**En producción:** el endpoint asíncrono de SageMaker `sistema-sueno` corre `sistema.analizar` con los modelos de `modelo/`. Cada PR que cambie `modelo/`, `sistema/`, `despliegue/` o `tests/` corre las pruebas y el E2E, que compara el candidato con producción en las 32 noches de test y deja la tabla en el PR. Cada merge a `main` que cambie lo que va dentro del endpoint (`modelo/`, `sistema/`, `despliegue/inference.py` o `despliegue/requirements.txt`) registra una versión nueva en MLflow y actualiza el endpoint (ver [despliegue/README.md](despliegue/README.md)).

## Partición

Toda partición es **por sujeto**: las dos noches de una persona van siempre juntas. Son 78 sujetos y 153 noches, porque los sujetos 13, 36 y 52 tienen una sola noche. La partición completa está en `particiones.json`.

<table>
  <tr><th>Grupo</th><th>Sujetos</th><th>Para qué</th></tr>
  <tr><td><b>Test</b> (16 sujetos, 32 noches)</td><td>0, 4, 18, 21, 25, 26, 28, 33, 37, 41, 43, 51, 57, 59, 60, 61</td><td>evaluación final; no se usa para ninguna decisión</td></tr>
  <tr><td>Fold 1</td><td>8, 16, 24, 29, 34, 49, 52, 53, 55, 56, 62, 64, 72</td><td rowspan="5">validación cruzada: cada fold se predice con un modelo entrenado con los otros 4</td></tr>
  <tr><td>Fold 2</td><td>2, 9, 17, 19, 22, 31, 32, 36, 46, 48, 63, 67, 73</td></tr>
  <tr><td>Fold 3</td><td>11, 12, 13, 14, 27, 30, 42, 50, 54, 77, 80, 81</td></tr>
  <tr><td>Fold 4</td><td>5, 10, 15, 23, 38, 40, 45, 70, 74, 75, 76, 82</td></tr>
  <tr><td>Fold 5</td><td>1, 3, 6, 7, 20, 35, 44, 47, 58, 65, 66, 71</td></tr>
</table>

Los 5 folds suman los 62 sujetos de desarrollo. En cada fold, 5 de los sujetos de entrenamiento se apartan como validación para la parada temprana (por ejemplo, en el fold 1 son 5, 40, 58, 75 y 80). El modelo final del XGBoost se entrena con los 62, sin validación; el de la CNN-Transformer y el del meta-modelo, con 56, y apartan 6 para elegir la mejor pasada o parar el XGBoost.

## Resultados

**CV:** predicciones fuera de fold de los 62 sujetos de desarrollo. **Test:** los 16 sujetos de test. MF1 = F1 macro, la métrica principal.

### Modelo A: XGBoost de 719 rasgos

| Canales | CV MF1 | Test MF1 | Test κ | Test F1 N1 |
|---|---|---|---|---|
| Solo Fpz-Cz | 0.769 | 0.727 | 0.702 | 0.405 |
| Solo Pz-Oz | 0.728 | 0.701 | 0.672 | 0.414 |
| Fpz-Cz + Pz-Oz | 0.784 | 0.745 | 0.721 | 0.452 |
| Fpz-Cz + EOG | 0.788 | 0.755 | 0.728 | 0.467 |
| Pz-Oz + EOG | 0.765 | 0.724 | 0.693 | 0.452 |
| Fpz-Cz + Pz-Oz + EOG | 0.799 | 0.761 | 0.736 | 0.478 |
| Fpz-Cz + EMG | 0.776 | 0.737 | 0.709 | 0.430 |
| Pz-Oz + EMG | 0.736 | 0.714 | 0.681 | 0.442 |
| Fpz-Cz + Pz-Oz + EMG | 0.789 | 0.753 | 0.728 | 0.469 |
| Fpz-Cz + EOG + EMG | 0.791 | 0.756 | 0.728 | 0.473 |
| Pz-Oz + EOG + EMG | 0.768 | 0.731 | 0.701 | 0.458 |
| **Los 4 canales** | **0.801** | **0.764** | **0.739** | **0.486** |

### Modelo A con las 100 columnas más importantes

Es el XGBoost que usa el sistema. Mismos hiperparámetros, semillas y folds que el de 719; solo cambia el número de columnas. Δ = frente al de 719.

| Canales | CV MF1 | Δ CV | Test MF1 | Δ test | Test κ | Test F1 N1 |
|---|---|---|---|---|---|---|
| Solo Fpz-Cz | 0.760 | −0.009 | 0.718 | −0.009 | 0.688 | 0.396 |
| Solo Pz-Oz | 0.716 | −0.012 | 0.690 | −0.011 | 0.661 | 0.374 |
| Fpz-Cz + Pz-Oz | 0.776 | −0.008 | 0.736 | −0.009 | 0.708 | 0.442 |
| Fpz-Cz + EOG | 0.770 | −0.018 | 0.735 | −0.020 | 0.705 | 0.429 |
| Pz-Oz + EOG | 0.742 | −0.023 | 0.705 | −0.020 | 0.675 | 0.407 |
| Fpz-Cz + Pz-Oz + EOG | 0.783 | −0.015 | 0.745 | −0.016 | 0.716 | 0.457 |
| Fpz-Cz + EMG | 0.769 | −0.007 | 0.731 | −0.006 | 0.699 | 0.426 |
| Pz-Oz + EMG | 0.728 | −0.008 | 0.706 | −0.007 | 0.673 | 0.415 |
| Fpz-Cz + Pz-Oz + EMG | 0.781 | −0.008 | 0.746 | −0.007 | 0.718 | 0.461 |
| Fpz-Cz + EOG + EMG | 0.778 | −0.014 | 0.744 | −0.012 | 0.713 | 0.454 |
| Pz-Oz + EOG + EMG | 0.750 | −0.018 | 0.718 | −0.014 | 0.685 | 0.438 |
| **Los 4 canales** | **0.788** | −0.013 | **0.754** | −0.010 | **0.726** | **0.471** |

Solo, pierde −0.013 de MF1 en la CV y −0.012 en el test, en promedio; más en las combinaciones con EOG, porque de sus 132 columnas quedan 5. En el sistema completo la pérdida desaparece: con canales al azar por noche, el meta-modelo da 0.767 con el de 100 frente a 0.768 con el de 719.

### Modelo B: CNN-Transformer

| Canales | CV MF1 | Test MF1 | Test κ | Test F1 N1 |
|---|---|---|---|---|
| Solo Fpz-Cz | 0.766 | 0.749 | 0.725 | 0.509 |
| Solo Pz-Oz | 0.737 | 0.710 | 0.672 | 0.478 |
| Fpz-Cz + Pz-Oz | 0.783 | 0.764 | 0.741 | 0.518 |
| Fpz-Cz + EOG | 0.780 | 0.758 | 0.727 | 0.523 |
| Pz-Oz + EOG | 0.770 | 0.737 | 0.698 | 0.496 |
| Fpz-Cz + Pz-Oz + EOG | 0.790 | 0.768 | 0.741 | 0.525 |
| Fpz-Cz + EMG | 0.772 | 0.752 | 0.727 | 0.514 |
| Pz-Oz + EMG | 0.740 | 0.713 | 0.674 | 0.480 |
| Fpz-Cz + Pz-Oz + EMG | 0.784 | 0.764 | 0.740 | 0.523 |
| Fpz-Cz + EOG + EMG | 0.781 | 0.756 | 0.724 | 0.522 |
| Pz-Oz + EOG + EMG | 0.769 | 0.740 | 0.702 | 0.495 |
| **Los 4 canales** | **0.789** | **0.764** | **0.736** | **0.524** |

### Meta-modelo

Test MF1 de A (el XGBoost de 100 columnas), de B, de la votación simple (promedio de A y B) y del meta-modelo. Entre corchetes, el IC 95 % por bootstrap agrupado por sujeto (2000 remuestras).

| Canales | CV MF1 | Test A | Test B | Test votación | **Test meta-modelo** | Test κ | Test F1 N1 |
|---|---|---|---|---|---|---|---|
| Solo Fpz-Cz | 0.799 | 0.718 | 0.749 | 0.764 | **0.772** [0.736, 0.800] | 0.748 | 0.512 |
| Solo Pz-Oz | 0.760 | 0.690 | 0.710 | 0.729 | **0.735** [0.682, 0.772] | 0.703 | 0.488 |
| Fpz-Cz + Pz-Oz | 0.809 | 0.736 | 0.764 | 0.776 | **0.780** [0.750, 0.805] | 0.757 | 0.519 |
| Fpz-Cz + EOG | 0.808 | 0.735 | 0.758 | 0.771 | **0.779** [0.746, 0.807] | 0.751 | 0.523 |
| Pz-Oz + EOG | 0.788 | 0.705 | 0.737 | 0.744 | **0.751** [0.697, 0.790] | 0.716 | 0.502 |
| Fpz-Cz + Pz-Oz + EOG | 0.817 | 0.745 | 0.768 | 0.779 | **0.784** [0.754, 0.810] | 0.758 | 0.527 |
| Fpz-Cz + EMG | 0.802 | 0.731 | 0.752 | 0.765 | **0.771** [0.735, 0.801] | 0.745 | 0.519 |
| Pz-Oz + EMG | 0.764 | 0.706 | 0.713 | 0.732 | **0.737** [0.683, 0.776] | 0.702 | 0.493 |
| Fpz-Cz + Pz-Oz + EMG | 0.810 | 0.746 | 0.764 | 0.776 | **0.780** [0.750, 0.805] | 0.755 | 0.525 |
| Fpz-Cz + EOG + EMG | 0.810 | 0.744 | 0.756 | 0.771 | **0.778** [0.745, 0.806] | 0.747 | 0.527 |
| Pz-Oz + EOG + EMG | 0.790 | 0.718 | 0.740 | 0.746 | **0.753** [0.704, 0.791] | 0.718 | 0.503 |
| **Los 4 canales** | **0.818** | 0.754 | 0.764 | 0.778 | **0.783** [0.752, 0.809] | 0.756 | 0.527 |

Diferencia de MF1 en el test, con su IC 95 % pareado (mismas remuestras para todos):

| Meta-modelo frente a | ΔMF1 en las 12 combinaciones | Significativas |
|---|---|---|
| A (XGBoost) | +0.029 a +0.053 | 12 de 12 |
| B (CNN-Transformer) | +0.013 a +0.025 | 7 de 12 |
| Votación simple | +0.004 a +0.008 | 1 de 12 |

### F1 por estadio en el test

| Modelo | Canales | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|
| A: XGBoost (100 columnas) | Solo Fpz-Cz | 0.878 | 0.396 | 0.803 | 0.770 | 0.745 |
| B: CNN-Transformer | Solo Fpz-Cz | 0.915 | 0.509 | 0.803 | 0.712 | 0.805 |
| **Meta-modelo** | Solo Fpz-Cz | **0.915** | **0.512** | **0.831** | **0.788** | **0.811** |
| A: XGBoost (100 columnas) | Los 4 canales | 0.907 | 0.471 | 0.809 | 0.787 | 0.796 |
| B: CNN-Transformer | Los 4 canales | 0.921 | 0.524 | 0.795 | 0.726 | 0.854 |
| **Meta-modelo** | Los 4 canales | **0.922** | **0.527** | **0.823** | **0.795** | 0.847 |

### Canales al azar por noche

En la práctica cada paciente llega con una combinación distinta de canales. En cada una de 2000 remuestras de sujetos del test, a cada noche se le sortea una de las 12 combinaciones:

| Modelo | MF1 [IC 95 %] | κ | F1 N1 | Exactitud |
|---|---|---|---|---|
| A: XGBoost (100 columnas) | 0.727 [0.681, 0.765] | 0.697 | 0.432 | 0.776 |
| B: CNN-Transformer | 0.748 [0.707, 0.779] | 0.717 | 0.509 | 0.787 |
| **Meta-modelo** | **0.767** [0.728, 0.797] | **0.738** | **0.514** | **0.804** |

El meta-modelo supera a A en +0.039 [+0.023, +0.061] y a B en +0.019 [+0.003, +0.036], las dos de forma significativa. Con el XGBoost de 719 daba 0.768 [0.730, 0.799].

### Confianza

La confianza de una época es la probabilidad de la clase elegida. Si es menor que el umbral, la época se marca **revisar** y el técnico recibe la 1.ª y la 2.ª opción. Las dos reglas se fijaron con la CV, antes de mirar el test:
- **umbral:** el menor de 0.50, 0.55, …, 0.95 con el que lo que entrega el sistema acierta al menos el 90 % en la CV. Resultado: **0.65** (en la CV entrega el 82.3 % de la noche con acierto 0.909);
- **aviso por noche:** si la fracción a revisar supera el percentil 90 de las noches de la CV, se sugiere revisar la noche completa. Resultado: **27.7 %**.

Los dos quedan en `modelo/sistema.json`. Test con el umbral 0.65:

| Canales | Entrega el sistema [IC 95 %] | Acierto en lo que entrega [IC 95 %] | Correcta entre 2 opciones en "revisar" | Acierto sin → con revisión |
|---|---|---|---|---|
| Solo Fpz-Cz | 79.9 % [76.5, 82.9] | 0.888 [0.866, 0.907] | 0.853 | 0.811 → 0.881 |
| Solo Pz-Oz | 78.3 % [75.4, 81.1] | 0.847 [0.771, 0.897] | 0.878 | 0.778 → 0.854 |
| Fpz-Cz + EOG | 80.5 % [77.8, 82.9] | 0.890 [0.870, 0.908] | 0.859 | 0.814 → 0.883 |
| Los 4 canales | 82.3 % [79.2, 84.9] | 0.886 [0.864, 0.906] | 0.868 | 0.817 → 0.883 |
| **Canales al azar por noche** | **80.2 %** [77.2, 82.9] | **0.878** [0.837, 0.904] | **0.861** | **0.804 → 0.874** |

"Con revisión" supone que el técnico elige bien entre las 2 opciones. En el test, 56 de las 384 noches × combinación disparan el aviso, y su acierto medio es 0.732, frente a 0.815 en las demás.

**Calibración (ECE):** 0.004 en la CV y 0.022 en el test, solo con Fpz-Cz. En el test el sistema está algo sobreconfiado: confianza media 0.834 frente a acierto 0.811.

### Semillas de entrenamiento

El sistema en producción se entrenó con la semilla 42. Para ver si el resultado fue suerte, se entrenó completo otra vez con las semillas 1, 2, 3 y 4: la red, el XGBoost y el meta-modelo, cada uno con la semilla de su sistema. La partición, los hiperparámetros, el preprocesamiento y las 100 columnas no cambian, y el bootstrap usa siempre la semilla 42, para medir las 5 con las mismas remuestras. Test:

| Métrica | 42 | 1 | 2 | 3 | 4 | Media ± DE |
|---|---|---|---|---|---|---|
| **MF1, canales al azar** | 0.767 | 0.770 | 0.764 | 0.769 | 0.770 | **0.768 ± 0.002** |
| κ, canales al azar | 0.738 | 0.742 | 0.734 | 0.742 | 0.738 | 0.739 ± 0.003 |
| F1 N1, canales al azar | 0.514 | 0.522 | 0.510 | 0.513 | 0.526 | 0.517 ± 0.007 |
| MF1, solo Fpz-Cz | 0.772 | 0.775 | 0.767 | 0.773 | 0.773 | 0.772 ± 0.003 |
| MF1, los 4 canales | 0.783 | 0.786 | 0.781 | 0.785 | 0.787 | 0.784 ± 0.002 |
| MF1, solo Pz-Oz | 0.735 | 0.737 | 0.734 | 0.736 | 0.738 | 0.736 ± 0.002 |
| Umbral de confianza | 0.65 | 0.65 | 0.65 | 0.65 | 0.65 | 0.65 |
| Entrega el sistema | 80.2 % | 81.6 % | 79.2 % | 80.9 % | 81.0 % | 80.6 % ± 0.9 |
| Acierto en lo que entrega | 0.878 | 0.875 | 0.878 | 0.878 | 0.874 | 0.877 ± 0.002 |

- **Es estable:** la desviación estándar (DE) del MF1 queda bajo 0.005, el límite fijado antes de ver los resultados. El rango con canales al azar (0.006) es mucho menor que el ancho del IC 95 % por tener 16 sujetos de test (0.070).
- **La 42 no fue suerte:** queda 4.ª de 5, a 0.001 de la media. Producción sigue con la 42; no se elige la mejor semilla, porque sería elegir con el test.
- Por piezas, el MF1 de test del XGBoost casi no cambia (DE de 0.002 o menos en cada combinación) y el de la red algo más (DE de 0.003 o menos en el test y hasta 0.007 en la CV), aunque su mejor pasada cambie mucho con la semilla.

La red de las semillas 1–4 se entrenó en una A100 de Colab, 6 entrenamientos a la vez. Las redes de los folds y las predicciones de cada semilla quedan en `data/procesados/semillas/`.

### De la CNN-BiLSTM a la CNN-Transformer

MF1 de la CV de las dos redes, con la misma partición, el mismo preprocesamiento y casi el mismo número de parámetros (659 685 frente a 660 069).

| Canales | CNN-BiLSTM (20 épocas) | CNN-Transformer (80 épocas) | Diferencia |
|---|---|---|---|
| Solo Fpz-Cz | 0.754 | 0.766 | +0.012 |
| Solo Pz-Oz | 0.721 | 0.737 | +0.016 |
| Fpz-Cz + Pz-Oz | 0.767 | 0.783 | +0.016 |
| Fpz-Cz + EOG | 0.780 | 0.780 | +0.000 |
| Pz-Oz + EOG | 0.769 | 0.770 | +0.001 |
| Fpz-Cz + Pz-Oz + EOG | 0.783 | 0.790 | +0.007 |
| Fpz-Cz + EMG | 0.756 | 0.772 | +0.016 |
| Pz-Oz + EMG | 0.725 | 0.740 | +0.015 |
| Fpz-Cz + Pz-Oz + EMG | 0.770 | 0.784 | +0.014 |
| Fpz-Cz + EOG + EMG | 0.779 | 0.781 | +0.002 |
| Pz-Oz + EOG + EMG | 0.769 | 0.769 | +0.000 |
| Los 4 canales | 0.784 | 0.789 | +0.005 |

**Qué significa:**

- Los tres modelos sirven para las 12 combinaciones, y a los tres les va mejor con más canales. El EOG es el canal que más aporta, y Fpz-Cz funciona mejor que Pz-Oz.
- **Mirar 80 épocas en lugar de 20 mejora la red:** en la CV, la CNN-Transformer supera a la CNN-BiLSTM en 10 de las 12 combinaciones y empata en las otras 2 (+0.009 en promedio). La ganancia está en las combinaciones sin EOG (+0.012 a +0.016); con EOG es pequeña (0 a +0.007).
- **A y B se equivocan en épocas distintas:** B comparte entre el 56 % y el 62 % de los errores de A en la CV, bajo la regla del 70 % del proyecto. A es mejor en N2 y N3, y B en N1 y REM.
- **El meta-modelo es el mejor modelo en las 12 combinaciones.** En la CV gana entre +0.018 y +0.033 de MF1 sobre el mejor de A y B. En el test supera de forma significativa a A en las 12 y a B en 7. Por estadio, mejora a los dos en casi todo; solo en REM con los 4 canales B sigue un poco mejor (0.854 frente a 0.847).
- **Frente a la votación simple, la ganancia es pequeña:** +0.006 a +0.010 en la CV y +0.004 a +0.008 en el test, significativa solo en 1 de las 12 combinaciones. Casi toda la mejora viene de juntar dos formas distintas de ver la noche, no de lo sofisticado del meta-modelo.
- **La meta del proyecto con un solo EEG es MF1 ≥ 0.75 y κ ≥ 0.70.** Con solo Fpz-Cz el meta-modelo la cumple (0.772 y 0.748), mientras que A y B por separado no. Con solo Pz-Oz no llega en MF1 (0.735), aunque sí en κ (0.703).
- **Con canales al azar por noche, el sistema da MF1 0.767** [0.728, 0.797] y κ 0.738: es la cifra que se espera con cualquier paciente, y supera de forma significativa a A y a B.
- **Con el umbral de 0.65, el sistema entrega solo ~80 % de la noche con ~88 % de acierto** y marca el ~20 % restante para revisión. Ahí la correcta está entre las 2 opciones el 86 % de las veces, así que con la revisión el acierto de la noche sube de 0.80 a 0.87.
- **El aviso por noche sirve:** las noches que lo disparan aciertan bastante menos (0.732 frente a 0.815).
