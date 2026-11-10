# Estadificación del sueño con canales variables

Clasificación automática de los estadios del sueño (W, N1, N2, N3, REM) en épocas de 30 s a partir de polisomnografía, con **cualquier combinación de canales que incluya al menos un EEG**:

| EEG (al menos uno) | EOG | EMG | Combinaciones |
|---|---|---|---|
| Fpz-Cz, Pz-Oz o los dos | con o sin | con o sin | 3 × 2 × 2 = **12** |

## Modelos

| Modelo | Entrada | Idea |
|---|---|---|
| **XGBoost** | 719 rasgos por época (espectro, forma de la señal, ondículas, pendiente 1/f y contexto de las épocas vecinas) | un único XGBoost para las 12 combinaciones: en el entrenamiento se apagan canales al azar (sus columnas se ponen en NaN) |
| **CNN-BiLSTM** | la señal cruda, 20 épocas seguidas | una CNN + Transformer resume cada época y canal en un vector, se promedian los canales presentes y una BiLSTM mira las épocas de antes y de después; en el entrenamiento se sortea la combinación de canales de cada trozo |

El codificador de época de la CNN-BiLSTM es una reimplementación propia de SleepEffFormer (Paul et al., 2026).

## Contenido

```
├── experimentacion/
│   ├── xgboost/
│   │   ├── 1_preprocesamiento.ipynb   lee las noches y calcula los 719 rasgos por época
│   │   └── 2_entrenamiento.ipynb      validación cruzada, modelo final y resultados
│   └── cnn_bilstm/
│       ├── 1_preprocesamiento.ipynb   filtra y normaliza la señal de cada noche
│       └── 2_entrenamiento.ipynb      arquitectura, validación cruzada, modelo final y resultados
├── modelo/
│   ├── xgboost/
│   │   ├── xgb.json                   el modelo entrenado (612 rondas)
│   │   └── columnas.json              los 719 rasgos de entrada, en orden
│   └── cnn_bilstm/
│       └── modelo_final.pt            pesos e hiperparámetros de la red (659 685 parámetros)
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

Con Python 3.11, los notebooks de cada modelo se corren en orden:

| Notebook | Qué hace |
|---|---|
| `xgboost/1_preprocesamiento.ipynb` | rasgos de las 153 noches → `data/procesados/xgboost/rasgos.parquet` |
| `xgboost/2_entrenamiento.ipynb` | 5 folds + modelo final → `modelo/xgboost/` |
| `cnn_bilstm/1_preprocesamiento.ipynb` | señal filtrada y normalizada → `data/procesados/cnn_bilstm/noches/` |
| `cnn_bilstm/2_entrenamiento.ipynb` | 5 folds + modelo final, en paralelo en una GPU → `modelo/cnn_bilstm/` |

El XGBoost corre en CPU. La CNN-BiLSTM necesita una GPU NVIDIA; se entrenó en una A100 de Google Colab. En Colab, el repositorio va en Google Drive, en `MyDrive/prueba_MAIA_Trabajo_de_Grado_DeepSleep_Stacking`, con los EDF en su `data/sleep-cassette/`.

**Reproducibilidad (semilla 42):**
- XGBoost: es determinista. Da 592, 656, 660, 508 y 644 rondas en los folds, 612 en el modelo final, y reproduce byte a byte los archivos de `modelo/xgboost/`.
- CNN-BiLSTM: en una A100, las mejores pasadas son 63, 59, 59, 90 y 44 en los folds y 94 en el modelo final. En otra GPU los decimales pueden cambiar un poco.

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

Los 5 folds suman los 62 sujetos de desarrollo. En cada fold, 5 de los sujetos de entrenamiento se apartan como validación para la parada temprana (por ejemplo, en el fold 1 son 5, 40, 58, 75 y 80). El modelo final del XGBoost se entrena con los 62, sin validación; el de la CNN-BiLSTM, con 56, y aparta 6 para elegir la mejor pasada.

## Resultados

**CV:** predicciones fuera de fold de los 62 sujetos de desarrollo. **Test:** los 16 sujetos de test. MF1 = F1 macro, la métrica principal.

### XGBoost

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

### CNN-BiLSTM

| Canales | CV MF1 | Test MF1 | Test κ | Test F1 N1 |
|---|---|---|---|---|
| Solo Fpz-Cz | 0.754 | 0.750 | 0.727 | 0.496 |
| Solo Pz-Oz | 0.721 | 0.698 | 0.659 | 0.452 |
| Fpz-Cz + Pz-Oz | 0.767 | 0.755 | 0.733 | 0.502 |
| Fpz-Cz + EOG | 0.780 | 0.761 | 0.733 | 0.522 |
| Pz-Oz + EOG | 0.769 | 0.727 | 0.687 | 0.491 |
| Fpz-Cz + Pz-Oz + EOG | 0.783 | 0.765 | 0.740 | 0.516 |
| Fpz-Cz + EMG | 0.756 | 0.751 | 0.725 | 0.501 |
| Pz-Oz + EMG | 0.725 | 0.707 | 0.668 | 0.470 |
| Fpz-Cz + Pz-Oz + EMG | 0.770 | 0.754 | 0.731 | 0.502 |
| Fpz-Cz + EOG + EMG | 0.779 | 0.758 | 0.730 | 0.514 |
| Pz-Oz + EOG + EMG | 0.769 | 0.727 | 0.688 | 0.479 |
| **Los 4 canales** | **0.784** | **0.762** | **0.736** | **0.512** |

### F1 por estadio en el test

| Modelo | Canales | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|---|
| XGBoost | Solo Fpz-Cz | 0.890 | 0.405 | 0.813 | 0.771 | 0.756 |
| CNN-BiLSTM | Solo Fpz-Cz | 0.918 | 0.496 | 0.813 | 0.738 | 0.784 |
| XGBoost | Los 4 canales | 0.918 | 0.486 | 0.817 | 0.789 | 0.811 |
| CNN-BiLSTM | Los 4 canales | 0.919 | 0.512 | 0.811 | 0.736 | 0.832 |

**Qué significa:**

- Los dos modelos sirven para las 12 combinaciones, y a los dos les va mejor con más canales. El EOG es el canal que más aporta, y Fpz-Cz funciona mejor que Pz-Oz.
- **En la CV, la CNN-BiLSTM queda por debajo del XGBoost** en MF1: entre 0.01 y 0.02 en 10 de las 12 combinaciones. Solo con Pz-Oz + EOG (con o sin EMG) queda igual o un poco arriba. En el test, en cambio, la red queda por encima con Fpz-Cz (0.750 frente a 0.727) y muy cerca con los 4 canales (0.762 frente a 0.764).
- **La red le gana al XGBoost en N1** en todas las combinaciones, entre +0.02 y +0.09 de F1 en el test, y también en REM. El XGBoost es mejor en N3.
- La meta del proyecto con un solo EEG es MF1 ≥ 0.75 y κ ≥ 0.70. Con solo Fpz-Cz, la CNN-BiLSTM llega justo a la meta (0.750 y 0.727); el XGBoost no la alcanza en MF1 (0.727). Con solo Pz-Oz, ninguno de los dos llega.
- Cada modelo tiene sus fortalezas: el XGBoost mira rasgos de cada época y es mejor en N3, y la red aprende de la señal cruda y de las épocas vecinas y es mejor en N1 y REM.
