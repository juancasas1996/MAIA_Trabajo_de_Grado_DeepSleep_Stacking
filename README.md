# Estadificación del sueño con canales variables

Clasificación automática de los estadios del sueño (W, N1, N2, N3, REM) en épocas de 30 s a partir de polisomnografía, con **cualquier combinación de canales que incluya al menos un EEG**:

| EEG (al menos uno) | EOG | EMG | Combinaciones |
|---|---|---|---|
| Fpz-Cz, Pz-Oz o los dos | con o sin | con o sin | 3 × 2 × 2 = **12** |

## Etapa actual: XGBoost con 719 rasgos

Cada época se describe con 719 rasgos calculados a partir de la señal (espectro, forma de la señal, ondículas, pendiente 1/f y contexto de las épocas vecinas). Un **único XGBoost** clasifica las épocas con cualquiera de las 12 combinaciones: en el entrenamiento se apagan canales al azar (sus columnas se ponen en NaN), y el modelo aprende a clasificar con lo que tenga.

## Contenido

```
├── experimentacion/
│   └── xgboost/
│       ├── 1_preprocesamiento.ipynb   lee las noches y calcula los 719 rasgos por época
│       └── 2_entrenamiento.ipynb      validación cruzada, modelo final y resultados
├── modelo/
│   └── xgboost/
│       ├── xgb.json                   el modelo entrenado (612 rondas)
│       └── columnas.json              los 719 rasgos de entrada, en orden
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

Lo que calculan los notebooks queda en `data/procesados/xgboost/`.

## Cómo correrlo

```bash
pip install -r requirements.txt
```

Con Python 3.11, se corren en orden:

| Notebook | Qué hace |
|---|---|
| `1_preprocesamiento.ipynb` | rasgos de las 153 noches → `data/procesados/xgboost/rasgos.parquet` |
| `2_entrenamiento.ipynb` | 5 folds + modelo final → `modelo/xgboost/` |

Con semilla 42, el entrenamiento es determinista: da 592, 656, 660, 508 y 644 rondas en los folds, 612 en el modelo final, y reproduce byte a byte los archivos de `modelo/xgboost/`.

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

Los 5 folds suman los 62 sujetos de desarrollo. En cada fold, 5 de los sujetos de entrenamiento se apartan como validación para el early stopping (por ejemplo, en el fold 1 son 5, 40, 58, 75 y 80). El modelo final se entrena con los 62, sin validación.

## Resultados

**CV:** predicciones fuera de fold de los 62 sujetos de desarrollo. **Test:** los 16 sujetos de test. MF1 = F1 macro, la métrica principal.

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

F1 por estadio en el test:

| Canales | W | N1 | N2 | N3 | REM |
|---|---|---|---|---|---|
| Solo Fpz-Cz | 0.890 | 0.405 | 0.813 | 0.771 | 0.756 |
| Los 4 canales | 0.918 | 0.486 | 0.817 | 0.789 | 0.811 |

**Qué significa:**

- Un solo modelo sirve para las 12 combinaciones. Con más canales le va mejor: de 0.701 (solo Pz-Oz) a 0.764 (los 4) de MF1 en el test.
- El EOG es el canal que más aporta, y Fpz-Cz funciona mejor que Pz-Oz.
- **N1 es el punto débil:** su F1 queda entre 0.40 y 0.49, porque se confunde con W y con N2.
- La meta del proyecto con un solo EEG es MF1 ≥ 0.75 y κ ≥ 0.70. Con solo Fpz-Cz, el XGBoost cumple en κ (0.702) pero no en MF1 (0.727).
