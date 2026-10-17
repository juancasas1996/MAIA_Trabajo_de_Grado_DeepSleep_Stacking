# Estadificación del sueño con canales variables

Proyecto de grado — Maestría en Inteligencia Artificial (MAIA), Universidad de los Andes · Grupo 3.

Clasificación automática de los estadios del sueño (W, N1, N2, N3, REM) en épocas de 30 s a partir de polisomnografía, con **cualquier combinación de canales que incluya al menos un EEG**:

| EEG (al menos uno) | EOG | EMG | Combinaciones |
|---|---|---|---|
| Fpz-Cz, Pz-Oz o los dos | con o sin | con o sin | 3 × 2 × 2 = **12** |

> **Rama experimental.** El paso 2 reemplaza la BiLSTM sobre 20 épocas (10 min) de la versión original por 2 capas de Transformer sobre 80 épocas (40 min). Sus predicciones van a `paso_3_meta_model/entradas/paso_2_transformer/`, y el paso 3 las lee con `PASO_2 = "paso_2_transformer"`. Los resultados de abajo son los de la versión original.

El sistema tiene tres piezas, y cada una es un solo modelo para las 12 combinaciones:

```
                    noche completa (los canales que traiga)
             ┌───────────────────┴────────────────────┐
  Paso 1: 911 rasgos por época             Paso 2: señal cruda, 80 épocas a la vez
  + XGBoost con canales en NaN             CNN → Transformer → Transformer entre épocas
             └───────────────┬────────────────────────┘
          probabilidades por época de cada uno (t−2 … t+2) + qué canales hay
                             │
           Paso 3: meta-modelo (regresión logística + XGBoost)
                             │
          estadio por época, confianza, 2.ª opción y "revisar"
```

## Contenido

```
├── particiones.json         partición por sujeto: 16 de test + 5 folds sobre 62, y las 12 combinaciones
├── data/                    los datos (no se suben al repositorio)
├── paso_1_xgboost/          paso_1_xgboost.ipynb
├── paso_2_deep_model/       paso_2_deep_model.ipynb
└── paso_3_meta_model/       paso_3_meta_model.ipynb
    └── entradas/            lo que dejan los pasos 1 y 2 para el paso 3
```

Cada notebook tiene todo su código, explicado sección por sección, y termina con sus matrices de confusión.

## Datos

[Sleep-EDF Database Expanded v1.0.0](https://physionet.org/content/sleep-edfx/1.0.0/), subconjunto Sleep Cassette: 153 noches de 78 personas sanas. Se ponen a mano en `data/`:

```
data/
├── sleep-cassette/                      los 306 EDF (SC4ssNE0-PSG.edf y SC4ssN*-Hypnogram.edf)
├── SC-subjects.xls
└── biot/EEG-SHHS+PREST-18-channels.ckpt pesos de BIOT, de https://github.com/ycq091044/BIOT/tree/main/pretrained-models
```

Los notebooks guardan sus cálculos intermedios en `data/procesados/`.

## Orden de ejecución

| Paso | Dónde | Tiempo aprox. |
|---|---|---|
| 1. `paso_1_xgboost.ipynb` | computador (CPU) | 2–2.5 h |
| 2. `paso_2_deep_model.ipynb` | Google Colab con GPU (A100) | 2–2.5 h |
| 3. `paso_3_meta_model.ipynb` | computador (CPU) | 20 min |

Los pasos 1 y 2 son independientes y pueden correr a la vez. El paso 3 necesita los dos.

**Paso 2 en Colab:**
- el repositorio debe estar en `MyDrive/repo`, con los datos en `repo/data/`;
- se abre el notebook con Colab, se elige una GPU A100 y se ejecuta todo;
- los 6 entrenamientos (5 folds + modelo final) corren en paralelo en la misma GPU.

Antes de la corrida larga conviene probar con `PRUEBA = True` (2 minutos). Si Colab se desconecta, al volver a correr el notebook se saltan los entrenamientos ya terminados.

```bash
pip install -r requirements.txt
```

## Resultados

Test de 16 personas que no se usaron para ninguna decisión. Entre corchetes, el IC 95 % por bootstrap agrupado por persona.

| Canales | MF1 | κ |
|---|---|---|
| Solo Fpz-Cz | 0.769 [0.733, 0.800] | 0.747 |
| Solo Pz-Oz | 0.732 [0.680, 0.770] | 0.699 |
| Fpz-Cz + EOG | 0.780 [0.745, 0.810] | 0.755 |
| Los 4 canales | 0.780 [0.750, 0.807] | 0.756 |
| **Combinación al azar por noche** | **0.764 [0.725, 0.797]** | 0.737 |

Con un umbral de confianza de 0.65, el sistema entrega ~81 % de la noche con ~88 % de acierto. El ~19 % restante lo marca para revisión, con dos opciones; la correcta está entre ellas el ~85 % de las veces.

**Reproducibilidad:** con semilla 42, los pasos 1 y 3 son deterministas, y el paso 2 entrenado en una A100 dio los mismos resultados que en el proyecto. La única diferencia posible está en los embeddings de BIOT: en el proyecto se calcularon en la GPU de un Mac, y en CPU difieren en ~10⁻⁷. Eso puede mover alguna cifra en la tercera decimal.

## Créditos

- BIOT (Yang, Westover y Sun, NeurIPS 2023), licencia MIT: el paso 1 usa su encoder y su checkpoint preentrenado sin modificarlos.
- El codificador de época del paso 2 es una reimplementación propia de SleepEffFormer (Paul et al., 2026).
