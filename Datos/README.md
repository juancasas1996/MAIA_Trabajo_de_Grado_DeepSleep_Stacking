# Datos

Los datos no se suben al repositorio: son los de [Sleep-EDF Database Expanded v1.0.0](https://physionet.org/content/sleep-edfx/1.0.0/) (PhysioNet, licencia Open Data Commons Attribution v1.0), subconjunto **Sleep Cassette**: 153 noches de 78 personas sanas. Se descargan de PhysioNet y se ponen a mano aquí:

```
Datos/
├── sleep-cassette/          los 306 EDF de Sleep Cassette
│   ├── SC4001E0-PSG.edf         señales de la noche: Fpz-Cz, Pz-Oz y EOG a 100 Hz; EMG submental (envolvente a 1 Hz), ...
│   ├── SC4001EC-Hypnogram.edf   estadios anotados por un experto cada 30 s (W, R, 1, 2, 3, 4, M y ?)
│   └── ...                      SC4ssN: ss = sujeto (00 a 82), N = noche (1 o 2)
├── SC-subjects.xls          una fila por noche: sujeto, noche, edad, sexo y hora de apagado de la luz
└── procesados/              lo que calculan los notebooks (se crea solo)
    ├── modelo_a_xgboost/            rasgos.parquet (719 rasgos por época) y folds/ (predicciones de cada fold)
    ├── modelo_a_xgboost_100_features/  folds/
    ├── modelo_b_red/                noches/ (señal filtrada y normalizada), logs/ y predicciones/
    ├── meta_modelo/                 predicciones.npz (CV y test, 12 combinaciones de canales)
    └── semillas/semilla_S/          lo mismo para las semillas 1 a 4
```

Muestra de `SC-subjects.xls`:

| subject | night | age | sex (F=1) | LightsOff |
|---|---|---|---|---|
| 0 | 1 | 33 | 1 | 00:38:00 |
| 0 | 2 | 33 | 1 | 21:57:00 |
| 1 | 1 | 33 | 1 | 22:44:00 |

Para predecir una noche nueva con los modelos de `Modelos/` basta con su EDF de señales (`*-PSG.edf`); el hipnograma y `SC-subjects.xls` son opcionales (ver "Predecir una noche" en el README principal).
