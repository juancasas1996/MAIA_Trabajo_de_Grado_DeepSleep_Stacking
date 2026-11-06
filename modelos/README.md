# Modelos entrenados

Los tres modelos del sistema en producción, entrenados con las 62 personas de desarrollo. Es la **v2**: el modelo A usa solo sus 100 columnas más importantes (en la v1, la del informe de avance, usaba las 719). Se generan con `python scripts/entrenamiento_completo.py --columnas 100`, que los escribe en esta carpeta; las versiones anteriores quedan en el historial de git.

| Carpeta | Archivo | Qué es |
|---|---|---|
| `modelo_a_xgboost/` | `xgb.json` | XGBoost enmascarado, 542 rondas (2 710 árboles) |
| | `columnas.json` | las 100 columnas de entrada, en orden (de las 719 que calcula `sueno.rasgos`); al predecir, `rasgos_noche(noche, columnas)` calcula solo estas |
| `modelo_b/` | `modelo_final.pt` | red CNN + Transformer (660 069 parámetros): pesos e hiperparámetros; igual que en la v1 |
| `meta_modelo/` | `meta_lr.joblib` | regresión logística (con su estandarización) |
| | `meta_xgb.json` | XGBoost de profundidad 4: se entrenaron 292 rondas y se usan las 242 primeras (la mejor ronda, guardada como `best_iteration`) |
| | `columnas.json` | las 54 entradas, en orden |
| `sistema.json` | | umbral de confianza (0.65), aviso por noche y cómo se combinan las piezas |

Para predecir una noche con las tres piezas juntas:

```python
from sueno.sistema import analizar, cargar_sistema

sistema = cargar_sistema("modelos")
salida = analizar(sistema, {"psg": "SC4001E0-PSG.edf", "canales": ["fpz", "eog"]})
```

Si se cargan a mano: el meta-XGBoost se limita a la mejor ronda, `meta_xgb.predict(dmatrix, iteration_range=(0, int(meta_xgb.attr("best_iteration")) + 1))`; con todas las rondas da otras probabilidades. La red del modelo B se reconstruye con `sueno.modelo_b.cargar("modelos/modelo_b/modelo_final.pt")`.
