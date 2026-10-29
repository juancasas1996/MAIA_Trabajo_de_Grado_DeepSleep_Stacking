# Modelos entrenados

Los tres modelos del sistema en producción, entrenados con las 62 personas de desarrollo. Son los del informe de avance. Se generan con `python scripts/entrenamiento_completo.py`, que los escribe en esta carpeta.

| Carpeta | Archivo | Qué es |
|---|---|---|
| `modelo_a_xgboost/` | `xgb.json` | XGBoost enmascarado, 612 rondas (3 060 árboles) |
| | `columnas.json` | las 719 columnas de entrada, en orden |
| `modelo_b/` | `modelo_final.pt` | red CNN + Transformer (660 069 parámetros): pesos e hiperparámetros |
| `meta_modelo/` | `meta_lr.joblib` | regresión logística (con su estandarización) |
| | `meta_xgb.json` | XGBoost de profundidad 4: se entrenaron 338 rondas y se usan las 288 primeras (la mejor ronda, guardada como `best_iteration`) |
| | `columnas.json` | las 54 entradas, en orden |
| `sistema.json` | | umbral de confianza (0.65), aviso por noche y cómo se combinan las piezas |

Para predecir una noche con las tres piezas juntas:

```python
from sueno.sistema import analizar, cargar_sistema

sistema = cargar_sistema("modelos")
salida = analizar(sistema, {"psg": "SC4001E0-PSG.edf", "canales": ["fpz", "eog"]})
```

Si se cargan a mano: el meta-XGBoost se limita a la mejor ronda, `meta_xgb.predict(dmatrix, iteration_range=(0, int(meta_xgb.attr("best_iteration")) + 1))`; con todas las rondas da otras probabilidades. La red del modelo B se reconstruye con `sueno.modelo_b.cargar("modelos/modelo_b/modelo_final.pt")`.
