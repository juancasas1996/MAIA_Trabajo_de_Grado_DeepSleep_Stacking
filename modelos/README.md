# Modelos entrenados

Los tres modelos finales del sistema, entrenados con las 62 personas de desarrollo. Son los del informe de avance.

| Carpeta | Archivo | Qué es |
|---|---|---|
| `modelo_a_xgboost/` | `xgb.json` | XGBoost enmascarado, 612 rondas (3 060 árboles) |
| | `columnas.json` | las 719 columnas de entrada, en orden |
| `modelo_b/` | `modelo_final.pt` | red CNN + Transformer (660 069 parámetros): pesos e hiperparámetros, guardado por el paso 2 |
| `meta_modelo/` | `meta_lr.joblib` | regresión logística (con su estandarización) |
| | `meta_xgb.json` | XGBoost de profundidad 4, 338 rondas |
| | `columnas.json` | las 54 entradas, en orden |
| `sistema.json` | | umbral de confianza (0.65), aviso por noche y cómo se combinan las piezas |

**Versión sin BIOT.** El modelo A y el meta-modelo son la versión sin los embeddings de BIOT (719 columnas en lugar de 911), que es la que se reporta. El paso 1 de este repositorio todavía calcula BIOT, así que al correrlo se obtiene la versión de 911 columnas, que no coincide con estos archivos.

Para cargarlos:

```python
import json, joblib, torch, xgboost as xgb

modelo_a = xgb.Booster(); modelo_a.load_model("modelos/modelo_a_xgboost/xgb.json")
ck = torch.load("modelos/modelo_b/modelo_final.pt", map_location="cpu", weights_only=False)
meta_lr = joblib.load("modelos/meta_modelo/meta_lr.joblib")
meta_xgb = xgb.Booster(); meta_xgb.load_model("modelos/meta_modelo/meta_xgb.json")
```

La red del modelo B se reconstruye con las clases del paso 2: `ModeloB(**ck["ARQUITECTURA"])` y `load_state_dict(ck["model_state_dict"])`.
