# Despliegue en SageMaker

El sistema se sirve en un **endpoint asíncrono de SageMaker** (`sistema-sueno`, una ml.m5.large sin GPU). Recibe un EDF que está en S3 y deja la respuesta en S3. Se apaga solo tras 5 min sin peticiones y se prende cuando llega una (~8 min). Cada versión desplegada está registrada en **MLflow** (app de MLflow de SageMaker, modelo `sistema-sueno`) con los tres modelos de `modelo/`, el código de `sistema/` y el commit de git del que salió.

## Flujo con GitHub Actions

```
merge de un PR a main que cambie modelo/, sistema/ o despliegue/
                                            │  el merge es la aprobación humana
                                            ▼
        workflow "Despliegue"   registra en MLflow ─► actualiza sistema-sueno ─► prueba de humo ─► tag modelo-v<versión>
```

- **El modelo nunca se despliega en producción sin el merge** de una persona.
- Si la actualización del endpoint falla, SageMaker se queda con la versión anterior.
- Para volver a una versión anterior: revertir el PR y hacer merge del revert, o `python despliegue/desplegar.py produccion <versión>`.

## Secret del repositorio

Un solo secret, en GitHub, *Settings → Secrets and variables → Actions → New repository secret*:

- **Name:** `AWS_CREDENCIALES`
- **Secret:** el bloque completo de AWS Academy (*AWS Details → AWS CLI → Show*), tal cual, desde `[default]` hasta el final del `aws_session_token`.

El workflow lo escribe en `~/.aws/credentials`, igual que en un computador. Las credenciales de AWS Academy vencen a las ~4 h: antes del merge hay que actualizar el secret (*Update secret*) con el bloque nuevo. Si vencieron, el workflow falla en el paso "Credenciales de AWS" sin tocar nada, y basta con actualizarlo y volver a correrlo (*Re-run jobs*).

La cuenta, el bucket (`maia-g3-sueno-<cuenta>`), el rol (`LabRole`) y la app de MLflow (`maia-g3-mlflow`) salen de las credenciales; se pueden cambiar con `BUCKET`, `ROL_SAGEMAKER` y `MLFLOW_ARN`.

## Archivos

| Archivo | Qué hace |
|---|---|
| `inference.py` | lo que corre dentro del endpoint: baja el EDF de S3 y llama a `sistema.analizar` |
| `modelo_mlflow.py` | el sistema como modelo de MLflow (pyfunc) |
| `registrar.py` | registra `modelo/` + `sistema/` como una versión nueva en MLflow |
| `desplegar.py` | `produccion [versión]`: actualiza el endpoint de producción |
| `humo.py` | prueba de humo después de desplegar: una noche de test con los 4 canales (las noches de test y sus estadios ya están en S3, en `sistema-sueno/test/`) |
| `cliente.py` | manda peticiones al endpoint asíncrono y recoge las respuestas |
| `config.py` | nombres de los recursos de AWS |
| `requirements.txt` | librerías del contenedor del endpoint |
| `requirements-aws.txt` | librerías para registrar y desplegar |

## Desde un computador

```bash
pip install -r requirements.txt -r despliegue/requirements-aws.txt
# credenciales en ~/.aws/credentials

python despliegue/registrar.py                     # versión nueva en MLflow
python despliegue/desplegar.py produccion          # la última versión registrada, a producción
python despliegue/humo.py                          # una noche de test al endpoint
```

**Petición al endpoint** (JSON en S3): `{"s3_uri": "s3://.../SC4001E0-PSG.edf", "canales": ["fpz", "eog"], "hora_apagado": "22:44:00", "epoca_inicio": 800, "epoca_fin": 1700}`. Solo `s3_uri` es obligatorio. La respuesta es la de `sistema.analizar`: por época, el estadio, la confianza, la 2.ª opción, su probabilidad, si hay que revisarla y las 5 probabilidades; para la noche, la fracción a revisar, el aviso y `segundos_proceso`.
