# Despliegue en SageMaker y CI/CD

El sistema se sirve en un **endpoint asíncrono de SageMaker** (`sistema-sueno`, una ml.m5.large sin GPU). Recibe un EDF que está en S3 y deja la respuesta en S3. Se apaga solo tras 5 min sin peticiones y se prende cuando llega una (~8 min). Cada versión desplegada está registrada en **MLflow** (app de MLflow de SageMaker, modelo `sistema-sueno`) con los tres modelos, el código de `src/sueno` y el commit de git del que salió.

## Flujo con GitHub Actions

```
rama con un modelo nuevo en modelos/  ──►  PR a main
                                            │
        workflow "Pruebas"                  ├─ pytest (pruebas unitarias y del modelo)
        workflow "E2E"                      ├─ pytest ─► si falla, se para
                                            │   └─ registra el candidato en MLflow
                                            │      despliega un endpoint aparte (sistema-sueno-pr-N)
                                            │      32 noches de test al candidato y a producción
                                            │      comenta en el PR: MF1, κ, F1 por estadio, matriz, latencia
                                            │      borra el endpoint candidato
                                            ▼
                          una persona revisa el comentario y decide
                                            │  merge = aprobación
                                            ▼
        workflow "Despliegue"   push a main ─ pytest ─► registra en MLflow ─► actualiza sistema-sueno
                                              ─► prueba de humo ─► tag modelo-v<versión>
```

- El E2E corre solo si el PR cambia `modelos/`, `src/` o `despliegue/`.
- **El modelo nunca se despliega en producción sin el merge** de una persona. El E2E no toca producción; solo le manda las 32 noches para comparar.
- Si algo falla (pruebas, E2E o despliegue), producción no cambia. Si la actualización del endpoint falla, SageMaker se queda con la versión anterior.
- Para volver a una versión anterior: revertir el PR y hacer merge del revert, o `python despliegue/desplegar.py produccion <versión>`.

**El E2E en detalle** (`e2e.py`):
- Las 32 noches de test (16 personas) se mandan con la ventana de sueño del entrenamiento. A cada noche se le asigna al azar una de las 12 combinaciones de canales (semilla 42; cada combinación sale 2 o 3 veces), la misma para el candidato y para producción, y la misma en todos los PR.
- Métricas: MF1, κ, exactitud y F1 por estadio, con IC 95 % por bootstrap agrupado por persona, y la diferencia pareada candidato − producción.
- Latencia: los segundos de proceso de cada noche dentro del contenedor (`segundos_proceso`), sin la cola ni el arranque de la máquina.
- Tarda ~30–40 min: ~10 min en crear el endpoint candidato, ~8 min en prender producción, y las noches de una en una.
- Costo: el endpoint candidato vive ~30 min (~US$0.06) y producción se prende ~15 min.

## Secret del repositorio

Un solo secret, en GitHub, *Settings → Secrets and variables → Actions → New repository secret*:

- **Name:** `AWS_CREDENCIALES`
- **Secret:** el bloque completo de AWS Academy (*AWS Details → AWS CLI → Show*), tal cual, desde `[default]` hasta el final del `aws_session_token`.

El workflow lo escribe en `~/.aws/credentials`, igual que en un computador. Las credenciales de AWS Academy vencen a las ~4 h: antes de abrir un PR con un modelo nuevo, o antes del merge, hay que actualizar el secret (*Update secret*) con el bloque nuevo. Si vencieron, el workflow falla en el paso "Credenciales de AWS" sin tocar nada, y basta con actualizarlo y volver a correrlo (*Re-run jobs*).

La cuenta, el bucket (`maia-g3-sueno-<cuenta>`), el rol (`LabRole`) y la app de MLflow (`maia-g3-mlflow`) salen de las credenciales; se pueden cambiar con `BUCKET`, `ROL_SAGEMAKER` y `MLFLOW_ARN`.

## Archivos

| Archivo | Qué hace |
|---|---|
| `inference.py` | lo que corre dentro del endpoint: baja el EDF de S3 y llama a `sueno.sistema.analizar` |
| `modelo_mlflow.py` | el sistema como modelo de MLflow (pyfunc) |
| `registrar.py` | registra `modelos/` + `src/sueno` como una versión nueva en MLflow |
| `desplegar.py` | `produccion [versión]`: actualiza el endpoint de producción; `borrar <endpoint>`: borra un candidato |
| `e2e.py` | el E2E de un candidato frente a producción |
| `humo.py` | prueba de humo después de desplegar: una noche con los 4 canales |
| `cliente.py` | manda peticiones al endpoint asíncrono y recoge las respuestas |
| `enviar_noche.py` | manda a mano una noche y guarda la respuesta |
| `preparar_test.py` | **una sola vez**: sube a S3 las 32 noches de test y su manifiesto (necesita `data/`) |
| `config.py` | nombres de los recursos de AWS |
| `requirements.txt` | librerías del contenedor del endpoint |
| `requirements-aws.txt` | librerías para registrar, desplegar y el E2E |

## Desde un computador

```bash
pip install -e . -r despliegue/requirements-aws.txt
# credenciales en ~/.aws/credentials

python despliegue/preparar_test.py                 # una vez
python despliegue/e2e.py --nombre prueba           # E2E de lo que hay en modelos/
python despliegue/enviar_noche.py SC4001E0-PSG.edf --canales fpz eog
```

**Petición al endpoint** (JSON en S3): `{"s3_uri": "s3://.../SC4001E0-PSG.edf", "canales": ["fpz", "eog"], "hora_apagado": "22:44:00", "epoca_inicio": 800, "epoca_fin": 1700}`. Solo `s3_uri` es obligatorio. La respuesta trae, por época, el estadio, la confianza, la 2.ª opción y si hay que revisarla, y `segundos_proceso`.
