"""Punto de entrada del endpoint de SageMaker (contenedor de inferencia de PyTorch).

El paquete del endpoint trae code/inference.py, code/sistema/ (el mismo código de sistema/) y modelo/.
El cuerpo de cada petición es un JSON con la noche ya subida a S3:

    {"s3_uri": "s3://bucket/noches/SC4001E0-PSG.edf",
     "canales": ["fpz", "eog"],          opcional: por defecto, los que tenga el EDF
     "hora_apagado": "22:44:00",         opcional
     "epoca_inicio": 800, "epoca_fin": 1700}   opcional: por defecto, todo el registro (los dos extremos incluidos)

La respuesta es el JSON de sistema.analizar(): estadio, confianza, 2.ª opción y "revisar" de cada época, y
segundos_proceso (lo que tardó el sistema dentro del contenedor, sin la cola ni la descarga del EDF).
Con el endpoint asíncrono, SageMaker la deja en S3.
"""
import json
import sys
import tempfile
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sistema import analizar, cargar_sistema  # noqa: E402

CAMPOS = ("canales", "hora_apagado", "epoca_inicio", "epoca_fin")


def model_fn(model_dir):
    return cargar_sistema(Path(model_dir) / "modelo")


def input_fn(body, content_type="application/json"):
    if content_type != "application/json":
        raise ValueError(f"se espera application/json, llegó {content_type}")
    pedido = json.loads(body)
    bucket, _, clave = pedido["s3_uri"].removeprefix("s3://").partition("/")
    destino = Path(tempfile.mkdtemp()) / Path(clave).name
    boto3.client("s3").download_file(bucket, clave, str(destino))
    return {"psg": destino, **{k: pedido[k] for k in CAMPOS if k in pedido}}


def predict_fn(entrada, sistema):
    try:
        return analizar(sistema, entrada)
    finally:
        Path(entrada["psg"]).unlink(missing_ok=True)


def output_fn(salida, accept="application/json"):
    return json.dumps(salida), "application/json"
