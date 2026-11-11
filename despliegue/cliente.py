"""Mandar peticiones a un endpoint asíncrono y recoger las respuestas de S3."""
import json
import sys
import time
import uuid
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402


def pedir(endpoint, pedido):
    """Deja el JSON de la petición en S3 y la manda al endpoint. Devuelve (uri de salida, uri de error)."""
    s3 = boto3.client("s3", region_name=C.REGION)
    clave = f"{endpoint}/peticiones/{uuid.uuid4().hex}.json"
    s3.put_object(Bucket=C.bucket(), Key=clave, Body=json.dumps(pedido).encode(), ContentType="application/json")
    r = boto3.client("sagemaker-runtime", region_name=C.REGION).invoke_endpoint_async(
        EndpointName=endpoint, InputLocation=f"s3://{C.bucket()}/{clave}", ContentType="application/json",
        InvocationTimeoutSeconds=3600)
    return r["OutputLocation"], r.get("FailureLocation", "")


def _leer(s3, uri):
    if not uri:
        return None
    bucket, _, clave = uri.removeprefix("s3://").partition("/")
    try:
        return s3.get_object(Bucket=bucket, Key=clave)["Body"].read()
    except s3.exceptions.NoSuchKey:
        return None


def esperar(pendientes, minutos=60, cada=15):
    """pendientes: id -> (uri de salida, uri de error). Devuelve id -> respuesta (dict) o id -> {"error": texto}.

    Las que no lleguen a tiempo quedan como error.
    """
    s3 = boto3.client("s3", region_name=C.REGION)
    resultado, faltan = {}, dict(pendientes)
    t0 = time.time()
    while faltan and time.time() - t0 < minutos * 60:
        for k, (salida, error) in list(faltan.items()):
            cuerpo = _leer(s3, salida)
            if cuerpo is not None:
                resultado[k] = json.loads(cuerpo)
            elif (cuerpo := _leer(s3, error)) is not None:
                resultado[k] = {"error": cuerpo.decode()[:2000]}
            else:
                continue
            faltan.pop(k)
        if faltan:
            time.sleep(cada)
    for k in faltan:
        resultado[k] = {"error": f"no llegó respuesta en {minutos} min"}
    return resultado
