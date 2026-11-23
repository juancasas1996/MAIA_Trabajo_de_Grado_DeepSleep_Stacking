"""API pública del sistema (función Lambda con URL). El endpoint de SageMaker sigue privado: la Lambda lo llama.

Todas las rutas piden el header  Authorization: Bearer <token>.

    POST /subir            {"archivo": "SC4001E0-PSG.edf"}
                           -> {"subir_a": url firmada para hacer PUT del EDF (1 h), "s3_uri": ...}
    POST /analizar         la petición del endpoint: {"s3_uri": ..., "canales": [...], "hora_apagado": ...,
                           "epoca_inicio": ..., "epoca_fin": ...}  -> {"id": ...}
    GET  /resultado/<id>   -> {"estado": "pendiente"} | {"estado": "listo", "resultado": {...}} |
                              {"estado": "error", "error": ...}

Variables de entorno: BUCKET, ENDPOINT, MODELO (opcional; si falta, ENDPOINT) y TOKENS (JSON nombre -> sha256 del token; los tokens no se guardan).
"""
import base64
import hashlib
import hmac
import json
import os
import re
import uuid

import boto3
from botocore.config import Config

BUCKET = os.environ["BUCKET"]
ENDPOINT = os.environ.get("ENDPOINT", "sistema-sueno")
MODELO = os.environ.get("MODELO", ENDPOINT)  # endpoint de SageMaker que atiende; ENDPOINT fija las carpetas
CARPETA = f"{ENDPOINT}/api"
CAMPOS = ("s3_uri", "canales", "hora_apagado", "epoca_inicio", "epoca_fin")

s3 = boto3.client("s3", config=Config(signature_version="s3v4"))
sm = boto3.client("sagemaker-runtime")


def _respuesta(codigo, cuerpo):
    return {"statusCode": codigo, "headers": {"Content-Type": "application/json"}, "body": json.dumps(cuerpo)}


def autorizado(headers):
    """El token del header, comparado (por su sha256) con los registrados."""
    token = (headers.get("authorization") or "").removeprefix("Bearer ").strip()
    if not token:
        return None
    h = hashlib.sha256(token.encode()).hexdigest()
    tokens = json.loads(os.environ.get("TOKENS") or "{}")
    return next((nombre for nombre, v in tokens.items() if hmac.compare_digest(v, h)), None)


def _leer(clave):
    try:
        return s3.get_object(Bucket=BUCKET, Key=clave)["Body"].read()
    except s3.exceptions.NoSuchKey:
        return None


def subir(datos, quien):
    nombre = re.sub(r"[^A-Za-z0-9._-]", "_", str(datos.get("archivo") or "noche.edf"))[:100]
    clave = f"{CARPETA}/subidas/{quien}/{uuid.uuid4().hex[:12]}/{nombre}"
    url = s3.generate_presigned_url("put_object", Params={"Bucket": BUCKET, "Key": clave}, ExpiresIn=3600)
    return _respuesta(200, {"subir_a": url, "s3_uri": f"s3://{BUCKET}/{clave}"})


def analizar(datos, quien):
    pedido = {k: datos[k] for k in CAMPOS if k in datos}
    # el endpoint solo lee EDF de este bucket, en las carpetas del sistema
    if not str(pedido.get("s3_uri", "")).startswith(f"s3://{BUCKET}/{ENDPOINT}/"):
        return _respuesta(400, {"error": f"s3_uri debe estar en s3://{BUCKET}/{ENDPOINT}/ (usar /subir)"})
    id_ = uuid.uuid4().hex
    clave = f"{CARPETA}/peticiones/{id_}.json"
    s3.put_object(Bucket=BUCKET, Key=clave, Body=json.dumps(pedido).encode(), ContentType="application/json")
    r = sm.invoke_endpoint_async(EndpointName=MODELO, InputLocation=f"s3://{BUCKET}/{clave}",
                                 ContentType="application/json", InvocationTimeoutSeconds=3600)
    destino = {"salida": r["OutputLocation"], "error": r.get("FailureLocation", ""), "quien": quien}
    s3.put_object(Bucket=BUCKET, Key=f"{CARPETA}/ids/{id_}.json", Body=json.dumps(destino).encode())
    return _respuesta(202, {"id": id_})


def resultado(id_):
    if not re.fullmatch(r"[0-9a-f]{32}", id_):
        return _respuesta(404, {"error": "id desconocido"})
    destino = _leer(f"{CARPETA}/ids/{id_}.json")
    if destino is None:
        return _respuesta(404, {"error": "id desconocido"})
    destino = json.loads(destino)
    clave = lambda uri: uri.removeprefix(f"s3://{BUCKET}/")  # noqa: E731
    if (cuerpo := _leer(clave(destino["salida"]))) is not None:
        return _respuesta(200, {"estado": "listo", "resultado": json.loads(cuerpo)})
    if destino["error"] and (cuerpo := _leer(clave(destino["error"]))) is not None:
        return _respuesta(200, {"estado": "error", "error": cuerpo.decode()[:2000]})
    return _respuesta(200, {"estado": "pendiente"})


def lambda_handler(event, context):
    quien = autorizado({k.lower(): v for k, v in (event.get("headers") or {}).items()})
    if quien is None:
        return _respuesta(401, {"error": "falta un token válido en Authorization: Bearer <token>"})
    metodo, ruta = event["requestContext"]["http"]["method"], event.get("rawPath", "/").rstrip("/")
    cuerpo = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        cuerpo = base64.b64decode(cuerpo).decode()
    try:
        datos = json.loads(cuerpo) if metodo == "POST" else {}
    except json.JSONDecodeError:
        datos = None
    if not isinstance(datos, dict):
        return _respuesta(400, {"error": "el cuerpo debe ser un objeto JSON"})
    if metodo == "POST" and ruta == "/subir":
        return subir(datos, quien)
    if metodo == "POST" and ruta == "/analizar":
        return analizar(datos, quien)
    if metodo == "GET" and ruta.startswith("/resultado/"):
        return resultado(ruta.removeprefix("/resultado/"))
    return _respuesta(404, {"error": "rutas: POST /subir, POST /analizar, GET /resultado/<id>"})
