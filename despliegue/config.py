"""Nombres de los recursos de AWS. Todo sale de variables de entorno o de la cuenta de las credenciales activas.

    MLFLOW_ARN       ARN de la app de MLflow de SageMaker; por defecto, el de la app llamada maia-g3-mlflow
    AWS_REGION       por defecto us-east-1
    BUCKET           por defecto maia-g3-sueno-<cuenta>
    ROL_SAGEMAKER    por defecto el LabRole de AWS Academy de la cuenta
"""
import os
from functools import lru_cache

import boto3

REGION = os.environ.get("AWS_REGION", "us-east-1")
MODELO = "sistema-sueno"          # nombre en el registro de MLflow y del endpoint de producción
INSTANCIA = "ml.m5.large"         # 2 vCPU, 8 GB, ~US$0.115 la hora mientras está prendida
# Contenedor de inferencia de PyTorch de SageMaker (la cuenta 763104351884 es la de AWS que publica estas imágenes)
IMAGEN = f"763104351884.dkr.ecr.{REGION}.amazonaws.com/pytorch-inference:2.5.1-cpu-py311-ubuntu22.04-sagemaker"
MINUTOS_SIN_USO = 5               # producción se apaga tras estos minutos sin peticiones
MLFLOW_APP = "maia-g3-mlflow"     # app de MLflow sin servidor de SageMaker
TEST = f"{MODELO}/test"           # en el bucket: las 32 noches de test y su manifiesto (ver preparar_test.py)


@lru_cache
def cuenta():
    return boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]


def bucket():
    return os.environ.get("BUCKET") or f"maia-g3-sueno-{cuenta()}"


def rol():
    return os.environ.get("ROL_SAGEMAKER") or f"arn:aws:iam::{cuenta()}:role/LabRole"


@lru_cache
def mlflow_arn():
    if os.environ.get("MLFLOW_ARN"):
        return os.environ["MLFLOW_ARN"]
    apps = boto3.client("sagemaker", region_name=REGION).list_mlflow_apps()["Summaries"]
    arns = [a["Arn"] for a in apps if a.get("Name") == MLFLOW_APP]
    if not arns:
        raise SystemExit(f"no está la app de MLflow {MLFLOW_APP} en {REGION}; pasa su ARN en MLFLOW_ARN")
    return arns[0]
