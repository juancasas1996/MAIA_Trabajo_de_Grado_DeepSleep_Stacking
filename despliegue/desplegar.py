"""Endpoints asíncronos de SageMaker a partir de una versión de MLflow de "sistema-sueno".

    python despliegue/desplegar.py produccion [versión]     despliega en el endpoint de producción (por defecto, la
                                                           última versión registrada) y le pone el alias "produccion"
    python despliegue/desplegar.py borrar <endpoint>       borra un endpoint candidato y su configuración

Producción (endpoint "sistema-sueno") se apaga sola tras unos minutos sin peticiones y se prende cuando llega una
(~8 min). Los endpoints candidatos (los crea e2e.py) tienen una máquina fija y se borran al terminar.
"""
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import boto3
import mlflow
from mlflow import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

DIM = "sagemaker:variant:DesiredInstanceCount"


def _sm():
    return boto3.client("sagemaker", region_name=C.REGION)


def paquete(version):
    """model.tar.gz con code/ (inference.py, sistema/, requirements.txt) y modelo/, bajado de MLflow y subido a S3."""
    mlflow.set_tracking_uri(C.mlflow_arn())
    tmp = Path(tempfile.mkdtemp())
    d = Path(mlflow.artifacts.download_artifacts(f"models:/{C.MODELO}/{version}", dst_path=str(tmp / "mlflow")))
    tar = tmp / "model.tar.gz"
    excluir = lambda t: None if "__pycache__" in t.name else t  # noqa: E731
    with tarfile.open(tar, "w:gz") as t:
        for f in ("inference.py", "sistema", "requirements.txt"):
            t.add(d / "code" / f, arcname=f"code/{f}", filter=excluir)
        t.add(d / "artifacts" / "modelo", arcname="modelo")
    clave = f"{C.MODELO}/v{version}/model.tar.gz"
    boto3.client("s3", region_name=C.REGION).upload_file(str(tar), C.bucket(), clave)
    print(f"paquete de la versión {version}: s3://{C.bucket()}/{clave} ({tar.stat().st_size / 1e6:.1f} MB)")
    return f"s3://{C.bucket()}/{clave}"


def configuracion(endpoint, version):
    """Crea el modelo y la configuración de SageMaker de esa versión; devuelve el nombre de la configuración."""
    sm = _sm()
    nombre = f"{endpoint}-v{version}-{time.strftime('%Y%m%d-%H%M%S')}"[:63]
    sm.create_model(ModelName=nombre, ExecutionRoleArn=C.rol(), PrimaryContainer={
        "Image": C.IMAGEN, "ModelDataUrl": paquete(version),
        "Environment": {"SAGEMAKER_PROGRAM": "inference.py", "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                        "SAGEMAKER_MODEL_SERVER_WORKERS": "1", "SAGEMAKER_MODEL_SERVER_TIMEOUT": "3600",
                        "TS_DEFAULT_RESPONSE_TIMEOUT": "3600", "MLFLOW_VERSION_SISTEMA": str(version)}})
    sm.create_endpoint_config(
        EndpointConfigName=nombre,
        ProductionVariants=[{"VariantName": "AllTraffic", "ModelName": nombre, "InstanceType": C.INSTANCIA,
                             "InitialInstanceCount": 1}],
        AsyncInferenceConfig={"OutputConfig": {"S3OutputPath": f"s3://{C.bucket()}/{endpoint}/salidas/",
                                               "S3FailurePath": f"s3://{C.bucket()}/{endpoint}/errores/"},
                              "ClientConfig": {"MaxConcurrentInvocationsPerInstance": 1}})
    return nombre


def estado(endpoint):
    try:
        return _sm().describe_endpoint(EndpointName=endpoint)["EndpointStatus"]
    except _sm().exceptions.ClientError:
        return None


def esperar(endpoint):
    _sm().get_waiter("endpoint_in_service").wait(EndpointName=endpoint, WaiterConfig={"Delay": 30, "MaxAttempts": 80})


def version_desplegada(endpoint):
    """Número de la versión de MLflow que sirve el endpoint (None si no existe)."""
    sm = _sm()
    try:
        cfg = sm.describe_endpoint(EndpointName=endpoint)["EndpointConfigName"]
    except sm.exceptions.ClientError:
        return None
    modelo = sm.describe_endpoint_config(EndpointConfigName=cfg)["ProductionVariants"][0]["ModelName"]
    return sm.describe_model(ModelName=modelo)["PrimaryContainer"]["Environment"].get("MLFLOW_VERSION_SISTEMA")


def crear_candidato(endpoint, version):
    """Endpoint aparte con una máquina fija, para el E2E. Producción no se toca."""
    _sm().create_endpoint(EndpointName=endpoint, EndpointConfigName=configuracion(endpoint, version))
    print(f"creando el endpoint candidato {endpoint} con la versión {version}…")
    esperar(endpoint)
    print(f"{endpoint} en servicio")


def borrar(endpoint):
    """Borra el endpoint, su configuración y su modelo de SageMaker (no toca MLflow ni S3)."""
    sm = _sm()
    try:
        cfg = sm.describe_endpoint(EndpointName=endpoint)["EndpointConfigName"]
    except sm.exceptions.ClientError:
        print(f"{endpoint} no existe")
        return
    modelos = [v["ModelName"] for v in sm.describe_endpoint_config(EndpointConfigName=cfg)["ProductionVariants"]]
    sm.delete_endpoint(EndpointName=endpoint)
    sm.delete_endpoint_config(EndpointConfigName=cfg)
    for m in modelos:
        sm.delete_model(ModelName=m)
    print(f"{endpoint} borrado")


# --- producción: apagado automático ---------------------------------------------------------------------------------

def _recurso(endpoint):
    return f"endpoint/{endpoint}/variant/AllTraffic"


def quitar_apagado(endpoint=C.MODELO):
    """Hay que quitar el escalado automático antes de actualizar el endpoint."""
    aas = boto3.client("application-autoscaling", region_name=C.REGION)
    boto3.client("cloudwatch", region_name=C.REGION).delete_alarms(AlarmNames=[f"{endpoint}-prender", f"{endpoint}-apagar"])
    try:
        aas.deregister_scalable_target(ServiceNamespace="sagemaker", ResourceId=_recurso(endpoint), ScalableDimension=DIM)
    except aas.exceptions.ObjectNotFoundException:
        pass


def poner_apagado(endpoint=C.MODELO):
    aas = boto3.client("application-autoscaling", region_name=C.REGION)
    cw = boto3.client("cloudwatch", region_name=C.REGION)
    rec = _recurso(endpoint)
    aas.register_scalable_target(ServiceNamespace="sagemaker", ResourceId=rec, ScalableDimension=DIM,
                                 MinCapacity=0, MaxCapacity=1)
    politica = lambda nombre, paso: aas.put_scaling_policy(  # noqa: E731
        PolicyName=nombre, ServiceNamespace="sagemaker", ResourceId=rec, ScalableDimension=DIM,
        PolicyType="StepScaling", StepScalingPolicyConfiguration={
            "AdjustmentType": "ExactCapacity", "Cooldown": 60, "MetricAggregationType": "Maximum",
            "StepAdjustments": [paso]})["PolicyARN"]
    prender = politica("prender", {"MetricIntervalLowerBound": 0, "ScalingAdjustment": 1})
    apagar = politica("apagar", {"MetricIntervalUpperBound": 0, "ScalingAdjustment": 0})
    dims = [{"Name": "EndpointName", "Value": endpoint}]
    # Prende cuando hay peticiones esperando y ninguna máquina
    cw.put_metric_alarm(AlarmName=f"{endpoint}-prender", Namespace="AWS/SageMaker", MetricName="HasBacklogWithoutCapacity",
                        Dimensions=dims, Statistic="Maximum", Period=60, EvaluationPeriods=1, Threshold=1,
                        ComparisonOperator="GreaterThanOrEqualToThreshold", TreatMissingData="notBreaching",
                        AlarmActions=[prender])
    # Apaga tras MINUTOS_SIN_USO minutos seguidos sin peticiones en cola ni en proceso
    cw.put_metric_alarm(AlarmName=f"{endpoint}-apagar", Namespace="AWS/SageMaker", MetricName="ApproximateBacklogSize",
                        Dimensions=dims, Statistic="Maximum", Period=60, EvaluationPeriods=C.MINUTOS_SIN_USO,
                        Threshold=0.5, ComparisonOperator="LessThanThreshold", TreatMissingData="breaching",
                        AlarmActions=[apagar])


def desplegar_produccion(version):
    """Crea o actualiza el endpoint de producción con esa versión. Si la actualización falla, SageMaker se queda con
    la versión anterior."""
    sm = _sm()
    cfg = configuracion(C.MODELO, version)
    actual = estado(C.MODELO)
    if actual is None:
        sm.create_endpoint(EndpointName=C.MODELO, EndpointConfigName=cfg)
        print(f"creando el endpoint {C.MODELO} con la versión {version}…")
    else:
        if actual != "InService":
            esperar(C.MODELO)
        quitar_apagado()
        sm.update_endpoint(EndpointName=C.MODELO, EndpointConfigName=cfg)
        print(f"actualizando el endpoint {C.MODELO} a la versión {version}…")
    try:
        esperar(C.MODELO)
    finally:
        poner_apagado()   # también si la actualización falló: el endpoint sigue con la versión anterior
    if version_desplegada(C.MODELO) != str(version):
        raise SystemExit(f"la actualización falló: {C.MODELO} sigue con la versión {version_desplegada(C.MODELO)}")
    mlflow.set_tracking_uri(C.mlflow_arn())
    MlflowClient().set_registered_model_alias(C.MODELO, "produccion", str(version))
    print(f"listo: {C.MODELO} en servicio con la versión {version} (alias 'produccion'); "
          f"se apaga tras {C.MINUTOS_SIN_USO} min sin uso")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("produccion", "borrar"):
        sys.exit(__doc__)
    if sys.argv[1] == "borrar":
        borrar(sys.argv[2])
        return
    mlflow.set_tracking_uri(C.mlflow_arn())
    version = sys.argv[2] if len(sys.argv) > 2 else max(
        int(v.version) for v in MlflowClient().search_model_versions(f"name='{C.MODELO}'"))
    desplegar_produccion(version)


if __name__ == "__main__":
    main()
