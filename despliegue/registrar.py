"""Registra el sistema en MLflow como una versión nueva de "sistema-sueno": el código de sistema/, los archivos de
modelo/ y las librerías, en un solo paquete.

    python despliegue/registrar.py [--etiqueta clave=valor ...]

Imprime el número de la versión registrada. Cada versión guarda el commit de git del que salió.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import mlflow

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(AQUI))
import config as C  # noqa: E402
from modelo_mlflow import SistemaSueno  # noqa: E402


def commit():
    """El commit del que sale el modelo. En un PR, GitHub Actions pasa el último commit de la rama en COMMIT_SHA."""
    if os.environ.get("COMMIT_SHA") or os.environ.get("GITHUB_SHA"):
        return os.environ.get("COMMIT_SHA") or os.environ["GITHUB_SHA"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=RAIZ, text=True).strip()
    except Exception:
        return "desconocido"


def describir_modelos(modelo=RAIZ / "modelo"):
    sistema = json.loads((modelo / "sistema.json").read_text())
    cols_a = json.loads((modelo / sistema.get("carpeta_modelo_a", "modelo_a_xgboost") / "columnas.json").read_text())
    cols_meta = json.loads((modelo / "meta_modelo" / "columnas.json").read_text())
    return {"modelo_a": f"XGBoost, {len(cols_a)} columnas", "modelo_b": "CNN + Transformer, 80 épocas",
            "meta_modelo": f"LR + XGBoost, {len(cols_meta)} entradas",
            "umbral_confianza": sistema["umbral_confianza"], "aviso_noche": sistema["aviso_noche_fraccion_revisar"]}


def registrar(etiquetas=None):
    """Registra modelo/ + sistema/ y devuelve el número de versión."""
    mlflow.set_tracking_uri(C.mlflow_arn())
    mlflow.set_experiment(C.MODELO)
    requisitos = [r for r in (AQUI / "requirements.txt").read_text().splitlines() if r and not r.startswith("#")]
    sha = commit()
    with mlflow.start_run(run_name=f"registro {sha[:7]}") as run:
        mlflow.set_tags({"git_commit": sha, **(etiquetas or {})})
        mlflow.log_params(describir_modelos())
        info = mlflow.pyfunc.log_model(
            name="sistema",
            python_model=SistemaSueno(),
            artifacts={"modelo": str(RAIZ / "modelo")},
            code_paths=[str(RAIZ / "sistema"), str(AQUI / "inference.py"), str(AQUI / "modelo_mlflow.py"),
                        str(AQUI / "requirements.txt")],
            pip_requirements=requisitos + ["torch==2.5.1"],
            registered_model_name=C.MODELO,
        )
    version = int(info.registered_model_version)
    cliente = mlflow.MlflowClient()
    for k, v in {"git_commit": sha, **(etiquetas or {})}.items():
        cliente.set_model_version_tag(C.MODELO, str(version), k, v)
    print(f"run {run.info.run_id} · versión {version} de {C.MODELO} (commit {sha[:7]})")
    return version


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--etiqueta", nargs="*", default=[], help="etiquetas de la versión, como clave=valor")
    a = ap.parse_args()
    registrar(dict(e.split("=", 1) for e in a.etiqueta))


if __name__ == "__main__":
    main()
