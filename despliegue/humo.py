"""Prueba de humo después de desplegar: una noche de test con los 4 canales al endpoint de producción.

    python despliegue/humo.py [--endpoint sistema-sueno]

Falla si el endpoint no responde, si no devuelve una predicción por época o si el MF1 de esa noche es menor que 0.5
(el sistema da ~0.78 en el test).
"""
import argparse
import json
import sys
from pathlib import Path

import boto3
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cliente  # noqa: E402
import config as C  # noqa: E402

MF1_MINIMO = 0.5
ESTADIOS = ["W", "N1", "N2", "N3", "REM"]


def manifiesto():
    """Las 32 noches de test, ya en S3: PSG, hora de apagado, ventana de sueño y estadio de cada época."""
    cuerpo = boto3.client("s3", region_name=C.REGION).get_object(Bucket=C.bucket(), Key=f"{C.TEST}/manifiesto.json")
    return json.loads(cuerpo["Body"].read())["noches"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=C.MODELO)
    a = ap.parse_args()
    x = manifiesto()[0]
    pedido = {"s3_uri": x["s3_uri"], "canales": ["fpz", "pz", "eog", "emg"], "hora_apagado": x["hora_apagado"],
              "epoca_inicio": x["epoca_inicio"], "epoca_fin": x["epoca_fin"]}
    r = cliente.esperar({x["noche"]: cliente.pedir(a.endpoint, pedido)}, minutos=30)[x["noche"]]
    if "error" in r:
        sys.exit(f"{a.endpoint} falló con {x['noche']}: {r['error']}")
    y = np.array(x["etiquetas"])
    if len(r["probabilidades"]) != len(y):
        sys.exit(f"{len(r['probabilidades'])} épocas en la respuesta y {len(y)} en el manifiesto")
    pred = np.array(r["probabilidades"]).argmax(1)
    v = y >= 0
    mf1 = f1_score(y[v], pred[v], labels=range(len(ESTADIOS)), average="macro", zero_division=0)
    kappa = cohen_kappa_score(y[v], pred[v])
    print(f"{a.endpoint} · {x['noche']} · {len(y)} épocas · MF1 {mf1:.3f} · κ {kappa:.3f} · "
          f"proceso {r.get('segundos_proceso', np.nan):.1f} s")
    if mf1 < MF1_MINIMO:
        sys.exit(f"MF1 {mf1:.3f} < {MF1_MINIMO}")


if __name__ == "__main__":
    main()
