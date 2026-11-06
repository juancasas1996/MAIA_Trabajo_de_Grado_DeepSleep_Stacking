"""Prueba de humo después de desplegar: una noche de test con los 4 canales al endpoint de producción.

    python despliegue/humo.py [--endpoint sistema-sueno]

Falla si el endpoint no responde, si no devuelve una predicción por época o si el MF1 de esa noche es menor que 0.5
(el sistema da ~0.78 en el test).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cliente  # noqa: E402
import config as C  # noqa: E402
from e2e import evaluar, manifiesto, predicciones  # noqa: E402

MF1_MINIMO = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=C.MODELO)
    a = ap.parse_args()
    x = manifiesto()[0]
    pedido = {"s3_uri": x["s3_uri"], "canales": ["fpz", "pz", "eog", "emg"], "hora_apagado": x["hora_apagado"],
              "epoca_inicio": x["epoca_inicio"], "epoca_fin": x["epoca_fin"]}
    r = cliente.esperar({x["noche"]: cliente.pedir(a.endpoint, pedido)}, minutos=30)
    preds, errores = predicciones([x], r)
    if errores:
        sys.exit(f"{a.endpoint} falló con {x['noche']}: {errores[x['noche']]}")
    m = evaluar(preds, n_boot=1)["candidato"]
    seg = r[x["noche"]].get("segundos_proceso", np.nan)
    print(f"{a.endpoint} · {x['noche']} · {len(preds[x['noche']][0])} épocas · MF1 {m['mf1']:.3f} · "
          f"κ {m['kappa']:.3f} · proceso {seg:.1f} s")
    if m["mf1"] < MF1_MINIMO:
        sys.exit(f"MF1 {m['mf1']:.3f} < {MF1_MINIMO}")


if __name__ == "__main__":
    main()
