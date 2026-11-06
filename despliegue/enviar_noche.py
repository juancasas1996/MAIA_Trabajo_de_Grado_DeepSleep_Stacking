"""Manda una noche al endpoint y espera la respuesta.

    python despliegue/enviar_noche.py SC4001E0-PSG.edf [--canales fpz eog] [--hora-apagado 22:44:00]
                                      [--inicio 800 --fin 1700] [--endpoint sistema-sueno] [--salida resultado.json]

Sube el EDF a S3 y manda la petición al endpoint asíncrono. Si el endpoint está apagado, tarda ~8 min en prender;
después, una noche toma menos de un minuto. La respuesta se guarda en un JSON.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cliente  # noqa: E402
import config as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edf")
    ap.add_argument("--canales", nargs="+", choices=["fpz", "pz", "eog", "emg"])
    ap.add_argument("--hora-apagado")
    ap.add_argument("--inicio", type=int)
    ap.add_argument("--fin", type=int)
    ap.add_argument("--endpoint", default=C.MODELO)
    ap.add_argument("--salida")
    a = ap.parse_args()

    edf = Path(a.edf)
    clave_edf = f"{C.MODELO}/noches/{edf.name}"
    boto3.client("s3", region_name=C.REGION).upload_file(str(edf), C.bucket(), clave_edf)
    pedido = {"s3_uri": f"s3://{C.bucket()}/{clave_edf}"}
    for k, v in (("canales", a.canales), ("hora_apagado", a.hora_apagado), ("epoca_inicio", a.inicio), ("epoca_fin", a.fin)):
        if v is not None:
            pedido[k] = v
    t0 = time.time()
    uris = cliente.pedir(a.endpoint, pedido)
    print(f"petición enviada a {a.endpoint} ({pedido}); esperando la respuesta en {uris[0]}")
    res = cliente.esperar({"noche": uris}, minutos=40)["noche"]
    if "error" in res:
        sys.exit(f"el endpoint falló: {res['error']}")
    destino = Path(a.salida or f"{edf.stem}_resultado.json")
    destino.write_text(json.dumps(res))
    conteo = {e: res["estadio"].count(e) for e in ("W", "N1", "N2", "N3", "REM")}
    print(f"listo en {(time.time() - t0) / 60:.1f} min · {len(res['estadio'])} épocas · {conteo} · "
          f"revisar {res['fraccion_revisar']:.1%} · aviso {res['aviso_noche']} · "
          f"proceso {res['segundos_proceso']:.1f} s · guardado en {destino}")


if __name__ == "__main__":
    main()
