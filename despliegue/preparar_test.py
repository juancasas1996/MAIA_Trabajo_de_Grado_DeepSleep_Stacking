"""Sube a S3 las 32 noches de test y su manifiesto, para el E2E. Se corre una sola vez, desde un computador con los
datos (data/sleep-cassette/ y data/SC-subjects.xls).

    python despliegue/preparar_test.py

En s3://<bucket>/sistema-sueno/test/ quedan:
  - noches/SC4ssNE0-PSG.edf: el PSG de cada noche de test (los 16 sujetos de test de particiones.json), salvo que
    ya esté en sistema-sueno/noches/ (donde se subieron las noches para el primer despliegue): entonces se usa esa;
  - manifiesto.json: por noche, la hora de apagado de la luz, la ventana de sueño que se analiza (la misma del
    entrenamiento: 30 min de vigilia a cada lado) y el estadio de cada época de esa ventana (-1 = sin estadio).

Con el manifiesto, el E2E no necesita los hipnogramas ni el código de lectura: manda el PSG con la ventana y
compara la respuesta con las etiquetas.
"""
import json
import sys
from pathlib import Path

import boto3

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "src"))
import config as C  # noqa: E402
from sueno import config as S  # noqa: E402
from sueno.lectura import horas_apagado, leer_noche, list_nights  # noqa: E402


def main():
    s3 = boto3.client("s3", region_name=C.REGION)
    test = set(S.particion()["final"]["test"])
    noches = [x for x in list_nights(S.EDF) if S.sujeto(x[0]) in test]
    assert len(noches) == 32, f"se esperaban 32 noches de test en {S.EDF}, hay {len(noches)}"
    horas = horas_apagado(S.SUJETOS_XLS)
    existentes = {o["Key"] for prefijo in (f"{C.TEST}/noches/", f"{C.MODELO}/noches/")
                  for p in s3.get_paginator("list_objects_v2").paginate(Bucket=C.bucket(), Prefix=prefijo)
                  for o in p.get("Contents", [])}
    manifiesto = []
    for n, psg, hyp in noches:
        noche = leer_noche({"psg": psg, "hipnograma": hyp, "canales": ["fpz"]})
        clave = f"{C.MODELO}/noches/{psg.name}"
        if clave not in existentes:
            clave = f"{C.TEST}/noches/{psg.name}"
            if clave not in existentes:
                s3.upload_file(str(psg), C.bucket(), clave)
        hora = horas[n]
        manifiesto.append({"noche": n, "sujeto": S.sujeto(n), "s3_uri": f"s3://{C.bucket()}/{clave}",
                           "hora_apagado": hora.isoformat() if hasattr(hora, "isoformat") else str(hora),
                           "epoca_inicio": int(noche.epocas[0]), "epoca_fin": int(noche.epocas[-1]),
                           "etiquetas": noche.etiquetas.tolist()})
        print(f"{n}: épocas {noche.epocas[0]}–{noche.epocas[-1]} ({len(noche.epocas)})")
    s3.put_object(Bucket=C.bucket(), Key=f"{C.TEST}/manifiesto.json", Body=json.dumps({"noches": manifiesto}).encode(),
                  ContentType="application/json")
    print(f"listo: {len(manifiesto)} noches en s3://{C.bucket()}/{C.TEST}/")


if __name__ == "__main__":
    main()
