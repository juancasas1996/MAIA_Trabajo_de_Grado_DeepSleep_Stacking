"""Convierte el reporte JUnit de pytest en una tabla para el resumen de GitHub Actions.

Escribe la tabla en la salida estándar (el workflow la manda a $GITHUB_STEP_SUMMARY) y, por cada prueba que falla,
una anotación de error (::error) que GitHub muestra en el PR junto a la línea de la prueba.

Uso: python .github/resumen_pruebas.py pruebas.xml
"""
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def estado(caso):
    for tag, icono, nombre in (("failure", "❌", "falló"), ("error", "💥", "error"), ("skipped", "⏭️", "omitida")):
        hijo = caso.find(tag)
        if hijo is not None:
            return icono, nombre, (hijo.get("message") or hijo.text or "").strip()
    return "✅", "pasó", ""


def main(ruta):
    if not Path(ruta).is_file():
        print("## ❌ Pruebas\n\nNo se generó el reporte: las pruebas no alcanzaron a correr (ver el paso anterior).")
        return
    casos = list(ET.parse(ruta).getroot().iter("testcase"))
    por_archivo = defaultdict(list)
    for c in casos:
        por_archivo[c.get("file") or c.get("classname")].append((c, *estado(c)))

    fallas = [(c, m) for filas in por_archivo.values() for c, _, n, m in filas if n in ("falló", "error")]
    pasaron = sum(n == "pasó" for filas in por_archivo.values() for _, _, n, _ in filas)
    titulo = "✅ Todas las pruebas pasaron" if not fallas else f"❌ {len(fallas)} prueba(s) no pasaron"
    print(f"## {titulo}\n\n**{pasaron} de {len(casos)}** pasaron.\n")

    for archivo, filas in sorted(por_archivo.items()):
        ok = sum(n == "pasó" for _, _, n, _ in filas)
        print(f"### {'✅' if ok == len(filas) else '❌'} `{archivo}` ({ok}/{len(filas)})\n")
        print("| | Prueba | Tiempo |\n|---|---|---|")
        for c, icono, _, _ in filas:
            print(f"| {icono} | `{c.get('name')}` | {float(c.get('time', 0)):.2f} s |")
        print()

    if fallas:
        print("### Detalle de las que no pasaron\n")
        for c, mensaje in fallas:
            print(f"<details><summary><code>{c.get('name')}</code></summary>\n\n```\n{mensaje[:2000]}\n```\n</details>\n")
            linea = int(c.get("line", 0)) + 1
            primera = mensaje.splitlines()[0] if mensaje else "falló"
            # va a stderr para que no quede en la tabla
            print(f"::error file={c.get('file')},line={linea},title={c.get('name')}::{primera}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])
