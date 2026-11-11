"""Prueba E2E de un modelo candidato frente al de producción, con las 32 noches de test.

    python despliegue/e2e.py --nombre pr-7 [--version 9] [--salida e2e]

1. Registra en MLflow lo que hay en modelo/ y sistema/ (o usa --version, si ya está registrado).
2. Lo despliega en un endpoint aparte (sistema-sueno-<nombre>). Producción no se toca.
3. Manda las 32 noches de test (las de S3, ver humo.py) al candidato y a producción. A cada noche se le sortea una de
   las 12 combinaciones de canales (semilla 42), la misma para los dos modelos.
4. Calcula MF1, κ, exactitud, F1 por estadio y matriz de confusión, con IC 95 % por bootstrap agrupado por persona,
   la diferencia pareada candidato − producción y la latencia (segundos de proceso dentro del contenedor).
5. Borra el endpoint candidato, aunque algo falle.

Deja en --salida e2e.md (el comentario del PR) y e2e.json. Sale con error si el candidato falla en alguna noche; si es
peor o mejor no decide nada: eso lo decide una persona al revisar el PR.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cliente  # noqa: E402
import config as C  # noqa: E402
import desplegar  # noqa: E402
from humo import ESTADIOS, manifiesto  # noqa: E402
from registrar import commit, describir_modelos, registrar  # noqa: E402

K = len(ESTADIOS)
SEED = 42
N_BOOT = 2000
COMBINACIONES = {
    "fpz": ("fpz",), "pz": ("pz",), "eeg2": ("fpz", "pz"),
    "fpz_eog": ("fpz", "eog"), "pz_eog": ("pz", "eog"), "eeg2_eog": ("fpz", "pz", "eog"),
    "fpz_emg": ("fpz", "emg"), "pz_emg": ("pz", "emg"), "eeg2_emg": ("fpz", "pz", "emg"),
    "fpz_eog_emg": ("fpz", "eog", "emg"), "pz_eog_emg": ("pz", "eog", "emg"), "eeg2_eog_emg": ("fpz", "pz", "eog", "emg"),
}


def sortear_canales(noches):
    """Una combinación por noche, siempre la misma (semilla fija): así los resultados de distintos PR se comparan.

    Cada una de las 12 combinaciones sale 2 o 3 veces en las 32 noches, en orden al azar.
    """
    rng = np.random.default_rng(SEED)
    cfgs = rng.permutation(np.resize(list(COMBINACIONES), len(noches)))
    return {x["noche"]: str(c) for x, c in zip(noches, cfgs)}


def correr(endpoint, noches, canales):
    """Manda todas las noches al endpoint y espera las respuestas: noche -> respuesta o {"error": ...}."""
    pendientes = {}
    for x in noches:
        pedido = {"s3_uri": x["s3_uri"], "canales": list(COMBINACIONES[canales[x["noche"]]]),
                  "hora_apagado": x["hora_apagado"], "epoca_inicio": x["epoca_inicio"], "epoca_fin": x["epoca_fin"]}
        pendientes[x["noche"]] = cliente.pedir(endpoint, pedido)
    print(f"{endpoint}: {len(pendientes)} noches enviadas")
    return cliente.esperar(pendientes, minutos=75)


def predicciones(noches, respuestas):
    """noche -> (etiquetas, probabilidades) de las noches que respondieron bien; y los errores."""
    preds, errores = {}, {}
    for x in noches:
        r = respuestas[x["noche"]]
        y = np.array(x["etiquetas"])
        if "error" in r:
            errores[x["noche"]] = r["error"]
        elif len(r["probabilidades"]) != len(y):
            errores[x["noche"]] = f"{len(r['probabilidades'])} épocas en la respuesta y {len(y)} en el manifiesto"
        else:
            preds[x["noche"]] = (y, np.array(r["probabilidades"], dtype=np.float32))
    return preds, errores


# --- métricas -----------------------------------------------------------------------------------------------------

def confusion(y, p):
    """Matriz de confusión de una noche (filas: real; columnas: predicho); las épocas con y = -1 no cuentan."""
    v = y >= 0
    return np.bincount(y[v] * K + p.argmax(1)[v], minlength=K * K).reshape(K, K)


def metricas_confusion(cm):
    """acc, MF1, κ y F1 por estadio a partir de una matriz de confusión."""
    cm = np.asarray(cm, dtype=np.float64)
    n, tp = cm.sum(), np.diag(cm)
    rows, cols = cm.sum(1), cm.sum(0)
    f1 = np.divide(2 * tp, rows + cols, out=np.zeros(K), where=(rows + cols) > 0)
    pe = (rows * cols).sum() / n**2
    return {"acc": tp.sum() / n, "mf1": f1.mean(), "kappa": (tp.sum() / n - pe) / (1 - pe),
            **{f"f1_{s}": v for s, v in zip(ESTADIOS, f1)}}


def _por_sujeto(preds):
    cms = {}
    for n, (y, p) in preds.items():
        s = int(n[3:5])
        cms[s] = cms.get(s, 0) + confusion(y, p)
    return cms


def evaluar(cand, prod=None, n_boot=N_BOOT, seed=SEED):
    """Métricas del candidato (y de producción) con IC 95 %, y la diferencia pareada en los mismos sujetos."""
    nombres = ["mf1", "kappa", "acc"] + [f"f1_{s}" for s in ESTADIOS]
    modelos = {"candidato": cand} | ({"produccion": prod} if prod else {})
    cms = {m: _por_sujeto(p) for m, p in modelos.items()}
    sujetos = sorted(set.intersection(*(set(c) for c in cms.values())))
    st = {m: np.stack([c[s] for s in sujetos]) for m, c in cms.items()}
    rng = np.random.default_rng(seed)
    boot = {m: [] for m in modelos}
    for _ in range(n_boot):
        idx = rng.integers(0, len(sujetos), len(sujetos))
        for m in modelos:
            boot[m].append(metricas_confusion(st[m][idx].sum(0)))
    ic = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]  # noqa: E731
    out = {}
    for m in modelos:
        centro = metricas_confusion(st[m].sum(0))
        out[m] = {k: float(centro[k]) for k in nombres}
        out[m]["ic95"] = {k: ic([b[k] for b in boot[m]]) for k in nombres}
        out[m]["confusion"] = st[m].sum(0).tolist()
    if prod:
        out["diferencia"] = {k: {"valor": out["candidato"][k] - out["produccion"][k],
                                 "ic95": ic([c[k] - p[k] for c, p in zip(boot["candidato"], boot["produccion"])])}
                             for k in nombres}
    out["sujetos"] = len(sujetos)
    return out


def latencia(respuestas):
    s = np.array([r["segundos_proceso"] for r in respuestas.values() if "segundos_proceso" in r])
    if not len(s):
        return {}
    return {"media": float(s.mean()), "mediana": float(np.median(s)), "p95": float(np.quantile(s, 0.95)),
            "maximo": float(s.max()), "total": float(s.sum())}


# --- informe ------------------------------------------------------------------------------------------------------

def _f(x, signo=False):
    return f"{x:+.3f}" if signo else f"{x:.3f}"


def _ic(v, signo=False):
    return f"[{_f(v[0], signo)}, {_f(v[1], signo)}]"


def _confusion_md(cm):
    cm = np.asarray(cm)
    filas = ["| real \\ predicho | " + " | ".join(ESTADIOS) + " |", "|---|" + "---|" * K]
    for i, s in enumerate(ESTADIOS):
        tot = max(cm[i].sum(), 1)
        filas.append(f"| **{s}** | " + " | ".join(f"{cm[i, j] / tot:.0%}" for j in range(K)) + " |")
    return "\n".join(filas)


def informe(r):
    c, p, d = r["metricas"].get("candidato"), r["metricas"].get("produccion"), r["metricas"].get("diferencia")
    vc, vp = r["version_candidato"], r["version_produccion"]
    lc, lp = r["latencia"]["candidato"], r["latencia"].get("produccion", {})
    L = [f"## E2E: candidato (MLflow v{vc}) frente a producción (v{vp or '—'})", "",
         f"Commit `{r['commit'][:7]}` · modelo A: {r['modelos']['modelo_a']} · {r['n_noches']} noches de test "
         f"({r['metricas']['sujetos']} personas), cada una con una combinación de canales sorteada (la misma para los "
         f"dos) · {r['minutos']:.0f} min", ""]
    if r["errores"]["candidato"]:
        L += [f"❌ **El candidato falló en {len(r['errores']['candidato'])} noche(s)**: "
              + ", ".join(sorted(r["errores"]["candidato"])), ""]
    filas = [("mf1", "**MF1**"), ("kappa", "κ"), ("acc", "Exactitud")] + [(f"f1_{s}", f"F1 {s}") for s in ESTADIOS]
    if p is None:
        L += ["⚠️ No hay comparación con producción: " + r.get("sin_produccion", "no respondió"), ""]
        L += ["| | Candidato [IC 95 %] |", "|---|---|"]
        for k, nombre in filas:
            L.append(f"| {nombre} | {_f(c[k])} {_ic(c['ic95'][k])} |")
    else:
        L += ["| | Producción | Candidato | Δ candidato − producción [IC 95 %] |", "|---|---|---|---|"]
        for k, nombre in filas:
            dif = d[k]
            marca = "" if dif["ic95"][0] <= 0 <= dif["ic95"][1] else (" ⬆️" if dif["valor"] > 0 else " ⬇️")
            L.append(f"| {nombre} | {_f(p[k])} | {_f(c[k])} | {_f(dif['valor'], True)} {_ic(dif['ic95'], True)}{marca} |")
    L += ["", f"**Latencia por noche** (segundos de proceso dentro del contenedor, {C.INSTANCIA}, sin la cola):", "",
          "| | Media | Mediana | p95 | Máximo |", "|---|---|---|---|---|"]
    for nombre, l in (("Producción", lp), ("Candidato", lc)):
        if l:
            L.append(f"| {nombre} | {l['media']:.1f} | {l['mediana']:.1f} | {l['p95']:.1f} | {l['maximo']:.1f} |")
        elif p is not None:
            L.append(f"| {nombre} | — | — | — | — |")
    L += ["", "<details><summary>Matrices de confusión (% por fila)</summary>", ""]
    for nombre, m in (("Producción", p), ("Candidato", c)):
        if m:
            L += [f"**{nombre}**", "", _confusion_md(m["confusion"]), ""]
    L += ["</details>", "",
          "⬆️/⬇️: diferencia significativa (el IC no cruza 0). Producción no se toca: este modelo se publica solo si se "
          "hace merge y el PR cambia lo que va dentro del endpoint (`modelo/`, `sistema/`, `despliegue/inference.py` o "
          "`despliegue/requirements.txt`)."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nombre", required=True, help="identifica el candidato, p. ej. pr-7 (endpoint sistema-sueno-pr-7)")
    ap.add_argument("--version", type=int, help="versión de MLflow ya registrada (si no, registra modelo/ y sistema/)")
    ap.add_argument("--salida", default="e2e")
    a = ap.parse_args()
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    noches = manifiesto()
    canales = sortear_canales(noches)
    version = a.version or registrar({"estado": "candidato", "candidato": a.nombre})
    endpoint = f"{C.MODELO}-{a.nombre}"[:63]
    v_prod = desplegar.version_desplegada(C.MODELO)

    try:
        desplegar.borrar(endpoint)   # por si quedó uno de una corrida anterior
        with ThreadPoolExecutor(2) as ex:
            # producción empieza a prenderse mientras se crea el candidato
            f_prod = ex.submit(correr, C.MODELO, noches, canales) if v_prod else None
            desplegar.crear_candidato(endpoint, version)
            r_cand = correr(endpoint, noches, canales)
            r_prod = f_prod.result() if f_prod else None
    finally:
        desplegar.borrar(endpoint)

    pc, ec = predicciones(noches, r_cand)
    pp, ep = predicciones(noches, r_prod) if r_prod else ({}, {})
    comunes = {n: pp[n] for n in pc if n in pp}
    resultado = {
        "commit": commit(), "version_candidato": version, "version_produccion": v_prod, "modelos": describir_modelos(),
        "n_noches": len(noches), "canales": canales, "minutos": (time.time() - t0) / 60,
        "errores": {"candidato": ec, "produccion": ep},
        "metricas": evaluar({n: pc[n] for n in comunes} if comunes else pc, comunes or None),
        "latencia": {"candidato": latencia(r_cand)} | ({"produccion": latencia(r_prod)} if r_prod else {}),
    }
    if not v_prod:
        resultado["sin_produccion"] = f"no existe el endpoint {C.MODELO}"
    elif not comunes:
        resultado["sin_produccion"] = f"producción falló en las {len(ep)} noches"
    (salida / "e2e.json").write_text(json.dumps(resultado, indent=1, ensure_ascii=False))
    (salida / "e2e.md").write_text(informe(resultado))
    print(informe(resultado))
    if ec:
        sys.exit(f"el candidato falló en {len(ec)} noche(s): {ec}")


if __name__ == "__main__":
    main()
