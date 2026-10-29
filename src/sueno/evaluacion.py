"""Métricas, intervalos por bootstrap agrupado por sujeto, confianza y matrices de confusión.

Las predicciones van como {noche: (y, probabilidades)}; las épocas con y = -1 no se evalúan.
"""
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from .config import ESTADIOS, K, SEED, sujeto

N_BOOT = 2000


def metricas(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, labels=range(K), average=None, zero_division=0)
    return {"acc": accuracy_score(y_true, y_pred), "mf1": f1.mean(), "kappa": cohen_kappa_score(y_true, y_pred),
            **{f"f1_{s}": v for s, v in zip(ESTADIOS, f1)},
            "confusion": confusion_matrix(y_true, y_pred, labels=range(K))}


def metricas_de(preds):
    y = np.concatenate([y for y, _ in preds.values()])
    p = np.concatenate([p.argmax(1) for _, p in preds.values()])
    return metricas(y[y >= 0], p[y >= 0])


def confusion(y, p):
    """Matriz de confusión de una noche (filas: real; columnas: predicho)."""
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


def bootstrap_canales_al_azar(preds, n_boot=N_BOOT, seed=SEED):
    """MF1 y κ del test con una combinación de canales al azar por noche, con IC 95 %.

    preds: combinación -> noche -> (y, p). En cada remuestra se eligen sujetos con reemplazo (con sus dos noches) y
    a cada noche se le sortea una de las combinaciones (1/12 cada una). El valor central usa todas las combinaciones
    en todas las noches con el mismo peso.
    """
    cfgs = list(preds)
    noches = sorted(preds[cfgs[0]])
    cm = np.stack([np.stack([confusion(*preds[c][n]) for c in cfgs]) for n in noches])   # (noches, cfgs, K, K)
    sujetos = sorted({sujeto(n) for n in noches})
    por_sujeto = {s: [i for i, n in enumerate(noches) if sujeto(n) == s] for s in sujetos}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = np.concatenate([por_sujeto[s] for s in rng.choice(sujetos, size=len(sujetos), replace=True)])
        pick = rng.integers(0, len(cfgs), size=len(idx))
        boot.append(metricas_confusion(cm[idx, pick].sum(0)))
    centro = metricas_confusion(cm.sum((0, 1)))
    ic = {m: [float(np.quantile([b[m] for b in boot], 0.025)), float(np.quantile([b[m] for b in boot], 0.975))]
          for m in ("mf1", "kappa")}
    return {**{m: float(v) for m, v in centro.items()}, "ic95": ic, "confusion": cm.sum((0, 1))}


def bootstrap_por_sujeto(preds, n_boot=N_BOOT, seed=SEED):
    """IC 95 % de MF1 y κ de una combinación: remuestras de sujetos con reemplazo."""
    cms = {}
    for n, (y, p) in preds.items():
        cms[sujeto(n)] = cms.get(sujeto(n), 0) + confusion(y, p)
    sujetos = sorted(cms)
    st = np.stack([cms[s] for s in sujetos])
    rng = np.random.default_rng(seed)
    boot = [metricas_confusion(st[rng.integers(0, len(sujetos), len(sujetos))].sum(0)) for _ in range(n_boot)]
    return {m: [float(np.quantile([b[m] for b in boot], 0.025)), float(np.quantile([b[m] for b in boot], 0.975))]
            for m in ("mf1", "kappa")}


# ---------------------------------------------------------------------------------------------------------------
# Confianza: cuándo pedir revisión
# ---------------------------------------------------------------------------------------------------------------

GRID = np.round(np.arange(0.50, 0.951, 0.05), 2)


def _estadisticas_noche(y, p, umbral):
    v = y >= 0
    y, p = y[v], p[v]
    o = np.argsort(-p, 1)
    conf, t1, t2 = p.max(1), o[:, 0], o[:, 1]
    seg, ok, top2 = conf >= umbral, t1 == y, (t1 == y) | (t2 == y)
    cm = np.bincount(y[seg] * K + t1[seg], minlength=K * K).reshape(K, K)
    return {"n": len(y), "seg": seg.sum(), "ok_seg": (ok & seg).sum(), "dud": (~seg).sum(), "ok_dud": (ok & ~seg).sum(),
            "top2_dud": (top2 & ~seg).sum(), "ok": ok.sum(), "combinado": (ok & seg).sum() + (top2 & ~seg).sum(), "cm_seg": cm}


def resumen_confianza(noches, umbral):
    """noches: lista de (y, p). Cobertura, acierto de lo entregado y de lo que va a revisar."""
    st = [_estadisticas_noche(y, p, umbral) for y, p in noches]
    t = {k: sum(s[k] for s in st) for k in st[0]}
    return {"cobertura": t["seg"] / t["n"], "acc_total": t["ok"] / t["n"], "acc_seguras": t["ok_seg"] / t["seg"],
            "mf1_seguras": metricas_confusion(t["cm_seg"])["mf1"], "acc_dudosas": t["ok_dud"] / max(t["dud"], 1),
            "top2_dudosas": t["top2_dud"] / max(t["dud"], 1), "acc_combinada": t["combinado"] / t["n"]}


def elegir_umbral(preds_cv, acierto_minimo=0.90):
    """Con la CV (12 combinaciones juntas): el menor umbral con el que lo entregado acierta al menos el 90 %, y el
    aviso por noche (percentil 90 de la fracción de la noche que va a revisar)."""
    noches = [v for c in preds_cv for v in preds_cv[c].values()]
    umbral = next((float(u) for u in GRID if resumen_confianza(noches, u)["acc_seguras"] >= acierto_minimo), float(GRID[-1]))
    frac = [s["dud"] / s["n"] for s in (_estadisticas_noche(y, p, umbral) for y, p in noches)]
    return umbral, float(np.quantile(frac, 0.90))


# ---------------------------------------------------------------------------------------------------------------
# Gráfica
# ---------------------------------------------------------------------------------------------------------------

def graficar_confusion(cm, titulo, ax):
    """Matriz de confusión con % por fila (cada fila es el estadio real y suma 100 %) y conteos."""
    cm = np.asarray(cm)
    cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{cmn[i, j]:.0%}\n{cm[i, j]:,}", ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > 0.6 else "black")
    ax.set_xticks(range(K), ESTADIOS)
    ax.set_yticks(range(K), ESTADIOS)
    ax.set_xlabel("predicho")
    ax.set_ylabel("real")
    ax.set_title(titulo, fontsize=10)
