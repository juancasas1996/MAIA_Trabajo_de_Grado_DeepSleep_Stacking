"""Entrena el sistema completo y deja los modelos en modelos/.

    python scripts/entrenamiento_completo.py                      todos los pasos
    python scripts/entrenamiento_completo.py --pasos a meta       solo algunos
    python scripts/entrenamiento_completo.py --prueba             pocas noches y pocas rondas, para ver que todo corre

Pasos, en orden:
  rasgos   rasgos por época de las 153 noches (modelo A)                     CPU, ~10 min con 8 procesos
  noches   señal preprocesada de cada noche (modelo B)                       CPU, ~5 min
  a        modelo A: 5 folds (predicciones fuera de fold) + modelo final     CPU, ~1-2 h
  b        modelo B: 5 folds + modelo final                                  GPU, ~2 h en una A100 con 6 a la vez
  meta     meta-modelo: 5 folds + modelo final                               CPU, ~10 min
  evaluar  métricas de CV y test, IC por bootstrap, umbral de confianza,
           matrices de confusión y modelos/sistema.json                      CPU, ~2 min

Cada paso guarda lo que calcula en data/procesados/ y, si se corta, al volver a correr retoma donde iba.
Necesita en data/ los EDF de Sleep Cassette (sleep-cassette/) y SC-subjects.xls. Partición: particiones.json.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sueno import config as C  # noqa: E402
from sueno import evaluacion, meta, modelo_a, modelo_b  # noqa: E402
from sueno.lectura import horas_apagado, leer_noche, list_nights  # noqa: E402

PASOS = ("rasgos", "noches", "a", "b", "meta", "evaluar")
ENTRENAMIENTOS_B = ["final", "fold_0", "fold_1", "fold_2", "fold_3", "fold_4"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def guardar_preds(path, preds):
    """preds: clave -> noche -> (y, p), con clave = combinación o (combinación, oof|test)."""
    arr = {}
    for clave, p in preds.items():
        for n, (y, pr) in p.items():
            arr[f"{clave}__{n}__y"], arr[f"{clave}__{n}__pred"] = y, pr
    np.savez_compressed(path, **arr)


def _redondear(x):
    """Floats a 4 decimales, dentro de dicts y listas (para el JSON de métricas)."""
    if isinstance(x, dict):
        return {k: _redondear(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_redondear(v) for v in x]
    return round(x, 4) if isinstance(x, float) else x


def cargar_preds(path):
    z = np.load(path)
    out = {}
    for k in z.files:
        clave, n, tipo = k.rsplit("__", 2)
        if tipo == "y":
            out.setdefault(clave, {})[n] = (z[k], z[f"{clave}__{n}__pred"])
    return out


class Proyecto:
    def __init__(self, prueba=False, n_procesos=8, en_paralelo=1, salida=None):
        self.prueba = prueba
        self.n_procesos = n_procesos
        self.en_paralelo = en_paralelo
        self.particion = C.particion()
        self.dir = C.PROCESADOS / ("prueba" if prueba else "")
        if prueba and salida is None:   # la prueba no pisa los modelos de verdad
            salida = self.dir
        self.modelos = Path(salida) / "modelos" if salida else C.MODELOS
        self.resultados = Path(salida) / "resultados" if salida else C.RESULTADOS
        self.noches = list_nights(C.EDF)
        if prueba:
            keep = {s for f in self.folds() + [self.final()] for k in ("train", "val", "test") for s in f[k]}
            self.noches = [x for x in self.noches if C.sujeto(x[0]) in keep]
        assert self.noches, f"no hay EDF en {C.EDF}"

    def folds(self):
        f = self.particion["folds"]
        if not self.prueba:
            return f
        # Prueba: 1 sujeto de test por fold; la validación es el de test del fold siguiente, para que el meta-modelo
        # tenga predicciones fuera de fold de sus sujetos de validación; 3 de entrenamiento
        tests = [x["test"][:1] for x in f]
        out = []
        for k, x in enumerate(f):
            val = tests[(k + 1) % len(f)]
            out.append({**x, "train": [s for s in x["train"] if s not in val][:3], "val": val, "test": tests[k]})
        return out

    def final(self):
        f = self.particion["final"]
        if not self.prueba:
            return f
        val = self.particion["folds"][0]["test"][:1]
        return {**f, "train": [s for s in f["train"] if s not in val][:3], "val": val, "test": f["test"][:1]}

    # --- rasgos y noches -----------------------------------------------------------------------------------------

    def rasgos(self):
        ruta = self.dir / "modelo_a" / "rasgos.parquet"
        if not ruta.exists():
            ruta.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            modelo_a.tabla(self.noches, horas_apagado(C.SUJETOS_XLS), self.n_procesos).to_parquet(ruta)
            log(f"rasgos de {len(self.noches)} noches ({time.time() - t0:.0f} s)")
        return pd.read_parquet(ruta)

    def preparar_noches(self):
        from joblib import Parallel, delayed

        d = self.dir / "modelo_b" / "noches"
        d.mkdir(parents=True, exist_ok=True)

        def una(nid, psg, hyp):
            np.savez(d / f"{nid}.npz", **modelo_b.preparar(leer_noche({"psg": psg, "hipnograma": hyp})))

        pendientes = [x for x in self.noches if not (d / f"{x[0]}.npz").exists()]
        Parallel(n_jobs=self.n_procesos)(delayed(una)(*x) for x in pendientes)
        log(f"noches del modelo B listas: {len(list(d.glob('SC4*.npz')))} en {d}")

    # --- modelo A ------------------------------------------------------------------------------------------------

    def modelo_a(self):
        df = self.rasgos()
        cols = modelo_a.columnas(df)
        d = self.dir / "modelo_a" / "folds"
        d.mkdir(parents=True, exist_ok=True)
        n_max = 50 if self.prueba else modelo_a.N_MAX
        rondas = []
        for f in self.folds():
            pred, info = d / f"fold_{f['fold']}.npz", d / f"fold_{f['fold']}.json"
            if not info.exists():
                t0 = time.time()
                m = modelo_a.entrenar(df, cols, np.random.default_rng([C.SEED, f["fold"]]), f["train"], f["val"], n_max)
                guardar_preds(pred, modelo_a.predecir_combinaciones(m, df, cols, f["test"]))
                info.write_text(json.dumps({"rondas": m.best_iteration + 1}))
                log(f"modelo A, fold {f['fold']}: {m.best_iteration + 1} rondas ({time.time() - t0:.0f} s)")
            rondas.append(json.loads(info.read_text())["rondas"])

        n_final = int(np.mean(rondas))   # el modelo final usa el promedio de rondas de los folds
        if not (d / "final.npz").exists():
            t0 = time.time()
            final = self.final()
            m = modelo_a.entrenar(df, cols, np.random.default_rng([C.SEED, 99]), final["train"] + final["val"],
                                  n_estimators=n_final)
            modelo_a.guardar(m, cols, self.modelos / "modelo_a_xgboost")
            guardar_preds(d / "final.npz", modelo_a.predecir_combinaciones(m, df, cols, final["test"]))
            log(f"modelo A final: {n_final} rondas ({time.time() - t0:.0f} s) -> {self.modelos / 'modelo_a_xgboost'}")

    def preds_a(self):
        """combinación -> 'oof'/'test' -> noche -> (y, p)"""
        d = self.dir / "modelo_a" / "folds"
        out = {c: {"oof": {}, "test": {}} for c in C.COMBINACIONES}
        for f in self.folds():
            for c, p in cargar_preds(d / f"fold_{f['fold']}.npz").items():
                out[c]["oof"].update(p)
        for c, p in cargar_preds(d / "final.npz").items():
            out[c]["test"] = p
        return out

    # --- modelo B ------------------------------------------------------------------------------------------------

    def _spec_b(self, nombre):
        return self.final() if nombre == "final" else self.folds()[int(nombre.split("_")[1])]

    def entrenar_b(self, nombre):
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log(f"modelo B, {nombre}: entrenando en {device}")
        modelo_b.entrenar_y_predecir(nombre, self._spec_b(nombre), self.dir / "modelo_b" / "noches",
                                     self.dir / "modelo_b" / "entrenamientos", device,
                                     max_epochs=2 if self.prueba else None)

    def modelo_b(self):
        sal = self.dir / "modelo_b" / "entrenamientos"
        listo = lambda n: (sal / f"{n}.npz").exists()  # noqa: E731
        pendientes = [n for n in ENTRENAMIENTOS_B if not listo(n)]
        if self.en_paralelo == 1:
            for n in pendientes:
                self.entrenar_b(n)
        else:
            # Cada entrenamiento en un proceso aparte; varios a la vez en la misma GPU
            sal.mkdir(parents=True, exist_ok=True)
            corriendo = {}
            while pendientes or corriendo:
                while pendientes and len(corriendo) < self.en_paralelo:
                    n = pendientes.pop(0)
                    cmd = [sys.executable, __file__, "--pasos", "b", "--entrenamiento", n] + (["--prueba"] if self.prueba else [])
                    corriendo[n] = subprocess.Popen(cmd, stdout=open(sal / f"{n}.out", "w"), stderr=subprocess.STDOUT)
                    log(f"modelo B, {n}: lanzado (salida en {sal / f'{n}.out'})")
                for n, p in list(corriendo.items()):
                    if p.poll() is not None:
                        corriendo.pop(n)
                        salida = sal / f"{n}.out"
                        log(f"modelo B, {n}: " + ("listo" if listo(n) else f"FALLÓ (ver {salida})"))
                time.sleep(30)
        faltan = [n for n in ENTRENAMIENTOS_B if not listo(n)]
        assert not faltan, f"faltan entrenamientos del modelo B: {faltan}"
        destino = self.modelos / "modelo_b"
        destino.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sal / "modelo_final.pt", destino / "modelo_final.pt")
        log(f"modelo B final -> {destino}")

    def preds_b(self, preds_a):
        """Como preds_a, en la misma ventana: las épocas sin etiqueta toman la predicción de la anterior."""
        sal = self.dir / "modelo_b" / "entrenamientos"
        out = {c: {"oof": {}, "test": {}} for c in C.COMBINACIONES}
        for nombre in ENTRENAMIENTOS_B:
            key = "test" if nombre == "final" else "oof"
            z = np.load(sal / f"{nombre}.npz")
            for k in z.files:
                if k.startswith(("y__", "pos__")):
                    continue
                cfg, n = k.split("__")
                yw = preds_a[cfg][key][n][0]
                assert np.array_equal(yw[yw >= 0], z[f"y__{n}"]), f"{n}: las etiquetas de A y B no coinciden"
                out[cfg][key][n] = (yw, meta.fill_window(yw >= 0, z[k]))
        return out

    # --- meta-modelo ---------------------------------------------------------------------------------------------

    def meta(self):
        ruta = self.dir / "meta" / "predicciones.npz"
        if ruta.exists():
            return
        ruta.parent.mkdir(parents=True, exist_ok=True)
        A = self.preds_a()
        B = self.preds_b(A)
        t0 = time.time()
        oof_in = meta.tabla({c: A[c]["oof"] for c in A}, {c: B[c]["oof"] for c in B})
        test_in = meta.tabla({c: A[c]["test"] for c in A}, {c: B[c]["test"] for c in B})
        cols = meta.columnas(oof_in)
        p_oof = np.zeros((len(oof_in), C.K), dtype=np.float32)
        sujeto = oof_in.night.str[3:5].astype(int)
        for f in self.folds():
            m = meta.entrenar(oof_in, cols, f["train"], f["val"])
            r = sujeto.isin(f["test"]).values
            p_oof[r] = m.predict_proba(oof_in.loc[r, cols])
            log(f"meta-modelo, fold {f['fold']} ({time.time() - t0:.0f} s)")
        final = self.final()
        m = meta.entrenar(oof_in, cols, final["train"], final["val"])
        meta.guardar(m, cols, self.modelos / "meta_modelo")
        p_test = m.predict_proba(test_in[cols])
        oof, test = meta.por_combinacion(oof_in, p_oof), meta.por_combinacion(test_in, p_test)
        guardar_preds(ruta, {**{f"{c}__oof": oof[c] for c in oof}, **{f"{c}__test": test[c] for c in test}})
        log(f"meta-modelo final ({m.xgb.best_iteration + 1} rondas de XGBoost) -> {self.modelos / 'meta_modelo'}")

    # --- evaluación ----------------------------------------------------------------------------------------------

    def evaluar(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        z = cargar_preds(self.dir / "meta" / "predicciones.npz")
        S = {c: {"oof": z[f"{c}__oof"], "test": z[f"{c}__test"]} for c in C.COMBINACIONES}
        A = self.preds_a()
        B = self.preds_b(A)

        umbral, aviso = evaluacion.elegir_umbral({c: S[c]["oof"] for c in S})
        sistema = {"estadios": C.ESTADIOS, "modalidades": list(C.MODALIDADES),
                   "meta_modelo": "promedio de las probabilidades de meta_lr y meta_xgb",
                   "contexto_meta": list(meta.CONTEXT),
                   "relleno_bordes": "log(1/5) en las épocas que no existen antes del inicio o después del final de la noche",
                   "umbral_confianza": umbral, "aviso_noche_fraccion_revisar": round(aviso, 4)}
        lineas = [f'  "{k}": {json.dumps(v, ensure_ascii=False)}' for k, v in sistema.items()]   # una clave por línea
        (self.modelos / "sistema.json").write_text("{\n" + ",\n".join(lineas) + "\n}\n")

        filas = []
        for c in C.COMBINACIONES:
            fila = {"combinacion": c}
            for nombre, P in (("a", A), ("b", B), ("sistema", S)):
                for key in ("oof", "test"):
                    m = evaluacion.metricas_de(P[c][key])
                    fila[f"{nombre}_{'cv' if key == 'oof' else 'test'}_mf1"] = float(m["mf1"])
            m = evaluacion.metricas_de(S[c]["test"])
            fila.update({"sistema_test_kappa": float(m["kappa"]), "sistema_test_f1_N1": float(m["f1_N1"]),
                         "sistema_test_mf1_ic95": evaluacion.bootstrap_por_sujeto(
                             {n: v for n, v in S[c]["test"].items()})["mf1"]})
            filas.append(fila)
        azar = evaluacion.bootstrap_canales_al_azar({c: S[c]["test"] for c in C.COMBINACIONES})
        conf = evaluacion.resumen_confianza([v for c in S for v in S[c]["test"].values()], umbral)

        self.resultados.mkdir(parents=True, exist_ok=True)
        resumen = {"por_combinacion": filas,
                   "canales_al_azar": {k: v for k, v in azar.items() if k != "confusion"},
                   "confianza_test": {k: float(v) for k, v in conf.items()},
                   "umbral_confianza": umbral, "aviso_noche_fraccion_revisar": round(aviso, 4)}
        (self.resultados / "metricas.json").write_text(json.dumps(_redondear(resumen), indent=2, ensure_ascii=False) + "\n")

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        for ax, c, nombre in zip(axes[:2], ["fpz", "eeg2_eog_emg"], ["solo Fpz-Cz", "4 canales"]):
            m = evaluacion.metricas_de(S[c]["test"])
            evaluacion.graficar_confusion(m["confusion"], f"Test · {nombre}\nMF1 {m['mf1']:.3f} · κ {m['kappa']:.3f}", ax)
        evaluacion.graficar_confusion(azar["confusion"], f"Test · canales al azar\nMF1 {azar['mf1']:.3f} · "
                                      f"κ {azar['kappa']:.3f}", axes[2])
        plt.tight_layout()
        fig.savefig(self.resultados / "matrices_confusion_test.png", dpi=150)

        t = pd.DataFrame(filas).set_index("combinacion")[["a_cv_mf1", "b_cv_mf1", "sistema_cv_mf1", "sistema_test_mf1"]]
        print(t.round(3).to_string())
        log(f"canales al azar: MF1 {azar['mf1']:.3f} [{azar['ic95']['mf1'][0]:.3f}, {azar['ic95']['mf1'][1]:.3f}] · "
            f"umbral {umbral} · aviso {aviso:.4f} · resultados en {self.resultados}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pasos", nargs="+", choices=PASOS, default=list(PASOS))
    ap.add_argument("--prueba", action="store_true", help="pocas noches, 50 rondas y 2 pasadas")
    ap.add_argument("--n-procesos", type=int, default=os.cpu_count(), help="procesos para rasgos y noches")
    ap.add_argument("--en-paralelo", type=int, default=1, help="entrenamientos del modelo B a la vez (GPU)")
    ap.add_argument("--salida", help="carpeta donde dejar modelos/ y resultados/ (por defecto, las del repositorio)")
    ap.add_argument("--entrenamiento", choices=ENTRENAMIENTOS_B, help=argparse.SUPPRESS)  # uso interno
    a = ap.parse_args()

    p = Proyecto(a.prueba, a.n_procesos, a.en_paralelo, a.salida)
    if a.entrenamiento:
        p.entrenar_b(a.entrenamiento)
        return
    log(f"{len(p.noches)} noches · pasos: {' '.join(a.pasos)}" + (" · PRUEBA" if a.prueba else ""))
    acciones = {"rasgos": p.rasgos, "noches": p.preparar_noches, "a": p.modelo_a, "b": p.modelo_b,
                "meta": p.meta, "evaluar": p.evaluar}
    for paso in PASOS:
        if paso in a.pasos:
            acciones[paso]()
    log("listo")


if __name__ == "__main__":
    main()
