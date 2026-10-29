"""particiones.json: partición por sujeto y sin cruces; las 12 combinaciones."""
from sueno.config import COMBINACIONES, particion

P = particion()


def test_combinaciones_iguales_a_particiones():
    assert {k: tuple(v) for k, v in P["combinaciones"].items()} == COMBINACIONES
    assert list(P["combinaciones"]) == list(COMBINACIONES)   # el orden importa para el sorteo de canales


def test_12_combinaciones_con_al_menos_un_eeg():
    assert len(COMBINACIONES) == 12
    assert all({"fpz", "pz"} & set(c) for c in COMBINACIONES.values())


def test_16_de_test_y_62_de_desarrollo_sin_cruce():
    test, desarrollo = set(P["sujetos_test"]), set(P["sujetos_desarrollo"])
    assert (len(test), len(desarrollo)) == (16, 62)
    assert not test & desarrollo


def test_cada_sujeto_de_desarrollo_es_test_en_un_solo_fold():
    vistos = [s for f in P["folds"] for s in f["test"]]
    assert sorted(vistos) == sorted(P["sujetos_desarrollo"])


def test_folds_sin_cruces_ni_sujetos_de_test():
    for f in P["folds"] + [P["final"]]:
        train, val, test = set(f["train"]), set(f["val"]), set(f["test"])
        assert not (train & val or train & test or val & test), f"fold {f['fold']}"
        assert not (train | val) & set(P["sujetos_test"]), f"fold {f['fold']}"
