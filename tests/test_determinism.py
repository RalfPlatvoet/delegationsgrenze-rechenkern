"""Bestimmtheit und Abhaengigkeitsgrenze, Abschnitt 6.4 und 4.3.

**Reichweite.** Abschnitt 6.4 verlangt zusaetzlich, dass die Reihenfolge der
Unternehmensbearbeitung das Querschnittsergebnis nicht veraendert. Das setzt das
agentenbasierte Modell voraus, das mit dem Zuschnitt vom 09.10.2026 gestrichen
ist. Der Kern der Stufe 1 enthaelt **keinen Zufall**; gepruaft wird deshalb, dass
wiederholte Auswertung bitgleich ist und dass die Reihenfolge der Stellen nur
dort wirkt, wo sie wirken soll.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dg.core import beliefs, chain, verify  # noqa: E402
from dg.core.types import Kette, Stelle, Umgebung  # noqa: E402

U = Umgebung(pi=0.3, R=100.0, L=50.0, lam=1000.0)
K = Kette(stellen=(Stelle(alpha=0.1, beta=0.2, c=1.0, tau=2.0),
                   Stelle(alpha=0.05, beta=0.1, c=3.0, tau=1.0),
                   Stelle(alpha=0.2, beta=0.3, c=2.0, tau=0.5)), W=100.0)


def test_wiederholte_auswertung_ist_bitgleich() -> None:
    """Zehn Auswertungen desselben Parametersatzes geben denselben Wert, bitgleich."""
    werte = [chain.ertrag(K, U) for _ in range(10)]
    assert len(set(werte)) == 1
    zeiten = [chain.durchlaufzeit(K, U) for _ in range(10)]
    assert len(set(zeiten)) == 1


def test_durchlass_haengt_nicht_von_der_reihenfolge_ab() -> None:
    """Die Durchlasswahrscheinlichkeiten sind Produkte und damit reihenfolgefrei."""
    umgekehrt = Kette(stellen=tuple(reversed(K.stellen)), W=K.W)
    assert math.isclose(chain.durchlass(K)[0], chain.durchlass(umgekehrt)[0], rel_tol=1e-15)
    assert math.isclose(chain.durchlass(K)[1], chain.durchlass(umgekehrt)[1], rel_tol=1e-15)


def test_kosten_und_zeit_haengen_sehr_wohl_von_der_reihenfolge_ab() -> None:
    """Kosten und Zeit sind reihenfolgeabhaengig, und das ist gewollt.

    Wer teuer und langsam ist, gehoert nach hinten, weil ihn dann weniger Faelle
    erreichen. Der Test haelt fest, dass der Kern diesen Unterschied **nicht**
    wegmittelt; er ist der Grund, warum die Verschaltung eine Entwurfsfrage ist.
    """
    umgekehrt = Kette(stellen=tuple(reversed(K.stellen)), W=K.W)
    assert not math.isclose(chain.durchlaufzeit(K, U),
                            chain.durchlaufzeit(umgekehrt, U), rel_tol=1e-9)
    assert not math.isclose(chain.ertrag(K, U), chain.ertrag(umgekehrt, U), rel_tol=1e-9)


def test_kern_laedt_kein_scipy() -> None:
    """Abhaengigkeitsgrenze nach Abschnitt 4.3, als Importpruefung verankert.

    Der Kern bleibt frei von SciPy, damit er im Browser ladbar bleibt. Diese
    Grenze weicht sonst unbemerkt auf.
    """
    for modul in ("scipy", "pandas", "matplotlib"):
        assert modul not in sys.modules, f"der Kern hat {modul} geladen"


def test_kern_benutzt_keinen_globalen_zufall() -> None:
    """Der Kern zieht keinen Zufall, also aendert ein gesetzter Startwert nichts."""
    np.random.seed(1)
    a = chain.ertrag(K, U)
    np.random.seed(999)
    b = chain.ertrag(K, U)
    assert a == b


# --------------------------------------------------- Stufe 2: Zufall nur aus dem Strom


def test_ziehen_haengt_allein_am_uebergebenen_strom() -> None:
    """Zwei Laeufe mit demselben Strom sind bitgleich, mit verschiedenem nicht."""
    def lauf(saat: int) -> tuple:
        rng = np.random.default_rng(saat)
        laufend = beliefs.gleichverteilte_zellen(1.0, 19.0)
        for _ in range(30):
            laufend = beliefs.aktualisieren(
                laufend,
                beliefs.ziehen(pi=0.2, alpha=0.35, beta=0.15, nu=0.3,
                               faelle=100, rng=rng),
            )
        return (laufend.a_u, laufend.a_v, laufend.a_w, laufend.c_1, laufend.c_0)

    assert lauf(20261010) == lauf(20261010)
    assert lauf(20261010) != lauf(20261011)


def test_beliefs_und_verify_benutzen_keinen_globalen_zufall() -> None:
    """Ein Eingriff in den globalen Zustand darf nichts aendern."""
    import random as standard_random

    def ohne_zufall() -> tuple:
        vor = beliefs.gleichverteilte_zellen(2.0, 8.0)
        zs = beliefs.erwartete_zaehlstaende(pi=0.2, alpha=0.35, beta=0.15,
                                            nu=0.3, faelle=100.0)
        nach = beliefs.aktualisieren(vor, zs)
        u, v, _, g = beliefs.erwartung(nach)
        return beliefs.vermutete_raten(nach) + verify.posteriors_aus_zellen(u, v, g)

    erst = ohne_zufall()
    np.random.seed(1)
    standard_random.seed(1)
    [np.random.random() for _ in range(10)]
    [standard_random.random() for _ in range(10)]
    assert ohne_zufall() == erst


def test_neue_module_laden_kein_scipy() -> None:
    """Die Abhaengigkeitsgrenze aus Abschnitt 4.3 gilt auch fuer Stufe 2."""
    for modul in (beliefs, verify):
        quelle = Path(modul.__file__).read_text(encoding="utf-8")
        assert "scipy" not in quelle
        assert "import pandas" not in quelle
        for verboten in ("print(", "open(", "logging"):
            assert verboten not in quelle, f"{verboten} in {modul.__name__}"
