"""AP L7: Der geschlossene Vergleichsmassstab fuer zwei Stellen.

Zweck nach Abschnitt 6.2 der Spezifikation: ein **unabhaengig** gerechneter Wert,
gegen den `chain.py` zu pruefen ist. Unabhaengig heisst hier woertlich: Dieses
Modul benutzt die Produktform aus Abschnitt 5.2 **nicht**, sondern zaehlt den
Entscheidungsbaum Blatt fuer Blatt ab. Wuerde es dieselbe Formel noch einmal
hinschreiben, pruefte der Test nichts.

Acht Blaetter: zwei Zustaende, mal zwei Urteile der ersten Stelle, mal zwei
Urteile der zweiten. Je Blatt werden Wahrscheinlichkeit, Auszahlung, Kosten und
Zeit getrennt gefuehrt und erst am Ende zusammengezogen.
"""

from __future__ import annotations

import math

from .chain import wirksame_raten, zeitfaktor
from .types import Kette, Umgebung


def ertrag_n2(kette: Kette, umgebung: Umgebung) -> float:
    """Erwarteter Periodenertrag einer Kette aus genau zwei Stellen.

    Gerechnet durch Abzaehlen des Entscheidungsbaums. Der Rueckgabewert muss mit
    `chain.ertrag` uebereinstimmen.
    """
    if kette.n != 2:
        raise ValueError(f"dieser Vergleichsmassstab gilt fuer n = 2, nicht n = {kette.n}")

    a1, b1, t1 = wirksame_raten(kette.stellen[0], kette.rho)
    a2, b2, t2 = wirksame_raten(kette.stellen[1], kette.rho)
    c1, c2 = kette.stellen[0].c, kette.stellen[1].c
    pi, R, L = umgebung.pi, umgebung.R, umgebung.L

    erwartete_auszahlung = 0.0
    erwartete_kosten = 0.0
    erwartete_zeit = 0.0

    for theta, p_theta in ((1, pi), (0, 1.0 - pi)):
        # Wahrscheinlichkeit, dass die erste Stelle weiterreicht.
        p_weiter_1 = (1.0 - a1) if theta == 1 else b1
        for d1, p_d1 in (("weiter", p_weiter_1), ("stopp", 1.0 - p_weiter_1)):
            pfad = p_theta * p_d1
            if pfad == 0.0:
                continue
            # Die erste Stelle bearbeitet jeden Fall.
            erwartete_kosten += pfad * c1 if d1 == "stopp" else 0.0
            erwartete_zeit += pfad * t1 if d1 == "stopp" else 0.0
            if d1 == "stopp":
                continue
            # Nur weitergereichte Faelle erreichen die zweite Stelle.
            erwartete_kosten += pfad * (c1 + c2)
            erwartete_zeit += pfad * (t1 + t2)
            p_weiter_2 = (1.0 - a2) if theta == 1 else b2
            for d2, p_d2 in (("weiter", p_weiter_2), ("stopp", 1.0 - p_weiter_2)):
                blatt = pfad * p_d2
                if d2 != "weiter":
                    continue
                erwartete_auszahlung += blatt * (R if theta == 1 else -L)

    roh = erwartete_auszahlung - erwartete_kosten
    return umgebung.lam * roh * zeitfaktor(kette, erwartete_zeit)


def durchlaufzeit_n2(kette: Kette, umgebung: Umgebung) -> float:
    """Die erwartete Durchlaufzeit, ebenfalls durch Abzaehlen."""
    if kette.n != 2:
        raise ValueError("gilt fuer n = 2")
    a1, b1, t1 = wirksame_raten(kette.stellen[0], kette.rho)
    _, _, t2 = wirksame_raten(kette.stellen[1], kette.rho)
    pi = umgebung.pi
    zeit = 0.0
    for theta, p_theta in ((1, pi), (0, 1.0 - pi)):
        p_weiter_1 = (1.0 - a1) if theta == 1 else b1
        zeit += p_theta * (1.0 - p_weiter_1) * t1
        zeit += p_theta * p_weiter_1 * (t1 + t2)
    return zeit


def ist_endlich(wert: float) -> bool:
    """Hilfsgroesse fuer die Invariantenpruefung."""
    return math.isfinite(wert)
