"""Anschluss an den analytischen Kern, Abschnitt 6.2 der Spezifikation.

**Reichweite.** Abschnitt 6.2 beschreibt den Vergleich des **agentenbasierten
Modells** mit dem geschlossenen Wert, samt Wiederholungszahl und Standardfehler.
Das agentenbasierte Modell ist mit dem Zuschnitt vom 09.10.2026 gestrichen, und
das Verzeichnis `abm/` wird nicht gebaut. Was bleibt und hier geprueft wird, ist
der Vergleich zweier **unabhaengiger** Rechnungen desselben Wertes: die
Produktform in `chain.py` gegen die Abzaehlung des Entscheidungsbaums in
`analytic.py`.

Das ist schwaecher als der urspruenglich vorgesehene Test, denn es ist keine
Pruefung gegen eine Ziehung. Es ist aber die staerkste Kontrolle, die der
reduzierte Zuschnitt noch zulaesst, und es faengt genau die Fehlerart, die hier
droht: eine falsche Formel, zweimal hingeschrieben.

Die Toleranz steht im Test und nicht in einer Konfigurationsdatei, damit eine
Lockerung im Diff sichtbar wird.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dg.core import analytic, beliefs, chain, verify  # noqa: E402
from dg.core.types import Kette, Stelle, Umgebung  # noqa: E402

TOLERANZ = 1e-10
ZIEHUNGEN = 5000


def test_produktform_und_abzaehlung_stimmen_ueberein() -> None:
    """Ueber einen Parameterschnitt stimmen beide Rechnungen ueberein."""
    rng = np.random.default_rng(20261010)
    groesste_abweichung = 0.0
    for _ in range(ZIEHUNGEN):
        kette = Kette(
            stellen=(
                Stelle(alpha=float(rng.uniform(0, 1)), beta=float(rng.uniform(0, 1)),
                       c=float(rng.uniform(0, 20)), tau=float(rng.uniform(0, 3))),
                Stelle(alpha=float(rng.uniform(0, 1)), beta=float(rng.uniform(0, 1)),
                       c=float(rng.uniform(0, 20)), tau=float(rng.uniform(0, 3))),
            ),
            W=float(rng.uniform(0.0, 8.0)),
            rho=float(rng.uniform(0, 1)),
            zeitregel="hart",
        )
        umgebung = Umgebung(pi=float(rng.uniform(0, 1)), R=float(rng.uniform(0, 100)),
                            L=float(rng.uniform(0, 100)), lam=float(rng.uniform(1, 1000)))
        a = chain.ertrag(kette, umgebung)
        b = analytic.ertrag_n2(kette, umgebung)
        skala = max(1.0, abs(a))
        groesste_abweichung = max(groesste_abweichung, abs(a - b) / skala)
        assert abs(a - b) <= TOLERANZ * skala
    assert groesste_abweichung <= TOLERANZ


def test_durchlaufzeit_stimmt_ueberein() -> None:
    """Auch die erwartete Durchlaufzeit wird von beiden Wegen gleich berechnet."""
    rng = np.random.default_rng(99)
    for _ in range(ZIEHUNGEN):
        kette = Kette(
            stellen=(Stelle(alpha=float(rng.uniform(0, 1)), beta=float(rng.uniform(0, 1)),
                            tau=float(rng.uniform(0, 3))),
                     Stelle(alpha=float(rng.uniform(0, 1)), beta=float(rng.uniform(0, 1)),
                            tau=float(rng.uniform(0, 3)))),
            W=float("inf"), rho=float(rng.uniform(0, 1)))
        umgebung = Umgebung(pi=float(rng.uniform(0, 1)), R=1.0, L=1.0)
        assert math.isclose(chain.durchlaufzeit(kette, umgebung),
                            analytic.durchlaufzeit_n2(kette, umgebung),
                            rel_tol=1e-12, abs_tol=1e-12)


def test_weiche_zeitregel_naehert_die_harte_an() -> None:
    """Mit wachsendem kappa naehert sich die weiche Fassung der harten.

    Das ist die Robustheitspruefung aus Abschnitt 5.3: Haengt ein Ergebnis an der
    Unstetigkeit, so faellt es hier auf.
    """
    stellen = (Stelle(alpha=0.1, beta=0.2, c=1.0, tau=2.0),
               Stelle(alpha=0.05, beta=0.1, c=3.0, tau=1.0))
    umgebung = Umgebung(pi=0.3, R=100.0, L=50.0, lam=1000.0)
    hart = Kette(stellen=stellen, W=1.0, zeitregel="hart")
    assert chain.ertrag(hart, umgebung) == 0.0
    vorher = math.inf
    for kappa in (0.5, 2.0, 10.0, 100.0):
        weich = Kette(stellen=stellen, W=1.0, zeitregel="weich", kappa=kappa)
        wert = chain.ertrag(weich, umgebung)
        assert 0.0 <= wert < vorher
        vorher = wert
    assert vorher < 1e-6 * 22720.0


# ------------------------------------------------- Zwei Wege zu denselben Posteriors


def test_beide_wege_zu_den_posteriors_stimmen_ueberein() -> None:
    """`posteriors` aus den Modellgroessen gegen `posteriors_aus_zellen`.

    Der zweite Weg benutzt die Abbildung des ersten **nicht**: Er liest `mu_w`
    als `u/(u+v)` und `mu_s` als `g` ab. Waere das dieselbe Formel, zweimal
    hingeschrieben, pruefte der Test nichts. Dieselbe Regel wie bei
    `analytic.py` gegen `chain.py`.
    """
    rng = np.random.default_rng(20261010)
    schlimmste = 0.0
    for _ in range(5000):
        pi = float(rng.uniform(0.02, 0.98))
        alpha = float(rng.uniform(0.01, 0.99))
        beta = float(rng.uniform(0.01, 0.99))
        u, v, _ = beliefs.zellen(pi, alpha, beta)
        g = beliefs.anteil_wirksam_unter_gestoppten(pi, alpha, beta)
        w1 = verify.posteriors(pi, alpha, beta)
        w2 = verify.posteriors_aus_zellen(u, v, g)
        schlimmste = max(schlimmste, abs(w1[0] - w2[0]), abs(w1[1] - w2[1]))
    assert schlimmste < 1e-12, f"groesste Abweichung {schlimmste}"


def test_umkehrabbildung_ist_die_inverse() -> None:
    """`(pi, alpha, beta) -> (u, v, g) -> (pi, alpha, beta)` ist die Identitaet."""
    rng = np.random.default_rng(4711)
    for _ in range(5000):
        pi = float(rng.uniform(0.02, 0.98))
        alpha = float(rng.uniform(0.01, 0.99))
        beta = float(rng.uniform(0.01, 0.99))
        u, v, _ = beliefs.zellen(pi, alpha, beta)
        g = beliefs.anteil_wirksam_unter_gestoppten(pi, alpha, beta)
        pi2, alpha2, beta2 = beliefs.nach_strukturell(u, v, g)
        assert math.isclose(pi2, pi, abs_tol=1e-10)
        assert math.isclose(alpha2, alpha, abs_tol=1e-10)
        assert math.isclose(beta2, beta, abs_tol=1e-10)


def _median_bis_zum_umschlag(
    *, vor, pi: float, alpha: float, beta: float, nu: float, verhaeltnis: float,
    laeufe: int, rng
) -> int:
    """Gezogener Median der gestoppten Faelle bis zum Umschlag der Ueberzeugung.

    **Aktualisiert wird Fall fuer Fall**, und das ist nicht beliebig: Wird in
    Bloecken gezogen, so kann der Umschlag erst an der Blockgrenze gesehen werden,
    und der gemessene Median waechst mit der Blockgroesse. Gemessen am 10.10.2026
    fuer `Beta(1,19)`: Median 79 bei Blockgroesse 1, 154 bei 50, 446 bei 200.
    **Die erste Fassung dieses Tests zog in Bloecken von 200 und hat daraufhin
    einen Fehler im Dauergesetz gemeldet, den es nicht gab.**
    """
    gemessen = []
    for _ in range(laeufe):
        laufend = vor
        gestoppt = 0
        while laufend.lage_g < verhaeltnis and gestoppt < 200_000:
            zs = beliefs.ziehen(pi=pi, alpha=alpha, beta=beta, nu=nu,
                                faelle=1, rng=rng)
            gestoppt += int(zs.n_w)
            laufend = beliefs.aktualisieren(laufend, zs)
        gemessen.append(gestoppt)
    gemessen.sort()
    return gemessen[len(gemessen) // 2]


def test_dauergesetz_trifft_die_ziehung_in_beiden_regimen() -> None:
    """Das Dauergesetz aus AP M3 gegen eine Ziehung, Regime A und Regime B.

    Geprueft wird die Zahl der **gestoppten Faelle** bis zum Umschlag und nicht die
    der Perioden: Jene ist volumenfrei, diese nicht. **Verglichen wird Median mit
    Median**, nicht Median mit Mittelwert: Die Dauer ist rechtsschief verteilt, in
    Regime A geometrisch. `median_der_dauer` rechnet um, und die Toleranz steht
    hier im Test.
    """
    pi, alpha, beta = 0.20, 0.35, 0.15
    _, _, w = beliefs.zellen(pi, alpha, beta)
    g = beliefs.anteil_wirksam_unter_gestoppten(pi, alpha, beta)
    verhaeltnis = 0.07

    for c_1, c_0, erwartetes_regime, laeufe in (
        (1.0, 19.0, "A", 60),
        (20.0, 380.0, "B", 12),
    ):
        vor = beliefs.gleichverteilte_zellen(c_1, c_0)
        regime, mittel = beliefs.dauer_regime(
            vor, g=g, schwellenverhaeltnis=verhaeltnis, zeta=0.9, w=w
        )
        assert regime == erwartetes_regime, f"{regime} statt {erwartetes_regime}"
        vorhersage = beliefs.median_der_dauer(regime, mittel)
        median = _median_bis_zum_umschlag(
            vor=vor, pi=pi, alpha=alpha, beta=beta, nu=0.1,
            verhaeltnis=verhaeltnis, laeufe=laeufe,
            rng=np.random.default_rng(20261010),
        )
        assert 0.6 * vorhersage <= median <= 1.7 * vorhersage, (
            f"Regime {regime}: Vorhersage {vorhersage:.0f}, Median {median}"
        )


def test_ein_gegenbeispiel_genuegt_trennt_die_regime() -> None:
    """Die Regimebedingung, und was sie nicht behauptet.

    Bei `Beta(1,19)` und `k/R = 0,07` genuegt ein frueh eintreffendes
    Gegenbeispiel, bei `Beta(20,380)` nicht. **Die Bedingung sagt nichts darueber,
    was geschieht, wenn viele unwirksame Faelle vorausgehen**; sie macht die
    Verteilung rechtsschief, und deshalb ist der Median die Kennzahl.
    """
    assert beliefs.ein_gegenbeispiel_genuegt(
        beliefs.gleichverteilte_zellen(1.0, 19.0), 0.07
    )
    assert not beliefs.ein_gegenbeispiel_genuegt(
        beliefs.gleichverteilte_zellen(20.0, 380.0), 0.07
    )
    # Sieben vorausgegangene unwirksame Faelle vertraegt die lose Meinung, acht nicht.
    lose = beliefs.gleichverteilte_zellen(1.0, 19.0)
    assert (lose.c_1 + 1.0) / (lose.staerke_g + 1.0 + 7.0) >= 0.07
    assert (lose.c_1 + 1.0) / (lose.staerke_g + 1.0 + 8.0) < 0.07
