"""Invariantenpruefung nach Abschnitt 6.3 der Spezifikation.

Die Spezifikation sieht `hypothesis` vor. Das Paket liegt nicht vor, deshalb
werden die Parametersaetze hier aus einem **festgelegten** Zufallsstrom gezogen.
Der Strom ist bestimmt, der Lauf also wiederholbar; der Unterschied zu
`hypothesis` ist, dass kein Gegenbeispiel verkleinert wird. Das ist in den
offenen Punkten vermerkt.

**Eine Invariante der Spezifikation ist beim Bau gefallen.** Sie steht unten mit
der Bedingung, unter der sie gilt, und mit einem Test, der ihr Scheitern im
allgemeinen Fall festhaelt.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dg.core import beliefs, chain, verify  # noqa: E402
from dg.core.types import Kette, Stelle, Umgebung  # noqa: E402

ZIEHUNGEN = 4000


def _parametersatz(rng: np.random.Generator, *, kosten: bool) -> tuple[Kette, Umgebung]:
    n = int(rng.integers(1, 5))
    stellen = tuple(
        Stelle(
            alpha=float(rng.uniform(0.0, 0.45)),
            beta=float(rng.uniform(0.0, 0.45)),
            c=float(rng.uniform(0.0, 20.0)) if kosten else 0.0,
            tau=float(rng.uniform(0.0, 3.0)),
        )
        for _ in range(n)
    )
    kette = Kette(stellen=stellen, W=float("inf"))
    umgebung = Umgebung(
        pi=float(rng.uniform(0.05, 0.95)),
        R=float(rng.uniform(0.0, 100.0)),
        L=float(rng.uniform(0.0, 100.0)),
        lam=1.0,
    )
    return kette, umgebung


def _mit_geaendertem(kette: Kette, i: int, *, alpha=None, beta=None) -> Kette:
    s = kette.stellen[i]
    neu = list(kette.stellen)
    neu[i] = Stelle(
        alpha=s.alpha if alpha is None else alpha,
        beta=s.beta if beta is None else beta,
        c=s.c, tau=s.tau, besetzung=s.besetzung, r=s.r, regel=s.regel,
    )
    return Kette(stellen=tuple(neu), W=kette.W, rho=kette.rho,
                 zeitregel=kette.zeitregel, kappa=kette.kappa)


def test_ertrag_faellt_in_beta_unbedingt() -> None:
    """Der Ertrag faellt in jedem beta_i, auch bei Kosten.

    Grund: Ein hoeheres beta_i hebt sowohl das Produkt der beta, also den
    Schaden, als auch die Erreichenswahrscheinlichkeiten der spaeteren Stellen,
    also die Kosten. Beide Wirkungen weisen in dieselbe Richtung.
    """
    rng = np.random.default_rng(1)
    for _ in range(ZIEHUNGEN):
        kette, umgebung = _parametersatz(rng, kosten=True)
        i = int(rng.integers(0, kette.n))
        vorher = chain.ertrag(kette, umgebung)
        nachher = chain.ertrag(
            _mit_geaendertem(kette, i, beta=min(kette.stellen[i].beta + 0.05, 1.0)), umgebung)
        assert nachher <= vorher + 1e-9


def test_ertrag_faellt_in_alpha_nur_ohne_kosten() -> None:
    """Der Ertrag faellt in jedem alpha_i, **sofern die Stellen nichts kosten**."""
    rng = np.random.default_rng(2)
    for _ in range(ZIEHUNGEN):
        kette, umgebung = _parametersatz(rng, kosten=False)
        i = int(rng.integers(0, kette.n))
        vorher = chain.ertrag(kette, umgebung)
        nachher = chain.ertrag(
            _mit_geaendertem(kette, i, alpha=min(kette.stellen[i].alpha + 0.05, 1.0)), umgebung)
        assert nachher <= vorher + 1e-9


def test_ertrag_faellt_in_alpha_bei_kosten_gerade_nicht() -> None:
    """**Berichtigung einer Invariante der Spezifikation.**

    Abschnitt 6.3 fordert, der Ertrag falle monoton in jedem alpha_i. Das gilt
    mit Kosten **nicht**. Ein hoeheres alpha_i senkt zwar den Ertrag aus
    wirksamen Faellen, stoppt aber zugleich mehr Faelle frueh und spart damit die
    Bearbeitungskosten der spaeteren Stellen. Sind die Kosten gross gegenueber
    dem Ertrag, ueberwiegt die Ersparnis.

    Dieser Test haelt das Scheitern fest, statt die Invariante abzuschwaechen und
    zu verschweigen. **Modellaussage, keine Weltaussage:** Eine schlechtere
    Stelle kann im Modell den erwarteten Ertrag heben, weil sie als billiger
    Vorfilter wirkt.
    """
    kette = Kette(
        stellen=(Stelle(alpha=0.096, beta=0.215, c=13.03),
                 Stelle(alpha=0.350, beta=0.302, c=18.17)),
        W=float("inf"))
    umgebung = Umgebung(pi=0.426, R=22.17, L=67.41, lam=1.0)
    vorher = chain.ertrag(kette, umgebung)
    nachher = chain.ertrag(_mit_geaendertem(kette, 0, alpha=0.146), umgebung)
    assert nachher > vorher, "Gegenbeispiel greift nicht mehr, Invariante neu pruefen"


def test_zusaetzliche_stelle_senkt_den_durchlass_wirksamer_faelle() -> None:
    """Eine zusaetzliche Stelle mit alpha > 0 senkt P(weiter | theta = 1)."""
    rng = np.random.default_rng(3)
    for _ in range(ZIEHUNGEN):
        kette, _ = _parametersatz(rng, kosten=True)
        vorher, _ = chain.durchlass(kette)
        erweitert = Kette(stellen=kette.stellen + (Stelle(alpha=0.2, beta=0.3),),
                          W=kette.W, rho=kette.rho)
        nachher, _ = chain.durchlass(erweitert)
        assert nachher <= vorher + 1e-12


def test_verknuepfungsregeln_wirken_gegenlaeufig() -> None:
    """Parallele senkt alpha(r) und hebt beta(r), Reihe umgekehrt, fuer r >= 2."""
    rng = np.random.default_rng(4)
    for _ in range(ZIEHUNGEN):
        a = float(rng.uniform(0.01, 0.9))
        b = float(rng.uniform(0.01, 0.9))
        r = int(rng.integers(2, 6))
        a_par, b_par, _ = chain.wirksame_raten(Stelle(alpha=a, beta=b, r=r, regel="parallele"), 1.0)
        a_rei, b_rei, _ = chain.wirksame_raten(Stelle(alpha=a, beta=b, r=r, regel="reihe"), 1.0)
        assert a_par < a and b_par > b
        assert a_rei > a and b_rei < b


def test_wahrscheinlichkeiten_und_erwartungswerte_bleiben_zulaessig() -> None:
    """Alle Wahrscheinlichkeiten liegen in [0,1], alle Erwartungswerte sind endlich."""
    rng = np.random.default_rng(5)
    for _ in range(ZIEHUNGEN):
        kette, umgebung = _parametersatz(rng, kosten=True)
        p1, p0 = chain.durchlass(kette)
        assert 0.0 <= p1 <= 1.0 and 0.0 <= p0 <= 1.0
        q = chain.erreichen(kette, umgebung)
        assert np.all(q >= -1e-12) and np.all(q <= 1.0 + 1e-12)
        assert math.isfinite(chain.ertrag(kette, umgebung))
        assert math.isfinite(chain.durchlaufzeit(kette, umgebung))


def test_zeitfaktor_bleibt_zwischen_null_und_eins() -> None:
    """Der Zeitfaktor liegt in [0,1], hart wie weich."""
    rng = np.random.default_rng(6)
    for _ in range(ZIEHUNGEN):
        kette, umgebung = _parametersatz(rng, kosten=True)
        T = chain.durchlaufzeit(kette, umgebung)
        for regel, kappa in (("hart", 0.0), ("weich", 0.5)):
            k2 = Kette(stellen=kette.stellen, W=float(rng.uniform(0.0, 5.0)),
                       rho=kette.rho, zeitregel=regel, kappa=kappa)
            f = chain.zeitfaktor(k2, T)
            assert 0.0 <= f <= 1.0


# --------------------------------------------------- Stufe 2: Ueberzeugung und Pruefung


def test_aktualisierung_ist_von_der_reihenfolge_unabhaengig() -> None:
    """Erschoepfende Statistik: nur die Summen zaehlen, nicht die Reihenfolge.

    Das ist die Pruefung darauf, dass die Aktualisierung wirklich konjugiert ist.
    Waere sie es nicht, so haette die Reihenfolge Wirkung, weil der Zustand mehr
    als fuenf Zahlen traegt.
    """
    rng = np.random.default_rng(20261010)
    vor = beliefs.gleichverteilte_zellen(2.0, 7.0)
    for _ in range(200):
        teile = [
            beliefs.ziehen(pi=0.2, alpha=0.35, beta=0.15, nu=0.4, faelle=int(
                rng.integers(1, 40)), rng=rng)
            for _ in range(5)
        ]
        vorwaerts = vor
        for teil in teile:
            vorwaerts = beliefs.aktualisieren(vorwaerts, teil)
        rueckwaerts = vor
        for teil in reversed(teile):
            rueckwaerts = beliefs.aktualisieren(rueckwaerts, teil)
        summe = teile[0]
        for teil in teile[1:]:
            summe = summe + teil
        einmal = beliefs.aktualisieren(vor, summe)
        for andere in (rueckwaerts, einmal):
            assert vorwaerts.a_u == andere.a_u
            assert vorwaerts.a_v == andere.a_v
            assert vorwaerts.a_w == andere.a_w
            assert vorwaerts.c_1 == andere.c_1
            assert vorwaerts.c_0 == andere.c_0


def test_schwellen_fallen_in_der_vermuteten_guete() -> None:
    """Beide Schwellen fallen, wenn die vermuteten Fehlerraten fallen.

    Das ist die Aussage aus `notiz-l6-verifikationsschwelle.md` Abschnitt 2, hier
    als Invariante ueber gezogene Parametersaetze. Gehen `alpha_hut` und `beta_hut`
    gegen null, so gehen beide Schwellen gegen null, und es wird nicht mehr
    geprueft.
    """
    rng = np.random.default_rng(4711)
    umgebung = Umgebung(pi=0.2, R=100.0, L=50.0, lam=1.0)
    for _ in range(2000):
        pi_hut = float(rng.uniform(0.05, 0.95))
        a_hoch = float(rng.uniform(0.05, 0.9))
        b_hoch = float(rng.uniform(0.05, 0.9))
        faktor = float(rng.uniform(0.1, 0.9))
        hoch = verify.schwellen(*verify.posteriors(pi_hut, a_hoch, b_hoch), umgebung)
        tief = verify.schwellen(
            *verify.posteriors(pi_hut, a_hoch * faktor, b_hoch * faktor), umgebung
        )
        assert tief[1] <= hoch[1] + 1e-12, "k*_s steigt bei besserer vermuteter Guete"
        assert tief[0] <= hoch[0] + 1e-12, "k*_w steigt bei besserer vermuteter Guete"
    # Der Grenzfall selbst.
    assert verify.schwellen(*verify.posteriors(0.2, 0.0, 0.0), umgebung) == (0.0, 0.0)


def test_aufdeckungsgrad_steigt_in_eps_und_faellt_in_zeta() -> None:
    """`nu = 1 - (1 - eps) * zeta`, monoton in beiden, und `nu = 1` beim Pruefen."""
    rng = np.random.default_rng(99)
    for _ in range(2000):
        zeta = float(rng.uniform(0.0, 1.0))
        eps1, eps2 = sorted(rng.uniform(0.0, 1.0, 2))
        assert beliefs.aufdeckungsgrad(zeta, eps1) <= beliefs.aufdeckungsgrad(zeta, eps2) + 1e-15
        z1, z2 = sorted(rng.uniform(0.0, 1.0, 2))
        assert beliefs.aufdeckungsgrad(z2, eps1) <= beliefs.aufdeckungsgrad(z1, eps1) + 1e-15
        assert beliefs.aufdeckungsgrad(zeta, eps1, geprueft=True) == 1.0
    assert beliefs.aufdeckungsgrad(1.0, 0.0) == 0.0
    assert beliefs.aufdeckungsgrad(0.0, 0.0) == 1.0


def test_fisher_information_ist_linear_im_aufdeckungsgrad() -> None:
    """`I(g) = nu * w / (g(1-g))`, also linear in `nu`.

    Daraus die Daempfung der Klassenaussage: Die Identifikation springt bei
    `nu = 0`, die Lerngeschwindigkeit ist stetig und bei kleinem `nu` beliebig
    langsam.
    """
    rng = np.random.default_rng(7)
    for _ in range(2000):
        w = float(rng.uniform(0.05, 0.95))
        g = float(rng.uniform(0.02, 0.98))
        nu = float(rng.uniform(0.01, 1.0))
        faktor = float(rng.uniform(0.1, 1.0))
        eins = beliefs.fisher_g(nu, w, g)
        zwei = beliefs.fisher_g(nu * faktor, w, g)
        assert math.isclose(zwei, eins * faktor, rel_tol=1e-12)
    assert beliefs.fisher_g(0.0, 0.5, 0.1) == 0.0


def test_dauer_steigt_in_staerke_und_dichtheit() -> None:
    """Die Dauer waechst linear in der Staerke der Meinung und in `1/(1-zeta)`."""
    g, verhaeltnis = 0.0933333, 0.07
    grund = beliefs.gleichverteilte_zellen(20.0, 380.0)
    eins = beliefs.dauer_mittelpfad(grund, g=g, schwellenverhaeltnis=verhaeltnis, zeta=0.9)
    doppelt = beliefs.dauer_mittelpfad(
        beliefs.gleichverteilte_zellen(40.0, 760.0), g=g,
        schwellenverhaeltnis=verhaeltnis, zeta=0.9,
    )
    assert math.isclose(doppelt, 2.0 * eins, rel_tol=1e-12)
    dichter = beliefs.dauer_mittelpfad(grund, g=g, schwellenverhaeltnis=verhaeltnis, zeta=0.95)
    assert math.isclose(dichter, 2.0 * eins, rel_tol=1e-12)
    assert beliefs.dauer_mittelpfad(grund, g=g, schwellenverhaeltnis=verhaeltnis, zeta=1.0) == math.inf
