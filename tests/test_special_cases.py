"""Sonderfalltests nach Abschnitt 6.1 der Spezifikation.

Jeder Test hat eine von Hand nachrechenbare Erwartung, und die Rechnung steht im
Testrumpf, nicht nur die Zahl.

**Reichweite, Stand 10.10.2026 nach AP M4.** Abschnitt 6.1 fuehrt elf Faelle.
**Zehn sind jetzt abgedeckt**, denn `verify.py` und `beliefs.py` liegen vor. Der
elfte, `gamma = 0`, haengt an `objects.py` und entfaellt mit der Streichung von
FF4. Die Zuordnung steht in `ABDECKUNG`.

**Eine Erwartung der Spezifikation ist berichtigt.** Abschnitt 6.1 verlangte fuer
`eps = 0`, der Posterior ueber `alpha` bleibe exakt am Prior. Das trifft nicht zu:
`alpha` steht in der Zelle `u`, und `u` wird aus jedem Fall gelernt. Exakt am
Prior bleibt der Posterior ueber `g`, und das ist die Groesse, auf die es
ankommt. Der Test prueft beides, also auch, dass die Ueberzeugung ueber `alpha`
sich **sehr wohl** bewegt. Begruendung in `notiz-m2-lernen-formalisiert.md`
Abschnitt 5.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from dg.core import analytic, beliefs, chain, verify  # noqa: E402
from dg.core.types import Kette, Stelle, Umgebung  # noqa: E402

ABDECKUNG = {
    "alpha = beta = 0": "hier",
    "alpha_i = 1": "hier",
    "W unendlich": "hier",
    "W < tau_1": "hier",
    "r = 1": "hier",
    "n = 1": "hier",
    "k = 0": "hier, seit AP M4",
    "k > max(L,R)": "hier, seit AP M4",
    "eps = 1": "hier, seit AP M4",
    "eps = 0": "hier, seit AP M4, mit berichtigter Erwartung",
    "gamma = 0": "entfaellt mit FF4",
}

U = Umgebung(pi=0.3, R=100.0, L=50.0, lam=1000.0)

# Die Lage, auf die sich die vier Faelle der Stufe 2 beziehen.
PI_M, ALPHA_M, BETA_M = 0.20, 0.35, 0.15
U_M = Umgebung(pi=PI_M, R=100.0, L=50.0, lam=1000.0)


def test_fehlerfrei_keine_fehlerkosten() -> None:
    """alpha = beta = 0: Ertrag = lam * (pi*R - sum q_i c_i), keine Fehlerkosten."""
    k = Kette(stellen=(Stelle(alpha=0.0, beta=0.0, c=1.0),
                       Stelle(alpha=0.0, beta=0.0, c=3.0)))
    # Von Hand: prod(1-alpha) = 1, prod(beta) = 0.
    # q_1 = 1; q_2 = pi*1 + (1-pi)*0 = 0.3.
    # Ertrag = 1000 * (0.3*100 - (1*1 + 0.3*3)) = 1000 * (30 - 1.9) = 28100.
    assert math.isclose(chain.ertrag(k, U), 28100.0, rel_tol=1e-12)
    assert math.isclose(chain.erreichen(k, U)[1], 0.3, rel_tol=1e-12)


def test_alpha_eins_loescht_den_ertrag_aus_wirksamen_faellen() -> None:
    """alpha_i = 1 fuer ein i: der Ertrag aus wirksamen Faellen ist exakt null."""
    k = Kette(stellen=(Stelle(alpha=1.0, beta=0.2), Stelle(alpha=0.05, beta=0.1)))
    p1, p0 = chain.durchlass(k)
    assert p1 == 0.0
    # Es bleibt allein der Schaden: -lam*(1-pi)*prod(beta)*L.
    erwartet = 1000.0 * (-0.7 * (0.2 * 0.1) * 50.0)
    assert math.isclose(chain.ertrag(k, U), erwartet, rel_tol=1e-12)


def test_fenster_unendlich_ist_wie_ohne_fenster() -> None:
    """W -> unendlich: die Zeitrestriktion ist unwirksam."""
    stellen = (Stelle(alpha=0.1, beta=0.2, c=1.0, tau=5.0),
               Stelle(alpha=0.05, beta=0.1, c=3.0, tau=7.0))
    offen = Kette(stellen=stellen, W=float("inf"))
    gross = Kette(stellen=stellen, W=1e12)
    assert chain.zeitfaktor(offen, chain.durchlaufzeit(offen, U)) == 1.0
    assert math.isclose(chain.ertrag(offen, U), chain.ertrag(gross, U), rel_tol=1e-12)


def test_fenster_kleiner_als_erste_stelle_gibt_null() -> None:
    """W < tau_1: der Ertrag ist null, unabhaengig von allen uebrigen Groessen."""
    # q_1 = 1, also ist T >= tau_1 > W fuer jede Belegung der uebrigen Groessen.
    for alpha2, beta2, c2 in ((0.0, 0.0, 0.0), (0.4, 0.9, 99.0), (1.0, 1.0, 1.0)):
        k = Kette(stellen=(Stelle(alpha=0.1, beta=0.2, tau=5.0),
                           Stelle(alpha=alpha2, beta=beta2, c=c2, tau=1.0)),
                  W=4.999, zeitregel="hart")
        assert chain.ertrag(k, U) == 0.0


def test_redundanz_eins_ist_die_grundfassung() -> None:
    """r = 1: beide Verknuepfungsregeln geben die Grundfassung zurueck."""
    for regel in ("reihe", "parallele"):
        s = Stelle(alpha=0.17, beta=0.23, tau=2.0, r=1, regel=regel)
        a, b, t = chain.wirksame_raten(s, rho=1.0)
        assert math.isclose(a, 0.17, rel_tol=1e-15)
        assert math.isclose(b, 0.23, rel_tol=1e-15)
        assert math.isclose(t, 2.0, rel_tol=1e-15)


def test_eine_stelle_ist_die_einzelstelle() -> None:
    """n = 1: die Kette entspricht der Einzelstelle, Produkt mit einem Faktor."""
    s = Stelle(alpha=0.1, beta=0.2, c=2.0, tau=1.0)
    k = Kette(stellen=(s,))
    p1, p0 = chain.durchlass(k)
    assert math.isclose(p1, 0.9, rel_tol=1e-15)
    assert math.isclose(p0, 0.2, rel_tol=1e-15)
    assert math.isclose(chain.erreichen(k, U)[0], 1.0, rel_tol=1e-15)
    erwartet = 1000.0 * (0.3 * 0.9 * 100.0 - 0.7 * 0.2 * 50.0 - 2.0)
    assert math.isclose(chain.ertrag(k, U), erwartet, rel_tol=1e-12)


def test_analytischer_massstab_stimmt_mit_der_kette_ueberein() -> None:
    """Der unabhaengig abgezaehlte Wert deckt sich mit der Produktform."""
    k = Kette(stellen=(Stelle(alpha=0.1, beta=0.2, c=1.0, tau=2.0),
                       Stelle(alpha=0.05, beta=0.1, c=3.0, tau=1.0)), W=100.0)
    assert math.isclose(chain.ertrag(k, U), analytic.ertrag_n2(k, U), rel_tol=1e-12)


# --------------------------------------------------------------- Stufe 2, AP M4


def test_k_null_laesst_die_autoritaet_beim_menschen() -> None:
    """k = 0: es wird stets geprueft, die reale Autoritaet bleibt beim Menschen."""
    # Von Hand: k*_w = (1-mu_w)*L und k*_s = mu_s*R sind beide echt positiv,
    # solange mu_w < 1 und mu_s > 0. Bei k = 0 liegt k unter beiden Schwellen.
    u, v, _ = beliefs.zellen(PI_M, ALPHA_M, BETA_M)
    g = beliefs.anteil_wirksam_unter_gestoppten(PI_M, ALPHA_M, BETA_M)
    mu_w, mu_s = verify.posteriors_aus_zellen(u, v, g)
    assert verify.pruefen(0.0, "weiter", mu_w, mu_s, U_M) is True
    assert verify.pruefen(0.0, "stopp", mu_w, mu_s, U_M) is True
    assert verify.reale_autoritaet(0.0, mu_w, mu_s, U_M) == "Mensch"


def test_k_ueber_max_l_r_laesst_nie_pruefen() -> None:
    """k > max(L, R): es wird nie geprueft, unabhaengig von der Ueberzeugung.

    Der Grund steht in den Schwellen selbst: `k*_w = (1-mu_w)*L <= L` und
    `k*_s = mu_s*R <= R`, also ist `max(L, R)` eine obere Schranke beider
    Schwellen. Gepruueft wird das ueber ein Gitter von Ueberzeugungen, damit die
    Unabhaengigkeit von `alpha_hut, beta_hut` wirklich geprueft ist.
    """
    k = max(U_M.L, U_M.R) + 1e-9
    for alpha_hut in (0.0, 0.1, 0.5, 0.9, 1.0):
        for beta_hut in (0.0, 0.1, 0.5, 0.9, 1.0):
            for pi_hut in (0.05, 0.3, 0.8):
                mu_w, mu_s = verify.posteriors(pi_hut, alpha_hut, beta_hut)
                assert verify.pruefen(k, "weiter", mu_w, mu_s, U_M) is False
                assert verify.pruefen(k, "stopp", mu_w, mu_s, U_M) is False
                assert verify.reale_autoritaet(k, mu_w, mu_s, U_M) == "Maschine"


def test_eps_eins_hebt_die_zensur_auf_und_g_konvergiert() -> None:
    """eps = 1: keine Zensur mehr, die Ueberzeugung ueber g laeuft auf den wahren Wert.

    Bei `eps = 1` wird jeder gestoppte Fall entgegen der Empfehlung behandelt,
    also ist der Aufdeckungsgrad eins, gleich wie dicht die Zensur ist.
    """
    g_wahr = beliefs.anteil_wirksam_unter_gestoppten(PI_M, ALPHA_M, BETA_M)
    for zeta in (1.0, 0.5, 0.0):
        assert beliefs.aufdeckungsgrad(zeta, 1.0) == 1.0
    laufend = beliefs.gleichverteilte_zellen(1.0, 19.0)
    assert abs(laufend.lage_g - 0.05) < 1e-12
    rng = np.random.default_rng(20261010)
    for _ in range(40):
        laufend = beliefs.aktualisieren(
            laufend,
            beliefs.ziehen(pi=PI_M, alpha=ALPHA_M, beta=BETA_M, nu=1.0,
                           faelle=500, rng=rng),
        )
    # 20.000 Faelle, davon drei Viertel gestoppt und alle aufgedeckt.
    assert abs(laufend.lage_g - g_wahr) < 0.01
    pi_hut, alpha_hut, beta_hut = beliefs.vermutete_raten(laufend)
    assert abs(pi_hut - PI_M) < 0.02
    assert abs(alpha_hut - ALPHA_M) < 0.05
    assert abs(beta_hut - BETA_M) < 0.02


def test_eps_null_laesst_g_exakt_am_prior_und_alpha_nicht() -> None:
    """eps = 0 bei zeta = 1: der Posterior ueber `g` bleibt exakt am Prior.

    **Berichtigte Erwartung.** Abschnitt 6.1 verlangte das fuer `alpha`. Fuer
    `alpha` ist es falsch, denn `alpha` steht in der Zelle `u`, und die wird aus
    jedem Fall gelernt. Der Test haelt beide Haelften fest: `g` bewegt sich
    **nicht**, `alpha_hut` bewegt sich **sehr wohl**, und zwar auf den wahren Wert
    der Flaeche zu.
    """
    vor = beliefs.gleichverteilte_zellen(1.0, 19.0)
    assert beliefs.aufdeckungsgrad(1.0, 0.0) == 0.0
    rng = np.random.default_rng(20261010)
    laufend = vor
    for _ in range(40):
        zaehlstaende = beliefs.ziehen(
            pi=PI_M, alpha=ALPHA_M, beta=BETA_M, nu=0.0, faelle=500, rng=rng
        )
        assert zaehlstaende.h_1 == 0.0 and zaehlstaende.h_0 == 0.0
        laufend = beliefs.aktualisieren(laufend, zaehlstaende)

    # Erste Haelfte: g ruehrt sich nicht, und zwar exakt.
    assert laufend.c_1 == vor.c_1
    assert laufend.c_0 == vor.c_0
    assert laufend.lage_g == vor.lage_g

    # Zweite Haelfte: alpha_hut bewegt sich. Der Wert, auf den es laeuft, ist der
    # Punkt der Flaeche, der zum Prior-Mittel von g gehoert.
    _, alpha_vor, _ = beliefs.vermutete_raten(vor)
    _, alpha_nach, _ = beliefs.vermutete_raten(laufend)
    u, v, _ = beliefs.zellen(PI_M, ALPHA_M, BETA_M)
    _, alpha_ziel, _ = beliefs.nach_strukturell(u, v, vor.lage_g)
    assert abs(alpha_nach - alpha_vor) > 0.1
    assert abs(alpha_nach - alpha_ziel) < 0.03
    # Und die wahre Rate wird dabei nicht getroffen: das ist die Kernaussage.
    assert abs(alpha_nach - ALPHA_M) > 0.1


def test_abdeckung_ist_vollstaendig_bis_auf_ff4() -> None:
    """Zehn der elf Faelle stehen hier, der elfte entfaellt mit FF4."""
    hier = [f for f, ort in ABDECKUNG.items() if ort.startswith("hier")]
    assert len(hier) == 10
    assert ABDECKUNG["gamma = 0"] == "entfaellt mit FF4"
