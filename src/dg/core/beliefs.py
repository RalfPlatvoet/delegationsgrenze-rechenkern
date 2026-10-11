"""AP M4: Die Ueberzeugung unter Zensur, ihre Aktualisierung, die Dauer und `eps*`.

Herkunft: `konzept-simulationswerkzeug.md` Abschnitt 2.5 in der Fassung vom
10.10.2026, `notiz-m2-lernen-formalisiert.md` Abschnitte 2 bis 6 und 8,
`notiz-m3-begriff-und-dauer.md` Abschnitte 5 und 6,
`anhang-a-annahmenkatalog.md` A2, E1, E3 bis E7.

**Die Koordinaten sind die Beobachtungskoordinaten und nicht die Modellgroessen.**
Gefuehrt wird

    (u, v, w) ~ Dirichlet(a_u, a_v, a_w)      g ~ Beta(c_1, c_0),  unabhaengig

mit `u = P(weiter, theta=1)`, `v = P(weiter, theta=0)`, `w = P(stopp)` und
`g = P(theta=1 | stopp)`. Die vermuteten Modellgroessen `pi`, `alpha`, `beta`
werden daraus abgeleitet und nicht unmittelbar gefuehrt. Der Grund steht in
`notiz-m2-lernen-formalisiert.md` Abschnitt 4: In diesen Koordinaten ist die
Aktualisierung **exakt konjugiert**, in `(alpha, beta)` nicht.

**`g` ist die Groesse, auf die es ankommt.** Sie ist die einzige, die unter
vollstaendiger Zensur nicht zu lernen ist, und sie ist gleichzeitig die
Nachverteilung `mu_s`, von der die Pruefung einer Ablehnung allein abhaengt.

Regeln nach Abschnitt 4.2 der Spezifikation: keine Ein- und Ausgabe, kein
globaler Zufall, nur unveraenderliche Eingaben, nur Standardbibliothek und NumPy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .types import Umgebung

# --------------------------------------------------------------------------- Typen


@dataclass(frozen=True, slots=True)
class Zaehlstaende:
    """Die erschoepfende Statistik, fuenf Zahlen.

    `n_u`, `n_v`, `n_w` zaehlen die Faelle je Empfehlungszelle, `h_1` und `h_0`
    die **aufgedeckten** gestoppten Faelle mit und ohne Wirksamkeit. Es gilt
    stets `h_1 + h_0 <= n_w`; der Rest der gestoppten Faelle bleibt verdeckt.
    """

    n_u: float = 0.0
    n_v: float = 0.0
    n_w: float = 0.0
    h_1: float = 0.0
    h_0: float = 0.0

    def __post_init__(self) -> None:
        for name in ("n_u", "n_v", "n_w", "h_1", "h_0"):
            wert = getattr(self, name)
            if wert < 0.0:
                raise ValueError(f"{name} darf nicht negativ sein, ist {wert}")
        if self.h_1 + self.h_0 > self.n_w + 1e-9:
            raise ValueError(
                f"aufgedeckte gestoppte Faelle ({self.h_1 + self.h_0}) koennen die "
                f"gestoppten ({self.n_w}) nicht uebersteigen"
            )

    def __add__(self, andere: "Zaehlstaende") -> "Zaehlstaende":
        return Zaehlstaende(
            n_u=self.n_u + andere.n_u,
            n_v=self.n_v + andere.n_v,
            n_w=self.n_w + andere.n_w,
            h_1=self.h_1 + andere.h_1,
            h_0=self.h_0 + andere.h_0,
        )


@dataclass(frozen=True, slots=True)
class Ueberzeugung:
    """Dirichlet ueber den drei Empfehlungszellen, Beta ueber `g`, unabhaengig.

    `a_u, a_v, a_w` sind die Dirichlet-Formparameter, `c_1, c_0` die der
    Beta-Verteilung. Nach E6 ist die Dirichlet-Vorverteilung gleichverteilt;
    **fuer `g` gilt E6 ausdruecklich nicht** (Entscheidung vom 10.10.2026 in
    AP M3), weil Gleichverteilung dort die Meinung waere, die Haelfte aller
    Ablehnungen sei falsch. Lage und Staerke der Beta-Verteilung sind Parameter
    mit Belegstand und hier ohne Vorbelegung zu uebergeben.
    """

    a_u: float
    a_v: float
    a_w: float
    c_1: float
    c_0: float

    def __post_init__(self) -> None:
        for name in ("a_u", "a_v", "a_w", "c_1", "c_0"):
            wert = getattr(self, name)
            if wert <= 0.0:
                raise ValueError(f"{name} muss positiv sein, ist {wert}")

    @property
    def staerke_zellen(self) -> float:
        return self.a_u + self.a_v + self.a_w

    @property
    def staerke_g(self) -> float:
        """Die Staerke der Meinung ueber `g`. Sie traegt die Dauer linear."""
        return self.c_1 + self.c_0

    @property
    def lage_g(self) -> float:
        """`g_0 = E[g]`. Sie entscheidet, **ob** der Ruhezustand besteht."""
        return self.c_1 / (self.c_1 + self.c_0)


def gleichverteilte_zellen(c_1: float, c_0: float) -> Ueberzeugung:
    """E6 fuer die Zellen, und fuer `g` die uebergebene Lage und Staerke."""
    return Ueberzeugung(a_u=1.0, a_v=1.0, a_w=1.0, c_1=c_1, c_0=c_0)


# --------------------------------------------------------- Abbildungen der Koordinaten


def aufdeckungsgrad(zeta: float, eps: float, *, geprueft: bool = False) -> float:
    """`nu = 1 - (1 - eps) * zeta`, und `nu = 1` sobald geprueft wird.

    `zeta` ist der Zensurgrad und gegeben (E3), `eps` der Erkundungsanteil und
    gewaehlt (E4). **Identifikatorisch treten beide ausschliesslich ueber `nu`
    ein**, entscheidungstheoretisch nicht: `zeta` kostet nichts, `eps` kostet je
    Fall. Die Pruefung deckt den Zustand ebenfalls auf und ist damit der zweite
    Informationskanal (`notiz-m2-lernen-formalisiert.md` Abschnitt 8).
    """
    if not (0.0 <= zeta <= 1.0):
        raise ValueError(f"zeta muss in [0,1] liegen, ist {zeta}")
    if not (0.0 <= eps <= 1.0):
        raise ValueError(f"eps muss in [0,1] liegen, ist {eps}")
    if geprueft:
        return 1.0
    return 1.0 - (1.0 - eps) * zeta


def zellen(pi: float, alpha: float, beta: float) -> tuple[float, float, float]:
    """Die drei Empfehlungszellen `(u, v, w)` aus den Modellgroessen."""
    u = pi * (1.0 - alpha)
    v = (1.0 - pi) * beta
    return u, v, 1.0 - u - v


def anteil_wirksam_unter_gestoppten(pi: float, alpha: float, beta: float) -> float:
    """`g = P(theta = 1 | stopp) = pi*alpha / w`."""
    u, v, w = zellen(pi, alpha, beta)
    if w <= 0.0:
        raise ValueError("ohne gestoppte Faelle ist g nicht definiert")
    return (pi - u) / w


def nach_strukturell(u: float, v: float, g: float) -> tuple[float, float, float]:
    """Die Umkehrung: `(u, v, g) -> (pi, alpha, beta)`.

    `pi = u + g*w`, `alpha = 1 - u/pi`, `beta = v/(1-pi)`. Die Abbildung ist eine
    Bijektion; nachgerechnet in `m2_identifikation.py` Pruefpunkt P7.
    """
    w = 1.0 - u - v
    pi = u + g * w
    if pi <= 0.0 or pi >= 1.0:
        raise ValueError(f"die Umkehrung gibt pi = {pi}, das liegt nicht in (0,1)")
    return pi, 1.0 - u / pi, v / (1.0 - pi)


def identifizierter_bereich(u: float, v: float) -> tuple[float, float]:
    """Der Bereich, in dem `pi` liegen kann: `[u, 1 - v]`.

    Aus `alpha in [0,1]` folgt `pi >= u`, aus `beta in [0,1]` folgt `pi <= 1-v`.
    `g` laeuft dabei genau ueber `[0, 1]`. **Die Daten schraenken `g` also auf das
    ganze Einheitsintervall ein, also auf nichts.**
    """
    return u, 1.0 - v


# ------------------------------------------------------------------- Aktualisierung


def aktualisieren(ueberzeugung: Ueberzeugung, zaehlstaende: Zaehlstaende) -> Ueberzeugung:
    """Die konjugierte Aktualisierung, exakt und in geschlossener Form.

    Dirichlet gewinnt die Zellzaehlstaende, Beta gewinnt die Zaehlstaende der
    aufgedeckten gestoppten Faelle. **Bei `nu = 0` sind `h_1` und `h_0` null, und
    die Beta-Verteilung bleibt Zeichen fuer Zeichen die Vorverteilung**, fuer jede
    Laufzeit. Das ist der Sonderfall `eps = 0` aus Abschnitt 6.1.
    """
    return Ueberzeugung(
        a_u=ueberzeugung.a_u + zaehlstaende.n_u,
        a_v=ueberzeugung.a_v + zaehlstaende.n_v,
        a_w=ueberzeugung.a_w + zaehlstaende.n_w,
        c_1=ueberzeugung.c_1 + zaehlstaende.h_1,
        c_0=ueberzeugung.c_0 + zaehlstaende.h_0,
    )


def erwartung(ueberzeugung: Ueberzeugung) -> tuple[float, float, float, float]:
    """Die Erwartungswerte `(u, v, w, g)` unter der Ueberzeugung.

    Fuer die Entscheidung genuegen sie, weil der Wert der Pruefung in `g` linear
    ist (`notiz-m2-lernen-formalisiert.md` Abschnitt 9).
    """
    s = ueberzeugung.staerke_zellen
    return (
        ueberzeugung.a_u / s,
        ueberzeugung.a_v / s,
        ueberzeugung.a_w / s,
        ueberzeugung.lage_g,
    )


def vermutete_raten(ueberzeugung: Ueberzeugung) -> tuple[float, float, float]:
    """Die vermuteten Modellgroessen `(pi_hut, alpha_hut, beta_hut)`.

    Abgeleitet aus den Erwartungswerten der Koordinaten. **Das ist eine Ableitung
    und keine eigene Ueberzeugung**: Der Erwartungswert einer nichtlinearen
    Abbildung ist nicht die Abbildung des Erwartungswerts, und wo das zaehlt, ist
    mit der Verteilung zu rechnen und nicht mit diesen drei Zahlen.
    """
    u, v, _, g = erwartung(ueberzeugung)
    return nach_strukturell(u, v, g)


def fisher_g(nu: float, w: float, g: float) -> float:
    """Die Fisher-Information ueber `g` je Fall: `nu * w / (g(1-g))`.

    **Linear in `nu`.** Der Fallbedarf fuer eine gewuenschte Genauigkeit waechst
    also wie `1/nu`. Daraus die Daempfung der Klassenaussage in
    `notiz-block-b-lernen.md` Abschnitt 5.3: Die Identifikationsklasse springt bei
    `nu = 0`, die Lerngeschwindigkeit ist stetig.
    """
    if not (0.0 < g < 1.0):
        raise ValueError(f"g muss in (0,1) liegen, ist {g}")
    return nu * w / (g * (1.0 - g))


def ziehen(
    *,
    pi: float,
    alpha: float,
    beta: float,
    nu: float,
    faelle: int,
    rng: np.random.Generator,
) -> Zaehlstaende:
    """Zieht die Zaehlstaende einer Periode aus der wahren Umgebung.

    Der Zufall kommt ausschliesslich aus `rng`, nach Abschnitt 4.2. Gezogen wird
    aus der Multinomialverteilung ueber den fuenf Zellen, in der Reihenfolge
    `(u, v, stopp-aufgedeckt-wirksam, stopp-aufgedeckt-unwirksam, stopp-verdeckt)`.
    """
    u, v, w = zellen(pi, alpha, beta)
    g = (pi - u) / w if w > 0.0 else 0.0
    p = np.array([u, v, w * nu * g, w * nu * (1.0 - g), w * (1.0 - nu)], dtype=float)
    p = np.clip(p, 0.0, None)
    summe = p.sum()
    if summe <= 0.0:
        raise ValueError("die Zellwahrscheinlichkeiten summieren sich nicht auf eins")
    p /= summe
    n_u, n_v, h_1, h_0, verdeckt = rng.multinomial(faelle, p)
    return Zaehlstaende(
        n_u=float(n_u),
        n_v=float(n_v),
        n_w=float(h_1 + h_0 + verdeckt),
        h_1=float(h_1),
        h_0=float(h_0),
    )


def erwartete_zaehlstaende(
    *, pi: float, alpha: float, beta: float, nu: float, faelle: float
) -> Zaehlstaende:
    """Dieselben Zaehlstaende auf dem Mittelpfad, ohne Zufall.

    Gebraucht fuer die geschlossene Rechnung der Dauer und von `eps*`. Der
    Mittelpfad ist **nicht** immer der richtige Begriff; wo ein einzelnes seltenes
    Ereignis den Vorgang ausloest, ist die Wartezeit zu rechnen (siehe
    `dauer_regime` und die fuenfte Lehre in `anhang-b-methodennotiz.md`).
    """
    u, v, w = zellen(pi, alpha, beta)
    g = (pi - u) / w if w > 0.0 else 0.0
    return Zaehlstaende(
        n_u=faelle * u,
        n_v=faelle * v,
        n_w=faelle * w,
        h_1=faelle * w * nu * g,
        h_0=faelle * w * nu * (1.0 - g),
    )


# ------------------------------------------------------------------------ Die Dauer


def ein_gegenbeispiel_genuegt(
    ueberzeugung: Ueberzeugung, schwellenverhaeltnis: float
) -> bool:
    """Hebt ein aufgedeckter wirksamer Fall die Ueberzeugung ueber `k/R`?

    Bedingung: `(c_1 + 1)/(c_1 + c_0 + 1) >= k/R`. Trifft sie zu, so ist die Dauer
    des Ruhezustands die **Wartezeit auf das erste Gegenbeispiel** und nicht der
    Mittelpfad (`notiz-m3-begriff-und-dauer.md` Abschnitt 5.2).

    **Die Bedingung gilt fuer den Fall, dass das Gegenbeispiel frueh kommt**, und
    das ist der haeufige. Kommen vorher viele aufgedeckte **unwirksame** Faelle, so
    ist die Ueberzeugung bereits gesunken und ein Gegenbeispiel genuegt nicht mehr;
    bei `Beta(1, 19)` und `k/R = 0,07` reicht es, solange nicht mehr als sieben
    unwirksame vorausgegangen sind. Das macht die Verteilung der Dauer
    rechtsschief und ist der Grund, warum der **Median** und nicht der Mittelwert
    die brauchbare Kennzahl ist, siehe `median_der_dauer`.
    """
    c1, c0 = ueberzeugung.c_1, ueberzeugung.c_0
    return (c1 + 1.0) / (c1 + c0 + 1.0) >= schwellenverhaeltnis


def median_der_dauer(regime: str, mittel: float) -> float:
    """Der Median der Dauer aus ihrem Mittelwert, je Regime.

    In Regime A ist die Wartezeit geometrisch, also ist der Median `ln 2` mal dem
    Mittelwert. In Regime B treibt eine grosse Zahl von Ereignissen den Umschlag,
    die Verteilung ist nahezu symmetrisch, und der Median liegt etwa zehn Prozent
    unter dem Mittelpfad.

    **Gemessen am 10.10.2026**, je 60 Laeufe mit Aktualisierung Fall fuer Fall:
    Regime A, `Beta(1,19)`, Mittelwert 107 gestoppte Faelle, Median gezogen 79,
    `ln 2`-Vorhersage 74. Regime B, `Beta(20,380)`, Mittelpfad 3.429, Median
    gezogen 3.052, Vorhersage 3.086. Die beiden Faktoren stehen hier und nicht in
    einer Konfigurationsdatei, damit eine Aenderung im Diff sichtbar wird.
    """
    if not math.isfinite(mittel):
        return mittel
    if regime == "A":
        return math.log(2.0) * mittel
    if regime == "B":
        return 0.90 * mittel
    return mittel


def besteht_ruhezustand(ueberzeugung: Ueberzeugung, schwellenverhaeltnis: float) -> bool:
    """`k >= R * E[g]`, also die Ungleichung aus `notiz-m2` Abschnitt 9.

    Bei vollstaendiger Zensur und ohne Erkundung ist der Zustand dann
    absorbierend: Wird nicht geprueft, faellt kein aufgedeckter Fall an, also
    bleibt die Beta-Verteilung unveraendert, also wird weiter nicht geprueft.
    """
    return ueberzeugung.lage_g < schwellenverhaeltnis


def dauer_mittelpfad(
    ueberzeugung: Ueberzeugung, *, g: float, schwellenverhaeltnis: float, zeta: float
) -> float:
    """Gestoppte Faelle bis zum Umschlag, Regime B (feste Meinung).

    `N = (c_1 + c_0) * (k/R - g_0) / [ (g - k/R) * (1 - zeta) ]`.

    **Volumenfrei:** Die Fallzahl haengt nicht vom Fallstrom ab, nur die
    Uebersetzung in Perioden tut es. Daraus die Aussage, der Ruhezustand sei ein
    Phaenomen der kleinen Zahl.
    """
    g_0 = ueberzeugung.lage_g
    if g_0 >= schwellenverhaeltnis:
        return 0.0
    if g <= schwellenverhaeltnis:
        return math.inf
    if zeta >= 1.0:
        return math.inf
    h = ueberzeugung.staerke_g * (schwellenverhaeltnis - g_0) / (g - schwellenverhaeltnis)
    return h / (1.0 - zeta)


def dauer_wartezeit(*, g: float, w: float, zeta: float) -> float:
    """Gestoppte Faelle bis zum ersten aufgedeckten wirksamen Fall, Regime A.

    Geometrisch mit der Rate `(1 - zeta) * g` je gestoppten Fall, im Mittel also
    `1 / [(1 - zeta) * g]` gestoppte Faelle. `w` wird nicht gebraucht und steht
    nur als Erinnerung daran, dass die Uebersetzung in Perioden durch `lam * w`
    teilt.
    """
    rate = (1.0 - zeta) * g
    return math.inf if rate <= 0.0 else 1.0 / rate


def dauer_regime(
    ueberzeugung: Ueberzeugung,
    *,
    g: float,
    schwellenverhaeltnis: float,
    zeta: float,
    w: float,
) -> tuple[str, float]:
    """Waehlt das Regime und gibt die Dauer in **gestoppten Faellen** zurueck.

    Rueckgabe `("A", ...)` fuer die Wartezeit auf das erste Gegenbeispiel,
    `("B", ...)` fuer den Mittelpfad, `("kein Ruhezustand", 0.0)`, wenn schon die
    Vorverteilung ueber der Schwelle liegt, `("wohlfahrtsneutral", inf)`, wenn das
    Nichtpruefen auch bei richtiger Ueberzeugung richtig waere, und
    `("Fixpunkt", inf)` bei vollstaendiger Zensur.

    **Der Rueckgabewert ist in beiden Regimen ein Mittelwert.** Die Verteilung der
    Dauer ist rechtsschief, der Median liegt also darunter. Wer gegen eine Ziehung
    prueft, vergleiche Median mit Median und rechne mit `median_der_dauer` um.
    """
    if not besteht_ruhezustand(ueberzeugung, schwellenverhaeltnis):
        return "kein Ruhezustand", 0.0
    if zeta >= 1.0:
        return "Fixpunkt", math.inf
    if g <= schwellenverhaeltnis:
        return "wohlfahrtsneutral", math.inf
    if ein_gegenbeispiel_genuegt(ueberzeugung, schwellenverhaeltnis):
        return "A", dauer_wartezeit(g=g, w=w, zeta=zeta)
    return "B", dauer_mittelpfad(
        ueberzeugung, g=g, schwellenverhaeltnis=schwellenverhaeltnis, zeta=zeta
    )


def perioden_aus_faellen(faelle: float, *, lam: float, w: float) -> float:
    """Uebersetzt gestoppte Faelle in Perioden: `T = N / (lam * w)`."""
    if lam <= 0.0 or w <= 0.0:
        return math.inf
    return faelle / (lam * w)


def verlust_je_gestoppten_fall(
    *, g: float, R: float, k: float
) -> float:
    """`max(0, g*R - k)`: was der Ruhezustand je gestoppten Fall kostet.

    Null, wenn das Nichtpruefen auch bei richtiger Ueberzeugung richtig waere;
    das ist der wohlfahrtsneutrale Ruhezustand. Positiv im kostspieligen Fall,
    und **kein anfallendes Datum zeigt ihn an**.
    """
    return max(0.0, g * R - k)


# ------------------------------------------------------------------------- `eps*`


def erkundungskosten(mu_s: float, umgebung: Umgebung) -> float:
    """Die erwarteten Kosten eines blind erkundeten gestoppten Falles.

    `(1 - mu_s) * L - mu_s * R`, positiv solange das Stoppen im Mittel richtig
    war. Fassung aus Konzept 2.5.
    """
    return (1.0 - mu_s) * umgebung.L - mu_s * umgebung.R


def erkundung_kommt_in_frage(k: float, mu_s: float, umgebung: Umgebung) -> bool:
    """Ist blinde Erkundung ueberhaupt der billigere Informationskanal?

    Beide Kanaele decken denselben Zustand auf. Die Pruefung ist der billigere
    genau dann, wenn `k < (1 - mu_s) * L`. **Nur wenn sie es nicht ist, kann
    `eps* > 0` sein** (`notiz-m2-lernen-formalisiert.md` Abschnitt 8). Das ist der
    Grund, warum `eps*` auf dem groessten Teil des Parameterraums null ist, und
    nicht die Hoehe der Erkundungskosten.
    """
    return k >= (1.0 - mu_s) * umgebung.L


# ----------------------------------------------- Der Pfad und die optimale Politik


@dataclass(frozen=True, slots=True)
class Politik:
    """Eine Politik: Erkundungsanteil und Pruefvorlauf.

    Zwei Instrumente, nicht eines. `eps` ist die blinde Erkundung aus Konzept 2.5.
    `pruef_vorlauf` ist die Zahl der Anfangsperioden, in denen eine Ablehnung
    **gegen das myopische Urteil** geprueft wird; danach gilt wieder die
    Schwellenregel. Beide Instrumente decken denselben Zustand auf. Dass das
    Konzept nur das erste fuehrte, ist der Befund aus
    `notiz-m2-lernen-formalisiert.md` Abschnitt 8.

    **Warum ein Vorlauf und nicht ein Schalter.** Dauerhaftes Pruefen gegen das
    myopische Urteil ist keine ernsthafte Politik: Es kauft Information, die nach
    wenigen Perioden gekauft ist, und zahlt dann weiter. Der Vorlauf ist die
    kleinste Politikklasse, die den Informationskauf von der laufenden
    Entscheidung trennt, und er ist zugleich die Form, in der eine Empfehlung
    aussprechbar ist: **die ersten `N` Ablehnungen pruefen, danach nach Schwelle.**
    """

    eps: float = 0.0
    pruef_vorlauf: int = 0

    def __post_init__(self) -> None:
        if not (0.0 <= self.eps <= 1.0):
            raise ValueError(f"eps muss in [0,1] liegen, ist {self.eps}")
        if self.pruef_vorlauf < 0:
            raise ValueError(f"pruef_vorlauf darf nicht negativ sein, ist {self.pruef_vorlauf}")

    def prueft_ohnehin(self, t: int) -> bool:
        """Gilt der Vorlauf in Periode `t`, nullbasiert gezaehlt?"""
        return t < self.pruef_vorlauf


def pfad_ertrag(
    *,
    pi: float,
    alpha: float,
    beta: float,
    umgebung: Umgebung,
    vor: Ueberzeugung,
    politik: Politik,
    zeta: float,
    k: float,
    T_H: int,
    delta: float,
) -> float:
    """Diskontierter Ertrag ueber `T_H` Perioden auf dem Mittelpfad der Ueberzeugung.

    Der Ablauf je Periode: Aus der laufenden Ueberzeugung folgt das Verhalten, aus
    dem Verhalten der Aufdeckungsgrad, aus der **wahren** Umgebung der Ertrag und
    die erwarteten Zaehlstaende, daraus die Ueberzeugung der naechsten Periode.

    **Gerechnet wird auf dem Mittelpfad und nicht gezogen.** Das ist zulaessig,
    solange die Ueberzeugung sich stetig bewegt; wo ein einzelnes seltenes
    Ereignis den Umschlag ausloest, ist der Mittelpfad der falsche Begriff, und
    `dauer_regime` sagt, wann dieser Fall vorliegt. Der Test gegen die Ziehung
    steht in `tests/test_invariants.py`.
    """
    if not (0.0 <= delta <= 1.0):
        raise ValueError(f"delta muss in [0,1] liegen, ist {delta}")
    if T_H < 1:
        raise ValueError(f"T_H muss mindestens 1 sein, ist {T_H}")

    from . import verify  # lokal, damit die Modulgrenze einseitig bleibt

    u_wahr, v_wahr, _ = zellen(pi, alpha, beta)
    g_wahr = anteil_wirksam_unter_gestoppten(pi, alpha, beta)

    laufend = vor
    summe = 0.0
    for t in range(T_H):
        u_hut, v_hut, _, g_hut = erwartung(laufend)
        mu_w_hut, mu_s_hut = verify.posteriors_aus_zellen(u_hut, v_hut, g_hut)
        vorlauf = politik.prueft_ohnehin(t)
        geprueft = vorlauf or verify.pruefen(k, "stopp", mu_w_hut, mu_s_hut, umgebung)
        ertrag = verify.periodenertrag(
            u=u_wahr,
            v=v_wahr,
            g=g_wahr,
            k=k,
            eps=politik.eps,
            umgebung=umgebung,
            mu_w_hut=mu_w_hut,
            mu_s_hut=mu_s_hut,
            pruefe_stopp_ohnehin=vorlauf,
        )
        summe += (delta ** t) * ertrag
        nu = aufdeckungsgrad(zeta, politik.eps, geprueft=geprueft)
        laufend = aktualisieren(
            laufend,
            erwartete_zaehlstaende(
                pi=pi, alpha=alpha, beta=beta, nu=nu, faelle=umgebung.lam
            ),
        )
    return summe


def pfad_ertrag_erwartet(
    *,
    umgebung: Umgebung,
    vor: Ueberzeugung,
    politik: Politik,
    zeta: float,
    k: float,
    T_H: int,
    delta: float,
    ziehungen: int,
    rng: np.random.Generator,
) -> float:
    """Der **erwartete** diskontierte Ertrag unter der eigenen Ueberzeugung.

    Das ist das Programm der Organisation, und `pfad_ertrag` ist es **nicht**:
    Jene Funktion rechnet mit den wahren Groessen, die die Organisation nicht
    kennt. Hier wird je Ziehung ein Umweltzustand aus der Vorverteilung gezogen
    und der Pfad dann mit den **erwarteten** Zaehlstaenden dieses Zustands gelaufen.

    **Gezogen wird allein der Umweltzustand, nicht die Stichprobe.** Das ist
    Absicht und nicht Bequemlichkeit: Der Wert der Information steckt in der
    Unsicherheit ueber `g`, und die ist durch die Ziehung des Zustands voll
    erfasst. Die Stichprobenstreuung darueber hinaus traegt zum Erkundungswert
    nahezu nichts bei, macht aber die Gittersuche unbrauchbar, weil die Zielgroesse
    in `eps` flach ist und das Rauschen den Unterschied verdeckt.

    **Warum nicht auf dem Mittelpfad, und das ist der Grund fuer diese Funktion.**
    Auf dem Mittelpfad der eigenen Ueberzeugung bewegt sich `E[g]` nicht: Die
    erwarteten Zaehlstaende sind gerade so, dass der Erwartungswert bleibt und nur
    die Streuung faellt. Also aendert sich das Verhalten nie, also hat Erkundung
    keinen Wert, also kommt `eps* = 0` heraus, und zwar immer und unabhaengig von
    allen Parametern. **Das ist kein Ergebnis, sondern ein Rechenfehler**, denn der
    Wert der Information steckt in der Streuung der Nachverteilung und nicht in
    ihrem Mittel. Genau das ist die strenge Konvexitaet der Wertfunktion, an der
    EK1988 Satz 9 haengt: Wer die Streuung wegmittelt, setzt die Konvexitaet auf
    null und findet keinen Erkundungswert.
    """
    from . import verify  # lokal, damit die Modulgrenze einseitig bleibt

    if ziehungen < 1:
        raise ValueError(f"ziehungen muss mindestens 1 sein, ist {ziehungen}")

    summe_ueber_ziehungen = 0.0
    gezaehlt = 0
    for _ in range(ziehungen):
        # Umweltzustand aus der eigenen Vorverteilung. **Nur hier wird gezogen.**
        u_z, v_z, _ = rng.dirichlet([vor.a_u, vor.a_v, vor.a_w])
        g_z = float(rng.beta(vor.c_1, vor.c_0))
        if not (0.0 < u_z + v_z < 1.0) or not (0.0 < g_z < 1.0):
            continue
        pi_z, alpha_z, beta_z = nach_strukturell(u_z, v_z, g_z)
        gezaehlt += 1

        laufend = vor
        wert = 0.0
        for t in range(T_H):
            u_hut, v_hut, _, g_hut = erwartung(laufend)
            mu_w_hut, mu_s_hut = verify.posteriors_aus_zellen(u_hut, v_hut, g_hut)
            vorlauf = politik.prueft_ohnehin(t)
            geprueft = vorlauf or verify.pruefen(
                k, "stopp", mu_w_hut, mu_s_hut, umgebung
            )
            wert += (delta ** t) * verify.periodenertrag(
                u=u_z, v=v_z, g=g_z, k=k, eps=politik.eps, umgebung=umgebung,
                mu_w_hut=mu_w_hut, mu_s_hut=mu_s_hut,
                pruefe_stopp_ohnehin=vorlauf,
            )
            nu = aufdeckungsgrad(zeta, politik.eps, geprueft=geprueft)
            laufend = aktualisieren(
                laufend,
                erwartete_zaehlstaende(
                    pi=pi_z, alpha=alpha_z, beta=beta_z, nu=nu, faelle=umgebung.lam
                ),
            )
        summe_ueber_ziehungen += wert
    if gezaehlt == 0:
        raise ValueError("keine brauchbare Ziehung aus der Vorverteilung")
    return summe_ueber_ziehungen / gezaehlt


def erkundungsfenster(
    vor: Ueberzeugung, *, g: float, umgebung: Umgebung
) -> tuple[float, float]:
    """Das Fenster der Pruefkosten, auf dem blinde Erkundung ueberhaupt lohnen kann.

    Drei Bedingungen muessen zugleich gelten, und sie schneiden ein Intervall aus:

    * `k >= R * g_0`, sonst wird ohnehin geprueft und es gibt keinen Ruhezustand,
    * `k >= (1 - g_0) * L`, sonst ist die Pruefung der billigere Informationskanal,
    * `k < g * R`, sonst ist das Lernen nichts wert.

    Also `max(R*g_0, (1 - g_0)*L) <= k < g*R`. **Das Fenster ist leer, sobald
    `(1 - g_0)*L >= g*R`**, also sobald der Schaden einer Fehlbehandlung nicht
    klein ist gegen den entgangenen Ertrag einer zu Unrecht gestoppten Behandlung.
    Rueckgabe `(untere Grenze, obere Grenze)`; ist die untere nicht kleiner als die
    obere, so ist das Fenster leer.
    """
    g_0 = vor.lage_g
    unten = max(umgebung.R * g_0, (1.0 - g_0) * umgebung.L)
    oben = g * umgebung.R
    return unten, oben


def eps_stern(
    *,
    umgebung: Umgebung,
    vor: Ueberzeugung,
    zeta: float,
    k: float,
    T_H: int,
    delta: float,
    rng: np.random.Generator,
    ziehungen: int = 200,
    gitter: int = 21,
) -> tuple[float, float]:
    """`eps*` als Gittersuche, bei **myopischer** Pruefregel.

    Das ist das Programm, das Konzept 2.5 stellt: der beste Erkundungsanteil,
    waehrend die Pruefung der myopischen Schwellenregel folgt. Rueckgabe
    `(eps*, Wert)`.

    **Die Zahl ist mit Vorbehalt zu lesen.** Sie beantwortet die Frage des
    Konzepts, und die Frage ist nach `notiz-m2-lernen-formalisiert.md` Abschnitt 8
    nur auf dem Bereich `k >= (1 - g_0)*L` die richtige. Sonst ist die Pruefung der
    billigere Kanal, und `politik_stern` findet ihn.
    """
    beste = (0.0, -math.inf)
    for i in range(gitter):
        eps = i / (gitter - 1)
        wert = pfad_ertrag_erwartet(
            umgebung=umgebung, vor=vor, politik=Politik(eps=eps), zeta=zeta, k=k,
            T_H=T_H, delta=delta, ziehungen=ziehungen,
            rng=np.random.default_rng(rng.bit_generator.state["state"]["state"]),
        )
        if wert > beste[1]:
            beste = (eps, wert)
    return beste


def politik_stern(
    *,
    umgebung: Umgebung,
    vor: Ueberzeugung,
    zeta: float,
    k: float,
    T_H: int,
    delta: float,
    rng: np.random.Generator,
    ziehungen: int = 200,
    gitter: int = 21,
    vorlaeufe: tuple[int, ...] | None = None,
) -> tuple[Politik, float]:
    """Die beste Politik ueber **beide** Instrumente.

    Gesucht wird ueber den Erkundungsanteil und ueber die Pruefung gegen das
    myopische Urteil. Das Ergebnis dominiert `eps_stern` schon deshalb, weil
    `eps_stern` eine Einschraenkung derselben Suche ist.

    **Jeder Gitterpunkt laeuft auf demselben Zufallsstrom**, damit die Werte
    vergleichbar sind und der Vergleich nicht die Ziehung misst.
    """
    zustand = rng.bit_generator.state["state"]["state"]
    if vorlaeufe is None:
        vorlaeufe = tuple(sorted({0, 1, 2, 3, 5, 8, 12, 20, T_H}))
    beste: tuple[Politik, float] = (Politik(), -math.inf)
    for vorlauf in vorlaeufe:
        for i in range(gitter):
            politik = Politik(eps=i / (gitter - 1), pruef_vorlauf=vorlauf)
            wert = pfad_ertrag_erwartet(
                umgebung=umgebung, vor=vor, politik=politik, zeta=zeta, k=k,
                T_H=T_H, delta=delta, ziehungen=ziehungen,
                rng=np.random.default_rng(zustand),
            )
            if wert > beste[1]:
                beste = (politik, wert)
    return beste
