"""AP M4: Die Verifikationsentscheidung, die beiden Schwellen und die reale Autoritaet.

Herkunft: `konzept-simulationswerkzeug.md` Abschnitt 2.4,
`kapitel-5-grundmodell.md` Abschnitt 5.4, `notiz-l6-verifikationsschwelle.md`,
`notiz-m2-lernen-formalisiert.md` Abschnitte 7 und 8.

**Zwei Wege zu denselben Nachverteilungen, und das ist Absicht.** `posteriors`
rechnet sie aus den vermuteten Modellgroessen `(pi_hut, alpha_hut, beta_hut)`, wie
Abschnitt 2.4 es hinschreibt. `posteriors_aus_zellen` rechnet sie aus den
Beobachtungskoordinaten `(u, v, g)`, wo `mu_w = u/(u+v)` und `mu_s = g` gilt. Die
beiden Wege muessen uebereinstimmen, und der Test darauf ist keine Tautologie,
weil der zweite Weg die Abbildung des ersten nicht benutzt. Dieselbe Regel wie
bei `analytic.py`.

**Die tragende Ablesung des Moduls.** `mu_w` haengt allein von `u` und `v` ab,
also von zwei **identifizierten** Groessen; `mu_s` haengt allein von `g` ab, der
einzigen **nicht identifizierten**. Also korrigiert sich `k*_w` selbst und `k*_s`
nie. Und die Pruefung einer Freigabe hat keinen Informationswert, denn fuer einen
weitergereichten Fall wird der Zustand ohnehin sichtbar: **Der einzige
Informationskanal der Organisation ist die Pruefung einer Ablehnung.**

Regeln nach Abschnitt 4.2 der Spezifikation: keine Ein- und Ausgabe, kein
globaler Zufall, nur unveraenderliche Eingaben, nur Standardbibliothek und NumPy.
"""

from __future__ import annotations

from typing import Literal

from .types import Umgebung

Empfehlung = Literal["weiter", "stopp"]
Autoritaet = Literal["Mensch", "Maschine", "geteilt"]


def posteriors(pi_hut: float, alpha_hut: float, beta_hut: float) -> tuple[float, float]:
    """`(mu_w, mu_s)` aus den vermuteten Modellgroessen, Fassung des Abschnitts 2.4.

    Die Groessen sind **vermutet** und nicht wahr, auch `pi_hut`: Nach A2 wird die
    Praevalenz mitgelernt. Der Abschnitt 2.4 schrieb bis zum 10.10.2026 `pi`; das
    war mit A2 unvereinbar und ist berichtigt.
    """
    for name, wert in (("pi_hut", pi_hut), ("alpha_hut", alpha_hut), ("beta_hut", beta_hut)):
        if not (0.0 <= wert <= 1.0):
            raise ValueError(f"{name} muss in [0,1] liegen, ist {wert}")
    nenner_w = pi_hut * (1.0 - alpha_hut) + (1.0 - pi_hut) * beta_hut
    nenner_s = pi_hut * alpha_hut + (1.0 - pi_hut) * (1.0 - beta_hut)
    mu_w = pi_hut * (1.0 - alpha_hut) / nenner_w if nenner_w > 0.0 else 0.0
    mu_s = pi_hut * alpha_hut / nenner_s if nenner_s > 0.0 else 0.0
    return mu_w, mu_s


def posteriors_aus_zellen(u: float, v: float, g: float) -> tuple[float, float]:
    """`(mu_w, mu_s)` aus den Beobachtungskoordinaten: `mu_w = u/(u+v)`, `mu_s = g`.

    Der zweite, unabhaengige Weg. Dass `mu_s = g` gilt, ist keine Naeherung,
    sondern die Definition von `g`; dass `mu_w` allein von `u` und `v` abhaengt,
    ist nachgerechnet (`m2_identifikation.py` Pruefpunkt P4).
    """
    if u + v <= 0.0:
        raise ValueError("ohne weitergereichte Faelle ist mu_w nicht definiert")
    return u / (u + v), g


def schwellen(mu_w: float, mu_s: float, umgebung: Umgebung) -> tuple[float, float]:
    """Die beiden Verifikationsschwellen `(k*_w, k*_s)`.

    `k*_w = (1 - mu_w) * L` ist die vermiedene Fehlbehandlung einer Freigabe,
    `k*_s = mu_s * R` der entgangene Ertrag einer Ablehnung. **Zwei Schwellen,
    nicht eine**, und sie haengen an verschiedenen Groessen: die eine am Schaden,
    die andere am Ertrag.
    """
    return (1.0 - mu_w) * umgebung.L, mu_s * umgebung.R


def schwellenverhaeltnis(k: float, umgebung: Umgebung) -> float:
    """`k/R`, die Groesse, mit der `E[g]` zu vergleichen ist.

    Ausgeschrieben und nicht abgekuerzt: `kappa` ist im Papier die Verfallsrate
    der weichen Zeitrestriktion (Parameterregister), und die Kollision ist am
    10.10.2026 so aufgeloest worden.
    """
    if umgebung.R <= 0.0:
        raise ValueError("ohne Ertrag ist das Schwellenverhaeltnis nicht definiert")
    return k / umgebung.R


def pruefen(k: float, empfehlung: Empfehlung, mu_w: float, mu_s: float,
            umgebung: Umgebung) -> bool:
    """Die Schwellenregel: pruefe genau dann, wenn `k` unter der Schwelle liegt.

    Der Uebergang ist scharf. Bei `k = 0` wird stets geprueft, bei
    `k > max(L, R)` nie, unabhaengig von der Ueberzeugung; beides sind
    Sonderfaelle aus Abschnitt 6.1.
    """
    k_w, k_s = schwellen(mu_w, mu_s, umgebung)
    if empfehlung == "weiter":
        return k < k_w
    if empfehlung == "stopp":
        return k < k_s
    raise ValueError(f"unbekannte Empfehlung: {empfehlung}")


def reale_autoritaet(k: float, mu_w: float, mu_s: float, umgebung: Umgebung) -> Autoritaet:
    """Wo die **reale** Autoritaet liegt, bei unveraenderter formaler.

    Geprueft wird beides, nur eines oder nichts. Wird nichts geprueft, so liegt
    die reale Autoritaet bei der Maschine; wird beides geprueft, beim Menschen.
    **Der haeufige Fall ist der geteilte**, und nach `notiz-l6` bricht bei seltenem
    wirksamem Fall die Schwelle fuer Ablehnungen zuerst ein. Die Organisation gibt
    also zuerst die Nachpruefung der Ablehnungen auf, und das ist die
    Fehlerrichtung, ueber die sie nicht lernen kann.
    """
    w = pruefen(k, "weiter", mu_w, mu_s, umgebung)
    s = pruefen(k, "stopp", mu_w, mu_s, umgebung)
    if w and s:
        return "Mensch"
    if not w and not s:
        return "Maschine"
    return "geteilt"


def pruefung_hat_informationswert(empfehlung: Empfehlung) -> bool:
    """Nur die Pruefung einer **Ablehnung** erzeugt Daten ueber `g`.

    Fuer einen weitergereichten Fall wird der Zustand ohnehin sichtbar; die
    Pruefung einer Freigabe hat allein den Wert der vermiedenen Fehlbehandlung.
    """
    return empfehlung == "stopp"


def billigerer_informationskanal(k: float, mu_s: float, umgebung: Umgebung) -> str:
    """Pruefung oder blinde Erkundung? Beide decken denselben Zustand auf.

    Pruefen und danach richtig handeln gibt `mu_s*R - k`, blind behandeln gibt
    `mu_s*R - (1 - mu_s)*L`. **Die Pruefung ist der billigere Weg genau dann, wenn
    `k < (1 - mu_s)*L`.** Daraus folgt, dass blinde Erkundung auf dem groessten
    Teil des Parameterraums nie optimal ist, und zwar nicht wegen ihrer Kosten,
    sondern weil es einen billigeren Kanal gibt.
    """
    return "Pruefung" if k < (1.0 - mu_s) * umgebung.L else "blinde Erkundung"


def periodenertrag(
    *,
    u: float,
    v: float,
    g: float,
    k: float,
    eps: float,
    umgebung: Umgebung,
    mu_w_hut: float | None = None,
    mu_s_hut: float | None = None,
    pruefe_stopp_ohnehin: bool = False,
) -> float:
    """Erwarteter Ertrag der Uebergabestelle je Periode.

    **Verhalten und Ertrag sind getrennt**, und das ist der Kern dieser Funktion.
    Die uebergebenen `(u, v, g)` sind die **wahren** Zellwahrscheinlichkeiten und
    bestimmen den Ertrag. `mu_w_hut` und `mu_s_hut` sind die **vermuteten**
    Nachverteilungen und bestimmen das Verhalten; bleiben sie leer, so wird mit
    richtiger Ueberzeugung gerechnet.

    `pruefe_stopp_ohnehin` hebt die myopische Regel fuer Ablehnungen auf. Das ist
    der zweite Informationskanal: eine Pruefung gegen das eigene myopische Urteil,
    die Information kauft. Nach `notiz-m3-begriff-und-dauer.md` Abschnitt 6 ist
    sie der Kanal, den das Konzept 2.5 nicht gefuehrt hat.

    Drei Zweige:

    * weiter: ungeprueft `u*R - v*L`, geprueft `u*R - (u+v)*k`
    * stopp, geprueft: `w*(g*R - k)`
    * stopp, ungeprueft: ein Anteil `eps` wird blind behandelt,
      `w*eps*(g*R - (1-g)*L)`, der Rest bringt nichts
    """
    w = 1.0 - u - v
    wahr_w, wahr_s = posteriors_aus_zellen(u, v, g)
    v_w = wahr_w if mu_w_hut is None else mu_w_hut
    v_s = wahr_s if mu_s_hut is None else mu_s_hut
    je_fall = 0.0

    if pruefen(k, "weiter", v_w, v_s, umgebung):
        je_fall += u * umgebung.R - (u + v) * k
    else:
        je_fall += u * umgebung.R - v * umgebung.L

    if pruefe_stopp_ohnehin or pruefen(k, "stopp", v_w, v_s, umgebung):
        je_fall += w * (g * umgebung.R - k)
    else:
        je_fall += w * eps * (g * umgebung.R - (1.0 - g) * umgebung.L)

    return umgebung.lam * je_fall
