"""AP L7: Durchlass, Ertrag, Redundanz und bindende Stelle.

Herkunft: `konzept-simulationswerkzeug.md` Abschnitt 2.1 bis 2.3,
`kapitel-5-grundmodell.md` Abschnitt 5.2 und 5.3.

Alles hier ist Modellrechnung ohne Zufall. Die Funktionen sind rein: gleiche
Eingabe, gleiche Ausgabe, kein Modulzustand.
"""

from __future__ import annotations

import math

import numpy as np

from .types import Kette, Stelle, Umgebung


def wirksame_raten(stelle: Stelle, rho: float) -> tuple[float, float, float]:
    """Die Raten und der Zeitverbrauch einer Stelle nach ihrer Redundanz.

    Parallele, also „eine genuegt":   alpha(r) = alpha^r,  beta(r) = 1 - (1-beta)^r
    Reihe,     also „alle muessen":   alpha(r) = 1-(1-alpha)^r,  beta(r) = beta^r

    Die Parallele senkt den Fehler erster Art und hebt den zweiten, die Reihe
    umgekehrt. Der Zeitverbrauch waechst mit `tau * (1 + (r-1)*rho)`.

    Bei `r = 1` geben beide Regeln die Grundfassung zurueck, und das ist einer
    der elf Sonderfaelle.
    """
    r = stelle.r
    if stelle.regel == "parallele":
        alpha = stelle.alpha ** r
        beta = 1.0 - (1.0 - stelle.beta) ** r
    else:
        alpha = 1.0 - (1.0 - stelle.alpha) ** r
        beta = stelle.beta ** r
    tau = stelle.tau * (1.0 + (r - 1) * rho)
    return alpha, beta, tau


def raten(kette: Kette) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Die drei Vektoren der wirksamen Raten und Zeiten ueber die Kette."""
    werte = [wirksame_raten(s, kette.rho) for s in kette.stellen]
    alpha = np.array([w[0] for w in werte], dtype=float)
    beta = np.array([w[1] for w in werte], dtype=float)
    tau = np.array([w[2] for w in werte], dtype=float)
    return alpha, beta, tau


def durchlass(kette: Kette) -> tuple[float, float]:
    """Durchlasswahrscheinlichkeit der ganzen Kette, je Zustand.

    Rueckgabe `(P(weiter | theta=1), P(weiter | theta=0))`, also das Produkt der
    `1 - alpha_i` und das Produkt der `beta_i`.
    """
    alpha, beta, _ = raten(kette)
    return float(np.prod(1.0 - alpha)), float(np.prod(beta))


def erreichen(kette: Kette, umgebung: Umgebung) -> np.ndarray:
    """Die Erreichenswahrscheinlichkeiten `q_i`, mit `q_1 = 1`.

    `q_i = pi * prod_{j<i}(1-alpha_j) + (1-pi) * prod_{j<i} beta_j`
    """
    alpha, beta, _ = raten(kette)
    vorher_1 = np.concatenate(([1.0], np.cumprod(1.0 - alpha)[:-1]))
    vorher_0 = np.concatenate(([1.0], np.cumprod(beta)[:-1]))
    return umgebung.pi * vorher_1 + (1.0 - umgebung.pi) * vorher_0


def durchlaufzeit(kette: Kette, umgebung: Umgebung) -> float:
    """Die erwartete Durchlaufzeit `T = sum_i q_i * tau_i`."""
    _, _, tau = raten(kette)
    return float(np.dot(erreichen(kette, umgebung), tau))


def zeitfaktor(kette: Kette, T: float) -> float:
    """Der Faktor, mit dem die Zeitrestriktion auf den Ertrag wirkt.

    Hart: 1 wenn `T <= W`, sonst 0. Weich: `exp(-kappa * max(0, T - W))`.
    """
    if kette.zeitregel == "hart":
        return 1.0 if T <= kette.W else 0.0
    return math.exp(-kette.kappa * max(0.0, T - kette.W))


def ertrag(kette: Kette, umgebung: Umgebung) -> float:
    """Der erwartete Periodenertrag nach Abschnitt 5.2.

    `lam * [ pi * prod(1-alpha) * R - (1-pi) * prod(beta) * L - sum q_i c_i ]`
    mal dem Zeitfaktor.
    """
    p1, p0 = durchlass(kette)
    q = erreichen(kette, umgebung)
    kosten = float(np.dot(q, np.array([s.c for s in kette.stellen], dtype=float)))
    roh = umgebung.pi * p1 * umgebung.R - (1.0 - umgebung.pi) * p0 * umgebung.L - kosten
    T = durchlaufzeit(kette, umgebung)
    return umgebung.lam * roh * zeitfaktor(kette, T)


def bindende_stelle(kette: Kette, umgebung: Umgebung, alpha_min: float = 0.0) -> int:
    """Die Stelle, deren Verbesserung den groessten Ertragszuwachs bringt.

    `i* = argmax_i [ Ertrag(alpha_i -> alpha_min) - Ertrag ]`

    Bei Gleichstand gewinnt der kleinere Index, damit die Ausgabe bestimmt ist.
    """
    grund = ertrag(kette, umgebung)
    bester, bester_zuwachs = 0, -math.inf
    for i, stelle in enumerate(kette.stellen):
        if stelle.alpha <= alpha_min:
            zuwachs = 0.0
        else:
            neu = list(kette.stellen)
            neu[i] = Stelle(
                alpha=alpha_min, beta=stelle.beta, c=stelle.c, tau=stelle.tau,
                besetzung=stelle.besetzung, r=stelle.r, regel=stelle.regel,
            )
            zuwachs = ertrag(
                Kette(stellen=tuple(neu), W=kette.W, rho=kette.rho,
                      zeitregel=kette.zeitregel, kappa=kette.kappa),
                umgebung,
            ) - grund
        if zuwachs > bester_zuwachs:
            bester, bester_zuwachs = i, zuwachs
    return bester
