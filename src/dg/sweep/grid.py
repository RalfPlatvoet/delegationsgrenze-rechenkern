"""AP S8: Gitterdefinition und Lauf.

Ein Gitter ist **vorab benannt** und unveraenderlich. Es traegt seine Achsen und
die festen Groessen, und es kann sich kanonisch ausschreiben; daran haengt die
Kennung des Laufs in `store.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..core import beliefs as b
from ..core.types import Umgebung


@dataclass(frozen=True, slots=True)
class Gitter:
    """Ein benanntes Gitter ueber Pruefkostenverhaeltnis und Zensurgrad.

    `k_R` ist das Verhaeltnis `k/R` und nicht `k`: Die Ergebnisse des Papiers
    haengen an diesem Verhaeltnis und nicht an den Einzelwerten, siehe Anhang C
    Teil 1. `R` ist deshalb fest, und `k` wird daraus gerechnet.
    """

    name: str
    k_R: tuple[float, ...]
    zeta: tuple[float, ...]
    # feste Groessen, alle als Beispielwerte nach Anhang C
    pi: float = 0.20
    alpha: float = 0.35
    beta: float = 0.15
    R: float = 100.0
    L: float = 50.0
    lam: float = 100.0
    g_0: float = 0.05
    staerke: float = 20.0
    delta: float = 0.95
    T_H: int = 30
    ziehungen: int = 150
    eps_gitter: int = 6
    vorlaeufe: tuple[int, ...] = (0, 1, 2, 3, 5, 8, 13, 21)

    @property
    def punkte(self) -> int:
        return len(self.k_R) * len(self.zeta)

    @property
    def g_wahr(self) -> float:
        """Der wahre Anteil wirksamer unter den gestoppten Faellen."""
        return b.anteil_wirksam_unter_gestoppten(self.pi, self.alpha, self.beta)

    @property
    def w_wahr(self) -> float:
        """Der wahre Anteil gestoppter Faelle, fuer die Umrechnung in Faelle."""
        return b.zellen(self.pi, self.alpha, self.beta)[2]

    def kanonisch(self) -> str:
        """Die kanonische Form, Grundlage der Lauf-Kennung.

        Umsortieren oder Umformatieren der Datei aendert die Kennung nicht, eine
        Wertaenderung sehr wohl. Deshalb sortierte Feldnamen und feste Rundung.
        """
        felder = []
        for feld in sorted(self.__slots__):
            wert = getattr(self, feld)
            if isinstance(wert, tuple):
                wert = ",".join(f"{x:.6g}" for x in wert)
            elif isinstance(wert, float):
                wert = f"{wert:.6g}"
            felder.append(f"{feld}={wert}")
        return ";".join(felder)


# Das Gitter der Abbildung, vorab benannt.
#
# Die Achse `k_R` ist um das Fenster `g_0 <= k/R < g` gelegt, in dem ein
# Pruefvorlauf ueberhaupt moeglich ist: unterhalb von `g_0 = 0,05` wird ohnehin
# geprueft, ab `g = 0,0933` ist das Lernen nichts wert. Die Achse `zeta` ist nach
# oben verdichtet, weil der Vorlauf nach AP M4 am Grenzfall haengt.
GITTER_PRUEFVORLAUF = Gitter(
    name="pruefvorlauf-ueber-k_R-und-zeta",
    k_R=(0.030, 0.040, 0.050, 0.060, 0.070, 0.080, 0.090, 0.100, 0.110),
    zeta=(1.000, 0.999, 0.990, 0.950, 0.900),
)


@dataclass(frozen=True, slots=True)
class Zeile:
    """Eine Zeile des Gitterlaufs."""

    k_R: float
    zeta: float
    eps: float
    pruef_vorlauf: int
    vorlauf_faelle: float
    wert: float

    def als_csv(self) -> tuple:
        return (f"{self.k_R:.3f}", f"{self.zeta:.4f}", f"{self.eps:.3f}",
                self.pruef_vorlauf, f"{self.vorlauf_faelle:.1f}", f"{self.wert:.4f}")


CSV_KOPF = ("k_R", "zeta", "eps_stern", "pruef_vorlauf_perioden",
            "pruef_vorlauf_faelle", "pfadertrag")


def laufen(gitter: Gitter = GITTER_PRUEFVORLAUF, *, seed: int = 20261011,
           melden=None) -> tuple[Zeile, ...]:
    """Zaehlt das Gitter ab und gibt eine Zeile je Punkt.

    **Jeder Punkt laeuft auf demselben Zufallsstrom**, derselbe Seed je Punkt.
    Sonst misst der Vergleich zwischen zwei Punkten die Ziehung und nicht den
    Parameter. Das ist dieselbe Regel, die `politik_stern` innerhalb eines
    Punktes befolgt.
    """
    umgebung_fest = Umgebung(pi=gitter.pi, R=gitter.R, L=gitter.L, lam=gitter.lam)
    vor = b.gleichverteilte_zellen(gitter.g_0 * gitter.staerke,
                                   (1.0 - gitter.g_0) * gitter.staerke)
    zeilen: list[Zeile] = []
    for k_R in gitter.k_R:
        for zeta in gitter.zeta:
            umgebung = replace(umgebung_fest)
            politik, wert = b.politik_stern(
                umgebung=umgebung,
                vor=vor,
                zeta=zeta,
                k=k_R * gitter.R,
                T_H=gitter.T_H,
                delta=gitter.delta,
                rng=np.random.default_rng(seed),
                ziehungen=gitter.ziehungen,
                gitter=gitter.eps_gitter,
                vorlaeufe=gitter.vorlaeufe,
            )
            faelle = politik.pruef_vorlauf * gitter.lam * gitter.w_wahr
            zeile = Zeile(k_R=k_R, zeta=zeta, eps=politik.eps,
                          pruef_vorlauf=politik.pruef_vorlauf,
                          vorlauf_faelle=faelle, wert=wert)
            zeilen.append(zeile)
            if melden is not None:
                melden(zeile)
    return tuple(zeilen)
