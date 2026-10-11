"""AP L7: Die Typen des Rechenkerns.

Herkunft aller Zeichen: `konzept-simulationswerkzeug.md` Abschnitt 2.1 bis 2.3,
`kapitel-5-grundmodell.md` Abschnitt 5.2 und 5.3.

Regeln fuer `core/` nach Abschnitt 4.2 der Spezifikation: keine Ein- und Ausgabe,
kein globaler Zufall, nur unveraenderliche Eingaben, nur Standardbibliothek und
NumPy.

**Die Pruefung findet im Konstruktor statt.** Ein unzulaessiger Parametersatz
entsteht gar nicht erst, statt spaeter eine sinnlose Zahl zu liefern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Besetzung = Literal["H", "M"]
Verknuepfung = Literal["reihe", "parallele"]
Zeitregel = Literal["hart", "weich"]


def _in_einheitsintervall(name: str, wert: float) -> None:
    if not (0.0 <= wert <= 1.0):
        raise ValueError(f"{name} muss in [0,1] liegen, ist {wert}")


def _nichtnegativ(name: str, wert: float) -> None:
    if wert < 0.0:
        raise ValueError(f"{name} darf nicht negativ sein, ist {wert}")


@dataclass(frozen=True, slots=True)
class Stelle:
    """Eine Stelle der Kette, nach Abschnitt 5.2.

    `alpha` ist der Fehler erster Art, also P(stopp | theta = 1).
    `beta` ist der Fehler zweiter Art, also P(weiter | theta = 0).
    `r` ist die Zahl paralleler oder gereihter Pruefungen an dieser Stelle,
    `regel` ihre Verknuepfung. Bei `r = 1` ist `regel` ohne Wirkung.
    """

    alpha: float
    beta: float
    c: float = 0.0
    tau: float = 0.0
    besetzung: Besetzung = "H"
    r: int = 1
    regel: Verknuepfung = "reihe"

    def __post_init__(self) -> None:
        _in_einheitsintervall("alpha", self.alpha)
        _in_einheitsintervall("beta", self.beta)
        _nichtnegativ("c", self.c)
        _nichtnegativ("tau", self.tau)
        if self.r < 1:
            raise ValueError(f"r muss mindestens 1 sein, ist {self.r}")
        if self.regel not in ("reihe", "parallele"):
            raise ValueError(f"unbekannte Verknuepfung: {self.regel}")
        if self.besetzung not in ("H", "M"):
            raise ValueError(f"unbekannte Besetzung: {self.besetzung}")


@dataclass(frozen=True, slots=True)
class Kette:
    """Die Kette aus `n` Stellen mit Handlungsfenster, nach Abschnitt 5.3.

    `rho` ist der Sequenzialitaetsgrad der Redundanz: 0 heisst gleichzeitige
    Pruefung ohne Zeitzuwachs, 1 heisst vollstaendig nacheinander.

    `zeitregel` waehlt zwischen der harten Fassung, in der der Ertrag bei
    Ueberschreitung des Fensters vollstaendig entfaellt, und der weichen, in der
    er mit `kappa` stetig verfaellt. Die harte Fassung ist die Grundfassung; die
    weiche dient der Pruefung, ob ein Ergebnis an der Unstetigkeit haengt.
    """

    stellen: tuple[Stelle, ...]
    W: float = float("inf")
    rho: float = 1.0
    zeitregel: Zeitregel = "hart"
    kappa: float = 0.0

    def __post_init__(self) -> None:
        if len(self.stellen) < 1:
            raise ValueError("eine Kette braucht mindestens eine Stelle")
        _nichtnegativ("W", self.W)
        _in_einheitsintervall("rho", self.rho)
        _nichtnegativ("kappa", self.kappa)
        if self.zeitregel not in ("hart", "weich"):
            raise ValueError(f"unbekannte Zeitregel: {self.zeitregel}")

    @property
    def n(self) -> int:
        return len(self.stellen)


@dataclass(frozen=True, slots=True)
class Umgebung:
    """Fall und Umgebung, nach Abschnitt 5.2.

    `pi` ist die Praevalenz. **Sie ist hier die wahre Praevalenz der Umgebung**,
    nicht die Vermutung der Organisation darueber; nach Annahme A2 des
    Annahmenkatalogs kennt die Organisation sie nicht. Die Unterscheidung zaehlt
    erst ab `beliefs.py` und ist hier als Vermerk gefuehrt, damit sie beim Bau
    jenes Moduls nicht uebersehen wird.
    """

    pi: float
    R: float
    L: float
    lam: float = 1.0

    def __post_init__(self) -> None:
        _in_einheitsintervall("pi", self.pi)
        _nichtnegativ("R", self.R)
        _nichtnegativ("L", self.L)
        _nichtnegativ("lam", self.lam)
