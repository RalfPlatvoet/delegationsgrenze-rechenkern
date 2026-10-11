"""AP S8: Die Gitterlaeufe.

Herkunft: Beschluss M1 vom 11.10.2026. Die vollstaendige Durchmusterung ist mit
dem Zuschnitt vom 09.10.2026 gestrichen; zurueckgeholt sind allein die
Gitterlaeufe fuer **eine** Abbildung, naemlich den Pruefvorlauf ueber Pruefkosten
und Zensurgrad. Der Inhalt ist aus der Arbeitsspur `m4_eps_stern.py` gehoben.

**Was hier nicht geschieht, und das ist Sperre 3 aus `abgrenzung.md` Abschnitt 8.**
Es wird **nicht nach Kippbereichen gesucht**. Das Gitter ist vorab benannt, seine
Achsen stehen in `grid.GITTER_PRUEFVORLAUF`, und es wird abgezaehlt und nicht
verfeinert. Wo das Ergebnis springt, wird der Sprung berichtet und nicht
nachgefahren.

**Abgrenzung gegen `core/`.** Dieses Paket liegt ausserhalb des Kerns und darf
schreiben; der Kern darf es nicht. Es benutzt ausser dem Kern nur die
Standardbibliothek und NumPy, also **kein** pandas, anders als im Konzept
Abschnitt 4.3 vorgesehen: Fuer ein Gitter dieser Groesse waere das eine
Abhaengigkeit ohne Gegenwert.
"""

from .grid import GITTER_PRUEFVORLAUF, Gitter, Zeile, laufen
from .store import lauf_kennung, schreiben

__all__ = ["GITTER_PRUEFVORLAUF", "Gitter", "Zeile", "laufen", "lauf_kennung", "schreiben"]
