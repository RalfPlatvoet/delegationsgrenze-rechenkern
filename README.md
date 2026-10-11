# Delegationsgrenze: der Rechenkern

Rechenkern zum Working Paper **„Die Delegationsgrenze. Maschinelle Bewertung, reale Autorität und
die Architektur betrieblicher Entscheidungsketten"** (PPI Working Paper 09/2026, Ralf Platvoet).

## Was das ist

Ein kleiner, geschlossen rechnender Kern für das Modell des Papiers. **Keine Simulation, kein
Agentenmodell, keine Oberfläche.** Er rechnet die Größen der Kapitel 5 bis 8 aus, prüft sich an von
Hand nachrechenbaren Sonderfällen und zieht dort, wo eine Verteilung zu prüfen ist.

| Modul | Inhalt | Kapitel |
|---|---|---|
| `types.py` | `Stelle`, `Kette`, `Umgebung` als unveränderliche Datenklassen mit Prüfung im Konstruktor | 5 |
| `chain.py` | Raten, Durchlass, Erreichenswahrscheinlichkeiten, Zeit, Ertrag, Redundanz, bindende Stelle | 6 |
| `analytic.py` | derselbe Ertrag für zwei Stellen durch Abzählen des Entscheidungsbaums, als Gegenrechnung | 6 |
| `verify.py` | Nachverteilungen, die beiden Prüfschwellen, die Schwellenregel, reale Autorität, Kanalwahl | 7 |
| `beliefs.py` | Zählstände und Überzeugung, Koordinatenabbildungen, konjugierte Aktualisierung, Fisher-Information, Dauergesetz, Erkundungsfenster, Prüfvorlauf | 8 |

Daneben liegt `src/dg/sweep/` für die Gitterläufe: `grid.py` mit dem vorab benannten Gitter und
seinem Lauf, `store.py` mit der Kennung eines Laufs, der Ergebnisablage und dem Laufmanifest. **Der
Kern schreibt nicht, das Gitter darf es**; der Kern kennt das Gitter nicht, das Gitter ruft den
Kern. Beide arbeiten ohne globalen Zufall.

## Was er nicht ist

Er trifft **keine Aussage über Betriebe**. Jedes Ergebnis steht unter den Annahmen des Kapitels 5,
und jede dieser Annahmen trägt in Anhang A des Papiers einen Vermerk darüber, was ihre Lockerung
ändert. Wer den Kern ohne das Papier benutzt, benutzt ein Rechenwerk ohne seinen Geltungsbereich.

## Benutzung

```
python3 -m pip install -e .
python3 tests/run_all.py
```

Einzige harte Abhängigkeit ist NumPy. `pytest` und `hypothesis` sind wahlfrei; fehlen sie, zieht
die Invariantenprüfung aus einem festgelegten Zufallsstrom, was schwächer ist und im Test vermerkt
steht.

Ein kurzes Beispiel, die Kette aus zwei Stellen und ihr Ertrag:

```python
from dg.core.types import Kette, Stelle, Umgebung
from dg.core.chain import ertrag, bindende_stelle

kette = Kette(
    stellen=(
        Stelle(alpha=0.08, beta=0.22, c=6.0, tau=4.0, besetzung="H"),
        Stelle(alpha=0.28, beta=0.07, c=3.0, tau=2.5, besetzung="M", r=2, regel="parallele"),
    ),
    W=6.0, rho=1.0,
)
umgebung = Umgebung(pi=0.20, R=400.0, L=60.0, lam=100.0)
print(ertrag(kette, umgebung), bindende_stelle(kette, umgebung))
```

## Vier Regeln für `core/`, und sie sind geprüft und nicht nur erklärt

- Keine Ein- und Ausgabe.
- Kein globaler Zufall. Jede Funktion, die Zufall braucht, nimmt einen Zufallsstrom als Argument.
- Nur unveränderliche Eingaben, damit ein Lauf nicht durch einen früheren verändert wird.
- Nur die Standardbibliothek und NumPy. Ein Test stellt fest, dass beim Laden der Kernmodule keine
  weitere Bibliothek in die geladenen Module gerät.

Die Prüfung findet im Konstruktor statt: Ein unzulässiger Parametersatz entsteht gar nicht erst.

## Prüfung

**39 Tests, alle bestanden**, in vier Ebenen: Sonderfälle mit von Hand nachrechenbarer Erwartung
(zehn von elf abgedeckt, der elfte entfällt mit der gestrichenen vierten Forschungsfrage), die
unabhängige Zweitrechnung über das Abzählen des Entscheidungsbaums (über 5.000 gezogene
Parametersätze auf `1e-10`), Invarianten über zufällig gezogene Parametersätze, und Bestimmtheit
bis zur Bitgleichheit.

**Drei Erwartungen der eigenen Spezifikation sind an diesen Tests gefallen**, und sie stehen im
Papier in Kapitel 9.4: die Monotonie des Ertrags in der Rate erster Art, die Erwartung zum Lernen
ohne Erkundung und die Form der Überzeugung. Die Gegenbeispiele sind bestehende Tests und stehen
nicht in einer Ausnahmeliste.

## Was hier nicht liegt

Die Arbeitsspuren des Papiers, also die Skripte, die die Zahlen der Kapitel 6 und 8 erzeugt haben,
sind **nicht** Teil des Kerns. Sie liegen beim Papier, und jede gerechnete Zahl des Papiers nennt
das Skript, aus dem sie stammt. Ebenso fehlen hier ein Lader für Parametersätze, eine gepinnte
Abhängigkeitsliste und ein Abbildungsmanifest; Kapitel 9.5 des Papiers legt das offen.

## Zitieren

Siehe `CITATION.cff`. Zu zitieren ist das Papier, und der Kern zusätzlich, wenn eine Rechnung
nachvollzogen wird.

## Lizenz

**MIT**, siehe `LICENSE`. Das Papier selbst steht unter CC BY 4.0; für den Code ist eine
CC-Lizenz die falsche Gattung, deshalb die Trennung.
