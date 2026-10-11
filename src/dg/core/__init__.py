"""Der Rechenkern des Working Papers „Die Delegationsgrenze".

Stand AP M4: `types`, `chain`, `analytic`, `verify`, `beliefs`. Das Modul
`objects` der Spezifikation entfaellt mit der Streichung von FF4.

* `types` Stelle, Kette, Umgebung als frozen dataclasses
* `chain` Durchlass, Ertrag, Redundanz, bindende Stelle
* `analytic` der unabhaengig abgezaehlte Vergleichsmassstab fuer `n = 2`
* `verify` Posteriors, Schwellen, reale Autoritaet, Kanalwahl, Periodenertrag
* `beliefs` Ueberzeugung unter Zensur, Aktualisierung, Dauergesetz, `eps*`

Regeln nach Abschnitt 4.2 der Spezifikation: keine Ein- und Ausgabe, kein
globaler Zufall, nur unveraenderliche Eingaben, nur Standardbibliothek und NumPy.
"""
