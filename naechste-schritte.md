# Nächste Schritte für dieses Repositorium

Angelegt am 11.10.2026 in AP S6 des Working Papers „Die Delegationsgrenze"
(`../working-paper-delegationsgrenze/veroeffentlichung.md`). Die Dateien liegen vollständig, die
Tests laufen, **committet ist nichts**. Lizenz ist am 11.10.2026 entschieden: MIT.

## 1. Erster Commit

```
cd ~/Code/Projekte/delegationsgrenze-rechenkern
git commit -m "Rechenkern zum Working Paper Die Delegationsgrenze, Fassung 0.1.0"
```

## 2. Entferntes Repositorium anlegen und schieben

Öffentlich, denn der Kern soll nachrechenbar sein.

```
gh repo create delegationsgrenze-rechenkern --public --source . --remote origin --push
```

## 3. Zenodo verbinden und Freigabe erzeugen

1. Bei Zenodo unter „GitHub" das Repositorium einschalten. **Das muss vor der Freigabe geschehen**,
   sonst greift der Haken nicht.
2. Freigabe im Repositorium erzeugen, Markierung `v0.1.0`, Titel „Fassung 0.1.0".
3. Zenodo erzeugt daraufhin den Eintrag und zwei Verweise: einen für diese Fassung und einen, der
   immer auf die jeweils letzte zeigt. **In das Papier gehört der Verweis auf diese Fassung.**

## 4. Zurück ins Papier

1. Den Verweis in `CITATION.cff` im Feld `doi` eintragen.
2. Den Verweis in Kapitel 9.5 des Papiers nachtragen, dort steht bisher nur die Ankündigung.
3. `build.py` laufen lassen, PDF neu erzeugen.
4. Erst danach der Upload nach `mpra-formular.md`. **Das ist Weg A, am 11.10.2026 entschieden.**

## Was hier nicht liegt

Die Arbeitsspuren des Papiers, also `k6_entwurf.py`, `m2_identifikation.py`, `m3_dauer.py`,
`m4_eps_stern.py`, `m5_feldregionen.py` und `s3_dsa.py`. Sie gehören zum Papier und nicht zum Kern.
**Offen ist, ob sie als zweites Archiv mitgehen**, denn jede gerechnete Zahl des Papiers nennt eines
dieser Skripte; der Punkt steht in `veroeffentlichung.md` Abschnitt 6.
