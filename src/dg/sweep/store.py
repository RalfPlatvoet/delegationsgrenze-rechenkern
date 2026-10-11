"""AP S8: Ergebnisablage, Lauf-Kennung und Laufmanifest.

Setzt Abschnitt 7 des Konzepts um, soweit der reduzierte Zuschnitt ihn traegt:
die Kennung eines Laufs aus drei Teilen, das Laufmanifest, und die Ergebnisse als
Textdatei. **Nicht** umgesetzt sind der Befehl, der alle Abbildungen aus einem
sauberen Stand erzeugt, und die gepinnte Abhaengigkeitsliste; Kapitel 9.5 des
Papiers legt das offen.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .grid import CSV_KOPF, Gitter, Zeile


def _git_kurz() -> str:
    """Die kurze Kennung des Stands, oder `ohne-git`, wenn keiner vorliegt."""
    try:
        aus = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        kurz = aus.stdout.strip()
        return kurz if aus.returncode == 0 and kurz else "ohne-git"
    except (OSError, subprocess.SubprocessError):
        return "ohne-git"


def lauf_kennung(gitter: Gitter, seed: int) -> str:
    """`<parameterhash>-<git-kurz>-<seed>` nach Konzept Abschnitt 7.

    Der Parameterhash geht ueber die kanonische Form des Gitters. Umsortieren
    oder Umformatieren aendert ihn nicht, eine Wertaenderung sehr wohl.
    """
    roh = gitter.kanonisch().encode("utf-8")
    kurz = hashlib.blake2b(roh, digest_size=6).hexdigest()
    return f"{kurz}-{_git_kurz()}-{seed}"


def schreiben(zeilen: tuple[Zeile, ...], gitter: Gitter, seed: int,
              *, wurzel: Path | str = "runs") -> Path:
    """Legt Ergebnisse und Manifest unter `runs/<kennung>/` ab."""
    kennung = lauf_kennung(gitter, seed)
    ordner = Path(wurzel) / kennung
    ordner.mkdir(parents=True, exist_ok=True)

    with io.open(ordner / "ergebnis.csv", "w", encoding="utf-8", newline="") as f:
        schreiber = csv.writer(f)
        schreiber.writerow(CSV_KOPF)
        for z in zeilen:
            schreiber.writerow(z.als_csv())

    manifest = {
        "lauf_kennung": kennung,
        "gitter": {"name": gitter.name, "punkte": gitter.punkte,
                   "kanonisch": gitter.kanonisch()},
        "abgeleitet": {"g_wahr": round(gitter.g_wahr, 6),
                       "w_wahr": round(gitter.w_wahr, 6),
                       "fenster": [gitter.g_0, round(gitter.g_wahr, 6)]},
        "seed": seed,
        "zeilen": len(zeilen),
        "zeitpunkt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "rechner": platform.node(),
    }
    io.open(ordner / "manifest.json", "w", encoding="utf-8").write(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return ordner
