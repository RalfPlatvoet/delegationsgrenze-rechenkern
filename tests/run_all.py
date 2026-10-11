"""Laeuft die Testmappe ohne pytest.

Grund: pytest liegt in dieser Umgebung nicht vor. Die Testdateien sind
pytest-tauglich geschrieben; dieser Laeufer ruft dieselben Funktionen auf, damit
die Mappe jetzt schon pruefbar ist.

Aufruf: python3 tests/run_all.py
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))

MAPPEN = ("test_special_cases", "test_analytic_match",
          "test_invariants", "test_determinism")


def main() -> int:
    bestanden = fehlgeschlagen = 0
    for name in MAPPEN:
        modul = importlib.import_module(name)
        print(f"\n{name}")
        for test in sorted(n for n in dir(modul) if n.startswith("test_")):
            try:
                getattr(modul, test)()
            except Exception as fehler:  # noqa: BLE001
                fehlgeschlagen += 1
                print(f"  FEHL {test}: {fehler}")
                traceback.print_exc()
            else:
                bestanden += 1
                print(f"  ok   {test}")
    print(f"\n{bestanden} bestanden, {fehlgeschlagen} fehlgeschlagen")
    return 1 if fehlgeschlagen else 0


if __name__ == "__main__":
    raise SystemExit(main())
