"""Startpunkt **innerhalb** der Sandbox: Sperre setzen, dann die Tests fahren.

Wird als eigener Prozess gestartet:

    python -m kibench.sandbox_start <arbeitsverzeichnis> <testpfad>

Warum ein eigener Prozess und nicht einfach pytest im Runner: Die Sperre aus
`wachhund` laesst sich nach der Installation nicht mehr entfernen - das ist ihr
Sinn. Im Runner-Prozess installiert wuerde sie auch den Runner selbst treffen,
der ja legitim nach `kibench.db` schreiben muss. Ein Unterprozess je Aufgabe
loest das und bringt zwei weitere Dinge mit: einen harten Timeout und einen
sauberen Zustand, falls generierter Code den Interpreter zerlegt.

Die letzte Zeile der Ausgabe ist eine JSON-Zeile mit dem Ergebnis. Alles davor
ist pytest-Ausgabe und wird als Protokoll mitgeschrieben.
"""
from __future__ import annotations

import json
import os
import sys

MARKE = "@@ERGEBNIS@@"


def main() -> int:
    if len(sys.argv) < 3:
        print("Aufruf: python -m kibench.sandbox_start <verzeichnis> <testpfad>",
              file=sys.stderr)
        return 2

    verzeichnis, testpfad = sys.argv[1], sys.argv[2]

    # pytest importieren, **bevor** die Sperre steht. Der Import liest nur, aber
    # er beruehrt hunderte Dateien; jede davon durch den Hook zu schicken kostet
    # unnoetig Zeit, und ein Fehlalarm dabei waere besonders aergerlich, weil er
    # wie ein Testfehler aussaehe.
    import pytest

    from kibench import wachhund
    from kibench.wachhund import SandboxVerstoss, aktivieren

    os.chdir(verzeichnis)
    # Der Loesungscode liegt in der Wurzel der Sandbox, die Tests darunter in
    # `tests\`. pytest legt beim Import nur das Verzeichnis der Testdatei auf
    # den Suchpfad - ohne die naechste Zeile findet `from loesung import ...`
    # nichts, und jede Aufgabe scheiterte an einem Importfehler statt an ihrem
    # Inhalt.
    sys.path.insert(0, os.path.abspath(verzeichnis))
    aktivieren(verzeichnis)

    try:
        code = pytest.main([
            testpfad,
            "-q",
            "--no-header",
            "-p", "no:cacheprovider",   # sonst legt pytest .pytest_cache an
            "--tb=short",
        ])
        verstoss = None
    except SandboxVerstoss as e:
        # Ein Ausbruchsversuch ist ein Ergebnis, kein Absturz: er wird
        # protokolliert und die Aufgabe gilt als nicht bestanden.
        code = 99
        verstoss = str(e)

    # pytest faengt Ausnahmen aus dem Testcode ab - ein abgewehrter Ausbruch
    # kaeme oben also gar nicht an, sondern nur als roter Test. Deshalb zaehlt
    # hier die Liste aus dem Wachhund, nicht das except.
    ergebnis = {
        "exit_code": int(code),
        "bestanden": int(code) == 0 and not wachhund.verstoesse,
        "sandbox_verstoss": verstoss,
        "verstoesse": list(wachhund.verstoesse),
    }
    print()
    print(MARKE + json.dumps(ergebnis, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
