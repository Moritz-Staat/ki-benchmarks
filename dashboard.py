"""Startet Sampler und Live-Ansicht und oeffnet den Browser.

    python dashboard.py              # Server starten, Browser oeffnen
    python dashboard.py --kein-browser
    python dashboard.py --port 8421

Laeuft auch unter `pythonw.exe`, also ohne Konsolenfenster - so startet die
Desktop-Verknuepfung. Siehe `_ausgabe_umleiten()`, das ist dort keine Kosmetik,
sondern Startbedingung.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DATEI = LOG_DIR / "dashboard.log"


def _ausgabe_umleiten() -> Path | None:
    """Unter pythonw.exe gibt es keine Standardausgabe - dann in eine Datei.

    `pythonw.exe` setzt `sys.stdout` und `sys.stderr` auf `None`. uvicorn baut
    beim Start seinen Log-Formatter auf und ruft dabei `sys.stdout.isatty()`;
    das schlaegt mit `AttributeError` fehl, der Formatter kann nicht konfiguriert
    werden, und der Server startet gar nicht erst. Ohne diese Umleitung startet
    die Desktop-Verknuepfung also den Browser, aber nichts, was er anzeigen
    koennte.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return None
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    datei = open(LOG_DATEI, "a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = datei
    if sys.stderr is None:
        sys.stderr = datei
    print(f"\n=== Start {time.strftime('%Y-%m-%d %H:%M:%S')} (ohne Konsole) ===")
    return LOG_DATEI


def main() -> None:
    log = _ausgabe_umleiten()

    p = argparse.ArgumentParser(description="ki-benchmarks Live-Dashboard")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--kein-browser", action="store_true")
    args = p.parse_args()

    # Erst nach der Umleitung importieren: uvicorn konfiguriert sein Logging
    # bereits beim Aufbau der Config.
    import uvicorn

    from kibench.config import HOST, PORT

    host = args.host or HOST
    port = args.port or PORT
    url = f"http://{host}:{port}/"

    if not args.kein_browser:
        # Erst oeffnen, wenn uvicorn wirklich lauscht - sonst zeigt der Browser
        # eine Fehlerseite und muss von Hand neu geladen werden.
        def oeffnen() -> None:
            time.sleep(1.5)
            webbrowser.open(url)

        threading.Thread(target=oeffnen, daemon=True).start()

    print(f"ki-benchmarks Live-Dashboard  ->  {url}")
    if log:
        print(f"Protokoll: {log}")
    else:
        print("Beenden mit Strg+C.")
    uvicorn.run("kibench.api:app", host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
