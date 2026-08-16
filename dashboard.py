"""Startet Sampler und Live-Ansicht und oeffnet den Browser.

    python dashboard.py              # Server starten, Browser oeffnen
    python dashboard.py --kein-browser
    python dashboard.py --port 8421
"""
from __future__ import annotations

import argparse
import threading
import time
import webbrowser

import uvicorn

from kibench.config import HOST, PORT


def main() -> None:
    p = argparse.ArgumentParser(description="ki-benchmarks Live-Dashboard")
    p.add_argument("--host", default=HOST)
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--kein-browser", action="store_true")
    args = p.parse_args()

    url = f"http://{args.host}:{args.port}/"
    if not args.kein_browser:
        # Erst oeffnen, wenn uvicorn wirklich lauscht - sonst zeigt der Browser
        # eine Fehlerseite und muss von Hand neu geladen werden.
        def oeffnen() -> None:
            time.sleep(1.5)
            webbrowser.open(url)

        threading.Thread(target=oeffnen, daemon=True).start()

    print(f"ki-benchmarks Live-Dashboard  ->  {url}")
    print("Beenden mit Strg+C.")
    uvicorn.run("kibench.api:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
