"""Abnahme: laeuft der Sampler ueber Stunden ohne Speicherleck und ohne Luecken?

Beobachtet den Dashboard-Prozess von aussen und schreibt eine CSV mit
Arbeitsspeicher, Handles und Threads. Ein Leck zeigt sich als monoton
steigender RSS-Wert; ein stabiler Wert ueber Stunden ist der Nachweis.

    python tools/dauerlauf_pruefen.py --minuten 120

Am Ende steht die Auswertung: RSS-Drift, Luecken in der Zeitreihe, Fehlerzahl.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import httpx
import psutil

BASIS = "http://127.0.0.1:8420"


def dashboard_prozess() -> psutil.Process | None:
    """Der Python-Prozess, der dashboard.py ausfuehrt."""
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            if p.info["name"] not in ("python.exe", "python"):
                continue
            if any("dashboard.py" in teil for teil in (p.info["cmdline"] or [])):
                return p
        except Exception:
            continue
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minuten", type=float, default=120.0)
    ap.add_argument("--takt", type=float, default=30.0, help="Sekunden zwischen zwei Beobachtungen")
    ap.add_argument("--csv", default="logs/dauerlauf-sampler.csv")
    args = ap.parse_args()

    proc = dashboard_prozess()
    if proc is None:
        print("Kein laufender dashboard.py-Prozess gefunden.", file=sys.stderr)
        return 1

    ziel = Path(args.csv)
    ziel.parent.mkdir(parents=True, exist_ok=True)
    ende = time.monotonic() + args.minuten * 60

    print(f"Beobachte PID {proc.pid} fuer {args.minuten:.0f} Minuten -> {ziel}")
    with ziel.open("w", newline="", encoding="utf-8") as f:
        s = csv.writer(f)
        s.writerow(["ts", "minute", "rss_mib", "vms_mib", "threads", "handles",
                    "messungen", "fehler", "ausgelassen", "db_zeilen", "dauer_ms"])
        start = time.monotonic()
        while time.monotonic() < ende:
            try:
                mi = proc.memory_info()
                rss, vms = mi.rss / 1024**2, mi.vms / 1024**2
                threads = proc.num_threads()
                handles = getattr(proc, "num_handles", lambda: None)()
            except psutil.NoSuchProcess:
                print("Prozess ist verschwunden - Dauerlauf abgebrochen.", file=sys.stderr)
                return 2
            try:
                st = httpx.get(f"{BASIS}/api/status", timeout=5.0).json()
            except Exception:
                st = {}
            s.writerow([
                round(time.time(), 1), round((time.monotonic() - start) / 60, 2),
                round(rss, 1), round(vms, 1), threads, handles,
                st.get("messungen"), st.get("fehler"), st.get("ausgelassen"),
                st.get("db_zeilen"), st.get("letzte_dauer_ms"),
            ])
            f.flush()
            time.sleep(args.takt)

    auswerten(ziel)
    return 0


def auswerten(ziel: Path) -> None:
    zeilen = list(csv.DictReader(ziel.open(encoding="utf-8")))
    if len(zeilen) < 2:
        print("Zu wenige Beobachtungen.")
        return
    rss = [float(z["rss_mib"]) for z in zeilen]
    erste, letzte = zeilen[0], zeilen[-1]
    dauer_min = float(letzte["minute"])

    print("\n=== Auswertung Dauerlauf ===")
    print(f"Laufzeit           {dauer_min:.1f} min ueber {len(zeilen)} Beobachtungen")
    print(f"RSS Start / Ende   {rss[0]:.1f} / {rss[-1]:.1f} MiB   (Drift {rss[-1]-rss[0]:+.1f} MiB)")
    print(f"RSS min / max      {min(rss):.1f} / {max(rss):.1f} MiB")
    print(f"Threads            {erste['threads']} -> {letzte['threads']}")
    print(f"Handles            {erste['handles']} -> {letzte['handles']}")
    print(f"Messungen          {erste['messungen']} -> {letzte['messungen']}")
    print(f"Fehler             {letzte['fehler']}")
    print(f"Ausgelassene Takte {letzte['ausgelassen']}")

    if dauer_min > 0:
        pro_stunde = (rss[-1] - rss[0]) / dauer_min * 60
        print(f"\nDrift hochgerechnet: {pro_stunde:+.1f} MiB/Stunde")
        if abs(pro_stunde) < 5:
            print("-> stabil, kein Leck")
        elif pro_stunde < 20:
            print("-> leichte Drift, ueber Nacht beobachten")
        else:
            print("-> VERDACHT AUF LECK")


if __name__ == "__main__":
    raise SystemExit(main())
