"""War die Maschine waehrend eines Messlaufs ruhig?

Der dense-Sweep aus Prompt A ist nicht deshalb unbrauchbar, weil er falsch
gerechnet haette, sondern weil niemand sagen konnte, was waehrenddessen sonst
auf der GPU los war. Ein Livestream im Hintergrund verschiebt jede Zahl, und
hinterher steht nur noch eine Tabelle da, der man nichts ansieht.

Dieses Werkzeug liest den Zeitraum eines Laufs aus `samples` und beantwortet
genau diese Frage aus Messdaten statt aus Erinnerung:

    python tools/sweep_umfeld.py --von 1787049278 --bis 1787052000
    python tools/sweep_umfeld.py --letzte-minuten 60

Ausgabe ist bewusst knapp: Fremd-VRAM, Temperatur, Takt, RAM, Auslagerung -
und ein Urteil.

Wichtig, und beim ersten Entwurf falsch gemacht: **die Spanne des Fremd-VRAM
taugt in einem Sweep-Fenster nicht als Kriterium.** `vram_fremd_mib` ist laut
`kibench/gpu.py` keine Direktmessung, sondern eine Zuordnung - NVML-Gesamtsumme
mal PDH-Anteil. Belegt `llama-server` 13 GB, waechst die Gesamtsumme um das
Achtfache, und die Desktop-Prozesse bekommen rechnerisch mehr ab, ohne ein Byte
zusaetzlich angefordert zu haben. Gemessen am Sweep vom 18.08.: dwm.exe steht bei
918 MiB im Leerlauf, bei 1602 MiB waehrend llama laedt und bei 322 MiB direkt
nach dem Entladen. Eine Min/Max-Spanne ueber so ein Fenster schlaegt immer Alarm.

Gefragt ist "hat jemand anderes die Karte mitbenutzt", und das beantworten zwei
Groessen, die von der Umverteilung nicht betroffen sind:

  * ist ein **neuer** Prozess mit nennenswertem VRAM dazugekommen?
  * hat die GPU **gerechnet, waehrend kein Modell geladen war**? Genau dann war
    es jemand anderes.

Der Fremd-VRAM wird weiter ausgegeben, aber als Zahl zum Ansehen, nicht als
Urteil. Nur wenn im ganzen Fenster nie ein Modell geladen war, ist seine Spanne
aussagekraeftig - dann gilt wieder die Schwelle des Dashboards.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kibench.db import verbindung  # noqa: E402

# Schwellen wie im Dashboard, damit Zahl und Anzeige dasselbe bedeuten.
FREMD_VRAM_SCHWANKUNG_MIB = 200
# Ab hier gilt ein waehrend des Laufs neu erschienener Prozess als Mitbenutzer.
NEUER_PROZESS_MIB = 200
# GPU-Last in Phasen ohne geladenes Modell. Kurze Spitzen entstehen beim Entladen
# und beim Aufbau des Desktops; erst Dauerlast ist ein fremder Rechenjob.
FREMDLAST_UTIL_PCT = 20
FREMDLAST_ANTEIL = 0.10
TEMP_GRENZE_C = 83
RAM_GRENZE_GIB = 56


def spanne(werte: list[float]) -> tuple[float, float, float]:
    return min(werte), max(werte), sum(werte) / len(werte)


def zeile(name: str, werte: list[float], einheit: str, nachkomma: int = 1) -> str:
    if not werte:
        return f"{name:<22} keine Daten"
    lo, hi, mit = spanne(werte)
    return (f"{name:<22} {mit:.{nachkomma}f} {einheit}"
            f"   (min {lo:.{nachkomma}f} / max {hi:.{nachkomma}f})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--von", type=float, help="Unix-Zeit Start")
    ap.add_argument("--bis", type=float, help="Unix-Zeit Ende")
    ap.add_argument("--letzte-minuten", type=float, help="statt --von/--bis")
    args = ap.parse_args()

    if args.letzte_minuten:
        bis = time.time()
        von = bis - args.letzte_minuten * 60
    else:
        if args.von is None:
            print("Bitte --von/--bis oder --letzte-minuten angeben.", file=sys.stderr)
            return 1
        von, bis = args.von, args.bis or time.time()

    conn = verbindung()
    rows = conn.execute(
        "SELECT * FROM samples WHERE ts BETWEEN ? AND ? ORDER BY ts", (von, bis)
    ).fetchall()

    if not rows:
        print("Keine Messpunkte in diesem Zeitraum - lief der Sampler?")
        return 2

    dauer_min = (rows[-1]["ts"] - rows[0]["ts"]) / 60
    # Der Sampler misst je Sekunde. Deutlich weniger Zeilen als Sekunden heisst:
    # es fehlen Punkte, und dann ist jede Aussage ueber "ruhig" nur halb belegt.
    erwartet = int(rows[-1]["ts"] - rows[0]["ts"]) + 1
    luecken = max(0, erwartet - len(rows))

    def spalte(name: str) -> list[float]:
        return [r[name] for r in rows if r[name] is not None]

    fremd = spalte("vram_fremd_mib")
    temp = spalte("gpu_temp_c")
    ram = spalte("ram_used_gib")
    page = spalte("pagefile_used_gib")

    print(f"\n=== Umfeld des Laufs ===")
    print(f"Zeitraum               {time.strftime('%H:%M:%S', time.localtime(rows[0]['ts']))}"
          f" - {time.strftime('%H:%M:%S', time.localtime(rows[-1]['ts']))}"
          f"  ({dauer_min:.1f} min, {len(rows)} Messpunkte)")
    if luecken:
        print(f"Fehlende Messpunkte    {luecken} von {erwartet} erwartet")
    print()
    print(zeile("Fremd-VRAM", fremd, "MiB", 0))
    print(zeile("VRAM gesamt", spalte("vram_used_mib"), "MiB", 0))
    print(zeile("GPU-Temperatur", temp, "C", 0))
    print(zeile("GPU-Takt SM", spalte("gpu_clock_sm_mhz"), "MHz", 0))
    print(zeile("GPU-Auslastung", spalte("gpu_util_pct"), "%", 0))
    print(zeile("Leistungsaufnahme", spalte("gpu_power_w"), "W", 1))
    print(zeile("RAM belegt", ram, "GiB", 1))
    print(zeile("Auslagerungsdatei", page, "GiB", 1))
    print(zeile("CPU", spalte("cpu_pct"), "%", 1))

    # Wer ausser den Runtimes lag im Grafikspeicher? Namen sagen mehr als eine Zahl.
    # Zusaetzlich festhalten, wer schon zu Beginn da war - nur Neuzugaenge sind
    # ein Hinweis auf Mitbenutzung.
    def fremde(row) -> dict[str, int]:
        try:
            return {str(p.get("name", "?")): int(p.get("mib", 0))
                    for p in json.loads(row["vram_prozesse_json"] or "[]")
                    if not any(k in str(p.get("name", "")).lower()
                               for k in ("llama-server", "ollama"))}
        except Exception:
            return {}

    fremde_namen: dict[str, int] = {}
    for r in rows:
        for name, mib in fremde(r).items():
            fremde_namen[name] = max(fremde_namen.get(name, 0), mib)
    zu_beginn = set(fremde(rows[0]))
    neue = {n: m for n, m in fremde_namen.items()
            if n not in zu_beginn and m >= NEUER_PROZESS_MIB}

    if fremde_namen:
        print("\nFremde Prozesse im VRAM (Spitzenwert der Zuordnung):")
        for name, mib in sorted(fremde_namen.items(), key=lambda x: -x[1])[:8]:
            marke = "  <- neu" if name in neue else ""
            print(f"  {mib:>6} MiB  {name}{marke}")

    # Lief ein Modell? Danach richtet sich, welche Kriterien ueberhaupt gelten.
    llama_max = max((r["vram_llama_mib"] or 0) for r in rows)
    leer = [r for r in rows if (r["vram_llama_mib"] or 0) < 100]
    util_leer = [r["gpu_util_pct"] for r in leer if r["gpu_util_pct"] is not None]
    fremdlast = [u for u in util_leer if u >= FREMDLAST_UTIL_PCT]

    if llama_max > 1000:
        print(f"\nModell geladen: bis {llama_max} MiB. Der Fremd-VRAM oben ist deshalb"
              f" eine Zuordnung,\nkeine Messung - siehe Modulkopf. Gewertet wird"
              f" stattdessen die GPU-Last in den {len(leer)} Punkten ohne geladenes Modell.")
        if util_leer:
            print(f"  GPU-Last ohne Modell: mittel {sum(util_leer)/len(util_leer):.1f} %,"
                  f" max {max(util_leer)} %,"
                  f" ueber {FREMDLAST_UTIL_PCT} % in {len(fremdlast)} von {len(util_leer)} Punkten")


    print("\n=== Urteil ===")
    urteile: list[str] = []

    if neue:
        for name, mib in sorted(neue.items(), key=lambda x: -x[1]):
            urteile.append(f"{name} kam waehrend des Laufs dazu und belegte bis {mib} MiB")

    if llama_max > 1000:
        # Ein Modell war geladen: der Fremd-VRAM ist verzerrt, gewertet wird die
        # Rechenlast in den Pausen dazwischen.
        if util_leer and len(fremdlast) > FREMDLAST_ANTEIL * len(util_leer):
            anteil = len(fremdlast) / len(util_leer) * 100
            urteile.append(f"GPU rechnete in {anteil:.0f} % der Pausen ohne geladenes Modell "
                           f"(ueber {FREMDLAST_UTIL_PCT} %) - da war ein fremder Job")
    elif fremd:
        # Kein Modell im Fenster: jetzt ist die Spanne aussagekraeftig.
        schwankung = max(fremd) - min(fremd)
        if schwankung > FREMD_VRAM_SCHWANKUNG_MIB:
            urteile.append(f"Fremd-VRAM schwankte um {schwankung:.0f} MiB "
                           f"(Schwelle {FREMD_VRAM_SCHWANKUNG_MIB}) - jemand hat mitbenutzt")
    if temp and max(temp) >= TEMP_GRENZE_C:
        urteile.append(f"GPU erreichte {max(temp):.0f} C - ab {TEMP_GRENZE_C} C drosselt sie")
    if ram and max(ram) >= RAM_GRENZE_GIB:
        urteile.append(f"RAM bis {max(ram):.1f} GiB - ab {RAM_GRENZE_GIB} GiB wird ausgelagert")
    if page and (max(page) - min(page)) > 1.0:
        urteile.append(f"Auslagerungsdatei wuchs um {max(page)-min(page):.1f} GiB")
    if luecken > dauer_min:  # mehr als eine fehlende Sekunde je Minute
        urteile.append(f"{luecken} fehlende Messpunkte - Zeitreihe ist nicht lueckenlos")

    if urteile:
        for u in urteile:
            print(f"  ! {u}")
        print("\n-> Der Lauf lief NICHT ungestoert. Zahlen entsprechend vorsichtig lesen.")
        return 3

    # "Ruhig" heisst nicht "leer": Desktop, Browser und Explorer belegen dauerhaft
    # rund 2 GB VRAM. Entscheidend ist, dass dieser Sockel sich nicht bewegt hat -
    # ein konstanter Verbraucher verschiebt alle Messwerte gleich und faellt beim
    # Vergleich heraus, ein schwankender nicht.
    if llama_max > 1000:
        print("  Kein neuer Prozess auf der Karte, und in den Pausen zwischen den Modellen")
        print("  hat nichts gerechnet - die GPU gehoerte dem Lauf allein.")
    elif fremd:
        print(f"  Fremd-VRAM lag stabil bei rund {sum(fremd)/len(fremd):.0f} MiB "
              f"(Schwankung {max(fremd)-min(fremd):.0f} MiB, Schwelle {FREMD_VRAM_SCHWANKUNG_MIB} MiB).")
    print("  Kein Temperaturlimit, keine Auslagerung, lueckenlose Zeitreihe.")
    print("\n-> Messwerte aus diesem Zeitraum sind untereinander vergleichbar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
