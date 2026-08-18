"""Haelt der gewaehlte Offload-Wert auch einen echten Arbeitskontext aus?

Die Luecke, die Prompt C ausdruecklich schliessen will: **Alle bisherigen
Messungen liefen mit einem Prompt von 1209 Token.** Der Betriebspunkt eines
Coding-Agenten liegt bei 10.000 bis 30.000. Ob `-ngl 58` das ueberlebt, ist
ungeprueft - und es gibt einen konkreten Verdacht, warum es schiefgehen koennte:
Der Absturz beim Wiederholungs-Sweep kam nicht beim Laden, sondern erst beim
Rechenpuffer fuer den langen Prompt. Ein groesserer Prompt heisst groesserer
Puffer.

Gemessen wird je Konfiguration und Prompt-Laenge:

* **Zeit bis zum ersten Token** - dafuer wird gestreamt, anders ist sie nicht
  zu bekommen
* Wanduhrzeit gesamt, Rechenzeit laut Server, Differenz (das Nachladen)
* VRAM vor und nach der Anfrage
* ob der Server ueberhaupt geantwortet hat

    python tools/langkontext.py                       alle llama.cpp-Konfigurationen
    python tools/langkontext.py --modelle qwen-dense-iq4
    python tools/langkontext.py --laengen 1000 8000 28000

**64k steht nicht zur Auswahl**, obwohl Prompt C es nennt: Die Server laufen mit
`-c 32768`. Ein 64k-Prompt wuerde nicht knapp scheitern, sondern gar nicht erst
angenommen. Dafuer muesste der Kontext hochgesetzt werden, und das verschiebt
den ganzen Offload-Punkt - eine eigene Messreihe, kein Nebenher.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from kibench import serverwechsel  # noqa: E402
from kibench.config import LLAMA_BASE, MODELLE, PROJEKT_DIR  # noqa: E402

# Grob gemessen an den Sweep-Prompts: der Fliesstext dort ergab 1209 Token bei
# 4886 Zeichen, also rund 4 Zeichen je Token. Fuer eine Laengensteuerung reicht
# das; die tatsaechliche Zahl kommt hinterher vom Server.
ZEICHEN_JE_TOKEN = 4.0

ABSATZ = (
    "Die Bandbreite zwischen Grafikspeicher und Arbeitsspeicher bestimmt bei lokaler "
    "Sprachmodell-Inferenz fast alles. Ein Modell, dessen Gewichte vollstaendig im "
    "Grafikspeicher liegen, liest diese mit mehreren hundert Gigabyte pro Sekunde. "
    "Muessen Teile der Gewichte aus dem Arbeitsspeicher nachgeladen werden, faellt die "
    "effektive Bandbreite um etwa eine Groessenordnung. Da bei der Generierung jedes "
    "einzelnen Tokens saemtliche aktiven Gewichte einmal gelesen werden muessen, wirkt "
    "sich das unmittelbar auf die Tokens pro Sekunde aus. "
)


def prompt_bauen(ziel_tokens: int) -> str:
    zeichen = int(ziel_tokens * ZEICHEN_JE_TOKEN)
    text = (ABSATZ * (zeichen // len(ABSATZ) + 1))[:zeichen]
    return ("Lies den folgenden Text und nenne in genau einem Satz sein Thema.\n\n"
            + text + "\n\nAntwort:")


def messen(alias: str, prompt: str, n_predict: int = 64,
           timeout_s: float = 900.0) -> dict:
    """Eine gestreamte Anfrage. Der Rueckgabewert unterscheidet Fehler von Werten."""
    koerper = {
        "model": alias,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n_predict,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        # Denken wuerde die Antwort um Tausende Token verlaengern und die Frage
        # verwaessern - hier geht es um die Verarbeitung des Prompts.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    ergebnis: dict = {"ok": False}
    t0 = time.perf_counter()
    erstes_token: float | None = None
    stuecke = 0
    usage = None
    timings = None

    try:
        with httpx.stream("POST", f"{LLAMA_BASE}/v1/chat/completions",
                          json=koerper, timeout=timeout_s) as r:
            if r.status_code != 200:
                ergebnis["fehler"] = f"http_{r.status_code}"
                return ergebnis
            for zeile in r.iter_lines():
                if not zeile or not zeile.startswith("data: "):
                    continue
                nutz = zeile[6:]
                if nutz.strip() == "[DONE]":
                    break
                try:
                    d = json.loads(nutz)
                except json.JSONDecodeError:
                    continue
                delta = ((d.get("choices") or [{}])[0].get("delta") or {})
                if delta.get("content"):
                    stuecke += 1
                    if erstes_token is None:
                        erstes_token = time.perf_counter() - t0
                if d.get("usage"):
                    usage = d["usage"]
                if d.get("timings"):
                    timings = d["timings"]
    except Exception as e:
        ergebnis["fehler"] = f"{type(e).__name__}: {e}"
        ergebnis["wanduhr_s"] = round(time.perf_counter() - t0, 2)
        return ergebnis

    wanduhr = time.perf_counter() - t0
    rechen = None
    if timings:
        rechen = (float(timings.get("prompt_ms", 0)) + float(timings.get("predicted_ms", 0))) / 1000

    ergebnis.update({
        "ok": stuecke > 0,
        "wanduhr_s": round(wanduhr, 2),
        "ttft_s": round(erstes_token, 2) if erstes_token is not None else None,
        "stuecke": stuecke,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "antwort_tokens": (usage or {}).get("completion_tokens"),
        "rechenzeit_s": round(rechen, 2) if rechen is not None else None,
        "nachladen_s": round(max(0.0, wanduhr - rechen), 2) if rechen is not None else None,
        "prompt_tps": round(float(timings["prompt_per_second"]), 1)
                      if timings and timings.get("prompt_per_second") else None,
        "gen_tps": round(float(timings["predicted_per_second"]), 1)
                   if timings and timings.get("predicted_per_second") else None,
    })
    if not ergebnis["ok"]:
        ergebnis["fehler"] = "keine_token"
    return ergebnis


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelle", nargs="*", default=None,
                    help="Aliase; ohne Angabe alle llama.cpp-Konfigurationen")
    ap.add_argument("--laengen", nargs="*", type=int, default=[1000, 8000, 28000])
    ap.add_argument("--csv", default=None)
    ap.add_argument("--kein-wechsel", action="store_true",
                    help="gegen den bereits laufenden Server messen, nichts umschalten")
    ap.add_argument("--marke", default=None,
                    help="Name fuer die CSV-Spalte, wenn der Server von Hand gestartet wurde")
    args = ap.parse_args()

    aliase = args.modelle or [m["alias"] for m in MODELLE if m["runtime"] == "llama.cpp"]
    ziel = Path(args.csv) if args.csv else (
        PROJEKT_DIR / "logs" / f"langkontext-{time.strftime('%Y%m%d-%H%M%S')}.csv")
    ziel.parent.mkdir(parents=True, exist_ok=True)

    try:
        from kibench.gpu import GpuMonitor
        gpu = GpuMonitor()
    except Exception:
        gpu = None

    def vram() -> int | None:
        try:
            return gpu.messen().vram_used_mib if gpu else None
        except Exception:
            return None

    zeilen: list[dict] = []
    print(f"Langkontext-Test  |  Laengen: {args.laengen}  |  CSV: {ziel}\n")

    if args.kein_wechsel:
        # Fuer Kandidatenwerte, die es als Konfiguration noch gar nicht gibt:
        # Server von Hand starten, hier nur messen. Der Alias kommt dann vom
        # Server selbst, damit in der CSV nicht der falsche Name steht.
        laeuft = serverwechsel.laufender_alias()
        if laeuft is None:
            print("Kein llama-server auf 8080 - mit --kein-wechsel muss einer laufen.",
                  file=sys.stderr)
            return 1
        aliase = [args.marke or laeuft]
        vermerk = f" (als {args.marke} protokolliert)" if args.marke else ""
        print(f"Messe gegen den laufenden Server: {laeuft}{vermerk}")
        print()

    for alias in aliase:
        print(f"--- {alias} ---")
        if args.kein_wechsel:
            w = serverwechsel.Wechselergebnis(ok=True, alias=alias, schon_geladen=True,
                                              vram_nachher_mib=vram())
        else:
            w = serverwechsel.bereitstellen(alias, gpu_monitor=gpu)
        if not w.ok:
            print(f"  konnte nicht bereitgestellt werden: {w.fehler}")
            zeilen.append({"modell": alias, "ziel_tokens": None, "fehler": w.fehler})
            continue
        print(f"  bereit ({'schon geladen' if w.schon_geladen else f'{w.ladezeit_s}s'}), "
              f"VRAM {w.vram_nachher_mib} MiB")

        for laenge in args.laengen:
            vorher = vram()
            m = messen(serverwechsel.laufender_alias() or alias, prompt_bauen(laenge))
            nachher = vram()
            zeile = {"modell": alias, "ziel_tokens": laenge,
                     "vram_vor_mib": vorher, "vram_nach_mib": nachher, **m}
            zeilen.append(zeile)

            if not m.get("ok"):
                # Genau der interessante Fall: haelt der Offload-Wert den langen
                # Prompt nicht aus, steht es hier - und dann muss der Wert nach
                # unten, statt die Suite auf einer Klippe zu fahren.
                print(f"  {laenge:>6} Token  FEHLGESCHLAGEN: {m.get('fehler')}"
                      f"   (VRAM {vorher} -> {nachher} MiB)")
                lebt = serverwechsel.laufender_alias()
                print(f"                 Server danach: {lebt or 'weg'}")
                if lebt is None:
                    break
                continue

            print(f"  {laenge:>6} Token  ({m['prompt_tokens']} echt)  "
                  f"TTFT {m['ttft_s']}s  Wanduhr {m['wanduhr_s']}s  "
                  f"Nachladen {m['nachladen_s']}s  "
                  f"Prompt {m['prompt_tps']} tok/s  Gen {m['gen_tps']} tok/s  "
                  f"VRAM {nachher} MiB")
        print()

    felder = sorted({k for z in zeilen for k in z})
    with ziel.open("w", newline="", encoding="utf-8") as f:
        s = csv.DictWriter(f, fieldnames=felder)
        s.writeheader()
        s.writerows(zeilen)
    print(f"Geschrieben: {ziel}")

    schlecht = [z for z in zeilen if z.get("ziel_tokens") and not z.get("ok")]
    if schlecht:
        print("\nNicht ueberlebt:")
        for z in schlecht:
            print(f"  {z['modell']} bei {z['ziel_tokens']} Token: {z.get('fehler')}")
        print("\n-> Die Offload-Werte aus Prompt B sind fuer den echten Einsatz zu hoch.")
        return 3
    lang = [z for z in zeilen if z.get("nachladen_s") and z["nachladen_s"] > 1.0]
    if lang:
        print("\nMit nennenswertem Nachladen (Messpunkt waere ungueltig):")
        for z in lang:
            print(f"  {z['modell']} bei {z['ziel_tokens']} Token: {z['nachladen_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
