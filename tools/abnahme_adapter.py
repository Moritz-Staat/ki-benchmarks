"""Abnahme: erreicht der Adapter alle fuenf Modelle, und gibt Umschalten VRAM frei?

Laeuft gegen das laufende Dashboard, nutzt also genau den Weg, den spaeter auch
der Benchmark aus Prompt C nimmt - nicht einen Testpfad daneben.

    python tools/abnahme_adapter.py                 # was gerade erreichbar ist
    python tools/abnahme_adapter.py --modell qwen-moe
    python tools/abnahme_adapter.py --nur-entladen

llama-server wird **nicht** von hier gestartet; das macht `modell-wechsel.ps1`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import httpx

BASIS = "http://127.0.0.1:8420"
PROMPT = "Nenne die Hauptstadt von Deutschland. Antworte mit einem Wort."


def vram() -> int | None:
    try:
        return httpx.get(f"{BASIS}/api/jetzt", timeout=10).json()["sample"]["vram_used_mib"]
    except Exception:
        return None


def erreichbar() -> dict:
    return httpx.get(f"{BASIS}/api/modelle", timeout=30).json()


def frage(modell: str) -> dict:
    t0 = time.perf_counter()
    r = httpx.post(
        f"{BASIS}/api/anfrage",
        json={"modell": modell, "prompt": PROMPT, "max_tokens": 4096, "thinking": True},
        timeout=1200,
    )
    d = r.json()
    d["_wanduhr_s"] = round(time.perf_counter() - t0, 1)
    return d


def zeile(modell: str, d: dict) -> str:
    if d.get("erfolg"):
        text = (d.get("text") or "").replace("\n", " ")[:40]
        return (
            f"  [OK]      {modell:30} {d['_wanduhr_s']:7.1f}s  "
            f"{d.get('antwort_tokens')} Antwort- / {d.get('denk_tokens') or 0} Denk-Token"
            f"  ->  {text!r}"
        )
    return f"  [FEHLER]  {modell:30} {d['_wanduhr_s']:7.1f}s  Grund: {d.get('fehlergrund')}"


def entladen_pruefen(pin_modell: str = "qwen2.5:14b-instruct-q8_0") -> tuple[bool, dict]:
    """Der eigentliche Nachweis: nicht 'Anfrage abgesetzt', sondern 'VRAM zurueck'.

    Erst ein Modell festhalten, sonst ist mit OLLAMA_KEEP_ALIVE=0 gar nichts
    geladen und ein bestandener Test beweist nichts.
    """
    leer = vram()
    fest = httpx.post(f"{BASIS}/api/festhalten", params={"modell": pin_modell}, timeout=900).json()
    if not fest.get("ok"):
        return False, {"fehler": fest.get("fehler"), "aussagekraeftig": False}
    vor = vram()

    d = httpx.post(f"{BASIS}/api/entladen", timeout=300).json()
    time.sleep(3)
    nach = vram()

    d["vram_leerlauf_mib"] = leer
    d["vram_mit_modell_mib"] = vor
    d["vram_nach_entladen_mib"] = nach
    d["belegt_durch_modell_mib"] = (vor - leer) if (vor and leer) else None
    d["tatsaechlich_frei_mib"] = (vor - nach) if (vor and nach) else None

    # Bestanden nur, wenn wirklich etwas geladen war, es danach weg ist und
    # der VRAM messbar zurueckkam.
    ok = (
        bool(d.get("aussagekraeftig"))
        and bool(d.get("bestaetigt"))
        and (d["tatsaechlich_frei_mib"] or 0) > 1000
    )
    return ok, d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modell", help="nur dieses Modell pruefen")
    ap.add_argument("--nur-entladen", action="store_true")
    args = ap.parse_args()

    try:
        httpx.get(f"{BASIS}/api/health", timeout=5)
    except Exception:
        print(f"Dashboard antwortet nicht auf {BASIS} - erst dashboard.py starten.", file=sys.stderr)
        return 1

    fehler = 0

    if not args.nur_entladen:
        e = erreichbar()
        offen = e["erreichbar"]
        print(f"\nErreichbar laut Runtimes: llama.cpp={offen['llama.cpp']}  ollama={len(offen['ollama'])} Modelle")

        alle = {m["alias"] for m in e["konfiguriert"]}
        erreichbare = set(offen["llama.cpp"]) | set(offen["ollama"])
        ziele = [args.modell] if args.modell else sorted(alle & erreichbare)

        print(f"\n--- Anfrage an {len(ziele)} Modell(e) ---")
        for m in ziele:
            d = frage(m)
            print(zeile(m, d), flush=True)
            if not d.get("erfolg"):
                fehler += 1

        nicht_erreichbar = alle - erreichbare
        if nicht_erreichbar:
            print(f"\n  nicht erreichbar (Runtime laeuft nicht): {sorted(nicht_erreichbar)}")

    print("\n--- Entladen: gibt Ollama den VRAM wirklich frei? ---")
    # Gemessen: laeuft llama-server daneben, verdraengt der WDDM-Treiber dessen
    # Gewichte in den Hauptspeicher, sobald Ollama ein grosses Modell laedt. Der
    # VRAM faellt dann um 14 GB, aber aus zwei Gruenden gleichzeitig - das
    # Ergebnis waere nicht dem Entladen zuzuschreiben.
    if httpx.get(f"{BASIS}/api/jetzt", timeout=10).json()["sample"]["llama_alive"]:
        print("  UEBERSPRUNGEN - llama-server laeuft. Erst stoppen:")
        print("      ..\\scripts\\modell-wechsel.ps1 stop")
        print("  Sonst misst der Test die Verdraengung mit, nicht nur das Entladen.")
        print(f"\n{'ALLES BESTANDEN' if fehler == 0 else f'{fehler} FEHLER'} (Entladen nicht geprueft)")
        return 0 if fehler == 0 else 2

    ok, d = entladen_pruefen()
    print(f"  war geladen        : {d.get('war_geladen')}")
    print(f"  entladen           : {d.get('entladen')}")
    print(f"  noch geladen       : {d.get('noch_geladen')}")
    print(f"  VRAM Leerlauf      : {d.get('vram_leerlauf_mib')} MiB")
    print(f"  VRAM mit Modell    : {d.get('vram_mit_modell_mib')} MiB "
          f"(+{d.get('belegt_durch_modell_mib')})")
    print(f"  VRAM nach Entladen : {d.get('vram_nach_entladen_mib')} MiB "
          f"(-{d.get('tatsaechlich_frei_mib')})")
    print(f"  aussagekraeftig    : {d.get('aussagekraeftig')}")
    print(f"  BESTANDEN          : {ok}")
    if not ok:
        fehler += 1

    print(f"\n{'ALLES BESTANDEN' if fehler == 0 else f'{fehler} FEHLER'}")
    return 0 if fehler == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
