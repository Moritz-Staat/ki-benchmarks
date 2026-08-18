"""Den richtigen llama-server hochfahren - ohne die Konsole zu blockieren.

Der Probelauf soll ueber alle sechs Konfigurationen laufen, "ohne manuellen
Eingriff". Drei davon sind llama.cpp-Konfigurationen, die sich Port 8080 und den
halben Grafikspeicher teilen; zwischen ihnen muss der Runner selbst umschalten.

Die vorhandenen Startskripte tun genau das Richtige - Standby aus, Ollama
raeumen, Flags setzen -, laufen aber im Vordergrund und blockieren die Konsole,
solange der Server lebt. Fuer den Runner werden sie deshalb losgeloest gestartet
und ueber `/health` abgewartet.

**Warum ueberhaupt umgeschaltet werden muss** statt beide Server nebeneinander
laufen zu lassen: Prompt B, Befund 5. Der WDDM-Treiber laesst zwei Runtimes
gleichzeitig zu, verdraengt aber die Gewichte der einen still in den
Hauptspeicher. Nichts stuerzt ab, keine Fehlermeldung - nur die Wartezeit
verzehnfacht sich, waehrend der Server weiter gesunde tok/s meldet.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import LLAMA_BASE, MODELLE_NACH_ALIAS, OLLAMA_BASE, PROJEKT_DIR

SCRIPTS_DIR = PROJEKT_DIR / "scripts"

# Ein 15-GB-Modell braucht von der Platte rund 7 s, aus dem Dateicache weniger.
# 300 s sind grosszuegig; laenger bedeutet, dass etwas anderes klemmt.
START_TIMEOUT_S = 300.0


@dataclass
class Wechselergebnis:
    ok: bool
    alias: str | None = None
    schon_geladen: bool = False
    ladezeit_s: float | None = None
    vram_vorher_mib: int | None = None
    vram_nachher_mib: int | None = None
    fehler: str | None = None


def laufender_alias(basis: str = LLAMA_BASE, timeout: float = 2.0) -> str | None:
    try:
        r = httpx.get(f"{basis}/props", timeout=timeout)
        if r.status_code == 200:
            return r.json().get("model_alias")
    except Exception:
        pass
    return None


def llama_stoppen() -> None:
    subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command",
         "Get-Process -Name llama-server -ErrorAction SilentlyContinue | "
         "Stop-Process -Force -ErrorAction SilentlyContinue"],
        capture_output=True, text=True,
    )
    # Der Treiber gibt den Speicher verzoegert frei. Ohne die Pause misst die
    # naechste VRAM-Abfrage den alten Stand und der naechste Start rechnet mit
    # zu wenig Platz.
    time.sleep(3.0)


def ollama_raeumen(basis: str = OLLAMA_BASE) -> list[str]:
    """Geladene Ollama-Modelle entladen. Siehe Modulkopf, warum das Pflicht ist."""
    entladen: list[str] = []
    try:
        r = httpx.get(f"{basis}/api/ps", timeout=5.0)
        modelle = [m.get("name") or m.get("model") for m in (r.json().get("models") or [])]
    except Exception:
        return entladen
    for name in [m for m in modelle if m]:
        try:
            httpx.post(f"{basis}/api/generate",
                       json={"model": name, "keep_alive": 0}, timeout=60.0)
            entladen.append(name)
        except Exception:
            pass
    if entladen:
        time.sleep(5.0)
    return entladen


def _startbefehl(alias: str) -> list[str] | None:
    ziel = (MODELLE_NACH_ALIAS.get(alias) or {}).get("wechsel_ziel")
    if not ziel:
        return None
    skript = SCRIPTS_DIR / ("qwen-moe.ps1" if ziel == "moe" else "qwen-dense.ps1")
    befehl = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
              "-File", str(skript)]
    if ziel == "dense-iq4":
        befehl.append("-Iq4")
    return befehl


def bereitstellen(
    alias: str,
    gpu_monitor=None,
    log_verzeichnis: Path | None = None,
) -> Wechselergebnis:
    """Dafuer sorgen, dass `alias` antwortet. Ollama-Modelle brauchen nichts."""
    eintrag = MODELLE_NACH_ALIAS.get(alias) or {}

    def vram() -> int | None:
        try:
            return gpu_monitor.messen().vram_used_mib if gpu_monitor else None
        except Exception:
            return None

    if eintrag.get("runtime") == "ollama":
        # Umgekehrte Richtung, gleicher Grund: llama-server haelt sonst 14 GB
        # fest, und Ollama laedt daneben in den Hauptspeicher.
        if laufender_alias():
            llama_stoppen()
        return Wechselergebnis(ok=True, alias=alias, vram_nachher_mib=vram())

    aktuell = laufender_alias()
    if aktuell == alias:
        return Wechselergebnis(ok=True, alias=alias, schon_geladen=True,
                               vram_nachher_mib=vram())

    befehl = _startbefehl(alias)
    if befehl is None:
        return Wechselergebnis(ok=False, fehler=f"Kein Startbefehl fuer {alias!r}")

    vorher = vram()
    if aktuell:
        llama_stoppen()
    ollama_raeumen()

    log_verzeichnis = log_verzeichnis or (PROJEKT_DIR / "ki-benchmarks" / "logs")
    log_verzeichnis.mkdir(parents=True, exist_ok=True)
    aus = open(log_verzeichnis / f"server-{alias}.log", "w", encoding="utf-8")
    subprocess.Popen(befehl, stdout=aus, stderr=subprocess.STDOUT,
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    t0 = time.monotonic()
    while time.monotonic() - t0 < START_TIMEOUT_S:
        time.sleep(1.0)
        try:
            h = httpx.get(f"{LLAMA_BASE}/health", timeout=2.0)
            if h.status_code == 200 and h.json().get("status") == "ok":
                break
        except Exception:
            continue
    else:
        return Wechselergebnis(ok=False, alias=alias, vram_vorher_mib=vorher,
                               fehler=f"Server kam in {START_TIMEOUT_S:.0f}s nicht hoch")

    ladezeit = round(time.monotonic() - t0, 1)
    gemeldet = laufender_alias()
    if gemeldet != alias:
        # Ein Server, der antwortet, ist nicht zwingend der richtige. Ohne diese
        # Pruefung liefe die halbe Messreihe gegen das falsche Modell.
        return Wechselergebnis(
            ok=False, alias=gemeldet, ladezeit_s=ladezeit, vram_vorher_mib=vorher,
            vram_nachher_mib=vram(),
            fehler=f"Server meldet {gemeldet!r}, erwartet war {alias!r}",
        )
    return Wechselergebnis(ok=True, alias=alias, ladezeit_s=ladezeit,
                           vram_vorher_mib=vorher, vram_nachher_mib=vram())
