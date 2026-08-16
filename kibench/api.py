"""FastAPI-Anwendung: JSON-Endpunkte plus die Live-Seite.

Warum Web und nicht Terminal: eine Webseite kann sich selbst verifizieren -
Endpunkt aufrufen, JSON gegen Erwartung pruefen. Bei einer TUI bliebe nur die
Behauptung, dass die Anzeige stimmt. Genau deshalb sind unten alle Kennzahlen
ueber `/api/...` abrufbar und nicht nur im HTML zusammengerechnet.
"""
from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import db
from .adapter import ModellAdapter
from .config import (
    MODELLE,
    MODELLE_NACH_ALIAS,
    RAM_TOTAL_GIB,
    SCHWELLE_FREMD_VRAM_SCHWANKUNG_MIB,
    SCHWELLE_GPU_TEMP_C,
    SCHWELLE_RAM_GIB,
    SCHWELLE_TPS_ANTEIL,
    SCHWELLE_VRAM_MIB,
    STATIC_DIR,
    VRAM_TOTAL_MIB,
)
from .sampler import Sampler

sampler = Sampler()
adapter: ModellAdapter | None = None


@asynccontextmanager
async def lebenszyklus(app: FastAPI):
    global adapter
    db.init_db()
    sampler.start()
    # Der Adapter teilt sich den GPU-Monitor des Samplers, damit nicht zwei
    # NVML-Handles offen sind.
    adapter = ModellAdapter(gpu_monitor=sampler._gpu)
    try:
        yield
    finally:
        sampler.stop()
        if adapter:
            adapter.schliessen()


app = FastAPI(title="ki-benchmarks Live-Dashboard", version="0.1.0", lifespan=lebenszyklus)


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _zeitfenster(fenster: str) -> float:
    """'5m' | '1h' | 'heute' -> Unix-Zeit, ab der gelesen wird."""
    jetzt = time.time()
    if fenster == "5m":
        return jetzt - 5 * 60
    if fenster == "15m":
        return jetzt - 15 * 60
    if fenster == "1h":
        return jetzt - 60 * 60
    if fenster == "6h":
        return jetzt - 6 * 60 * 60
    if fenster == "heute":
        lokal = time.localtime(jetzt)
        return time.mktime((lokal.tm_year, lokal.tm_mon, lokal.tm_mday, 0, 0, 0, 0, 0, -1))
    raise HTTPException(status_code=400, detail=f"unbekanntes Zeitfenster: {fenster}")


def _warnungen(zeile: dict, schwankung_mib: int) -> list[dict]:
    """Die Warnschwellen aus Prompt B, an einer Stelle ausgewertet."""
    w: list[dict] = []
    vram = zeile.get("vram_used_mib")
    if vram and vram > SCHWELLE_VRAM_MIB:
        w.append({
            "stufe": "kritisch",
            "feld": "vram",
            "text": f"VRAM {vram/1024:.1f} GB über {SCHWELLE_VRAM_MIB/1024:.1f} GB — OOM-Gefahr",
        })
    ram = zeile.get("ram_used_gib")
    if ram and ram > SCHWELLE_RAM_GIB:
        w.append({
            "stufe": "kritisch",
            "feld": "ram",
            "text": f"RAM {ram:.1f} GB über {SCHWELLE_RAM_GIB:.0f} GB — Auslagerung droht",
        })
    temp = zeile.get("gpu_temp_c")
    if temp and temp > SCHWELLE_GPU_TEMP_C:
        w.append({
            "stufe": "kritisch",
            "feld": "temp",
            "text": f"GPU {temp} °C über {SCHWELLE_GPU_TEMP_C} °C — thermisches Throttling",
        })

    # tok/s unter der Haelfte des Sweep-Werts aus SETUP.md
    modell = zeile.get("llama_model")
    tps = zeile.get("llama_gen_tps")
    eintrag = MODELLE_NACH_ALIAS.get(modell or "")
    if eintrag and eintrag.get("sweep_gen_tps") and tps:
        grenze = eintrag["sweep_gen_tps"] * SCHWELLE_TPS_ANTEIL
        if tps < grenze:
            w.append({
                "stufe": "warnung",
                "feld": "tps",
                "text": (
                    f"{tps:.1f} tok/s unter der Hälfte des Sweep-Werts "
                    f"({eintrag['sweep_gen_tps']:.1f} tok/s) — etwas stimmt nicht"
                ),
            })

    # Fremd-VRAM-Schwankung: der Befund aus Prompt A, der den dense-Sweep
    # unbrauchbar gemacht hat.
    if schwankung_mib > SCHWELLE_FREMD_VRAM_SCHWANKUNG_MIB:
        w.append({
            "stufe": "warnung",
            "feld": "fremd_vram",
            "text": (
                f"Fremd-VRAM schwankt um {schwankung_mib} MiB — "
                "die Maschine ist nicht ruhig, Messwerte sind unsicher"
            ),
        })
    return w


def _row_dict(row) -> dict[str, Any]:
    return {k: row[k] for k in row.keys()} if row is not None else {}


# ---------------------------------------------------------------------------
# Endpunkte
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    """Kurzer Lebenszeichen-Endpunkt fuer Tests und Startskripte."""
    return {
        "ok": True,
        "sampler_laeuft": sampler.status.laeuft,
        "messungen": sampler.status.messungen,
    }


@app.get("/api/status")
def status() -> dict:
    """Zustand des Samplers selbst - Grundlage der Abnahme 'laeuft 2 Stunden
    ohne Leck, Zeitreihe ohne Luecken'."""
    conn = db.verbindung()
    zeilen = conn.execute("SELECT COUNT(*) n, MIN(ts) a, MAX(ts) b FROM samples").fetchone()
    laufzeit = (time.time() - sampler.status.gestartet_ts) if sampler.status.gestartet_ts else 0
    return {
        "laeuft": sampler.status.laeuft,
        "gestartet_ts": sampler.status.gestartet_ts,
        "laufzeit_s": round(laufzeit, 1),
        "messungen": sampler.status.messungen,
        "fehler": sampler.status.fehler,
        "letzter_fehler": sampler.status.letzter_fehler,
        "ausgelassen": sampler.status.ausgelassen,
        "letzte_dauer_ms": sampler.status.letzte_dauer_ms,
        "quellen": sampler.status.quellen,
        "db_zeilen": zeilen["n"],
        "db_von_ts": zeilen["a"],
        "db_bis_ts": zeilen["b"],
    }


@app.get("/api/jetzt")
def jetzt() -> dict:
    """Die Kennzahlen fuer die Kachelreihe."""
    conn = db.verbindung()
    row = _row_dict(db.letzter_sample(conn))
    if not row:
        return {"vorhanden": False, "warnungen": []}

    schwankung = sampler.fremd_vram_schwankung_mib()
    row["vram_prozesse"] = json.loads(row.pop("vram_prozesse_json") or "[]")
    row["cpu_kerne"] = json.loads(row.pop("cpu_kerne_json") or "[]")
    row["ollama_modelle"] = json.loads(row.pop("ollama_modelle_json") or "[]")
    row["llama_metrics"] = json.loads(row.pop("llama_metrics_json") or "{}")

    return {
        "vorhanden": True,
        "alter_s": round(time.time() - row["ts"], 1),
        "vram_total_mib": VRAM_TOTAL_MIB,
        "ram_total_gib": RAM_TOTAL_GIB,
        "fremd_vram_schwankung_mib": schwankung,
        "fremd_vram_unruhig": schwankung > SCHWELLE_FREMD_VRAM_SCHWANKUNG_MIB,
        "warnungen": _warnungen(row, schwankung),
        "sample": row,
    }


VERLAUF_SPALTEN = [
    "ts",
    "gpu_util_pct",
    "vram_used_mib",
    "vram_fremd_mib",
    "gpu_temp_c",
    "gpu_clock_sm_mhz",
    "gpu_power_w",
    "cpu_pct",
    "ram_used_gib",
    "llama_gen_tps",
    "llama_prompt_tps",
    "llama_kv_cache_pct",
]


@app.get("/api/verlauf")
def verlauf(
    fenster: str = Query("5m", pattern="^(5m|15m|1h|6h|heute)$"),
    max_punkte: int = Query(1200, ge=50, le=5000),
) -> dict:
    """Zeitreihe fuer die Diagramme, spaltenweise statt zeilenweise.

    Spaltenweise, weil uPlot genau dieses Format erwartet und weil es bei 1200
    Punkten rund ein Drittel weniger JSON ist als eine Liste von Objekten.
    """
    ab = _zeitfenster(fenster)
    conn = db.verbindung()
    zeilen = db.samples_zeitraum(conn, ab, VERLAUF_SPALTEN, max_punkte=max_punkte)
    spalten: dict[str, list] = {name: [] for name in VERLAUF_SPALTEN}
    for z in zeilen:
        for name in VERLAUF_SPALTEN:
            spalten[name].append(z[name])
    return {
        "fenster": fenster,
        "punkte": len(zeilen),
        "ab_ts": ab,
        "spalten": spalten,
        "schwellen": {
            "vram_mib": SCHWELLE_VRAM_MIB,
            "ram_gib": SCHWELLE_RAM_GIB,
            "gpu_temp_c": SCHWELLE_GPU_TEMP_C,
            "vram_total_mib": VRAM_TOTAL_MIB,
            "ram_total_gib": RAM_TOTAL_GIB,
        },
    }


@app.get("/api/modelle")
def modelle() -> dict:
    """Die fuenf Modelle aus SETUP.md plus was gerade tatsaechlich antwortet."""
    return {
        "konfiguriert": MODELLE,
        "erreichbar": adapter.verfuegbare_modelle() if adapter else {},
    }


class AnfrageEingabe(BaseModel):
    modell: str
    prompt: str
    max_tokens: int = Field(default=4096, ge=1, le=32768)
    thinking: bool = True
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


@app.post("/api/anfrage")
def anfrage(eingabe: AnfrageEingabe) -> dict:
    """Eine einzelne Anfrage ueber den Adapter - fuer den Erreichbarkeitstest.

    Bewusst kein Benchmark: die Aufgabensammlung kommt in Prompt C.
    """
    if adapter is None:
        raise HTTPException(status_code=503, detail="Adapter nicht bereit")
    m = adapter.anfragen(
        eingabe.modell,
        [{"role": "user", "content": eingabe.prompt}],
        max_tokens=eingabe.max_tokens,
        temperature=eingabe.temperature,
        thinking=eingabe.thinking,
    )
    return {
        "erfolg": m.erfolg,
        "fehlergrund": m.fehlergrund,
        "text": m.text,
        "denktext_zeichen": len(m.denktext),
        "tool_calls": m.tool_calls,
        "tool_call_fliesstext": m.tool_call_fliesstext,
        "finish_reason": m.finish_reason,
        "prompt_tokens": m.prompt_tokens,
        "denk_tokens": m.denk_tokens,
        "antwort_tokens": m.antwort_tokens,
        "dauer_s": m.dauer_s,
        "gen_tps": m.gen_tps,
    }


@app.post("/api/entladen")
def entladen(modell: str | None = None) -> dict:
    """Ollama-Modelle entladen und nachweisen, dass der VRAM frei ist."""
    if adapter is None:
        raise HTTPException(status_code=503, detail="Adapter nicht bereit")
    return adapter.entladen(modell)


# ---------------------------------------------------------------------------
# Live-Seite
# ---------------------------------------------------------------------------

@app.get("/")
def seite() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.exception_handler(404)
async def nicht_gefunden(request, exc):
    return JSONResponse(status_code=404, content={"fehler": "nicht gefunden", "pfad": str(request.url.path)})
