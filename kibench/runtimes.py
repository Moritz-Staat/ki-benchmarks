"""Abfrage der beiden Runtimes: llama-server /metrics und /props, Ollama /api/ps.

Alles hier ist bewusst fehlertolerant. Der Sampler soll auch dann weiterlaufen,
wenn gerade kein llama-server aktiv ist - das ist eine Abnahmebedingung aus
Prompt B, kein Komfort.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import httpx
import psutil

from .config import (
    LLAMA_BASE,
    MESS_TIMEOUT_S,
    OLLAMA_BASE,
    PROZESS_CACHE_S,
    RUNTIME_BACKOFF_S,
)

MIB = 1024 * 1024

LLAMA_PROZESS = "llama-server.exe"
OLLAMA_PROZESS = "ollama.exe"

_prozess_cache: tuple[float, set[str]] = (0.0, set())


def laufende_prozessnamen() -> set[str]:
    """Namen aller laufenden Prozesse, hoechstens einmal je Sekunde erhoben.

    Kostet rund 1 ms bei 300 Prozessen. Das ist der Preis dafuer, nicht in einen
    TCP-Timeout von 800 ms zu laufen, wenn der Server gar nicht existiert.
    """
    global _prozess_cache
    jetzt = time.monotonic()
    stand, namen = _prozess_cache
    if jetzt - stand < PROZESS_CACHE_S and namen:
        return namen
    try:
        namen = {p.info["name"] for p in psutil.process_iter(["name"]) if p.info["name"]}
    except Exception:
        namen = set()
    _prozess_cache = (jetzt, namen)
    return namen


def metrics_parsen(text: str) -> dict[str, float]:
    """Prometheus-Textformat -> flaches Dict.

    Bewusst generisch: llama.cpp benennt Metriken zwischen Builds um, und ein
    fest verdrahteter Parser wuerde das still verschlucken. Alles, was nicht
    ausdruecklich in eine Spalte wandert, landet als JSON in `llama_metrics_json`.
    """
    werte: dict[str, float] = {}
    for zeile in text.splitlines():
        zeile = zeile.strip()
        if not zeile or zeile.startswith("#"):
            continue
        # Labels ignorieren - llama.cpp setzt nur model_name, und der steht in /props.
        if "{" in zeile:
            name, rest = zeile.split("{", 1)
            _, _, wert = rest.rpartition("}")
        else:
            teile = zeile.rsplit(None, 1)
            if len(teile) != 2:
                continue
            name, wert = teile
        try:
            werte[name.strip()] = float(wert.strip())
        except ValueError:
            continue
    return werte


@dataclass
class LlamaMessung:
    alive: bool = False
    model: str | None = None
    gen_tps: float | None = None
    prompt_tps: float | None = None
    kv_cache_pct: float | None = None
    kv_cache_tokens: int | None = None
    requests_processing: int | None = None
    requests_deferred: int | None = None
    uebrige: dict[str, float] = field(default_factory=dict)


@dataclass
class OllamaMessung:
    alive: bool = False
    modelle: list[dict] = field(default_factory=list)


# Zuordnung Metrikname -> Feld. Die Namen stammen aus llama.cpp b10437.
_ZUORDNUNG = {
    "llamacpp:predicted_tokens_seconds": "gen_tps",
    "llamacpp:prompt_tokens_seconds": "prompt_tps",
    "llamacpp:kv_cache_usage_ratio": "kv_cache_pct",
    "llamacpp:kv_cache_tokens": "kv_cache_tokens",
    "llamacpp:requests_processing": "requests_processing",
    "llamacpp:requests_deferred": "requests_deferred",
}


class RuntimeMonitor:
    """Haelt HTTP-Verbindungen offen, damit nicht jede Sekunde neu verbunden wird."""

    def __init__(self, llama_base: str = LLAMA_BASE, ollama_base: str = OLLAMA_BASE) -> None:
        self.llama_base = llama_base.rstrip("/")
        self.ollama_base = ollama_base.rstrip("/")
        self._client = httpx.Client(timeout=MESS_TIMEOUT_S)
        # Sperre nach einem Fehlschlag. Waehrend llama-server ein Modell laedt,
        # existiert der Prozess schon, der Port antwortet aber noch nicht -
        # das dauert 8 bis 12 Sekunden und darf nicht jede Sekunde in den
        # Timeout laufen.
        self._llama_sperre_bis = 0.0
        self._ollama_sperre_bis = 0.0

    def _gesperrt(self, bis: float) -> bool:
        return time.monotonic() < bis

    def llama(self) -> LlamaMessung:
        m = LlamaMessung()
        if LLAMA_PROZESS not in laufende_prozessnamen():
            return m
        if self._gesperrt(self._llama_sperre_bis):
            return m
        try:
            r = self._client.get(f"{self.llama_base}/metrics")
            if r.status_code != 200:
                self._llama_sperre_bis = time.monotonic() + RUNTIME_BACKOFF_S
                return m
            roh = metrics_parsen(r.text)
        except Exception:
            self._llama_sperre_bis = time.monotonic() + RUNTIME_BACKOFF_S
            return m

        m.alive = True
        for name, wert in roh.items():
            feld = _ZUORDNUNG.get(name)
            if feld is None:
                m.uebrige[name] = wert
                continue
            if feld == "kv_cache_pct":
                setattr(m, feld, round(wert * 100.0, 2))
            elif feld in ("kv_cache_tokens", "requests_processing", "requests_deferred"):
                setattr(m, feld, int(wert))
            else:
                setattr(m, feld, round(wert, 2))

        # Welches Modell geladen ist, steht nicht in /metrics.
        try:
            p = self._client.get(f"{self.llama_base}/props")
            if p.status_code == 200:
                daten = p.json()
                m.model = daten.get("model_alias") or daten.get("model_path")
        except Exception:
            pass
        return m

    def ollama(self) -> OllamaMessung:
        m = OllamaMessung()
        if OLLAMA_PROZESS not in laufende_prozessnamen():
            return m
        if self._gesperrt(self._ollama_sperre_bis):
            return m
        try:
            r = self._client.get(f"{self.ollama_base}/api/ps")
            if r.status_code != 200:
                self._ollama_sperre_bis = time.monotonic() + RUNTIME_BACKOFF_S
                return m
            daten = r.json()
        except Exception:
            self._ollama_sperre_bis = time.monotonic() + RUNTIME_BACKOFF_S
            return m

        m.alive = True
        for eintrag in daten.get("models") or []:
            m.modelle.append(
                {
                    "name": eintrag.get("name") or eintrag.get("model"),
                    "size_mib": int((eintrag.get("size") or 0) / MIB),
                    "size_vram_mib": int((eintrag.get("size_vram") or 0) / MIB),
                    "bis": eintrag.get("expires_at"),
                }
            )
        return m

    def schliessen(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
