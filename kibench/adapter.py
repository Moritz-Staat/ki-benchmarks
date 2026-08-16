"""Einheitlicher Zugriff auf beide Runtimes.

llama-server und Ollama sprechen beide OpenAI-kompatibel, verhalten sich aber in
drei Punkten unterschiedlich genug, dass eine gemeinsame Schicht noetig ist.
Die drei Fallstricke aus Prompt B, Teil 3, sind hier jeweils an einer Stelle
geloest:

1. **Ollama haelt Modelle im VRAM.** `entladen()` schickt `keep_alive: 0` und
   **prueft anschliessend ueber NVML nach**, ob der Speicher wirklich zurueckkam.
   Ein Aufruf, der nur die Anfrage absetzt und Erfolg meldet, waere wertlos.

2. **Ollamas Standardkontext ist 4096.** `num_ctx` wird bei jeder Anfrage
   explizit gesetzt. Ollama schneidet sonst still ab, und die Messung misst dann
   etwas anderes als gedacht.

3. **Eine leere Antwort ist kein Erfolg.** Genau das ist in der Verifikation von
   Prompt A passiert: `chat/completions` galt als "OK", obwohl der Antworttext
   leer war - beide grossen Modelle denken per Default, und `max_tokens: 20`
   war schon vom Denkblock aufgebraucht. `AntwortMessung.erfolg` ist deshalb nur
   dann wahr, wenn tatsaechlich sichtbarer Text oder ein Tool-Aufruf ankam.
   Diese Fehlerart ist die gefaehrlichste im Projekt, weil sie Erfolge zaehlt,
   die keine sind.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import LLAMA_BASE, MODELLE_NACH_ALIAS, OLLAMA_BASE

# Grosszuegig, weil beide Qwen3-Varianten per Default denken. Gemessen in
# Prompt A: eine Ein-Wort-Antwort kostet mit aktivem Denken 846 Token.
MAX_TOKENS_STANDARD = 4096
KONTEXT_STANDARD = 32768

# Ein Denkmodell braucht bei 5,4 tok/s fuer 846 Token rund zweieinhalb Minuten.
ANTWORT_TIMEOUT_S = 900.0


@dataclass
class AntwortMessung:
    erfolg: bool = False
    fehlergrund: str | None = None

    text: str = ""
    denktext: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    tool_call_fliesstext: bool = False

    finish_reason: str | None = None
    prompt_tokens: int | None = None
    antwort_tokens: int | None = None
    denk_tokens: int | None = None
    # True, wenn die Denk-Token aus der Textlaenge geschaetzt wurden, weil die
    # Runtime `completion_tokens_details.reasoning_tokens` nicht liefert.
    denk_tokens_geschaetzt: bool = False
    gesamt_tokens: int | None = None

    dauer_s: float | None = None
    gen_tps: float | None = None

    http_status: int | None = None
    roh: dict[str, Any] = field(default_factory=dict)


class ModellAdapter:
    """Eine Instanz je Prozess. Haelt einen HTTP-Client offen."""

    def __init__(
        self,
        llama_base: str = LLAMA_BASE,
        ollama_base: str = OLLAMA_BASE,
        gpu_monitor=None,
    ) -> None:
        self.llama_base = llama_base.rstrip("/")
        self.ollama_base = ollama_base.rstrip("/")
        self._client = httpx.Client(timeout=ANTWORT_TIMEOUT_S)
        # Fuer die Nachpruefung bei entladen(). Optional, damit der Adapter auch
        # ohne GPU testbar bleibt.
        self._gpu = gpu_monitor

    # --- Hilfsmittel --------------------------------------------------------

    def runtime_von(self, alias: str) -> str:
        eintrag = MODELLE_NACH_ALIAS.get(alias)
        if eintrag:
            return eintrag["runtime"]
        # Ollama-Modelle heissen 'name:tag', llama.cpp-Aliase nicht.
        return "ollama" if ":" in alias else "llama.cpp"

    def basis_von(self, alias: str) -> str:
        return self.ollama_base if self.runtime_von(alias) == "ollama" else self.llama_base

    # --- Erreichbarkeit -----------------------------------------------------

    def verfuegbare_modelle(self) -> dict[str, list[str]]:
        """Welche Modelle antworten gerade? Fuer die Abnahme 'Adapter erreicht
        alle fuenf Modelle' und fuer die Statusanzeige."""
        ergebnis: dict[str, list[str]] = {"llama.cpp": [], "ollama": []}
        try:
            r = self._client.get(f"{self.llama_base}/v1/models", timeout=2.0)
            if r.status_code == 200:
                ergebnis["llama.cpp"] = [m["id"] for m in r.json().get("data", [])]
        except Exception:
            pass
        try:
            r = self._client.get(f"{self.ollama_base}/api/tags", timeout=5.0)
            if r.status_code == 200:
                ergebnis["ollama"] = [m["name"] for m in r.json().get("models", [])]
        except Exception:
            pass
        return ergebnis

    # --- Anfrage ------------------------------------------------------------

    def anfragen(
        self,
        alias: str,
        nachrichten: list[dict],
        *,
        max_tokens: int = MAX_TOKENS_STANDARD,
        temperature: float = 0.0,
        werkzeuge: list[dict] | None = None,
        thinking: bool = True,
        kontext: int = KONTEXT_STANDARD,
    ) -> AntwortMessung:
        m = AntwortMessung()
        basis = self.basis_von(alias)
        runtime = self.runtime_von(alias)

        koerper: dict[str, Any] = {
            "model": alias,
            "messages": nachrichten,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if werkzeuge:
            koerper["tools"] = werkzeuge
            koerper["tool_choice"] = "auto"
        if not thinking:
            # llama.cpp und Ollama nehmen beide den Jinja-Schalter des Templates.
            koerper["chat_template_kwargs"] = {"enable_thinking": False}
        if runtime == "ollama":
            # Fallstrick 2: ohne num_ctx schneidet Ollama bei 4096 still ab.
            koerper["options"] = {"num_ctx": kontext}

        t0 = time.perf_counter()
        try:
            r = self._client.post(f"{basis}/v1/chat/completions", json=koerper)
            m.http_status = r.status_code
            if r.status_code != 200:
                m.fehlergrund = f"http_{r.status_code}"
                m.dauer_s = round(time.perf_counter() - t0, 3)
                return m
            daten = r.json()
        except Exception as e:
            m.fehlergrund = f"verbindung: {type(e).__name__}"
            m.dauer_s = round(time.perf_counter() - t0, 3)
            return m

        m.dauer_s = round(time.perf_counter() - t0, 3)
        m.roh = daten
        return self._auswerten(m, daten)

    def _auswerten(self, m: AntwortMessung, daten: dict) -> AntwortMessung:
        wahl = (daten.get("choices") or [{}])[0]
        nachricht = wahl.get("message") or {}
        m.finish_reason = wahl.get("finish_reason")

        m.text = (nachricht.get("content") or "").strip()
        # llama.cpp und Ollama benennen den Denkblock unterschiedlich.
        m.denktext = (
            nachricht.get("reasoning_content") or nachricht.get("thinking") or ""
        ).strip()
        m.tool_calls = list(nachricht.get("tool_calls") or [])

        nutzung = daten.get("usage") or {}
        m.prompt_tokens = nutzung.get("prompt_tokens")
        m.gesamt_tokens = nutzung.get("total_tokens")
        gesamt_antwort = nutzung.get("completion_tokens")

        # Denk-Token von Antwort-Token trennen. Manche Builds liefern die Zahl in
        # completion_tokens_details; sonst bleibt nur eine Schaetzung aus der
        # Textlaenge, und die ist als solche markiert.
        details = nutzung.get("completion_tokens_details") or {}
        m.denk_tokens = details.get("reasoning_tokens")
        if m.denk_tokens is None and m.denktext:
            geschaetzt = max(1, round(len(m.denktext) / 3.6))
            # Die Schaetzung kann die tatsaechliche Zahl uebersteigen - gemessen
            # bei qwen-moe: 190 geschaetzt gegen 187 wirklich, was ohne Deckel
            # zu -3 Antwort-Token fuehrte. Der Deckel ist Pflicht, nicht Kosmetik:
            # negative Tokenzahlen wuerden in Prompt C stillschweigend in die
            # Auswertung wandern.
            m.denk_tokens = min(geschaetzt, gesamt_antwort) if gesamt_antwort else geschaetzt
            m.denk_tokens_geschaetzt = True
        if gesamt_antwort is not None:
            m.antwort_tokens = max(0, gesamt_antwort - (m.denk_tokens or 0))
            if m.dauer_s and m.dauer_s > 0:
                m.gen_tps = round(gesamt_antwort / m.dauer_s, 2)

        # Fallstrick 1 aus Prompt A: ein Tool-Aufruf, der als Fliesstext im
        # content steht statt in tool_calls. Meist fehlendes --jinja, bei
        # kleinen Modellen aber auch schlicht Ueberforderung.
        if not m.tool_calls and m.text.startswith("{") and '"name"' in m.text:
            m.tool_call_fliesstext = True

        # Fallstrick 3: leer ist kein Erfolg.
        if m.text or m.tool_calls:
            m.erfolg = True
        elif m.finish_reason == "length":
            m.fehlergrund = "leere_antwort_max_tokens"
        elif m.denktext:
            m.fehlergrund = "leere_antwort_nur_denkblock"
        else:
            m.fehlergrund = "leere_antwort"
        return m

    # --- Entladen -----------------------------------------------------------

    def entladen(self, alias: str | None = None, warten_s: float = 8.0) -> dict:
        """Modelle aus dem VRAM werfen und **nachpruefen**, dass er frei ist.

        Ohne die Nachpruefung waere der Rueckgabewert eine Behauptung: Ollama
        bestaetigt die Anfrage sofort, gibt den Speicher aber verzoegert frei.
        """
        vorher = self._vram_used()
        entladen: list[str] = []

        try:
            r = self._client.get(f"{self.ollama_base}/api/ps", timeout=5.0)
            geladen = [x.get("name") or x.get("model") for x in (r.json().get("models") or [])]
        except Exception:
            geladen = []

        ziele = [alias] if alias and ":" in alias else geladen
        for name in [z for z in ziele if z]:
            try:
                self._client.post(
                    f"{self.ollama_base}/api/generate",
                    json={"model": name, "keep_alive": 0},
                    timeout=60.0,
                )
                entladen.append(name)
            except Exception:
                pass

        # Dem Treiber Zeit geben, sonst misst die Nachpruefung den alten Stand.
        frist = time.monotonic() + warten_s
        rest_geladen: list[str] = []
        while time.monotonic() < frist:
            time.sleep(0.5)
            try:
                r = self._client.get(f"{self.ollama_base}/api/ps", timeout=5.0)
                rest_geladen = [
                    x.get("name") or x.get("model") for x in (r.json().get("models") or [])
                ]
            except Exception:
                rest_geladen = []
            if not rest_geladen:
                break

        nachher = self._vram_used()
        return {
            "war_geladen": geladen,
            "entladen": entladen,
            "noch_geladen": rest_geladen,
            "vram_vorher_mib": vorher,
            "vram_nachher_mib": nachher,
            "freigegeben_mib": (vorher - nachher) if (vorher and nachher) else None,
            # Der eigentliche Nachweis, nicht die Bestaetigung der Anfrage.
            "bestaetigt": not rest_geladen,
            # Ohne dieses Feld liest sich "bestaetigt: true" wie ein Nachweis,
            # obwohl gar nichts geladen war - mit OLLAMA_KEEP_ALIVE=0 der
            # Normalfall. Ein Aufruf ohne geladenes Modell beweist nichts.
            "aussagekraeftig": bool(geladen),
        }

    def modell_festhalten(self, alias: str, keep_alive: str = "120s") -> dict:
        """Ollama-Modell laden und im VRAM halten - nur zum Pruefen von `entladen()`.

        Im Betrieb ist `OLLAMA_KEEP_ALIVE=0` gesetzt, damit Ollama den Speicher
        fuer llama-server frei macht. Genau deshalb laesst sich die Freigabe
        sonst nicht pruefen: es ist nie etwas zu entladen da.
        """
        try:
            self._client.post(
                f"{self.ollama_base}/api/generate",
                json={"model": alias, "prompt": "Hi", "stream": False, "keep_alive": keep_alive},
                timeout=600.0,
            )
        except Exception as e:
            return {"ok": False, "fehler": f"{type(e).__name__}: {e}"}
        time.sleep(1.0)
        return {"ok": True, "vram_mib": self._vram_used()}

    def _vram_used(self) -> int | None:
        if self._gpu is None:
            return None
        try:
            return self._gpu.messen().vram_used_mib
        except Exception:
            return None

    def schliessen(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
