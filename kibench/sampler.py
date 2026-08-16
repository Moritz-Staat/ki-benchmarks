"""Der Sampler: ein Hintergrund-Thread, eine Messung pro Sekunde, nach SQLite.

Zwei Eigenschaften sind Abnahmebedingungen aus Prompt B und bestimmen den Aufbau:

1. **Laeuft auch ohne llama-server.** Jede Quelle wird einzeln gekapselt; faellt
   eine aus, fehlen ihre Spalten, der Takt laeuft weiter.
2. **Kein Speicherleck ueber Stunden.** Deshalb gibt es hier keine wachsenden
   Listen. Der einzige Zustand ist ein Ringpuffer fester Laenge fuer die
   Fremd-VRAM-Schwankung; alles andere geht sofort in die Datenbank.

Der Takt haengt an `time.monotonic()` und nicht an `sleep(1)`, sonst driftet die
Zeitreihe ueber Stunden um die Summe aller Messdauern davon.
"""
from __future__ import annotations

import collections
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

from . import db
from .config import (
    FREMD_VRAM_FENSTER_S,
    OLLAMA_PS_INTERVALL_S,
    SAMPLE_INTERVALL_S,
)
from .gpu import GpuMonitor
from .runtimes import RuntimeMonitor
from .sysinfo import SysMonitor


@dataclass
class SamplerStatus:
    laeuft: bool = False
    gestartet_ts: float | None = None
    messungen: int = 0
    fehler: int = 0
    letzter_fehler: str | None = None
    ausgelassen: int = 0            # Takte, die zu lange gedauert haben
    letzte_dauer_ms: float | None = None
    quellen: dict = field(default_factory=dict)


class Sampler:
    def __init__(self, intervall_s: float = SAMPLE_INTERVALL_S, db_pfad=None) -> None:
        self.intervall_s = intervall_s
        self._db_pfad = db_pfad
        self._stop = threading.Event()
        # Wird gesetzt, sobald Datenbank und Messquellen offen sind. start()
        # wartet darauf, sonst greift der Aufrufer auf ein halb aufgebautes
        # Objekt zu - die API holt sich direkt nach start() den GpuMonitor.
        self._bereit = threading.Event()
        self._thread: threading.Thread | None = None
        self.status = SamplerStatus()

        # Ringpuffer fester Laenge - waechst nicht, egal wie lange der Sampler laeuft.
        self._fremd_vram = collections.deque(maxlen=int(FREMD_VRAM_FENSTER_S / intervall_s) or 60)

        self._gpu: GpuMonitor | None = None
        self._sys: SysMonitor | None = None
        self._rt: RuntimeMonitor | None = None
        self._ollama_zuletzt = 0.0
        self._ollama_cache = None

    # --- Steuerung ----------------------------------------------------------

    def start(self, warten_s: float = 10.0) -> bool:
        """Startet den Sampler und wartet, bis er wirklich misst.

        Gibt zurueck, ob er rechtzeitig hochkam. Ein `start()`, das vor der
        Initialisierung zurueckkehrt, wuerde falsche Zusagen machen: der
        Statusendpunkt meldete "laeuft nicht", und die API haette sich einen
        GpuMonitor geholt, den es noch gar nicht gibt.
        """
        if self._thread and self._thread.is_alive():
            return True
        self._stop.clear()
        self._bereit.clear()
        self._thread = threading.Thread(target=self._schleife, name="sampler", daemon=True)
        self._thread.start()
        return self._bereit.wait(timeout=warten_s)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        self._bereit.clear()
        self.status.laeuft = False

    # --- Kennzahl fuer die Live-Ansicht -------------------------------------

    def fremd_vram_schwankung_mib(self) -> int:
        """Spannweite des Fremd-VRAM im Beobachtungsfenster.

        Prompt B will sichtbar markiert haben, wenn waehrend eines Vorgangs
        fremder VRAM um mehr als 200 MiB schwankt - ein Videostream im Browser
        belegt genau diese Groessenordnung und hat in Prompt A den dense-Sweep
        unbrauchbar gemacht, ohne dass es beim Messen aufgefallen waere.
        """
        if len(self._fremd_vram) < 2:
            return 0
        return max(self._fremd_vram) - min(self._fremd_vram)

    # --- Schleife -----------------------------------------------------------

    def _schleife(self) -> None:
        conn = db.init_db(self._db_pfad)
        self._gpu = GpuMonitor()
        self._sys = SysMonitor()
        self._rt = RuntimeMonitor()

        self.status.laeuft = True
        self.status.gestartet_ts = time.time()
        naechster = time.monotonic()
        # Ab hier sind Datenbank und alle Messquellen offen.
        self._bereit.set()

        try:
            while not self._stop.is_set():
                t0 = time.perf_counter()
                try:
                    werte = self._messen()
                    db.sample_schreiben(conn, werte)
                    conn.commit()
                    self.status.messungen += 1
                except Exception as e:
                    self.status.fehler += 1
                    self.status.letzter_fehler = f"{type(e).__name__}: {e}"
                self.status.letzte_dauer_ms = round((time.perf_counter() - t0) * 1000, 2)

                naechster += self.intervall_s
                rest = naechster - time.monotonic()
                if rest < 0:
                    # Der Takt ist davongelaufen (Systemlast, Standby). Nicht
                    # aufholen, sondern neu ausrichten - sonst laeuft der Sampler
                    # danach im Dauerlauf, um die Luecke zu schliessen.
                    ausgelassen = int(-rest / self.intervall_s) + 1
                    self.status.ausgelassen += ausgelassen
                    naechster = time.monotonic() + self.intervall_s
                    rest = self.intervall_s
                self._stop.wait(rest)
        finally:
            self.status.laeuft = False
            for teil in (self._gpu, self._rt):
                try:
                    teil.schliessen()  # type: ignore[union-attr]
                except Exception:
                    pass
            try:
                conn.commit()
            except Exception:
                pass

    def _messen(self) -> dict:
        jetzt = time.time()
        g = self._gpu.messen()          # type: ignore[union-attr]
        s = self._sys.messen()          # type: ignore[union-attr]
        l = self._rt.llama()            # type: ignore[union-attr]

        # Ollama seltener fragen: /api/ps ist teurer als NVML und aendert sich selten.
        if jetzt - self._ollama_zuletzt >= OLLAMA_PS_INTERVALL_S or self._ollama_cache is None:
            self._ollama_cache = self._rt.ollama()   # type: ignore[union-attr]
            self._ollama_zuletzt = jetzt
        o = self._ollama_cache

        if g.zuordnung_verfuegbar:
            self._fremd_vram.append(g.vram_fremd_mib)

        self.status.quellen = {
            "nvml": g.vram_used_mib is not None,
            "pdh": g.zuordnung_verfuegbar,
            "psutil": s.cpu_pct is not None,
            "llama": l.alive,
            "ollama": o.alive,
        }

        return {
            "ts": jetzt,
            "ts_iso": datetime.fromtimestamp(jetzt).isoformat(timespec="seconds"),
            "gpu_util_pct": g.gpu_util_pct,
            "gpu_mem_util_pct": g.gpu_mem_util_pct,
            "vram_used_mib": g.vram_used_mib,
            "vram_free_mib": g.vram_free_mib,
            "gpu_temp_c": g.gpu_temp_c,
            "gpu_clock_sm_mhz": g.gpu_clock_sm_mhz,
            "gpu_clock_mem_mhz": g.gpu_clock_mem_mhz,
            "gpu_power_w": g.gpu_power_w,
            "vram_llama_mib": g.vram_llama_mib if g.zuordnung_verfuegbar else None,
            "vram_ollama_mib": g.vram_ollama_mib if g.zuordnung_verfuegbar else None,
            "vram_fremd_mib": g.vram_fremd_mib if g.zuordnung_verfuegbar else None,
            "vram_prozesse_json": json.dumps(g.prozesse, ensure_ascii=False) if g.prozesse else None,
            "cpu_pct": s.cpu_pct,
            "cpu_kerne_json": json.dumps(s.cpu_kerne) if s.cpu_kerne else None,
            "ram_used_gib": s.ram_used_gib,
            "ram_pct": s.ram_pct,
            "pagefile_used_gib": s.pagefile_used_gib,
            "pagefile_pct": s.pagefile_pct,
            "disk_read_mibs": s.disk_read_mibs,
            "disk_write_mibs": s.disk_write_mibs,
            "llama_alive": 1 if l.alive else 0,
            "llama_model": l.model,
            "llama_gen_tps": l.gen_tps,
            "llama_prompt_tps": l.prompt_tps,
            "llama_kv_cache_pct": l.kv_cache_pct,
            "llama_kv_cache_tokens": l.kv_cache_tokens,
            "llama_requests_processing": l.requests_processing,
            "llama_requests_deferred": l.requests_deferred,
            "llama_metrics_json": json.dumps(l.uebrige) if l.uebrige else None,
            "ollama_alive": 1 if o.alive else 0,
            "ollama_modelle_json": json.dumps(o.modelle, ensure_ascii=False) if o.modelle else None,
        }
