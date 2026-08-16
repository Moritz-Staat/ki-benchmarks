"""GPU-Messung: NVML fuer die Karte, Windows-PDH fuer den VRAM je Prozess.

Warum zwei Quellen?

NVML liefert auf dieser Maschine **keinen** VRAM je Prozess.
`nvmlDeviceGetComputeRunningProcesses` gibt zwar alle 32 GPU-Prozesse zurueck,
aber `usedGpuMemory` ist bei jedem einzelnen `None` - in v1, v2 und v3. Das ist
die bekannte WDDM-Einschraenkung: unter Windows koennen GeForce-Karten im
WDDM-Treibermodell den Speicher nicht je Prozess zuordnen, nur im TCC-Modus,
den GeForce nicht anbietet. `nvidia-smi` zeigt an derselben Stelle `[N/A]`.

Prompt B verlangt den Fremd-VRAM ausdruecklich als Pflichtfeld. Die Windows-
Leistungsindikatoren liefern ihn: der Zaehlersatz `GPU Process Memory` mit
`Dedicated Usage` je Instanz `pid_<PID>_luid_..._phys_<n>` ist dieselbe Quelle,
aus der auch der Task-Manager seine GPU-Spalte speist. Ueber `win32pdh` kostet
eine Abfrage rund 0,03 ms - guenstiger als NVML selbst und weit unter dem, was
ein `nvidia-smi`-Subprozess kosten wuerde.

Die Summe ueber alle PDH-Instanzen liegt systematisch **ueber**
`nvmlDeviceGetMemoryInfo().used`, weil PDH je physischer Engine zaehlt und
gemeinsam genutzte Allokationen mehrfach auftauchen. Gemessen: 4446 MiB PDH-Summe
gegen 2099 MiB laut NVML, also gut der doppelte Wert.

Beide Zahlen unveraendert nebeneinanderzustellen ergibt Unsinn - im ersten
Entwurf lag der "Fremd-VRAM" ueber der Gesamtbelegung. Deshalb gilt hier:

* **NVML bestimmt die Gesamtsumme.** Sie ist die einzige verlaessliche Zahl.
* **PDH bestimmt nur die Aufteilung.** Die Rohwerte werden auf die NVML-Summe
  normiert, jeder Gruppenwert ist also `NVML-Gesamt x PDH-Anteil`.

Das Ergebnis ist eine **Zuordnung, keine Direktmessung** - der Anteil stimmt, die
absolute Zahl je Prozess ist auf die reale Gesamtbelegung heruntergerechnet. Fuer
die Frage, die Prompt B beantwortet haben will ("war die Maschine waehrend dieses
Laufs ruhig?"), ist genau das die brauchbare Groesse: sie ist in echten MiB
ausgedrueckt und durch die Kartenkapazitaet begrenzt.

Der unnormierte Rohwert bleibt je Prozess als `mib_roh` erhalten, damit die
Herkunft nachvollziehbar bleibt.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field

import psutil
import pynvml

from .config import EIGENE_PROZESSE

MIB = 1024 * 1024

# Instanzname der PDH-Zaehler: pid_1234_luid_0x00000000_0x0000D229_phys_0
_PID_RE = re.compile(r"^pid_(\d+)_")


@dataclass
class GpuMessung:
    gpu_util_pct: int | None = None
    gpu_mem_util_pct: int | None = None
    vram_used_mib: int | None = None
    vram_free_mib: int | None = None
    gpu_temp_c: int | None = None
    gpu_clock_sm_mhz: int | None = None
    gpu_clock_mem_mhz: int | None = None
    gpu_power_w: float | None = None

    # Auf die NVML-Gesamtsumme normiert, siehe Modulkopf.
    vram_llama_mib: int = 0
    vram_ollama_mib: int = 0
    vram_fremd_mib: int = 0
    prozesse: list[dict] = field(default_factory=list)
    # Nachvollziehbarkeit der Normierung.
    vram_roh_summe_mib: int = 0
    zuordnung_faktor: float = 1.0

    # True, solange die PDH-Quelle Daten liefert. Faellt sie aus, sind die
    # drei Zuordnungsfelder oben 0 und duerfen nicht als "nichts los" gelesen werden.
    zuordnung_verfuegbar: bool = False


class GpuMonitor:
    """Haelt NVML- und PDH-Handles offen. Eine Instanz je Prozess."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._nvml_ok = False
        self._handle = None
        self._pdh_query = None
        self._pdh_counter = None
        self._pdh_ok = False
        # PID -> Prozessname. GPU-Prozesse wechseln selten, psutil.Process() je
        # Sekunde fuer 58 Instanzen waere unnoetig teuer.
        self._namen: dict[int, str] = {}

        self._nvml_start()
        self._pdh_start()

    # --- NVML ---------------------------------------------------------------

    def _nvml_start(self) -> None:
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml_ok = True
        except Exception:
            self._nvml_ok = False

    def _nvml_lesen(self, m: GpuMessung) -> None:
        if not self._nvml_ok:
            return
        h = self._handle
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            m.vram_used_mib = mem.used // MIB
            m.vram_free_mib = mem.free // MIB
        except Exception:
            pass
        try:
            u = pynvml.nvmlDeviceGetUtilizationRates(h)
            m.gpu_util_pct = u.gpu
            m.gpu_mem_util_pct = u.memory
        except Exception:
            pass
        try:
            m.gpu_temp_c = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            pass
        try:
            m.gpu_power_w = round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0, 1)
        except Exception:
            pass
        try:
            m.gpu_clock_sm_mhz = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
            m.gpu_clock_mem_mhz = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM)
        except Exception:
            pass

    # --- PDH ----------------------------------------------------------------

    def _pdh_start(self) -> None:
        try:
            import win32pdh

            self._pdh_query = win32pdh.OpenQuery()
            self._pdh_counter = win32pdh.AddCounter(
                self._pdh_query, r"\GPU Process Memory(*)\Dedicated Usage"
            )
            # PDH braucht eine erste Sammlung, bevor formatierte Werte kommen.
            win32pdh.CollectQueryData(self._pdh_query)
            self._pdh_ok = True
        except Exception:
            self._pdh_ok = False

    def _prozessname(self, pid: int) -> str:
        name = self._namen.get(pid)
        if name is not None:
            return name
        try:
            name = psutil.Process(pid).name()
        except Exception:
            name = f"pid {pid}"
        self._namen[pid] = name
        return name

    def _pdh_lesen(self, m: GpuMessung) -> None:
        if not self._pdh_ok:
            return
        try:
            import win32pdh

            win32pdh.CollectQueryData(self._pdh_query)
            roh = win32pdh.GetFormattedCounterArray(self._pdh_counter, win32pdh.PDH_FMT_LARGE)
        except Exception:
            # Der Zaehlersatz kann kurzzeitig verschwinden, etwa beim Treiberwechsel.
            # Kein Grund, den Sampler zu beenden - beim naechsten Takt erneut versuchen.
            return

        je_pid: dict[int, int] = {}
        for instanz, wert in (roh.items() if isinstance(roh, dict) else roh):
            treffer = _PID_RE.match(instanz)
            if not treffer or not wert:
                continue
            pid = int(treffer.group(1))
            # Mehrere phys_N-Instanzen je PID: die groesste zaehlt, nicht die Summe.
            # Summieren wuerde denselben Speicher mehrfach zaehlen.
            je_pid[pid] = max(je_pid.get(pid, 0), int(wert))

        lebende = set(psutil.pids())
        self._namen = {p: n for p, n in self._namen.items() if p in lebende}

        roh_summe = sum(je_pid.values()) // MIB
        if roh_summe <= 0:
            return

        # PDH liefert nur die Aufteilung, NVML die Summe. Ohne NVML-Wert bleibt
        # der Rohmassstab - dann ist die Zahl zwar zu gross, aber die einzige,
        # die es gibt.
        faktor = (m.vram_used_mib / roh_summe) if m.vram_used_mib else 1.0

        llama = ollama = fremd = 0.0
        prozesse: list[dict] = []
        for pid, byte in sorted(je_pid.items(), key=lambda kv: kv[1], reverse=True):
            roh_mib = byte // MIB
            if roh_mib <= 0:
                continue
            mib = roh_mib * faktor
            name = self._prozessname(pid)
            prozesse.append(
                {"pid": pid, "name": name, "mib": round(mib), "mib_roh": roh_mib}
            )
            if name == "llama-server.exe":
                llama += mib
            elif name in EIGENE_PROZESSE:
                ollama += mib
            else:
                fremd += mib

        m.vram_llama_mib = round(llama)
        m.vram_ollama_mib = round(ollama)
        m.vram_fremd_mib = round(fremd)
        m.vram_roh_summe_mib = roh_summe
        m.zuordnung_faktor = round(faktor, 4)
        # Nur die groessten Verbraucher speichern - 58 Instanzen je Sekunde
        # waeren ueber Stunden mehr Text als Messwerte.
        m.prozesse = prozesse[:10]
        m.zuordnung_verfuegbar = True

    # --- oeffentlich --------------------------------------------------------

    def messen(self) -> GpuMessung:
        m = GpuMessung()
        with self._lock:
            self._nvml_lesen(m)
            self._pdh_lesen(m)
        return m

    def schliessen(self) -> None:
        with self._lock:
            if self._pdh_query is not None:
                try:
                    import win32pdh

                    win32pdh.CloseQuery(self._pdh_query)
                except Exception:
                    pass
                self._pdh_query = None
                self._pdh_ok = False
            if self._nvml_ok:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
                self._nvml_ok = False
