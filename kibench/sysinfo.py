"""CPU, RAM, Pagefile und Disk-I/O ueber psutil."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import psutil

GIB = 1024 ** 3
MIB = 1024 * 1024


@dataclass
class SysMessung:
    cpu_pct: float | None = None
    cpu_kerne: list[float] = field(default_factory=list)
    ram_used_gib: float | None = None
    ram_pct: float | None = None
    pagefile_used_gib: float | None = None
    pagefile_pct: float | None = None
    disk_read_mibs: float | None = None
    disk_write_mibs: float | None = None


class SysMonitor:
    """psutil-Aufrufe mit den beiden Zustaenden, die sie brauchen.

    `cpu_percent` misst immer gegen den vorigen Aufruf. Der erste Aufruf liefert
    deshalb 0,0 und wird im Konstruktor verbraucht, damit die erste echte Messung
    schon stimmt. Disk-I/O ist ein Zaehler seit Systemstart und muss selbst
    abgeleitet werden.
    """

    def __init__(self) -> None:
        psutil.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None, percpu=True)
        self._letzte_disk = psutil.disk_io_counters()
        self._letzte_zeit = time.monotonic()

    def messen(self) -> SysMessung:
        m = SysMessung()
        try:
            m.cpu_pct = psutil.cpu_percent(interval=None)
            m.cpu_kerne = [round(v, 1) for v in psutil.cpu_percent(interval=None, percpu=True)]
        except Exception:
            pass
        try:
            vm = psutil.virtual_memory()
            m.ram_used_gib = round(vm.used / GIB, 2)
            m.ram_pct = vm.percent
        except Exception:
            pass
        try:
            sw = psutil.swap_memory()
            m.pagefile_used_gib = round(sw.used / GIB, 2)
            m.pagefile_pct = sw.percent
        except Exception:
            pass
        try:
            jetzt = time.monotonic()
            d = psutil.disk_io_counters()
            dt = jetzt - self._letzte_zeit
            if d and self._letzte_disk and dt > 0:
                m.disk_read_mibs = round((d.read_bytes - self._letzte_disk.read_bytes) / MIB / dt, 2)
                m.disk_write_mibs = round((d.write_bytes - self._letzte_disk.write_bytes) / MIB / dt, 2)
            self._letzte_disk = d
            self._letzte_zeit = jetzt
        except Exception:
            pass
        return m
