"""Ein frisches, abgeschottetes Arbeitsverzeichnis je Aufgabe und Durchgang.

Zwei Sperren, weil es zwei Wege nach draussen gibt:

1. **Der Runner schreibt, was das Modell verlangt.** Ein Modell, das
   `..\\..\\wichtig.txt` als Zielpfad nennt, bekommt hier ein Nein - `schreiben()`
   prueft den Pfad, bevor irgendetwas passiert. Das ist die haeufigere Richtung:
   nicht boeser Wille, sondern ein Modell, das den Arbeitsordner nicht begriffen
   hat.

2. **Der generierte Code laeuft.** Dagegen hilft kein Pfadcheck im Runner, denn
   der Code macht, was er will. Dafuer gibt es `wachhund.py`, das im
   Unterprozess einen Audit-Hook setzt.

Verzeichnisse liegen unter `runs\\<run_id>\\<aufgabe>_<durchgang>\\` und werden
nach dem Lauf verworfen - ausser der Lauf ist gescheitert, dann bleiben sie
liegen. Ein Fehlschlag, den man nicht mehr ansehen kann, ist nur halb
protokolliert.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import REPO_DIR

RUNS_DIR = REPO_DIR / "runs"

# Vorschlag aus Prompt C. Grosszuegig, weil ein dense-Modell bei 7 tok/s fuer
# eine mehrstufige Aufgabe lange braucht - der Timeout soll Endlosschleifen
# fangen, nicht langsame Modelle bestrafen.
TIMEOUT_STANDARD_S = 600.0

# Grenze fuer eine einzelne vom Modell geschriebene Datei. Ein Modell, das in
# eine Endlosschleife geraet, schreibt sonst die Platte voll.
MAX_DATEI_BYTES = 1_000_000


class SandboxFehler(Exception):
    """Der Runner wollte etwas tun, das die Sandbox nicht zulaesst."""


@dataclass
class TestErgebnis:
    bestanden: bool = False
    exit_code: int | None = None
    dauer_s: float = 0.0
    ausgabe: str = ""
    verstoesse: list[str] = field(default_factory=list)
    timeout: bool = False
    fehler: str | None = None


class Sandbox:
    def __init__(
        self,
        run_id: int,
        aufgabe: str,
        durchgang: int = 1,
        wurzel: Path | None = None,
    ) -> None:
        basis = wurzel or RUNS_DIR
        self.verzeichnis = (basis / f"run_{run_id}" / f"{aufgabe}_{durchgang}").resolve()
        self._wurzel_aufloesung: Path | None = None

    # --- Aufbau und Abbau ---------------------------------------------------

    def anlegen(self, startdateien: dict[str, str] | None = None) -> Path:
        """Frisches Verzeichnis. Ein vorhandenes wird geloescht, nicht ergaenzt.

        Sonst sieht ein Wiederholungsdurchgang die Loesung des vorherigen und
        misst etwas anderes als gedacht.
        """
        if self.verzeichnis.exists():
            shutil.rmtree(self.verzeichnis, ignore_errors=True)
        self.verzeichnis.mkdir(parents=True, exist_ok=True)
        # Einmal aufloesen und merken: unter Windows ist der Pfad ueber
        # AppData\Local\Temp oft ein Symlink, und ein Vergleich gegen den
        # nicht aufgeloesten Pfad wuerde jeden Schreibzugriff ablehnen.
        self._wurzel_aufloesung = self.verzeichnis.resolve()
        for pfad, inhalt in (startdateien or {}).items():
            self.schreiben(pfad, inhalt)
        return self.verzeichnis

    def verwerfen(self) -> None:
        shutil.rmtree(self.verzeichnis, ignore_errors=True)

    # --- Pfadpruefung -------------------------------------------------------

    def _pruefen(self, pfad: str) -> Path:
        wurzel = self._wurzel_aufloesung or self.verzeichnis.resolve()
        ziel = (self.verzeichnis / pfad).resolve()
        if ziel != wurzel and wurzel not in ziel.parents:
            raise SandboxFehler(f"Pfad liegt ausserhalb der Sandbox: {pfad!r}")
        return ziel

    # --- Dateien ------------------------------------------------------------

    def schreiben(self, pfad: str, inhalt: str) -> Path:
        if len(inhalt.encode("utf-8")) > MAX_DATEI_BYTES:
            raise SandboxFehler(
                f"Datei zu gross ({len(inhalt)} Zeichen), Grenze {MAX_DATEI_BYTES} Byte"
            )
        ziel = self._pruefen(pfad)
        ziel.parent.mkdir(parents=True, exist_ok=True)
        ziel.write_text(inhalt, encoding="utf-8")
        return ziel

    def lesen(self, pfad: str) -> str:
        ziel = self._pruefen(pfad)
        if not ziel.is_file():
            raise SandboxFehler(f"Datei nicht gefunden: {pfad!r}")
        return ziel.read_text(encoding="utf-8", errors="replace")

    def dateien(self) -> list[str]:
        """Alle Dateien relativ zur Sandbox - fuer Protokoll und Modellkontext."""
        if not self.verzeichnis.exists():
            return []
        return sorted(
            str(p.relative_to(self.verzeichnis)).replace("\\", "/")
            for p in self.verzeichnis.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and ".tmp" not in p.parts
        )

    # --- Tests --------------------------------------------------------------

    def tests_fahren(
        self, testpfad: str = "tests", timeout_s: float = TIMEOUT_STANDARD_S
    ) -> TestErgebnis:
        """pytest in einem eigenen Prozess mit gesetzter Sperre.

        Der Rueckgabewert unterscheidet drei Faelle, die man nicht vermischen
        darf: bestanden, nicht bestanden, und gar nicht erst gelaufen. Der
        dritte ist ein Ergebnis - Prompt C, Fallstrick 4: ein Timeout ist ein
        Datenpunkt, kein fehlender.
        """
        from .sandbox_start import MARKE

        e = TestErgebnis()
        t0 = time.perf_counter()

        # pytest schreibt beim Start in das Temp-Verzeichnis des Systems - das
        # liegt ausserhalb der Sandbox, und die Sperre hat den ersten Versuch
        # prompt abgefangen. Das Temp-Verzeichnis deswegen freizugeben waere die
        # falsche Richtung: dann duerfte auch generierter Code dorthin. Also
        # bekommt der Unterprozess ein eigenes Temp *in* der Sandbox.
        temp = self.verzeichnis / ".tmp"
        temp.mkdir(parents=True, exist_ok=True)
        umgebung = dict(os.environ)
        umgebung.update({
            "TMP": str(temp), "TEMP": str(temp), "TMPDIR": str(temp),
            "PYTHONDONTWRITEBYTECODE": "1",
            # Kein Proxy, keine Telemetrie - das Netz ist ohnehin gesperrt,
            # aber ein Timeout beim Verbindungsversuch kostet trotzdem Zeit.
            "NO_PROXY": "*", "HTTP_PROXY": "", "HTTPS_PROXY": "",
        })

        try:
            p = subprocess.run(
                [sys.executable, "-m", "kibench.sandbox_start",
                 str(self.verzeichnis), testpfad],
                cwd=str(REPO_DIR),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
                env=umgebung,
            )
            e.ausgabe = (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr else "")
        except subprocess.TimeoutExpired as t:
            e.timeout = True
            e.fehler = f"Timeout nach {timeout_s:.0f}s"
            e.ausgabe = (t.stdout or b"").decode("utf-8", "replace") if isinstance(
                t.stdout, bytes) else (t.stdout or "")
            e.dauer_s = round(time.perf_counter() - t0, 2)
            return e

        e.dauer_s = round(time.perf_counter() - t0, 2)

        for zeile in reversed(e.ausgabe.splitlines()):
            if zeile.startswith(MARKE):
                try:
                    d = json.loads(zeile[len(MARKE):])
                except json.JSONDecodeError:
                    break
                e.bestanden = bool(d.get("bestanden"))
                e.exit_code = d.get("exit_code")
                e.verstoesse = list(d.get("verstoesse") or [])
                return e

        # Keine Ergebniszeile: der Unterprozess ist unterwegs gestorben. Das als
        # "nicht bestanden" zu verbuchen waere richtig, aber zu wenig - es ist
        # ein anderer Fall als ein roter Test, und die Auswertung muss ihn
        # unterscheiden koennen.
        e.fehler = "Unterprozess lieferte kein Ergebnis"
        e.exit_code = p.returncode
        return e
