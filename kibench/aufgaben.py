"""Aufgaben von der Platte lesen und in die `tasks`-Tabelle spiegeln.

Eine Aufgabe ist ein Verzeichnis:

    suiten/<suite>/<schluessel>/
        aufgabe.json      Titel, Auftrag, Schwierigkeit, erwartete Schritte
        start/            Ausgangszustand - wird in die Sandbox kopiert
        tests/            die Testsuite, entscheidet ueber bestanden/nicht

**Die Tests liegen bewusst getrennt vom Startzustand.** Beides in einem Ordner
waere bequemer, aber dann sieht das Modell die Tests und kann sie umschreiben,
statt die Aufgabe zu loesen. Der Runner kopiert `start/` in die Sandbox und legt
`tests/` erst unmittelbar vor dem Pruefen dazu.

Die Dateien sind die Quelle der Wahrheit, nicht die Datenbank: `synchronisieren()`
liest die Verzeichnisse und schreibt sie in `tasks`. So bleibt eine Aufgabe
versionierbar, und ein Lauf von gestern behaelt trotzdem seine `task_id`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import REPO_DIR

SUITEN_DIR = REPO_DIR / "suiten"

SCHWIERIGKEITEN = ("leicht", "mittel", "schwer")


class AufgabenFehler(Exception):
    """Eine Aufgabe auf der Platte ist unvollstaendig oder widerspruechlich."""


@dataclass
class Aufgabe:
    schluessel: str
    titel: str
    prompt: str
    suite: str
    kategorie: str = "sonstiges"
    schwierigkeit: str = "mittel"
    erwartete_schritte: int = 1
    max_tokens: int = 4096
    timeout_s: float = 600.0
    erwartung: str | None = None
    quelle: str | None = None
    verzeichnis: Path | None = None
    startdateien: dict[str, str] = field(default_factory=dict)
    testdateien: dict[str, str] = field(default_factory=dict)

    @property
    def dateiliste(self) -> list[str]:
        return sorted(self.startdateien)


def _dateien_lesen(wurzel: Path) -> dict[str, str]:
    if not wurzel.is_dir():
        return {}
    ergebnis: dict[str, str] = {}
    for p in sorted(wurzel.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            rel = str(p.relative_to(wurzel)).replace("\\", "/")
            ergebnis[rel] = p.read_text(encoding="utf-8")
    return ergebnis


def aufgabe_laden(verzeichnis: Path, suite: str) -> Aufgabe:
    datei = verzeichnis / "aufgabe.json"
    if not datei.is_file():
        raise AufgabenFehler(f"{verzeichnis}: aufgabe.json fehlt")
    d = json.loads(datei.read_text(encoding="utf-8"))

    for pflicht in ("titel", "prompt"):
        if not d.get(pflicht):
            raise AufgabenFehler(f"{verzeichnis}: '{pflicht}' fehlt oder ist leer")

    schwierigkeit = d.get("schwierigkeit", "mittel")
    if schwierigkeit not in SCHWIERIGKEITEN:
        raise AufgabenFehler(
            f"{verzeichnis}: Schwierigkeit {schwierigkeit!r} unbekannt, "
            f"erlaubt sind {SCHWIERIGKEITEN}"
        )

    tests = _dateien_lesen(verzeichnis / "tests")
    if not tests:
        # Ohne Tests waere die Bewertung eine Geschmacksfrage - und Prompt C
        # verlangt ausdruecklich binaer: gruen oder rot.
        raise AufgabenFehler(f"{verzeichnis}: keine Tests, damit ist nichts messbar")

    return Aufgabe(
        schluessel=d.get("schluessel") or verzeichnis.name,
        titel=d["titel"],
        prompt=d["prompt"],
        suite=suite,
        kategorie=d.get("kategorie", "sonstiges"),
        schwierigkeit=schwierigkeit,
        erwartete_schritte=int(d.get("erwartete_schritte", 1)),
        max_tokens=int(d.get("max_tokens", 4096)),
        timeout_s=float(d.get("timeout_s", 600.0)),
        erwartung=d.get("erwartung"),
        quelle=d.get("quelle"),
        verzeichnis=verzeichnis,
        startdateien=_dateien_lesen(verzeichnis / "start"),
        testdateien=tests,
    )


def suite_laden(suite: str, wurzel: Path | None = None) -> list[Aufgabe]:
    basis = (wurzel or SUITEN_DIR) / suite
    if not basis.is_dir():
        raise AufgabenFehler(f"Suite nicht gefunden: {basis}")
    aufgaben = [
        aufgabe_laden(p, suite)
        for p in sorted(basis.iterdir())
        if p.is_dir() and not p.name.startswith(".")
    ]
    doppelt = {a.schluessel for a in aufgaben if
               sum(1 for b in aufgaben if b.schluessel == a.schluessel) > 1}
    if doppelt:
        raise AufgabenFehler(f"Doppelte Schluessel in {suite}: {sorted(doppelt)}")
    return aufgaben


def suiten_auflisten(wurzel: Path | None = None) -> list[str]:
    basis = wurzel or SUITEN_DIR
    if not basis.is_dir():
        return []
    return sorted(p.name for p in basis.iterdir() if p.is_dir() and not p.name.startswith("."))


def synchronisieren(conn, aufgaben: list[Aufgabe]) -> dict[str, int]:
    """Aufgaben in `tasks` spiegeln und die Zuordnung Schluessel -> id liefern.

    `schluessel` ist UNIQUE; ein zweiter Lauf aktualisiert also, statt zu
    verdoppeln. Damit bleiben `results` aus frueheren Laeufen zuordenbar, auch
    wenn der Aufgabentext inzwischen geschaerft wurde.
    """
    ids: dict[str, int] = {}
    for a in aufgaben:
        conn.execute(
            """
            INSERT INTO tasks (schluessel, kategorie, titel, prompt, erwartung,
                               max_tokens, quelle, aktiv, schwierigkeit,
                               erwartete_schritte, timeout_s)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
            ON CONFLICT(schluessel) DO UPDATE SET
                kategorie          = excluded.kategorie,
                titel              = excluded.titel,
                prompt             = excluded.prompt,
                erwartung          = excluded.erwartung,
                max_tokens         = excluded.max_tokens,
                quelle             = excluded.quelle,
                aktiv              = 1,
                schwierigkeit      = excluded.schwierigkeit,
                erwartete_schritte = excluded.erwartete_schritte,
                timeout_s          = excluded.timeout_s
            """,
            (a.schluessel, a.kategorie, a.titel, a.prompt, a.erwartung,
             a.max_tokens, a.quelle or a.suite, a.schwierigkeit,
             a.erwartete_schritte, a.timeout_s),
        )
        zeile = conn.execute(
            "SELECT id FROM tasks WHERE schluessel = ?", (a.schluessel,)
        ).fetchone()
        ids[a.schluessel] = zeile["id"]
    conn.commit()
    return ids


def verteilung(aufgaben: list[Aufgabe]) -> dict[str, int]:
    """Wie viele Aufgaben je Schwierigkeit? Prompt C verlangt etwa ein Drittel je Stufe."""
    z = {s: 0 for s in SCHWIERIGKEITEN}
    for a in aufgaben:
        z[a.schwierigkeit] = z.get(a.schwierigkeit, 0) + 1
    return z
