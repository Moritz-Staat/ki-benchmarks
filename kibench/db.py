"""SQLite-Zugriff.

Ein Schreiber (der Sampler), beliebig viele Leser (die API). Deshalb WAL:
Leser blockieren den Schreiber nicht und umgekehrt.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from .config import DB_PATH

SCHEMA_VERSION = "1"
_SCHEMA_SQL = Path(__file__).resolve().parent / "schema.sql"

# sqlite3-Verbindungen sind nicht thread-sicher. Jeder Thread bekommt seine eigene.
_lokal = threading.local()


def verbindung(pfad: Path | None = None) -> sqlite3.Connection:
    """Verbindung fuer den aufrufenden Thread, wird wiederverwendet."""
    pfad = pfad or DB_PATH
    vorhanden = getattr(_lokal, "conn", None)
    if vorhanden is not None and getattr(_lokal, "pfad", None) == str(pfad):
        return vorhanden

    conn = sqlite3.connect(str(pfad), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    _lokal.conn = conn
    _lokal.pfad = str(pfad)
    return conn


def init_db(pfad: Path | None = None) -> sqlite3.Connection:
    """Legt das Schema an. Idempotent - laeuft bei jedem Start."""
    pfad = pfad or DB_PATH
    pfad.parent.mkdir(parents=True, exist_ok=True)
    conn = verbindung(pfad)
    conn.executescript(_SCHEMA_SQL.read_text(encoding="utf-8"))
    conn.execute(
        "INSERT INTO meta(schluessel, wert) VALUES('schema_version', ?) "
        "ON CONFLICT(schluessel) DO UPDATE SET wert = excluded.wert",
        (SCHEMA_VERSION,),
    )
    conn.commit()
    return conn


def tabellen(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [r["name"] for r in rows]


def spalten(conn: sqlite3.Connection, tabelle: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({tabelle})").fetchall()
    return [r["name"] for r in rows]


def sample_schreiben(conn: sqlite3.Connection, werte: dict[str, Any]) -> None:
    """Ein Messpunkt. Die Spaltenliste kommt aus dem Dict, nicht aus einer
    zweiten Aufzaehlung - so kann das Schema wachsen, ohne dass hier etwas bricht."""
    felder = list(werte.keys())
    platzhalter = ", ".join("?" for _ in felder)
    sql = f"INSERT INTO samples ({', '.join(felder)}) VALUES ({platzhalter})"
    conn.execute(sql, [werte[f] for f in felder])


def letzter_sample(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM samples ORDER BY ts DESC LIMIT 1").fetchone()


def samples_zeitraum(
    conn: sqlite3.Connection, ab_ts: float, spalten_liste: list[str], max_punkte: int = 1200
) -> list[sqlite3.Row]:
    """Messpunkte ab einem Zeitpunkt, gleichmaessig ausgeduennt.

    Ohne Ausduennen liefert "heute" 86400 Punkte an den Browser. Das Ausduennen
    passiert in SQL ueber die Zeilennummer, damit nicht erst alles geladen wird.
    """
    gesamt = conn.execute("SELECT COUNT(*) AS n FROM samples WHERE ts >= ?", (ab_ts,)).fetchone()["n"]
    schritt = max(1, gesamt // max_punkte)
    felder = ", ".join(spalten_liste)
    sql = f"""
        SELECT {felder} FROM (
            SELECT {felder}, ROW_NUMBER() OVER (ORDER BY ts) AS rn
            FROM samples WHERE ts >= ?
        ) WHERE rn % ? = 0 ORDER BY ts
    """
    return conn.execute(sql, (ab_ts, schritt)).fetchall()
