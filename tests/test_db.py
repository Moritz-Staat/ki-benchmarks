"""Das Schema muss vollstaendig sein - auch die Tabellen, die erst Prompt C fuellt.

Genau das ist eine Abnahmebedingung aus Prompt B, Teil 2: jetzt anlegen, damit
spaeter keine Migration noetig wird.
"""
from __future__ import annotations

import time

from kibench import db

ERWARTETE_TABELLEN = {"samples", "runs", "tasks", "results", "raw_logs", "meta"}


def test_alle_tabellen_vorhanden(test_db):
    assert ERWARTETE_TABELLEN <= set(db.tabellen(test_db))


def test_schema_ist_idempotent(test_db, tmp_path):
    """init_db laeuft bei jedem Start - zweimal darf nichts kaputt machen."""
    vorher = db.tabellen(test_db)
    db.init_db(tmp_path / "test.db")
    assert db.tabellen(test_db) == vorher


def test_samples_hat_die_pflichtfelder(test_db):
    spalten = set(db.spalten(test_db, "samples"))
    # Die Felder, ohne die das Dashboard nichts anzeigen kann.
    for feld in (
        "ts", "vram_used_mib", "vram_free_mib", "gpu_temp_c", "gpu_util_pct",
        "cpu_pct", "ram_used_gib", "llama_gen_tps", "llama_prompt_tps",
        "llama_kv_cache_pct", "llama_model",
    ):
        assert feld in spalten, feld


def test_fremd_vram_ist_pflichtfeld(test_db):
    """Prompt B nennt den Fremd-VRAM ausdruecklich 'nicht optional'."""
    spalten = set(db.spalten(test_db, "samples"))
    assert {"vram_fremd_mib", "vram_llama_mib", "vram_ollama_mib", "vram_prozesse_json"} <= spalten


def test_prompt_c_tabellen_haben_ihre_felder(test_db):
    """Damit Prompt C nicht doch migrieren muss."""
    assert {"modell_alias", "offload", "thinking", "status"} <= set(db.spalten(test_db, "runs"))
    assert {"schluessel", "prompt", "erwartung", "werkzeuge_json"} <= set(db.spalten(test_db, "tasks"))
    ergebnis = set(db.spalten(test_db, "results"))
    assert {"run_id", "task_id", "durchgang", "erfolg", "dauer_s"} <= ergebnis
    # Die Trennung von Denk- und Antwort-Token ist der Kern von Fallstrick 3.
    assert {"denk_tokens", "antwort_tokens", "fehlergrund"} <= ergebnis
    assert {"tool_calls_erhalten", "tool_call_fliesstext"} <= ergebnis
    assert {"result_id", "richtung", "inhalt"} <= set(db.spalten(test_db, "raw_logs"))


def test_sample_schreiben_und_lesen(test_db):
    jetzt = time.time()
    db.sample_schreiben(test_db, {
        "ts": jetzt, "ts_iso": "2026-08-16T12:00:00",
        "vram_used_mib": 15116, "ram_used_gib": 20.1, "llama_alive": 1,
    })
    test_db.commit()
    zeile = db.letzter_sample(test_db)
    assert zeile["vram_used_mib"] == 15116
    assert zeile["llama_alive"] == 1
    # Nicht gesetzte Spalten muessen NULL sein, nicht 0 - sonst ist
    # "nicht gemessen" von "gemessen: null" nicht unterscheidbar.
    assert zeile["gpu_temp_c"] is None


def test_zeitraum_duennt_aus(test_db):
    """500 Messpunkte duerfen nicht als 500 Punkte an den Browser gehen."""
    jetzt = time.time()
    for i in range(500):
        db.sample_schreiben(test_db, {
            "ts": jetzt - 500 + i, "ts_iso": "x", "vram_used_mib": i,
        })
    test_db.commit()
    zeilen = db.samples_zeitraum(test_db, jetzt - 600, ["ts", "vram_used_mib"], max_punkte=50)
    assert 0 < len(zeilen) <= 60
    # Und die Reihenfolge muss stimmen, sonst zeichnet uPlot Zickzack.
    ts = [z["ts"] for z in zeilen]
    assert ts == sorted(ts)
