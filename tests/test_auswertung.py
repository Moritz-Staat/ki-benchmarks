"""Aggregation und die Auswertungs-Endpunkte.

Der wichtigste Fall hier ist der, den Prompt C ausdruecklich verlangt: Eine
Aufgabe, die mal geloest wird und mal nicht, darf nicht als "50 %" verschwinden.
Sie muss als **manchmal** sichtbar bleiben.
"""
from __future__ import annotations

import time

from kibench import auswertung


def _lauf(conn, alias="qwen-moe", thinking=1, notiz="rauchtest", quant="UD-Q4_K_XL"):
    cur = conn.execute(
        """INSERT INTO runs (modell_alias, modell_name, runtime, quant, offload,
                             kontext, thinking, ts_start, ts_ende, status, notiz)
           VALUES (?,?,?,?,?,?,?,?,?,'fertig',?)""",
        (alias, alias, "llama.cpp", quant, "-ngl 99", 32768, thinking,
         time.time() - 60, time.time(), notiz),
    )
    conn.commit()
    return cur.lastrowid


def _aufgabe(conn, schluessel, schwierigkeit="leicht", kategorie="bugfix"):
    conn.execute(
        """INSERT INTO tasks (schluessel, kategorie, titel, prompt, aktiv,
                              schwierigkeit, erwartete_schritte)
           VALUES (?,?,?,?,1,?,2)""",
        (schluessel, kategorie, f"Titel {schluessel}", "Mach was", schwierigkeit),
    )
    conn.commit()
    return conn.execute("SELECT id FROM tasks WHERE schluessel = ?",
                        (schluessel,)).fetchone()["id"]


def _ergebnis(conn, run_id, task_id, durchgang, erfolg, **kw):
    felder = {
        "gen_tps": 50.0, "dauer_s": 12.0, "tool_calls_erhalten": 3,
        "tool_calls_korrekt": 3, "messpunkt_gueltig": 1, "nachladen_s": 0.1,
        "schritte": 3, "denk_tokens": 200, "antwort_tokens": 100, "timeout": 0,
    }
    felder.update(kw)
    spalten = ", ".join(felder)
    platz = ", ".join("?" * len(felder))
    conn.execute(
        f"""INSERT INTO results (run_id, task_id, durchgang, erfolg, ts_start,
                                 ts_ende, {spalten})
            VALUES (?,?,?,?,?,?,{platz})""",
        (run_id, task_id, durchgang, int(erfolg), time.time() - 30, time.time(),
         *felder.values()),
    )
    conn.commit()


# --- Streuung ---------------------------------------------------------------

def test_schwankende_aufgabe_bleibt_als_manchmal_sichtbar(test_db):
    run = _lauf(test_db)
    sicher = _aufgabe(test_db, "immer-gut")
    wackelig = _aufgabe(test_db, "mal-so-mal-so")
    hoffnungslos = _aufgabe(test_db, "nie-geschafft", schwierigkeit="schwer")

    for d in (1, 2, 3):
        _ergebnis(test_db, run, sicher, d, True)
        _ergebnis(test_db, run, hoffnungslos, d, False)
    _ergebnis(test_db, run, wackelig, 1, True)
    _ergebnis(test_db, run, wackelig, 2, False)
    _ergebnis(test_db, run, wackelig, 3, True)

    k = auswertung.vergleich(test_db)["konfigurationen"][0]
    assert k["aufgaben_immer"] == 1
    assert k["aufgaben_manchmal"] == 1
    assert k["aufgaben_nie"] == 1
    # Der Mittelwert allein wuerde alle drei Faelle zu einer Zahl verruehren.
    assert k["erfolgsquote"] == round(5 / 9, 3)


def test_erfolgsquote_je_schwierigkeit(test_db):
    run = _lauf(test_db)
    leicht = _aufgabe(test_db, "leicht-1", schwierigkeit="leicht")
    schwer = _aufgabe(test_db, "schwer-1", schwierigkeit="schwer")
    _ergebnis(test_db, run, leicht, 1, True)
    _ergebnis(test_db, run, schwer, 1, False)

    k = auswertung.vergleich(test_db)["konfigurationen"][0]
    assert k["je_schwierigkeit"]["leicht"]["quote"] == 1.0
    assert k["je_schwierigkeit"]["schwer"]["quote"] == 0.0


# --- Ungueltige Messpunkte --------------------------------------------------

def test_verworfene_messpunkte_zaehlen_beim_erfolg_aber_nicht_beim_tempo(test_db):
    """Der Befund aus Prompt B als Auswertungsregel.

    Ein Lauf mit Nachladen misst die Wartezeit des Treibers, nicht das Modell -
    fuer tok/s also unbrauchbar. Geloest hat das Modell die Aufgabe trotzdem.
    """
    run = _lauf(test_db)
    t = _aufgabe(test_db, "eine-aufgabe")
    _ergebnis(test_db, run, t, 1, True, gen_tps=60.0, dauer_s=10.0)
    _ergebnis(test_db, run, t, 2, True, gen_tps=3.0, dauer_s=200.0,
              messpunkt_gueltig=0, nachladen_s=17.4)

    k = auswertung.vergleich(test_db)["konfigurationen"][0]
    assert k["erfolgsquote"] == 1.0          # beide Durchgaenge waren erfolgreich
    assert k["messpunkte_verworfen"] == 1
    assert k["gen_tps_median"] == 60.0       # der langsame Wert faellt heraus
    assert k["dauer_median_s"] == 10.0


# --- Trennschaerfe ----------------------------------------------------------

def test_aufgabe_die_alle_loesen_trennt_nicht(test_db):
    schnell = _lauf(test_db, alias="qwen-moe")
    langsam = _lauf(test_db, alias="qwen-dense")
    t = _aufgabe(test_db, "zu-leicht")
    _ergebnis(test_db, schnell, t, 1, True)
    _ergebnis(test_db, langsam, t, 1, True)

    a = auswertung.vergleich(test_db)["aufgaben"][0]
    assert a["trennt"] is False


def test_aufgabe_mit_unterschied_trennt(test_db):
    schnell = _lauf(test_db, alias="qwen-moe")
    klein = _lauf(test_db, alias="llama3.2:3b")
    t = _aufgabe(test_db, "unterscheidet", schwierigkeit="schwer")
    _ergebnis(test_db, schnell, t, 1, True)
    _ergebnis(test_db, klein, t, 1, False)

    a = auswertung.vergleich(test_db)["aufgaben"][0]
    assert a["trennt"] is True
    assert a["quote_min"] == 0.0 and a["quote_max"] == 1.0


def test_thinking_und_non_thinking_bleiben_getrennt(test_db):
    mit = _lauf(test_db, alias="qwen-moe", thinking=1)
    ohne = _lauf(test_db, alias="qwen-moe", thinking=0)
    t = _aufgabe(test_db, "denkprobe")
    _ergebnis(test_db, mit, t, 1, True, denk_tokens=2000)
    _ergebnis(test_db, ohne, t, 1, False, denk_tokens=0)

    ks = auswertung.vergleich(test_db)["konfigurationen"]
    assert len(ks) == 2
    denkend = next(k for k in ks if k["thinking"])
    stumm = next(k for k in ks if not k["thinking"])
    assert denkend["erfolgsquote"] == 1.0 and stumm["erfolgsquote"] == 0.0
    assert denkend["denk_tokens_median"] == 2000


# --- Endpunkte --------------------------------------------------------------

def test_endpunkt_vergleich_ist_leer_aber_gueltig(client):
    d = client.get("/api/vergleich").json()
    assert d["konfigurationen"] == [] and d["aufgaben"] == []
    assert d["gesamt_ergebnisse"] == 0


def test_endpunkt_laeufe_liefert_liste(client):
    d = client.get("/api/laeufe").json()
    assert "laeufe" in d and isinstance(d["laeufe"], list)


def test_endpunkt_unbekannter_lauf_meldet_404(client):
    assert client.get("/api/lauf/9999").status_code == 404


def test_auswertungsseiten_werden_ausgeliefert(client):
    for pfad in ("/vergleich", "/lauf"):
        r = client.get(pfad)
        assert r.status_code == 200
        assert "ki-benchmarks" in r.text


def test_lauf_detail_verknuepft_hardwarekurve(test_db):
    """Der eigentliche Zweck der gemeinsamen Datenbank: ein SELECT statt zweier Systeme."""
    run = _lauf(test_db)
    t = _aufgabe(test_db, "mit-kurve")
    _ergebnis(test_db, run, t, 1, True)
    jetzt = time.time()
    for i in range(5):
        test_db.execute(
            "INSERT INTO samples (ts, ts_iso, vram_used_mib, gpu_util_pct) VALUES (?,?,?,?)",
            (jetzt - 30 + i, "x", 14000 + i, 50),
        )
    test_db.commit()

    d = auswertung.lauf_detail(test_db, run)
    assert d["lauf"]["id"] == run
    assert len(d["ergebnisse"]) == 1
    assert d["kurve_punkte_gesamt"] >= 5
    assert d["kurve"][0]["vram_used_mib"] >= 14000
