"""Der Runner gegen einen erfundenen Adapter.

Kein Modell, keine GPU: geprueft wird die Mechanik. Ein echtes Modell wuerde
hier nichts beweisen, was ein Skript nicht auch beweist - und es waere nicht
wiederholbar. Die Modelle kommen im Probelauf dran.
"""
from __future__ import annotations

import json

from kibench.aufgaben import suite_laden, synchronisieren
from kibench.runner import Runner


class FakeAntwort:
    """Genug von `AntwortMessung`, damit der Runner damit arbeiten kann."""

    def __init__(self, tool_calls=None, text="", timings=True, erfolg=True):
        self.tool_calls = tool_calls or []
        self.text = text
        self.denktext = ""
        self.erfolg = erfolg or bool(tool_calls)
        self.fehlergrund = None if self.erfolg else "leere_antwort"
        self.finish_reason = "stop"
        self.prompt_tokens = 100
        self.antwort_tokens = 50
        self.denk_tokens = 0
        self.gesamt_tokens = 150
        self.dauer_s = 0.2
        self.gen_tps = 250.0
        self.tool_call_fliesstext = False
        self.roh = {
            "choices": [{"message": {"content": text, "tool_calls": self.tool_calls}}],
        }
        if timings:
            # llama.cpp haengt das an; damit ist der Nachladeanteil rechenbar.
            self.roh["timings"] = {"prompt_ms": 100.0, "predicted_ms": 90.0}


def _aufruf(name, **args):
    return {"id": f"call_{name}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


class SkriptAdapter:
    """Spielt eine feste Folge von Antworten ab."""

    def __init__(self, antworten):
        self.antworten = list(antworten)
        self.anfragen_gesehen = []

    def anfragen(self, alias, nachrichten, **kw):
        self.anfragen_gesehen.append(nachrichten[-1])
        if self.antworten:
            return self.antworten.pop(0)
        return FakeAntwort(text="Ich bin fertig.")


def _aufgabe(schluessel, suite="rauchtest"):
    return next(a for a in suite_laden(suite) if a.schluessel == schluessel)


def _lauf(test_db, tmp_path, adapter, aufgabe, alias="qwen-moe"):
    ids = synchronisieren(test_db, [aufgabe])
    r = Runner(adapter, conn=test_db, sandbox_wurzel=tmp_path)
    run_id = r.lauf_anlegen(alias, thinking=False, suite=aufgabe.suite)
    erg = r.aufgabe_fahren(run_id, aufgabe, ids[aufgabe.schluessel], 1, alias, thinking=False)
    return r, run_id, erg


# --- Der gute Fall ----------------------------------------------------------

def test_geloeste_aufgabe_wird_als_erfolg_gebucht(test_db, tmp_path):
    a = _aufgabe("summe-bugfix")
    adapter = SkriptAdapter([
        FakeAntwort([_aufruf("datei_lesen", pfad="rechner.py")]),
        FakeAntwort([_aufruf("datei_schreiben", pfad="rechner.py",
                             inhalt="def addiere(a, b):\n    return a + b\n\n"
                                    "def multipliziere(a, b):\n    return a * b\n")]),
        FakeAntwort([_aufruf("fertig", zusammenfassung="Vorzeichen korrigiert")]),
    ])
    _, run_id, erg = _lauf(test_db, tmp_path, adapter, a)

    assert erg.erfolg is True
    assert erg.fehlergrund is None
    assert erg.statistik.korrekt == 3
    assert erg.statistik.erfundene_namen == []
    zeile = test_db.execute("SELECT * FROM results WHERE run_id = ?", (run_id,)).fetchone()
    assert zeile["erfolg"] == 1
    assert zeile["schritte"] == 3


def test_falsche_loesung_wird_als_misserfolg_gebucht(test_db, tmp_path):
    a = _aufgabe("summe-bugfix")
    adapter = SkriptAdapter([
        FakeAntwort([_aufruf("datei_schreiben", pfad="rechner.py",
                             inhalt="def addiere(a, b):\n    return a * b\n\n"
                                    "def multipliziere(a, b):\n    return a * b\n")]),
        FakeAntwort([_aufruf("fertig")]),
    ])
    _, _, erg = _lauf(test_db, tmp_path, adapter, a)
    assert erg.erfolg is False
    assert erg.fehlergrund == "tests_rot"


# --- Werkzeugfehler ---------------------------------------------------------

def test_erfundener_werkzeugname_wird_gezaehlt(test_db, tmp_path):
    a = _aufgabe("summe-bugfix")
    adapter = SkriptAdapter([
        FakeAntwort([_aufruf("datei_patchen", pfad="rechner.py")]),
        FakeAntwort([_aufruf("datei_schreiben", pfad="rechner.py",
                             inhalt="def addiere(a, b):\n    return a + b\n\n"
                                    "def multipliziere(a, b):\n    return a * b\n")]),
        FakeAntwort([_aufruf("fertig")]),
    ])
    _, _, erg = _lauf(test_db, tmp_path, adapter, a)
    assert erg.statistik.erfundene_namen == ["datei_patchen"]
    assert erg.erfolg is True     # der Fehltritt allein macht die Loesung nicht kaputt


def test_fehlende_pflichtfelder_werden_gezaehlt(test_db, tmp_path):
    a = _aufgabe("summe-bugfix")
    adapter = SkriptAdapter([
        FakeAntwort([_aufruf("datei_schreiben", pfad="rechner.py")]),   # inhalt fehlt
        FakeAntwort([_aufruf("fertig")]),
    ])
    _, _, erg = _lauf(test_db, tmp_path, adapter, a)
    assert erg.statistik.parameterfehler == 1


def test_endlosschleife_wird_abgebrochen(test_db, tmp_path):
    """Prompt C nennt 'derselbe Call mehrfach' ausdruecklich als Fehlerbild."""
    a = _aufgabe("summe-bugfix")
    gleich = [FakeAntwort([_aufruf("datei_lesen", pfad="rechner.py")]) for _ in range(10)]
    adapter = SkriptAdapter(gleich)
    _, _, erg = _lauf(test_db, tmp_path, adapter, a)
    assert erg.fehlergrund == "endlosschleife_werkzeug"
    assert erg.statistik.wiederholungen >= 3


def test_antwort_ohne_werkzeug_bricht_ab(test_db, tmp_path):
    a = _aufgabe("summe-bugfix")
    adapter = SkriptAdapter([
        FakeAntwort(text="Hier ist der Code: def addiere(a, b): return a + b"),
        FakeAntwort(text="Wie gesagt, einfach das Vorzeichen tauschen."),
    ])
    _, _, erg = _lauf(test_db, tmp_path, adapter, a)
    assert erg.fehlergrund == "antwortet_ohne_werkzeug"
    assert erg.statistik.abbruch_ohne_ergebnis is True
    assert erg.erfolg is False


def test_pfad_ausserhalb_wird_dem_modell_zurueckgemeldet(test_db, tmp_path):
    """Kein Absturz: das Modell soll die Chance bekommen, es richtig zu machen."""
    a = _aufgabe("summe-bugfix")
    adapter = SkriptAdapter([
        FakeAntwort([_aufruf("datei_schreiben", pfad="../../boese.py", inhalt="x = 1")]),
        FakeAntwort([_aufruf("datei_schreiben", pfad="rechner.py",
                             inhalt="def addiere(a, b):\n    return a + b\n\n"
                                    "def multipliziere(a, b):\n    return a * b\n")]),
        FakeAntwort([_aufruf("fertig")]),
    ])
    _, _, erg = _lauf(test_db, tmp_path, adapter, a)
    assert erg.erfolg is True
    assert not (tmp_path / "boese.py").exists()


# --- Nachladen und Messpunktgueltigkeit -------------------------------------

def test_ohne_server_timings_gilt_nachladen_als_ungeprueft(test_db, tmp_path):
    """Ollama liefert keine timings - das darf nicht als 'kein Nachladen' durchgehen."""
    a = _aufgabe("summe-bugfix")
    adapter = SkriptAdapter([
        FakeAntwort([_aufruf("datei_schreiben", pfad="rechner.py",
                             inhalt="def addiere(a, b):\n    return a + b\n\n"
                                    "def multipliziere(a, b):\n    return a * b\n")],
                    timings=False),
        FakeAntwort([_aufruf("fertig")], timings=False),
    ])
    _, _, erg = _lauf(test_db, tmp_path, adapter, a, alias="qwen3:8b")
    assert erg.nachladen_pruefbar is False
    assert erg.nachladen_s == 0.0


def test_langes_nachladen_macht_den_messpunkt_ungueltig(test_db, tmp_path):
    a = _aufgabe("summe-bugfix")
    langsam = FakeAntwort([_aufruf("datei_schreiben", pfad="rechner.py",
                                   inhalt="def addiere(a, b):\n    return a + b\n\n"
                                          "def multipliziere(a, b):\n    return a * b\n")])
    langsam.dauer_s = 20.0                                   # so lange gewartet
    langsam.roh["timings"] = {"prompt_ms": 100.0, "predicted_ms": 400.0}   # halbe Sekunde gerechnet
    adapter = SkriptAdapter([langsam, FakeAntwort([_aufruf("fertig")])])
    _, run_id, erg = _lauf(test_db, tmp_path, adapter, a)

    assert erg.nachladen_s > 1.0
    assert erg.messpunkt_gueltig is False
    zeile = test_db.execute("SELECT * FROM results WHERE run_id = ?", (run_id,)).fetchone()
    assert zeile["messpunkt_gueltig"] == 0
    # Der Punkt bleibt trotzdem in der Tabelle - ein verworfener Messpunkt ist
    # eine Information, kein Loch.
    assert zeile["erfolg"] == 1


# --- Sandbox im echten Durchlauf --------------------------------------------

def test_ausbruchsversuch_scheitert_und_wird_protokolliert(test_db, tmp_path):
    """Die Abnahmebedingung aus Prompt C, ueber den vollen Weg des Runners."""
    a = _aufgabe("ausbruchsversuch", suite="sandbox-probe")
    adapter = SkriptAdapter([
        FakeAntwort([_aufruf("datei_schreiben", pfad="egal.py",
                             inhalt="def tuwas():\n    return 1\n")]),
        FakeAntwort([_aufruf("fertig")]),
    ])
    _, run_id, erg = _lauf(test_db, tmp_path, adapter, a)

    assert erg.erfolg is False
    assert erg.fehlergrund == "sandbox_ausbruch"
    assert erg.sandbox_verstoesse
    zeile = test_db.execute("SELECT * FROM results WHERE run_id = ?", (run_id,)).fetchone()
    assert "Schreibzugriff ausserhalb" in zeile["sandbox_verstoesse"]


# --- Protokoll und Wiederaufnahme -------------------------------------------

def test_jede_anfrage_und_antwort_landet_in_raw_logs(test_db, tmp_path):
    a = _aufgabe("summe-bugfix")
    adapter = SkriptAdapter([
        FakeAntwort([_aufruf("datei_schreiben", pfad="rechner.py",
                             inhalt="def addiere(a, b):\n    return a + b\n\n"
                                    "def multipliziere(a, b):\n    return a * b\n")]),
        FakeAntwort([_aufruf("fertig")]),
    ])
    _, run_id, _ = _lauf(test_db, tmp_path, adapter, a)
    n = test_db.execute(
        "SELECT COUNT(*) AS n FROM raw_logs WHERE run_id = ?", (run_id,)
    ).fetchone()["n"]
    assert n == 4        # zwei Schritte, je Anfrage und Antwort


def test_erledigte_durchgaenge_ermoeglichen_wiederaufnahme(test_db, tmp_path):
    a = _aufgabe("summe-bugfix")
    adapter = SkriptAdapter([FakeAntwort([_aufruf("fertig")])])
    r, run_id, _ = _lauf(test_db, tmp_path, adapter, a)
    erledigt = r.erledigte_durchgaenge(run_id)
    assert len(erledigt) == 1
    assert list(erledigt)[0][1] == 1
