"""Die Sandbox ist die einzige Stelle im Projekt mit echtem Schadenspotenzial.

Entsprechend wird sie nicht nur im Normalfall geprueft, sondern gegen den
Ausbruch: jeder dieser Tests laesst absichtlich generierten Code etwas tun, das
er nicht darf, und besteht nur, wenn es scheitert.
"""
from __future__ import annotations

import pytest

from kibench.sandbox import Sandbox, SandboxFehler

# Jeder dieser Faelle startet einen frischen Interpreter und importiert pytest,
# kostet also ein bis zwei Sekunden. Fuer eine Sicherung, die sonst nur behauptet
# waere, ist das gut angelegt.


def _box(tmp_path, name="aufgabe"):
    s = Sandbox(run_id=1, aufgabe=name, durchgang=1, wurzel=tmp_path)
    s.anlegen()
    return s


# --- Pfadsperre im Runner ---------------------------------------------------

def test_schreiben_legt_datei_an(tmp_path):
    s = _box(tmp_path)
    s.schreiben("src/rechner.py", "def f():\n    return 1\n")
    assert s.lesen("src/rechner.py").startswith("def f()")
    assert "src/rechner.py" in s.dateien()


def test_schreiben_ausserhalb_wird_abgelehnt(tmp_path):
    s = _box(tmp_path)
    for pfad in ("../ausbruch.txt", "../../ausbruch.txt", "unterordner/../../weg.txt"):
        with pytest.raises(SandboxFehler):
            s.schreiben(pfad, "sollte nicht ankommen")
    assert not (tmp_path / "ausbruch.txt").exists()


def test_absoluter_pfad_wird_abgelehnt(tmp_path):
    s = _box(tmp_path)
    with pytest.raises(SandboxFehler):
        s.schreiben(str(tmp_path / "woanders.txt"), "nein")


def test_zu_grosse_datei_wird_abgelehnt(tmp_path):
    s = _box(tmp_path)
    with pytest.raises(SandboxFehler):
        s.schreiben("gross.txt", "x" * 2_000_000)


def test_anlegen_raeumt_den_vorherigen_durchgang_weg(tmp_path):
    s = _box(tmp_path)
    s.schreiben("alt.txt", "vom letzten Durchgang")
    s.anlegen()
    assert s.dateien() == []


# --- Tests fahren -----------------------------------------------------------

def test_gruene_tests_werden_als_bestanden_gemeldet(tmp_path):
    s = _box(tmp_path)
    s.schreiben("loesung.py", "def addiere(a, b):\n    return a + b\n")
    s.schreiben("tests/test_loesung.py",
                "from loesung import addiere\n\n"
                "def test_addiere():\n    assert addiere(2, 3) == 5\n")
    e = s.tests_fahren(timeout_s=120)
    assert e.bestanden is True
    assert e.exit_code == 0
    assert e.verstoesse == []


def test_rote_tests_werden_als_nicht_bestanden_gemeldet(tmp_path):
    s = _box(tmp_path)
    s.schreiben("loesung.py", "def addiere(a, b):\n    return a - b\n")
    s.schreiben("tests/test_loesung.py",
                "from loesung import addiere\n\n"
                "def test_addiere():\n    assert addiere(2, 3) == 5\n")
    e = s.tests_fahren(timeout_s=120)
    assert e.bestanden is False
    assert e.exit_code != 0
    assert e.fehler is None       # roter Test ist ein Ergebnis, kein Fehler


def test_timeout_ist_ein_ergebnis_kein_absturz(tmp_path):
    """Fallstrick 4 aus Prompt C: ein Timeout ist ein Datenpunkt."""
    s = _box(tmp_path)
    s.schreiben("tests/test_endlos.py",
                "import time\n\ndef test_haengt():\n    time.sleep(60)\n")
    e = s.tests_fahren(timeout_s=8)
    assert e.timeout is True
    assert e.bestanden is False
    assert "Timeout" in (e.fehler or "")


# --- Ausbruchsversuche aus dem generierten Code -----------------------------

def test_schreiben_ausserhalb_wird_zur_laufzeit_gesperrt(tmp_path):
    """Die Abnahmebedingung aus Prompt C, wortwoertlich."""
    s = _box(tmp_path)
    ziel = (tmp_path / "eingebrochen.txt").as_posix()
    s.schreiben("tests/test_ausbruch.py",
                "def test_schreibt_nach_draussen():\n"
                f"    open({ziel!r}, 'w').write('hier war ich')\n")
    e = s.tests_fahren(timeout_s=120)
    assert e.bestanden is False
    assert e.verstoesse, "Der Ausbruch muss protokolliert werden, nicht nur scheitern"
    assert "Schreibzugriff ausserhalb" in e.verstoesse[0]
    assert not (tmp_path / "eingebrochen.txt").exists()


def test_loeschen_ausserhalb_wird_gesperrt(tmp_path):
    s = _box(tmp_path)
    opfer = tmp_path / "wichtig.txt"
    opfer.write_text("bitte nicht loeschen", encoding="utf-8")
    s.schreiben("tests/test_loescht.py",
                "import os\n\ndef test_loescht():\n"
                f"    os.remove({opfer.as_posix()!r})\n")
    e = s.tests_fahren(timeout_s=120)
    assert e.bestanden is False
    assert any("os.remove" in v for v in e.verstoesse)
    assert opfer.exists(), "Die Datei ausserhalb wurde tatsaechlich geloescht"


def test_netzwerk_wird_gesperrt(tmp_path):
    s = _box(tmp_path)
    s.schreiben("tests/test_netz.py",
                "import socket\n\ndef test_verbindet():\n"
                "    socket.create_connection(('1.1.1.1', 80), timeout=3)\n")
    e = s.tests_fahren(timeout_s=120)
    assert e.bestanden is False
    assert any("Netzwerk" in v for v in e.verstoesse)


def test_prozessstart_wird_gesperrt(tmp_path):
    """Ohne diese Sperre waere die Sandbox in einer Zeile zu umgehen."""
    s = _box(tmp_path)
    s.schreiben("tests/test_prozess.py",
                "import subprocess\n\ndef test_startet():\n"
                "    subprocess.run(['cmd', '/c', 'echo hi'])\n")
    e = s.tests_fahren(timeout_s=120)
    assert e.bestanden is False
    assert any("Prozessstart" in v for v in e.verstoesse)


def test_schreiben_innerhalb_bleibt_erlaubt(tmp_path):
    """Die Gegenprobe: eine Sperre, die alles blockt, ist auch kaputt."""
    s = _box(tmp_path)
    s.schreiben("tests/test_schreibt_intern.py",
                "from pathlib import Path\n\ndef test_schreibt():\n"
                "    Path('ergebnis.txt').write_text('ok', encoding='utf-8')\n"
                "    assert Path('ergebnis.txt').read_text(encoding='utf-8') == 'ok'\n")
    e = s.tests_fahren(timeout_s=120)
    assert e.bestanden is True
    assert e.verstoesse == []
