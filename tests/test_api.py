"""Die API-Endpunkte, automatisiert geprueft.

Genau dafuer ist die Oberflaeche eine Webseite und kein Terminal-Dashboard: die
Anzeige kann sich selbst verifizieren, statt dass jemand behauptet, sie stimme.
"""
from __future__ import annotations

import time

import pytest


def test_health(client):
    d = client.get("/api/health").json()
    assert d["ok"] is True
    assert d["sampler_laeuft"] is True


def test_status_meldet_laufenden_sampler(client):
    d = client.get("/api/status").json()
    assert d["laeuft"] is True
    assert d["fehler"] == 0
    # start() muss warten, bis wirklich gemessen wird - sonst greift die API
    # auf einen GpuMonitor zu, den es noch nicht gibt.
    assert d["quellen"]["nvml"] is True
    # Der Sampler muss die Quellen benennen, sonst ist "keine Daten" nicht
    # von "Quelle kaputt" unterscheidbar.
    assert {"nvml", "pdh", "psutil", "llama", "ollama"} <= set(d["quellen"])


def test_sampler_misst_tatsaechlich(client):
    """Nicht nur 'laeuft', sondern: es kommen Zeilen in der Datenbank an."""
    vorher = client.get("/api/status").json()["db_zeilen"]
    time.sleep(2.5)
    nachher = client.get("/api/status").json()["db_zeilen"]
    assert nachher > vorher


def test_jetzt_liefert_kennzahlen(client):
    time.sleep(1.5)
    d = client.get("/api/jetzt").json()
    assert d["vorhanden"] is True
    assert d["vram_total_mib"] == 16376
    assert d["ram_total_gib"] == 64
    s = d["sample"]
    assert s["vram_used_mib"] > 0
    assert s["ram_used_gib"] > 0
    assert isinstance(d["warnungen"], list)


def test_vram_zuordnung_uebersteigt_nie_die_gesamtsumme(client):
    """Der Fehler aus dem ersten Entwurf: PDH-Rohwerte summierten sich auf das
    Doppelte der echten Belegung, der 'Fremd-VRAM' lag ueber dem Gesamtwert."""
    time.sleep(1.5)
    s = client.get("/api/jetzt").json()["sample"]
    if s["vram_fremd_mib"] is None:
        pytest.skip("keine PDH-Zuordnung auf diesem System")
    summe = s["vram_llama_mib"] + s["vram_ollama_mib"] + s["vram_fremd_mib"]
    # Rundung je Prozess erlaubt ein paar MiB Abweichung.
    assert summe <= s["vram_used_mib"] + 20
    assert s["vram_fremd_mib"] <= s["vram_used_mib"] + 20


def test_prozessliste_hat_rohwert_und_zuordnung(client):
    time.sleep(1.5)
    prozesse = client.get("/api/jetzt").json()["sample"]["vram_prozesse"]
    if not prozesse:
        pytest.skip("keine GPU-Prozesse gemeldet")
    p = prozesse[0]
    assert {"pid", "name", "mib", "mib_roh"} <= set(p)
    # Die Normierung schrumpft die Rohwerte, sie darf sie nie vergroessern.
    assert p["mib"] <= p["mib_roh"] + 1


def test_verlauf_liefert_spalten(client):
    time.sleep(2.0)
    d = client.get("/api/verlauf?fenster=5m").json()
    assert d["punkte"] > 0
    for feld in ("ts", "vram_used_mib", "ram_used_gib", "gpu_temp_c", "llama_gen_tps"):
        assert feld in d["spalten"]
        assert len(d["spalten"][feld]) == d["punkte"]
    assert d["schwellen"]["vram_total_mib"] == 16376


@pytest.mark.parametrize("fenster", ["5m", "15m", "1h", "6h", "heute"])
def test_alle_zeitfenster(client, fenster):
    r = client.get(f"/api/verlauf?fenster={fenster}")
    assert r.status_code == 200
    assert r.json()["fenster"] == fenster


def test_unbekanntes_zeitfenster_wird_abgelehnt(client):
    assert client.get("/api/verlauf?fenster=ewig").status_code == 422


def test_verlauf_deckelt_die_punktzahl(client):
    time.sleep(2.0)
    d = client.get("/api/verlauf?fenster=heute&max_punkte=50").json()
    assert d["punkte"] <= 60


def test_modelle_listet_alle_sechs_konfigurationen(client):
    """Fuenf Modelle, sechs Konfigurationen - dense laeuft in zwei Quantisierungen.

    Der Test hiess bis Prompt C `..._alle_fuenf` und pruefte `len == 5`. Mit der
    sechsten Konfiguration musste die Zusicherung mitwandern; ein Test, dessen
    Name eine andere Zahl behauptet als sein Rumpf, ist schlimmer als keiner.
    """
    d = client.get("/api/modelle").json()
    aliase = [m["alias"] for m in d["konfiguriert"]]
    assert len(aliase) == 6
    assert "qwen-dense" in aliase and "qwen-moe" in aliase
    assert "qwen-dense-iq4" in aliase
    assert "llama3.2:3b" in aliase
    # Die Sweep-Werte aus SETUP.md sind die Grundlage der tok/s-Warnschwelle.
    moe = next(m for m in d["konfiguriert"] if m["alias"] == "qwen-moe")
    assert moe["sweep_gen_tps"] == 64.40
    iq4 = next(m for m in d["konfiguriert"] if m["alias"] == "qwen-dense-iq4")
    assert iq4["quant"] == "IQ4_XS" and iq4["offload"] == "-ngl 58"


def test_llama_konfigurationen_haben_ein_wechselziel(client):
    """Ohne `wechsel_ziel` kann der Runner den Server nicht selbst umschalten -
    und der Probelauf ueber alle sechs Konfigurationen braucht genau das."""
    d = client.get("/api/modelle").json()
    for m in d["konfiguriert"]:
        if m["runtime"] == "llama.cpp":
            assert m.get("wechsel_ziel"), f"{m['alias']} hat kein Wechselziel"


def test_seite_wird_ausgeliefert(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "ki-benchmarks" in r.text
    # uPlot muss lokal liegen - kein CDN, sonst laeuft die Seite offline nicht.
    assert "/static/vendor/uPlot.iife.min.js" in r.text
    assert "cdn." not in r.text


def test_uplot_liegt_lokal(client):
    r = client.get("/static/vendor/uPlot.iife.min.js")
    assert r.status_code == 200
    assert len(r.content) > 10000


def test_unbekannter_pfad_gibt_json_404(client):
    r = client.get("/gibtsnicht")
    assert r.status_code == 404
    assert r.json()["fehler"] == "nicht gefunden"


# --- Warnschwellen -----------------------------------------------------------

def test_warnschwellen_greifen():
    from kibench.api import _warnungen

    w = _warnungen({"vram_used_mib": 16000, "ram_used_gib": 60.0, "gpu_temp_c": 88}, 0)
    felder = {x["feld"] for x in w}
    assert {"vram", "ram", "temp"} <= felder
    assert all(x["stufe"] == "kritisch" for x in w)


def test_tps_warnung_nutzt_den_sweep_wert():
    from kibench.api import _warnungen

    # qwen-moe: Sweep 64,40 tok/s -> Schwelle 32,2
    w = _warnungen({"llama_model": "qwen-moe", "llama_gen_tps": 20.0}, 0)
    assert any(x["feld"] == "tps" for x in w)
    w = _warnungen({"llama_model": "qwen-moe", "llama_gen_tps": 60.0}, 0)
    assert not any(x["feld"] == "tps" for x in w)


def test_fremd_vram_schwankung_warnt():
    from kibench.api import _warnungen

    assert not any(x["feld"] == "fremd_vram" for x in _warnungen({}, 150))
    assert any(x["feld"] == "fremd_vram" for x in _warnungen({}, 350))


def test_ruhige_maschine_gibt_keine_warnung():
    from kibench.api import _warnungen

    assert _warnungen({"vram_used_mib": 2000, "ram_used_gib": 20.0, "gpu_temp_c": 50}, 30) == []
