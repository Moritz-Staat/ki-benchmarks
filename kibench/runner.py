"""Der Benchmark-Runner: Aufgabe an ein Modell geben, Ergebnis messen.

Ablauf je Aufgabe und Durchgang:

    Sandbox anlegen  ->  Startdateien hinein  ->  Modell arbeiten lassen
    ->  Tests dazulegen  ->  pytest im Unterprozess  ->  Ergebnis in die DB

Das Modell arbeitet ueber Werkzeuge, nicht ueber Freitext. Es haette auch
gereicht, Codebloecke aus der Antwort zu schneiden - aber Prompt C will die
Tool-Call-Zuverlaessigkeit als eigene Messdimension, und die laesst sich nur
messen, wenn tatsaechlich Werkzeuge im Spiel sind. Nebenbei faellt damit die
haeufigste Fehlerquelle weg: ein Modell, das seinen Code in Prosa einbettet.

## Was hier gemessen wird und warum es so gemessen wird

**Wanduhrzeit, nicht die gemeldeten tok/s.** Der Befund aus Prompt B: Der Server
meldete 75,5 tok/s, waehrend der Anrufer 21,6 s wartete - der Treiber hatte die
Gewichte in den Hauptspeicher ausgelagert und lud sie zurueck. `/metrics` sieht
das nicht. Deshalb steht in jedem Ergebnis die gewartete Zeit, die gerechnete
Zeit und die Differenz; ueberschreitet die Differenz eine Sekunde, gilt der
Messpunkt als ungueltig und wird nicht mitgemittelt. Er bleibt trotzdem in der
Tabelle - ein verworfener Messpunkt ist eine Information, kein Loch.

**Leere Antworten sind Fehlschlaege.** Das erledigt der Adapter; hier wird nur
darauf geachtet, den Rueckgabewert nicht zu uebergehen.

**Fehlschlaege werden protokolliert, nicht uebersprungen.** Timeout, Absturz,
Ausbruchsversuch, Endlosschleife - alles sind Ergebniszeilen.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any

from . import db, serverwechsel
from .aufgaben import Aufgabe
from .config import (
    MODELLE_NACH_ALIAS,
    SCHWELLE_FREMD_VRAM_SCHWANKUNG_MIB,
)
from .sandbox import Sandbox

# Ab hier gilt ein Messpunkt als vom Nachladen verfaelscht. Siehe Modulkopf.
NACHLADEN_GRENZE_S = 1.0

# Sicherheitsnetz gegen Endlosschleifen im Werkzeuggebrauch. Prompt C nennt
# "derselbe Call mehrfach" ausdruecklich als Fehlerbild.
SCHRITTE_MINDESTENS = 6
SCHRITTE_FAKTOR = 3

SYSTEMTEXT = """Du bearbeitest eine Programmieraufgabe in einem leeren Arbeitsverzeichnis.

Arbeite ausschliesslich ueber die bereitgestellten Werkzeuge. Schreibe keinen Code
in deine Antwort - Code, der nicht ueber datei_schreiben geht, wird nicht gewertet.

Regeln:
- Pfade sind immer relativ zum Arbeitsverzeichnis, zum Beispiel "rechner.py".
- Schreibe jede Datei vollstaendig; es gibt kein teilweises Aendern.
- Wenn du fertig bist, rufe fertig() auf.
- Es gibt keine Testdateien im Verzeichnis. Deine Loesung wird anschliessend
  gegen eine Testsuite geprueft, die du nicht siehst."""

WERKZEUGE = [
    {
        "type": "function",
        "function": {
            "name": "dateien_auflisten",
            "description": "Listet alle Dateien im Arbeitsverzeichnis auf.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "datei_lesen",
            "description": "Liest eine Datei aus dem Arbeitsverzeichnis.",
            "parameters": {
                "type": "object",
                "properties": {"pfad": {"type": "string", "description": "relativer Pfad"}},
                "required": ["pfad"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "datei_schreiben",
            "description": "Schreibt eine Datei vollstaendig neu.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pfad": {"type": "string", "description": "relativer Pfad"},
                    "inhalt": {"type": "string", "description": "vollstaendiger Dateiinhalt"},
                },
                "required": ["pfad", "inhalt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fertig",
            "description": "Meldet, dass die Aufgabe geloest ist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "zusammenfassung": {"type": "string", "description": "kurz, was getan wurde"}
                },
                "required": [],
            },
        },
    },
]

_WERKZEUG_NAMEN = {w["function"]["name"] for w in WERKZEUGE}
_PFLICHTFELDER = {
    w["function"]["name"]: set(w["function"]["parameters"].get("required") or [])
    for w in WERKZEUGE
}


def _rechenzeit_aus(m) -> float | None:
    """Vom Server gemeldete Rechenzeit einer Antwort, in Sekunden.

    llama.cpp haengt an die OpenAI-Antwort ein `timings`-Objekt an; darin stehen
    `prompt_ms` und `predicted_ms`. Ollama tut das nicht - dort gibt es keine
    Vergleichsgroesse, und der Aufrufer muss das wissen, statt eine Null
    geliefert zu bekommen. Genau diese stille Null waere der Fehler, den Prompt B
    beschreibt: eine Kennzahl, die gesund aussieht, weil sie nichts gemessen hat.
    """
    t = (m.roh or {}).get("timings")
    if not isinstance(t, dict):
        return None
    try:
        return (float(t.get("prompt_ms", 0.0)) + float(t.get("predicted_ms", 0.0))) / 1000.0
    except (TypeError, ValueError):
        return None


@dataclass
class Schrittstatistik:
    schritte: int = 0
    erhalten: int = 0
    korrekt: int = 0
    erfundene_namen: list[str] = field(default_factory=list)
    parameterfehler: int = 0
    wiederholungen: int = 0
    fliesstext: int = 0
    ohne_werkzeug: int = 0          # Antwort ohne Werkzeugaufruf
    abbruch_ohne_ergebnis: bool = False


@dataclass
class Aufgabenergebnis:
    erfolg: bool = False
    fehlergrund: str | None = None
    dauer_s: float = 0.0            # Wanduhr des gesamten Arbeitsschritts
    anfragezeit_s: float = 0.0      # davon: gewartet auf Antworten des Modells
    rechenzeit_s: float = 0.0       # davon: vom Server als Rechnung gemeldet
    nachladen_s: float = 0.0        # die Differenz - das ist der Befund
    nachladen_pruefbar: bool = True
    messpunkt_gueltig: bool = True
    timeout: bool = False
    sandbox_verstoesse: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    antwort_tokens: int = 0
    denk_tokens: int = 0
    gesamt_tokens: int = 0
    gen_tps: float | None = None
    prompt_tps: float | None = None
    finish_reason: str | None = None
    statistik: Schrittstatistik = field(default_factory=Schrittstatistik)
    testausgabe: str = ""
    vram_max_mib: int | None = None
    vram_fremd_max_mib: int | None = None
    maschine_ruhig: bool | None = None


class Runner:
    def __init__(self, adapter, conn=None, gpu_monitor=None,
                 sandbox_wurzel=None) -> None:
        self.adapter = adapter
        self.conn = conn or db.init_db()
        self.gpu = gpu_monitor
        # Nur fuer Tests: sonst liegen die Arbeitsverzeichnisse unter runs\.
        self.sandbox_wurzel = sandbox_wurzel

    # --- Laeufe -------------------------------------------------------------

    def lauf_anlegen(self, alias: str, thinking: bool, suite: str,
                     notiz: str | None = None) -> int:
        e = MODELLE_NACH_ALIAS.get(alias) or {}
        cur = self.conn.execute(
            """INSERT INTO runs (modell_alias, modell_name, runtime, quant, offload,
                                 kontext, thinking, ts_start, status, notiz)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'laufend', ?)""",
            (alias, e.get("name"), e.get("runtime"), e.get("quant"), e.get("offload"),
             32768, int(thinking), time.time(), notiz or suite),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def lauf_abschliessen(self, run_id: int, status: str = "fertig") -> None:
        self.conn.execute(
            "UPDATE runs SET ts_ende = ?, status = ? WHERE id = ?",
            (time.time(), status, run_id),
        )
        self.conn.commit()

    def erledigte_durchgaenge(self, run_id: int) -> set[tuple[int, int]]:
        """Was in diesem Lauf schon gemessen wurde - Grundlage der Wiederaufnahme."""
        return {
            (r["task_id"], r["durchgang"])
            for r in self.conn.execute(
                "SELECT task_id, durchgang FROM results WHERE run_id = ?", (run_id,)
            )
        }

    # --- Eine Aufgabe -------------------------------------------------------

    def aufgabe_fahren(
        self,
        run_id: int,
        aufgabe: Aufgabe,
        task_id: int,
        durchgang: int,
        alias: str,
        thinking: bool = True,
    ) -> Aufgabenergebnis:
        erg = Aufgabenergebnis()
        box = Sandbox(run_id=run_id, aufgabe=aufgabe.schluessel, durchgang=durchgang,
                      wurzel=self.sandbox_wurzel)
        box.anlegen(aufgabe.startdateien)

        ts_start = time.time()
        t0 = time.perf_counter()
        stat = erg.statistik

        nachrichten: list[dict] = [
            {"role": "system", "content": SYSTEMTEXT},
            {"role": "user", "content": self._auftragstext(aufgabe, box)},
        ]
        max_schritte = max(SCHRITTE_MINDESTENS, aufgabe.erwartete_schritte * SCHRITTE_FAKTOR)
        gesehene_aufrufe: set[str] = set()
        fertig_gemeldet = False

        for _ in range(max_schritte):
            m = self.adapter.anfragen(
                alias, nachrichten,
                max_tokens=aufgabe.max_tokens,
                werkzeuge=WERKZEUGE,
                thinking=thinking,
                temperature=0.0,
            )
            stat.schritte += 1
            self._roh_schreiben(run_id, "anfrage", nachrichten[-1])
            self._roh_schreiben(run_id, "antwort", m.roh or {"fehler": m.fehlergrund})

            erg.anfragezeit_s += m.dauer_s or 0.0
            rechen = _rechenzeit_aus(m)
            if rechen is None:
                # Ollama liefert keine Server-Timings. Ohne sie laesst sich das
                # Nachladen nicht ausrechnen - das ist kein Beinbruch, denn
                # Ollama laedt ohnehin bei jeder Anfrage neu. Es darf nur nicht
                # als "kein Nachladen" durchgehen.
                erg.nachladen_pruefbar = False
            else:
                erg.rechenzeit_s += rechen
            erg.prompt_tokens += m.prompt_tokens or 0
            erg.antwort_tokens += m.antwort_tokens or 0
            erg.denk_tokens += m.denk_tokens or 0
            erg.gesamt_tokens += m.gesamt_tokens or 0
            erg.finish_reason = m.finish_reason
            if m.tool_call_fliesstext:
                stat.fliesstext += 1

            if not m.erfolg and not m.tool_calls:
                erg.fehlergrund = m.fehlergrund or "keine_antwort"
                break

            if not m.tool_calls:
                stat.ohne_werkzeug += 1
                # Antwort ohne Werkzeugaufruf: einmal nachfassen, dann aufgeben.
                # Zweimal dieselbe Aufforderung zu schicken misst nur noch
                # Geduld, nicht Faehigkeit.
                if stat.ohne_werkzeug >= 2:
                    stat.abbruch_ohne_ergebnis = True
                    erg.fehlergrund = "antwortet_ohne_werkzeug"
                    break
                nachrichten.append(self._assistenznachricht(m))
                nachrichten.append({
                    "role": "user",
                    "content": "Bitte benutze die Werkzeuge. Code in der Antwort wird "
                               "nicht gewertet. Rufe fertig() auf, wenn du fertig bist.",
                })
                continue

            nachrichten.append(self._assistenznachricht(m))
            for aufruf in m.tool_calls:
                antwort, ist_fertig = self._werkzeug_ausfuehren(
                    aufruf, box, stat, gesehene_aufrufe
                )
                nachrichten.append({
                    "role": "tool",
                    "tool_call_id": aufruf.get("id") or "0",
                    "content": antwort,
                })
                fertig_gemeldet = fertig_gemeldet or ist_fertig
            if fertig_gemeldet:
                break
            if stat.wiederholungen >= 3:
                # Prompt C nennt das ausdruecklich als Fehlerbild: derselbe Call
                # immer wieder. Weiterlaufen zu lassen kostet nur Zeit.
                erg.fehlergrund = "endlosschleife_werkzeug"
                break
        else:
            erg.fehlergrund = erg.fehlergrund or "schrittgrenze_erreicht"

        # Der Befund aus Prompt B als Rechnung: gewartete Zeit minus gerechnete
        # Zeit. Verglichen wird gegen die Summe der Anfragezeiten, nicht gegen
        # die Gesamtdauer - das Schreiben der Dateien dazwischen ist Arbeit des
        # Runners und hat mit dem Modell nichts zu tun.
        erg.dauer_s = round(time.perf_counter() - t0, 2)
        erg.anfragezeit_s = round(erg.anfragezeit_s, 2)
        erg.rechenzeit_s = round(erg.rechenzeit_s, 2)
        if erg.nachladen_pruefbar:
            erg.nachladen_s = round(max(0.0, erg.anfragezeit_s - erg.rechenzeit_s), 2)
            erg.messpunkt_gueltig = erg.nachladen_s <= NACHLADEN_GRENZE_S
        else:
            erg.nachladen_s = 0.0
            erg.messpunkt_gueltig = True
        if erg.dauer_s > 0 and erg.antwort_tokens:
            erg.gen_tps = round(erg.antwort_tokens / erg.dauer_s, 2)

        # Tests erst jetzt dazulegen - vorher haette das Modell sie sehen und
        # umschreiben koennen, statt die Aufgabe zu loesen.
        for pfad, inhalt in aufgabe.testdateien.items():
            box.schreiben(f"tests/{pfad}", inhalt)

        t = box.tests_fahren(timeout_s=aufgabe.timeout_s)
        erg.erfolg = t.bestanden
        erg.timeout = t.timeout
        erg.sandbox_verstoesse = t.verstoesse
        erg.testausgabe = t.ausgabe[-8000:]
        if t.timeout:
            erg.fehlergrund = erg.fehlergrund or "test_timeout"
        elif t.fehler:
            erg.fehlergrund = erg.fehlergrund or t.fehler
        elif not t.bestanden and not erg.fehlergrund:
            erg.fehlergrund = "tests_rot"
        if t.verstoesse:
            erg.fehlergrund = "sandbox_ausbruch"

        self._umfeld_ergaenzen(erg, ts_start, time.time())
        self._ergebnis_schreiben(run_id, task_id, durchgang, erg, ts_start)
        if erg.erfolg:
            box.verwerfen()
        return erg

    # --- Werkzeuge ----------------------------------------------------------

    def _werkzeug_ausfuehren(
        self, aufruf: dict, box: Sandbox, stat: Schrittstatistik, gesehen: set[str]
    ) -> tuple[str, bool]:
        stat.erhalten += 1
        funktion = (aufruf.get("function") or {})
        name = funktion.get("name") or ""
        roh = funktion.get("arguments")

        if name not in _WERKZEUG_NAMEN:
            stat.erfundene_namen.append(name)
            return (f"Unbekanntes Werkzeug {name!r}. Verfuegbar: "
                    f"{', '.join(sorted(_WERKZEUG_NAMEN))}."), False

        try:
            args = json.loads(roh) if isinstance(roh, str) else (roh or {})
            if not isinstance(args, dict):
                raise ValueError("Argumente sind kein Objekt")
        except Exception as e:
            stat.parameterfehler += 1
            return f"Argumente nicht lesbar ({e}). Erwartet wird JSON.", False

        fehlend = _PFLICHTFELDER[name] - set(args)
        if fehlend:
            stat.parameterfehler += 1
            return f"Pflichtfelder fehlen: {', '.join(sorted(fehlend))}.", False

        kennung = name + json.dumps(args, sort_keys=True)[:400]
        if kennung in gesehen:
            stat.wiederholungen += 1
        gesehen.add(kennung)
        stat.korrekt += 1

        try:
            if name == "dateien_auflisten":
                dateien = box.dateien()
                return (", ".join(dateien) if dateien else "(leer)"), False
            if name == "datei_lesen":
                return box.lesen(str(args["pfad"])), False
            if name == "datei_schreiben":
                inhalt = args["inhalt"]
                if not isinstance(inhalt, str):
                    stat.parameterfehler += 1
                    return "Feld 'inhalt' muss eine Zeichenkette sein.", False
                box.schreiben(str(args["pfad"]), inhalt)
                return f"Geschrieben: {args['pfad']} ({len(inhalt)} Zeichen)", False
            if name == "fertig":
                return "Verstanden.", True
        except Exception as e:
            # Ein abgelehnter Pfad ist kein Absturz, sondern eine Antwort an das
            # Modell - es soll die Chance bekommen, es richtig zu machen.
            return f"Fehlgeschlagen: {e}", False
        return "Unbehandelt.", False

    # --- Hilfsmittel --------------------------------------------------------

    def _auftragstext(self, aufgabe: Aufgabe, box: Sandbox) -> str:
        dateien = box.dateien()
        teile = [aufgabe.prompt.strip()]
        if dateien:
            teile.append("Dateien im Arbeitsverzeichnis: " + ", ".join(dateien))
        else:
            teile.append("Das Arbeitsverzeichnis ist leer.")
        return "\n\n".join(teile)

    def _assistenznachricht(self, m) -> dict:
        """Die Antwort des Modells so zurueckgeben, wie sie kam.

        Wichtig fuer den naechsten Schritt: `tool_calls` muessen unveraendert
        mitlaufen, sonst findet die Runtime die zugehoerigen tool-Antworten
        nicht mehr zu und faengt an zu halluzinieren.
        """
        roh = ((m.roh.get("choices") or [{}])[0].get("message") or {}) if m.roh else {}
        nachricht: dict[str, Any] = {"role": "assistant", "content": roh.get("content") or ""}
        if m.tool_calls:
            nachricht["tool_calls"] = m.tool_calls
        return nachricht

    def _roh_schreiben(self, run_id: int, richtung: str, inhalt: Any) -> None:
        try:
            self.conn.execute(
                "INSERT INTO raw_logs (run_id, ts, richtung, endpunkt, inhalt) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, time.time(), richtung, "/v1/chat/completions",
                 json.dumps(inhalt, ensure_ascii=False)[:200000]),
            )
        except Exception:
            pass

    def _umfeld_ergaenzen(self, erg: Aufgabenergebnis, von: float, bis: float) -> None:
        """Was hat die Maschine waehrend des Laufs getan? Aus `samples`.

        Das ist der Punkt, an dem sich die gemeinsame Datenbank auszahlt: kein
        Join ueber zwei Systeme, ein SELECT.
        """
        try:
            r = self.conn.execute(
                "SELECT MAX(vram_used_mib) AS v, MAX(vram_fremd_mib) AS f, "
                "MIN(vram_fremd_mib) AS fmin, COUNT(*) AS n "
                "FROM samples WHERE ts BETWEEN ? AND ?", (von, bis),
            ).fetchone()
        except Exception:
            return
        if not r or not r["n"]:
            return
        erg.vram_max_mib = r["v"]
        erg.vram_fremd_max_mib = r["f"]
        if r["f"] is not None and r["fmin"] is not None:
            erg.maschine_ruhig = (r["f"] - r["fmin"]) <= SCHWELLE_FREMD_VRAM_SCHWANKUNG_MIB

    def _ergebnis_schreiben(self, run_id: int, task_id: int, durchgang: int,
                            erg: Aufgabenergebnis, ts_start: float) -> None:
        s = erg.statistik
        self.conn.execute(
            """INSERT INTO results (
                   run_id, task_id, durchgang, erfolg, fehlergrund,
                   ts_start, ts_ende, dauer_s,
                   prompt_tokens, denk_tokens, antwort_tokens, gesamt_tokens,
                   gen_tps, prompt_tps, finish_reason,
                   tool_calls_erwartet, tool_calls_erhalten, tool_calls_korrekt,
                   tool_call_fliesstext,
                   vram_max_mib, vram_fremd_max_mib, maschine_ruhig,
                   schritte, rechenzeit_s, nachladen_s, messpunkt_gueltig,
                   timeout, sandbox_verstoesse)
               VALUES (?,?,?,?,?, ?,?,?, ?,?,?,?, ?,?,?, ?,?,?,?, ?,?,?, ?,?,?,?,?,?)""",
            (run_id, task_id, durchgang, int(erg.erfolg), erg.fehlergrund,
             ts_start, time.time(), erg.dauer_s,
             erg.prompt_tokens, erg.denk_tokens, erg.antwort_tokens, erg.gesamt_tokens,
             erg.gen_tps, erg.prompt_tps, erg.finish_reason,
             None, s.erhalten, s.korrekt, int(bool(s.fliesstext)),
             erg.vram_max_mib, erg.vram_fremd_max_mib,
             None if erg.maschine_ruhig is None else int(erg.maschine_ruhig),
             s.schritte, erg.rechenzeit_s, erg.nachladen_s, int(erg.messpunkt_gueltig),
             int(erg.timeout),
             json.dumps(erg.sandbox_verstoesse, ensure_ascii=False) if erg.sandbox_verstoesse else None),
        )
        self.conn.commit()


def reihenfolge_mischen(paare: list, saat: int | None = None) -> list:
    """Reihenfolge randomisieren - Fallstrick 2 aus Prompt C.

    Ohne das misst der spaetere Lauf einen warmen Prompt-Cache mit und sieht
    schneller aus, ohne es zu sein. Die Saat wird protokolliert, damit ein Lauf
    reproduzierbar bleibt.
    """
    r = random.Random(saat)
    kopie = list(paare)
    r.shuffle(kopie)
    return kopie


def modell_bereitstellen(alias: str, gpu_monitor=None):
    return serverwechsel.bereitstellen(alias, gpu_monitor=gpu_monitor)
