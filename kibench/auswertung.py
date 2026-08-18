"""Aggregation der Ergebnisse - fuer die Vergleichs- und die Einzellaufansicht.

Zwei Dinge unterscheiden das hier von einer Handvoll `AVG()`-Abfragen:

**Streuung statt Mittelwert.** Prompt C ist an der Stelle deutlich: Drei stark
schwankende Durchgaenge sehen gemittelt aus wie ein verlaesslicher Wert. Jede
Erfolgsquote kommt deshalb mit der Zahl der Durchgaenge und der Spanne, und je
Aufgabe steht da, ob sie **immer**, **manchmal** oder **nie** geloest wurde.
"Manchmal" ist die interessanteste Kategorie und verschwindet in jedem
Mittelwert.

**Ungueltige Messpunkte fliessen nicht in die Tempowerte.** Ein Lauf, in dem der
Treiber Gewichte nachgeladen hat, misst die Wartezeit des Treibers und nicht die
Geschwindigkeit des Modells. Fuer die Erfolgsquote zaehlt er trotzdem - das
Modell hat die Aufgabe ja geloest oder eben nicht.
"""
from __future__ import annotations

import statistics
from typing import Any


# Suiten, die nicht bewertet werden. `sandbox-probe` enthaelt genau eine Aufgabe,
# die per Konstruktion niemals bestehen kann - sie prueft die Sperre, nicht das
# Modell. Zaehlte sie mit, wuerde ausgerechnet die Konfiguration schlechter
# dastehen, mit der die Abnahme gefahren wurde. Aufgefallen beim ersten echten
# Lauf: qwen-moe kam auf 0,889 statt 1,0, ohne eine einzige Aufgabe verfehlt zu
# haben.
NICHT_GEWERTET = ("sandbox-probe",)


def _quote(treffer: int, gesamt: int) -> float | None:
    return round(treffer / gesamt, 3) if gesamt else None


def laeufe(conn, limit: int = 100) -> list[dict]:
    zeilen = conn.execute(
        """SELECT r.*,
                  (SELECT COUNT(*) FROM results x WHERE x.run_id = r.id)            AS ergebnisse,
                  (SELECT COUNT(*) FROM results x WHERE x.run_id = r.id AND x.erfolg = 1) AS erfolge
           FROM runs r ORDER BY r.ts_start DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    aus = []
    for z in zeilen:
        d = {k: z[k] for k in z.keys()}
        d["erfolgsquote"] = _quote(z["erfolge"], z["ergebnisse"])
        d["dauer_s"] = round((z["ts_ende"] or 0) - z["ts_start"], 1) if z["ts_ende"] else None
        aus.append(d)
    return aus


def _ergebniszeilen(conn, run_ids: list[int] | None = None) -> list[dict]:
    sql = """
        SELECT res.*, t.schluessel, t.titel, t.kategorie, t.schwierigkeit,
               t.erwartete_schritte,
               r.modell_alias, r.modell_name, r.quant, r.thinking, r.runtime, r.notiz
        FROM results res
        JOIN tasks t ON t.id = res.task_id
        JOIN runs  r ON r.id = res.run_id
    """
    parameter: tuple = ()
    if run_ids:
        sql += f" WHERE res.run_id IN ({','.join('?' * len(run_ids))})"
        parameter = tuple(run_ids)
    return [{k: z[k] for k in z.keys()} for z in conn.execute(sql, parameter)]


def vergleich(conn, suite: str | None = None,
              alles: bool = False) -> dict[str, Any]:
    """Alle Konfigurationen nebeneinander.

    Gruppiert wird nach Modell **und** Thinking-Modus - das sind unterschiedliche
    Betriebsarten desselben Modells, und Prompt C will sie getrennt sehen.

    `alles=True` nimmt auch die Abnahme-Suiten mit hinein; ohne das bleiben sie
    draussen, siehe NICHT_GEWERTET.
    """
    zeilen = _ergebniszeilen(conn)
    if suite:
        zeilen = [z for z in zeilen if (z.get("notiz") or "") == suite]
    elif not alles:
        zeilen = [z for z in zeilen if (z.get("notiz") or "") not in NICHT_GEWERTET]

    gruppen: dict[tuple, list[dict]] = {}
    for z in zeilen:
        gruppen.setdefault((z["modell_alias"], bool(z["thinking"])), []).append(z)

    konfigurationen = []
    for (alias, thinking), rows in sorted(gruppen.items(), key=lambda x: (x[0][0], x[0][1])):
        konfigurationen.append(_eine_konfiguration(alias, thinking, rows))

    return {
        "konfigurationen": konfigurationen,
        "aufgaben": _je_aufgabe(zeilen),
        "gesamt_ergebnisse": len(zeilen),
    }


def _eine_konfiguration(alias: str, thinking: bool, rows: list[dict]) -> dict[str, Any]:
    gesamt = len(rows)
    erfolge = sum(1 for z in rows if z["erfolg"])

    # Tempo nur aus gueltigen Messpunkten. Siehe Modulkopf.
    gueltig = [z for z in rows if z.get("messpunkt_gueltig", 1)]
    tps = [z["gen_tps"] for z in gueltig if z.get("gen_tps")]
    dauern = [z["dauer_s"] for z in gueltig if z.get("dauer_s")]

    # Erfolgsquote je Aufgabe, daraus die Streuung ueber die Durchgaenge.
    je_aufgabe: dict[str, list[int]] = {}
    for z in rows:
        je_aufgabe.setdefault(z["schluessel"], []).append(1 if z["erfolg"] else 0)
    immer = sum(1 for v in je_aufgabe.values() if v and all(v))
    nie = sum(1 for v in je_aufgabe.values() if v and not any(v))
    manchmal = len(je_aufgabe) - immer - nie

    erhalten = sum(z.get("tool_calls_erhalten") or 0 for z in rows)
    korrekt = sum(z.get("tool_calls_korrekt") or 0 for z in rows)

    return {
        "alias": alias,
        "modell": rows[0].get("modell_name") if rows else alias,
        "quant": rows[0].get("quant") if rows else None,
        "thinking": thinking,
        "ergebnisse": gesamt,
        "erfolge": erfolge,
        "erfolgsquote": _quote(erfolge, gesamt),
        "je_schwierigkeit": _nach(rows, "schwierigkeit"),
        "je_kategorie": _nach(rows, "kategorie"),
        # Das Herzstueck der Streuung: "manchmal" ist die Kategorie, die jeder
        # Mittelwert verschluckt.
        "aufgaben_immer": immer,
        "aufgaben_manchmal": manchmal,
        "aufgaben_nie": nie,
        "gen_tps_median": round(statistics.median(tps), 2) if tps else None,
        "gen_tps_spanne": [round(min(tps), 2), round(max(tps), 2)] if tps else None,
        "dauer_median_s": round(statistics.median(dauern), 1) if dauern else None,
        "dauer_spanne_s": [round(min(dauern), 1), round(max(dauern), 1)] if dauern else None,
        "messpunkte_verworfen": gesamt - len(gueltig),
        "tool_calls_erhalten": erhalten,
        "tool_calls_korrekt": korrekt,
        "tool_call_quote": _quote(korrekt, erhalten),
        "tool_call_fliesstext": sum(1 for z in rows if z.get("tool_call_fliesstext")),
        "timeouts": sum(1 for z in rows if z.get("timeout")),
        "sandbox_verstoesse": sum(1 for z in rows if z.get("sandbox_verstoesse")),
        "denk_tokens_median": _median_von(rows, "denk_tokens"),
        "antwort_tokens_median": _median_von(rows, "antwort_tokens"),
        "schritte_median": _median_von(rows, "schritte"),
        "fehlergruende": _haeufigkeit(z.get("fehlergrund") for z in rows if not z["erfolg"]),
    }


def _nach(rows: list[dict], feld: str) -> dict[str, dict]:
    aus: dict[str, dict] = {}
    for z in rows:
        k = z.get(feld) or "unbekannt"
        e = aus.setdefault(k, {"gesamt": 0, "erfolge": 0})
        e["gesamt"] += 1
        e["erfolge"] += 1 if z["erfolg"] else 0
    for e in aus.values():
        e["quote"] = _quote(e["erfolge"], e["gesamt"])
    return dict(sorted(aus.items()))


def _median_von(rows: list[dict], feld: str) -> float | None:
    werte = [z[feld] for z in rows if z.get(feld) is not None]
    return round(statistics.median(werte), 1) if werte else None


def _haeufigkeit(werte) -> dict[str, int]:
    aus: dict[str, int] = {}
    for w in werte:
        if w:
            aus[w] = aus.get(w, 0) + 1
    return dict(sorted(aus.items(), key=lambda x: -x[1]))


def _je_aufgabe(zeilen: list[dict]) -> list[dict]:
    """Welche Aufgabe trennt die Modelle, welche nicht?

    Eine Aufgabe, die alle loesen oder keiner loest, traegt zum Vergleich nichts
    bei. Prompt C macht daraus eine Abnahmebedingung, also gehoert die Zahl
    sichtbar in die Ansicht - nicht in eine Fussnote.
    """
    nach_aufgabe: dict[str, list[dict]] = {}
    for z in zeilen:
        nach_aufgabe.setdefault(z["schluessel"], []).append(z)

    aus = []
    for schluessel, rows in sorted(nach_aufgabe.items()):
        konfigurationen = {}
        for z in rows:
            k = f"{z['modell_alias']}{'' if z['thinking'] else ' (ohne Denken)'}"
            e = konfigurationen.setdefault(k, {"gesamt": 0, "erfolge": 0})
            e["gesamt"] += 1
            e["erfolge"] += 1 if z["erfolg"] else 0
        quoten = [e["erfolge"] / e["gesamt"] for e in konfigurationen.values() if e["gesamt"]]
        aus.append({
            "schluessel": schluessel,
            "titel": rows[0]["titel"],
            "kategorie": rows[0]["kategorie"],
            "schwierigkeit": rows[0]["schwierigkeit"],
            "konfigurationen": konfigurationen,
            "quote_min": round(min(quoten), 3) if quoten else None,
            "quote_max": round(max(quoten), 3) if quoten else None,
            # Genau das ist die Frage aus der Abnahme: differenziert die Aufgabe?
            "trennt": bool(quoten) and (max(quoten) - min(quoten)) > 0.01,
        })
    return aus


def lauf_detail(conn, run_id: int) -> dict[str, Any]:
    lauf = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if lauf is None:
        return {}
    zeilen = _ergebniszeilen(conn, [run_id])
    von = lauf["ts_start"]
    bis = lauf["ts_ende"] or (max((z["ts_ende"] or 0) for z in zeilen) if zeilen else von)

    # Die Hardwarekurve zum Lauf - der Punkt, an dem sich die gemeinsame
    # Datenbank auszahlt: ein SELECT statt einer Korrelation ueber zwei Systeme.
    kurve = [
        {k: z[k] for k in z.keys()}
        for z in conn.execute(
            "SELECT ts, vram_used_mib, vram_fremd_mib, gpu_util_pct, gpu_temp_c, "
            "       ram_used_gib, llama_gen_tps "
            "FROM samples WHERE ts BETWEEN ? AND ? ORDER BY ts", (von, bis or von),
        )
    ]
    schritt = max(1, len(kurve) // 600)
    return {
        "lauf": {k: lauf[k] for k in lauf.keys()},
        "ergebnisse": zeilen,
        "kurve": kurve[::schritt],
        "kurve_punkte_gesamt": len(kurve),
    }


def protokoll(conn, run_id: int, limit: int = 200) -> list[dict]:
    return [
        {k: z[k] for k in z.keys()}
        for z in conn.execute(
            "SELECT id, ts, richtung, endpunkt, http_status, inhalt FROM raw_logs "
            "WHERE run_id = ? ORDER BY id LIMIT ?", (run_id, limit),
        )
    ]
