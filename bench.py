"""Benchmark-Kommandozeile.

    python bench.py aufgaben                        Suiten und Aufgaben anzeigen
    python bench.py run --modell alle --suite rauchtest
    python bench.py run --modell qwen-moe --durchgaenge 3 --thinking beide
    python bench.py resume                          letzten abgebrochenen Lauf fortsetzen
    python bench.py compare                         Vergleich ueber alle Laeufe

Der Runner schaltet llama-server selbst zwischen den Konfigurationen um; das
Dashboard muss dafuer nicht laufen. Es sollte aber: nur dann steht hinterher in
jedem Ergebnis, was die Maschine waehrenddessen getan hat.
"""
from __future__ import annotations

import argparse
import sys
import time

from kibench import auswertung, db, serverwechsel
from kibench.adapter import ModellAdapter
from kibench.aufgaben import AufgabenFehler, suite_laden, suiten_auflisten, synchronisieren, verteilung
from kibench.config import MODELLE
from kibench.runner import Runner, reihenfolge_mischen


def _gpu():
    try:
        from kibench.gpu import GpuMonitor
        return GpuMonitor()
    except Exception:
        return None


def _konfigurationen(auswahl: str) -> list[str]:
    if auswahl in ("alle", "all"):
        return [m["alias"] for m in MODELLE]
    return [t.strip() for t in auswahl.split(",") if t.strip()]


def _modi(auswahl: str) -> list[bool]:
    return {"beide": [True, False], "ja": [True], "nein": [False]}[auswahl]


# --- Befehle ----------------------------------------------------------------

def befehl_aufgaben(args) -> int:
    for suite in ([args.suite] if args.suite else suiten_auflisten()):
        try:
            aufgaben = suite_laden(suite)
        except AufgabenFehler as e:
            print(f"{suite}: {e}")
            continue
        v = verteilung(aufgaben)
        print(f"\n=== {suite} ===  {len(aufgaben)} Aufgaben  "
              f"(leicht {v['leicht']} / mittel {v['mittel']} / schwer {v['schwer']})")
        for a in aufgaben:
            print(f"  {a.schluessel:<24} {a.schwierigkeit:<8} {a.kategorie:<28} {a.titel}")
    return 0


def befehl_run(args) -> int:
    aufgaben = suite_laden(args.suite)
    conn = db.init_db()
    ids = synchronisieren(conn, aufgaben)
    gpu = _gpu()
    adapter = ModellAdapter(gpu_monitor=gpu)
    runner = Runner(adapter, conn=conn, gpu_monitor=gpu)

    aliase = _konfigurationen(args.modell)
    modi = _modi(args.thinking)
    saat = args.saat if args.saat is not None else int(time.time())

    print(f"Suite {args.suite}: {len(aufgaben)} Aufgaben x {args.durchgaenge} Durchgaenge "
          f"x {len(aliase)} Konfigurationen x {len(modi)} Modus/Modi")
    print(f"Mischsaat {saat} (fuer eine identische Reihenfolge: --saat {saat})\n")

    for alias in aliase:
        for thinking in modi:
            code = _eine_konfiguration_fahren(
                runner, alias, thinking, aufgaben, ids, args, saat, gpu
            )
            if code != 0:
                return code

    print("\nFertig. Auswertung:  python bench.py compare")
    return 0


def _eine_konfiguration_fahren(runner, alias, thinking, aufgaben, ids, args, saat, gpu) -> int:
    kopf = f"{alias}{'' if thinking else ' (ohne Denken)'}"
    print(f"--- {kopf} ---")

    w = serverwechsel.bereitstellen(alias, gpu_monitor=gpu)
    if not w.ok:
        # Nicht stillschweigend weiterlaufen: eine halbe Messreihe gegen das
        # falsche Modell ist schlimmer als ein Abbruch.
        print(f"  Konnte {alias} nicht bereitstellen: {w.fehler}", file=sys.stderr)
        return 1
    if w.ladezeit_s:
        print(f"  geladen in {w.ladezeit_s}s, VRAM {w.vram_nachher_mib} MiB")

    run_id = runner.lauf_anlegen(alias, thinking, args.suite, notiz=args.suite)
    paare = reihenfolge_mischen(
        [(a, d) for a in aufgaben for d in range(1, args.durchgaenge + 1)], saat
    )
    for a, durchgang in paare:
        t0 = time.perf_counter()
        erg = runner.aufgabe_fahren(run_id, a, ids[a.schluessel], durchgang, alias, thinking)
        zeichen = "gruen" if erg.erfolg else "rot  "
        hinweis = ""
        if not erg.messpunkt_gueltig:
            hinweis = f"  [Messpunkt verworfen, {erg.nachladen_s}s Nachladen]"
        elif erg.sandbox_verstoesse:
            hinweis = "  [Sandbox-Ausbruch abgewehrt]"
        elif erg.fehlergrund:
            hinweis = f"  [{erg.fehlergrund}]"
        print(f"  {zeichen} {a.schluessel:<22} D{durchgang} "
              f"{time.perf_counter() - t0:6.1f}s  {erg.statistik.schritte} Schritte{hinweis}")
    runner.lauf_abschliessen(run_id)
    return 0


def befehl_resume(args) -> int:
    conn = db.init_db()
    zeile = conn.execute(
        "SELECT * FROM runs WHERE status = 'laufend' ORDER BY ts_start DESC LIMIT 1"
        if args.run is None else "SELECT * FROM runs WHERE id = ?",
        () if args.run is None else (args.run,),
    ).fetchone()
    if zeile is None:
        print("Kein abgebrochener Lauf gefunden.")
        return 0

    suite = zeile["notiz"] or "rauchtest"
    aufgaben = suite_laden(suite)
    ids = synchronisieren(conn, aufgaben)
    gpu = _gpu()
    runner = Runner(ModellAdapter(gpu_monitor=gpu), conn=conn, gpu_monitor=gpu)

    erledigt = runner.erledigte_durchgaenge(zeile["id"])
    offen = [(a, d) for a in aufgaben for d in range(1, args.durchgaenge + 1)
             if (ids[a.schluessel], d) not in erledigt]
    print(f"Lauf {zeile['id']} ({zeile['modell_alias']}): {len(erledigt)} erledigt, "
          f"{len(offen)} offen")
    if not offen:
        runner.lauf_abschliessen(zeile["id"])
        return 0

    w = serverwechsel.bereitstellen(zeile["modell_alias"], gpu_monitor=gpu)
    if not w.ok:
        print(f"  Konnte {zeile['modell_alias']} nicht bereitstellen: {w.fehler}", file=sys.stderr)
        return 1
    for a, d in offen:
        erg = runner.aufgabe_fahren(zeile["id"], a, ids[a.schluessel], d,
                                    zeile["modell_alias"], bool(zeile["thinking"]))
        print(f"  {'gruen' if erg.erfolg else 'rot  '} {a.schluessel} D{d}")
    runner.lauf_abschliessen(zeile["id"])
    return 0


def befehl_compare(args) -> int:
    conn = db.init_db()
    d = auswertung.vergleich(conn, suite=args.suite)
    if not d["konfigurationen"]:
        print("Noch keine Ergebnisse.")
        return 0

    print(f"\n{'Konfiguration':<28} {'Erfolg':>8} {'immer/manchmal/nie':>20} "
          f"{'tok/s':>8} {'Dauer':>8} {'Tool-Calls':>11} {'verworfen':>10}")
    print("-" * 100)
    for k in d["konfigurationen"]:
        name = k["alias"] + ("" if k["thinking"] else " (o. Denken)")
        quote = f"{k['erfolgsquote']:.0%}" if k["erfolgsquote"] is not None else "-"
        tps = f"{k['gen_tps_median']}" if k["gen_tps_median"] else "-"
        dauer = f"{k['dauer_median_s']}s" if k["dauer_median_s"] else "-"
        tq = f"{k['tool_call_quote']:.0%}" if k["tool_call_quote"] is not None else "-"
        # immer / manchmal / nie geloest - die mittlere Zahl ist die
        # interessanteste und faellt in jedem Mittelwert unter den Tisch.
        streuung = f"{k['aufgaben_immer']}/{k['aufgaben_manchmal']}/{k['aufgaben_nie']}"
        print(f"{name:<28} {quote:>8} {streuung:>20} "
              f"{tps:>8} {dauer:>8} {tq:>11} {k['messpunkte_verworfen']:>10}")

    trennend = [a for a in d["aufgaben"] if a["trennt"]]
    print(f"\nAufgaben, die zwischen den Konfigurationen trennen: "
          f"{len(trennend)} von {len(d['aufgaben'])}")
    if d["aufgaben"] and not trennend:
        print("  ACHTUNG: keine einzige Aufgabe differenziert. Dann misst die Suite nichts -")
        print("  siehe Abnahme in Prompt C.")
    print("\nAusfuehrlich im Dashboard: http://127.0.0.1:8420/vergleich")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="ki-benchmarks")
    unter = p.add_subparsers(dest="befehl", required=True)

    a = unter.add_parser("aufgaben", help="Suiten und Aufgaben anzeigen")
    a.add_argument("--suite")
    a.set_defaults(funktion=befehl_aufgaben)

    r = unter.add_parser("run", help="Suite gegen Modelle fahren")
    r.add_argument("--modell", default="alle", help="Alias, kommagetrennt, oder 'alle'")
    r.add_argument("--suite", default="rauchtest")
    r.add_argument("--durchgaenge", type=int, default=3)
    r.add_argument("--thinking", default="ja", choices=("beide", "ja", "nein"))
    r.add_argument("--saat", type=int, default=None)
    r.set_defaults(funktion=befehl_run)

    w = unter.add_parser("resume", help="abgebrochenen Lauf fortsetzen")
    w.add_argument("--run", type=int, default=None)
    w.add_argument("--durchgaenge", type=int, default=3)
    w.set_defaults(funktion=befehl_resume)

    v = unter.add_parser("compare", help="Vergleich ueber alle Laeufe")
    v.add_argument("--suite")
    v.set_defaults(funktion=befehl_compare)

    args = p.parse_args()
    return args.funktion(args)


if __name__ == "__main__":
    raise SystemExit(main())
