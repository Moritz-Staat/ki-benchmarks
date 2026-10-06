"""Was bringt der eingebaute Vorhersagekopf (MTP) auf dieser Karte wirklich?

Qwen3.8-27B traegt einen `nextn`-Kopf direkt in der GGUF-Datei (Tensoren
`blk.64.nextn.*`, Metadatum `qwen35.nextn_predict_layers = 1`). `--spec-type
draft-mtp` verdrahtet ihn. Der Kopf schlaegt Token vor, das grosse Modell prueft
sie in einem Durchgang - angenommene Vorschlaege sind geschenkte Token.

Warum ein eigenes Werkzeug und nicht `langkontext.py`:

* Der **Nettoeffekt** zaehlt, nicht das Flag. MTP kostet selbst VRAM. Wenn dafuer
  Schichten in den RAM muessen, frisst der Verlust den Gewinn. Deshalb startet
  dieses Skript den Server **selbst** mit exakt den Argumenten einer Zeile der
  Messmatrix und misst jede Zeile mit ihrem eigenen `-ngl`.
* Die **Qualitaetspruefung** braucht den vollstaendigen Antworttext, nicht nur
  die tok/s. `langkontext.py` wirft ihn weg.
* MTP-Gewinn zeigt sich erst ueber mehr als ein paar Dutzend Token. 64 Token wie
  in `langkontext.py` sind zu wenig.

Gemessen wird gestreamt gegen `/v1/chat/completions`: Wanduhrzeit, Zeit bis zum
ersten Token, erzeugte Token je gewarteter Sekunde. Die Server-timings kommen
mit, aber sie entscheiden nichts - siehe `Invoke-LlamaMeasure` in
`scripts\\llama-lib.ps1`, warum.

    python tools/mtp_messung.py tiefe        # --spec-draft-n-max 1/2/3/4
    python tools/mtp_messung.py kontext      # 8k und 28k, mit und ohne MTP
    python tools/mtp_messung.py qualitaet    # temperature 0, Ausgabe zeichenweise vergleichen
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import ctypes
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from kibench.config import LLAMA_BASE, PROJEKT_DIR  # noqa: E402

LLAMA_SERVER = Path(r"C:\Users\morit\llama.cpp\llama-server.exe")
MODELLE = {
    "iq4": Path(r"C:\Users\morit\models\Qwen3.8-27B-IQ4_XS.gguf"),
    "q5": Path(r"C:\Users\morit\models\Qwen3.8-27B-UD-Q5_K_XL.gguf"),
    "moe": Path(r"C:\Users\morit\models\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"),
}
LOG_DIR = PROJEKT_DIR / "logs" / "mtp"

# Alles, was laut KLARSTELLUNG.md nicht angetastet wird: Denkmodus an, KV-Cache
# f16, Kontext 32768. MTP ist der einzige Hebel in diesem Lauf.
KONTEXT = 32768


# --- Prompts ----------------------------------------------------------------

ABSATZ = (
    "Die Bandbreite zwischen Grafikspeicher und Arbeitsspeicher bestimmt bei lokaler "
    "Sprachmodell-Inferenz fast alles. Ein Modell, dessen Gewichte vollstaendig im "
    "Grafikspeicher liegen, liest diese mit mehreren hundert Gigabyte pro Sekunde. "
    "Muessen Teile der Gewichte aus dem Arbeitsspeicher nachgeladen werden, faellt die "
    "effektive Bandbreite um etwa eine Groessenordnung. Da bei der Generierung jedes "
    "einzelnen Tokens saemtliche aktiven Gewichte einmal gelesen werden muessen, wirkt "
    "sich das unmittelbar auf die Tokens pro Sekunde aus. "
)
ZEICHEN_JE_TOKEN = 4.0


def fuelltext(ziel_tokens: int) -> str:
    zeichen = int(ziel_tokens * ZEICHEN_JE_TOKEN)
    return (ABSATZ * (zeichen // len(ABSATZ) + 1))[:zeichen]


def kontext_prompt(ziel_tokens: int) -> str:
    return ("Lies den folgenden Text und nenne in genau einem Satz sein Thema.\n\n"
            + fuelltext(ziel_tokens) + "\n\nAntwort:")


# Drei Prompts fuer die Qualitaetspruefung. Einer mit Code, einer mit langem
# Kontext - so verlangt es Aufgabe 5. Der dritte ist Fliesstext, damit nicht
# beide Faelle dieselbe Sorte Token erzeugen.
QUALITAETS_PROMPTS = {
    "code": (
        "Hier ist eine Python-Funktion mit einem Fehler:\n\n"
        "```python\n"
        "def gleitender_mittelwert(werte, fenster):\n"
        "    ergebnis = []\n"
        "    for i in range(len(werte)):\n"
        "        abschnitt = werte[i:i + fenster]\n"
        "        ergebnis.append(sum(abschnitt) / fenster)\n"
        "    return ergebnis\n"
        "```\n\n"
        "Benenne den Fehler und gib die korrigierte Funktion aus."
    ),
    "text": (
        "Erklaere in hoechstens zehn Saetzen, warum ein Mixture-of-Experts-Modell "
        "mit ausgelagerten Experten schneller bleiben kann als ein kleineres "
        "dichtes Modell mit ausgelagerten Schichten."
    ),
    "langkontext": (
        "Lies den folgenden Text sorgfaeltig und beantworte danach die Frage.\n\n"
        + fuelltext(8000)
        + "\n\nFrage: Welcher physikalische Kennwert entscheidet dem Text zufolge "
          "ueber die Geschwindigkeit, und wodurch unterscheidet sich dabei ein "
          "dichtes Modell von einem Mixture-of-Experts-Modell? Antworte in drei Saetzen."
    ),
}


# --- Server -----------------------------------------------------------------

ES_CONTINUOUS       = 0x80000000
ES_SYSTEM_REQUIRED  = 0x00000001


@contextlib.contextmanager
def standby_verhindern():
    """Haelt den Rechner wach, solange gemessen wird.

    **Warum es das gibt:** Der Q5-Lauf vom 19.08.2026 lief in den Standby. Der
    Rechner schlief um 17:00:51 ein und wurde erst um 21:53 per Netzschalter
    geweckt; eine einzelne Anfrage stand dadurch 4 h 55 min still (17 688 s
    Wanduhr fuer 1045 Token, waehrend das 3-Sekunden-Mittel normale 10 tok/s
    zeigte). Der Lauf war unbrauchbar und wurde verworfen.

    Der Fehler war strukturell: `qwen-dense.ps1` ruft `standby-aus.ps1` auf,
    dieses Skript startet `llama-server.exe` aber direkt und ging damit an der
    Absicherung vorbei. Solange jemand am Rechner sitzt, faellt das nicht auf -
    Tastatur und Maus setzen den Leerlaufzaehler zurueck.

    **Warum SetThreadExecutionState und nicht `standby-aus.ps1`:** Das Skript
    schreibt den Standby-Wert dauerhaft auf 0 und braucht `standby-an.ps1`, um
    ihn zurueckzusetzen. Stirbt der Messlauf dazwischen, bleibt der Rechner auf
    Dauer wach - genau das Risiko, das man bei einem unbeaufsichtigten Lauf
    nicht will. Diese Anforderung haengt dagegen am Prozess und verfaellt
    automatisch, wenn er endet, egal wie.

    Der Bildschirm darf weiter ausgehen (kein ES_DISPLAY_REQUIRED) - das
    unterbricht keinen Lauf, so haelt es auch `standby-aus.ps1`.
    """
    gesetzt = False
    try:
        # Ohne restype liefert ctypes einen vorzeichenbehafteten int, und
        # ES_CONTINUOUS (0x80000000) kommt als negative Zahl zurueck.
        fn = ctypes.windll.kernel32.SetThreadExecutionState
        fn.restype = ctypes.c_uint
        vorher = fn(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        gesetzt = vorher != 0
    except Exception as e:                                  # kein Windows
        print(f"    Standby-Sperre nicht gesetzt: {e}")
    if gesetzt:
        print("Standby ist fuer die Dauer des Laufs gesperrt "
              "(verfaellt automatisch beim Beenden).")
    else:
        print("WARNUNG: Standby konnte nicht gesperrt werden. Bei einem langen "
              "Lauf ohne Benutzer am Rechner sind die Zeiten wertlos.")
    try:
        yield gesetzt
    finally:
        if gesetzt:
            try:
                fn(ES_CONTINUOUS)
                print("Standby-Sperre wieder freigegeben.")
            except Exception:
                pass


def vram_used_mib() -> int | None:
    try:
        aus = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True)
        return int(aus.stdout.strip().splitlines()[0])
    except Exception:
        return None


@dataclass
class Konfiguration:
    """Eine Zeile der Messmatrix - genau ein Serverstart."""
    marke: str
    modell: str                       # Schluessel in MODELLE
    ngl: int
    mtp: bool = False
    n_max: int = 2
    zusatz: list[str] = field(default_factory=list)

    def argumente(self) -> list[str]:
        args = [
            "-m", str(MODELLE[self.modell]),
            "--host", "127.0.0.1", "--port", "8080",
            "-ngl", str(self.ngl),
            "-c", str(KONTEXT),
            "-fa", "on",
            "--jinja",
            "--metrics",
            "--alias", self.marke,
            # KV-Cache bleibt f16 (Vorgabe) - das ist der llama.cpp-Standard,
            # hier nur zur Sicherheit ausgeschrieben.
            "--cache-type-k", "f16", "--cache-type-v", "f16",
            # Gehoert **nicht** zu MTP, sondern in jede Zeile der Matrix. Ohne
            # das legt llama-server automatisch vier Plaetze mit gemeinsamem
            # KV-Cache an (n_slots = 4, kv_unified = true), MTP erzwingt aber
            # einen. Der erste Anlauf dieser Messreihe hatte es nur bei MTP
            # gesetzt - damit war nicht mehr das Flag der einzige Unterschied,
            # und der Basislauf widersprach sich zwischen zwei Durchgaengen
            # selbst, weil der Server den Platz per LRU wechselt.
            "--parallel", "1",
        ]
        if self.mtp:
            args += ["--spec-type", "draft-mtp",
                     "--spec-draft-n-max", str(self.n_max)]
        return args + self.zusatz


class Server:
    """Startet llama-server, wartet auf /health, raeumt zuverlaessig auf."""

    def __init__(self, konf: Konfiguration):
        self.konf = konf
        self.proc: subprocess.Popen | None = None
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stempel = time.strftime("%Y%m%d-%H%M%S")
        self.log = LOG_DIR / f"{konf.marke}-{stempel}.log"

    def __enter__(self) -> "Server":
        self.datei = self.log.open("w", encoding="utf-8", errors="replace")
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(
            [str(LLAMA_SERVER)] + self.konf.argumente(),
            stdout=self.datei, stderr=subprocess.STDOUT)
        frist = time.time() + 300
        while time.time() < frist:
            if self.proc.poll() is not None:
                self.ladezeit = None
                raise RuntimeError(
                    f"Server beendet mit Code {self.proc.returncode}\n"
                    + self.log_ende(30))
            try:
                r = httpx.get(f"{LLAMA_BASE}/health", timeout=3)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    self.ladezeit = round(time.perf_counter() - t0, 1)
                    return self
            except Exception:
                pass
            time.sleep(0.7)
        raise RuntimeError("Timeout beim Warten auf /health\n" + self.log_ende(30))

    def __exit__(self, *_):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        try:
            self.datei.close()
        except Exception:
            pass
        # Der Treiber gibt VRAM verzoegert frei - sonst misst die naechste
        # Konfiguration den Speicher der vorherigen mit.
        time.sleep(4)
        return False

    def lebt(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def log_ende(self, zeilen: int = 25) -> str:
        try:
            self.datei.flush()
        except Exception:
            pass
        try:
            return "\n".join(self.log.read_text(encoding="utf-8", errors="replace")
                             .splitlines()[-zeilen:])
        except Exception:
            return "(kein Log)"

    def entwurfs_statistik(self) -> dict:
        """llama-server protokolliert die Annahmequote des Vorhersagekopfs.

        Ohne diese Zahl ist ein ausbleibender Gewinn nicht zu deuten: entweder
        der Kopf schlaegt schlecht vor, oder er schlaegt gut vor und der
        Pruefdurchgang kostet zu viel.
        """
        text = self.log_ende(4000)
        werte = {}
        for schluessel, muster in (
            ("n_draft", r"n_draft\s*=\s*(\d+)"),
            ("n_predict", r"n_predict\s*=\s*(\d+)"),
            ("n_accept", r"n_accept\s*=\s*(\d+)"),
            ("annahme_prozent", r"accept(?:ance)?[^0-9%]*([\d.]+)\s*%"),
        ):
            treffer = re.findall(muster, text, re.I)
            if treffer:
                werte[schluessel] = treffer[-1]
        return werte


# --- Messung ----------------------------------------------------------------

def messen(marke: str, prompt: str, max_tokens: int, denken: bool,
           bis_zum_ende: bool = False, timeout_s: float = 1800.0) -> dict:
    """Eine gestreamte Anfrage.

    `bis_zum_ende` setzt `ignore_eos` - fuer Geschwindigkeitsmessungen die
    einzig brauchbare Einstellung. Der erste Versuch dieser Messreihe lief ohne
    und erzeugte 35 Token, weil das Modell die Frage in einem Satz beantwortet
    hat. Auf 35 Token ist der Unterschied zwischen 12 und 22 tok/s nicht von
    Rauschen zu unterscheiden. Fuer die Qualitaetspruefung bleibt es aus, dort
    soll die Antwort natuerlich enden.
    """
    koerper = {
        "model": marke,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": denken},
    }
    if bis_zum_ende:
        koerper["ignore_eos"] = True
    ergebnis: dict = {"ok": False}
    t0 = time.perf_counter()
    ttft = None
    text = []
    denk_text = []
    usage = timings = None

    try:
        with httpx.stream("POST", f"{LLAMA_BASE}/v1/chat/completions",
                          json=koerper, timeout=timeout_s) as r:
            if r.status_code != 200:
                ergebnis["fehler"] = f"http_{r.status_code}"
                return ergebnis
            for zeile in r.iter_lines():
                if not zeile.startswith("data: "):
                    continue
                nutz = zeile[6:]
                if nutz.strip() == "[DONE]":
                    break
                try:
                    d = json.loads(nutz)
                except json.JSONDecodeError:
                    continue
                delta = ((d.get("choices") or [{}])[0].get("delta") or {})
                stueck = delta.get("content")
                denk = delta.get("reasoning_content")
                if stueck or denk:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                if stueck:
                    text.append(stueck)
                if denk:
                    denk_text.append(denk)
                if d.get("usage"):
                    usage = d["usage"]
                if d.get("timings"):
                    timings = d["timings"]
    except Exception as e:
        ergebnis["fehler"] = f"{type(e).__name__}: {e}"
        ergebnis["wanduhr_s"] = round(time.perf_counter() - t0, 2)
        return ergebnis

    wanduhr = time.perf_counter() - t0
    ausgabe = "".join(text)
    denken_roh = "".join(denk_text)
    erzeugt = (usage or {}).get("completion_tokens")
    rechen = None
    if timings:
        rechen = (float(timings.get("prompt_ms", 0))
                  + float(timings.get("predicted_ms", 0))) / 1000

    ergebnis.update({
        "ok": bool(ausgabe or denken_roh),
        "wanduhr_s": round(wanduhr, 2),
        "ttft_s": round(ttft, 2) if ttft is not None else None,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "antwort_tokens": erzeugt,
        # Die einzige Zahl, die der Anwender erlebt: erzeugte Token durch
        # gewartete Sekunden, Prompt-Verarbeitung eingerechnet.
        "gen_tps_wanduhr": round(erzeugt / wanduhr, 2) if erzeugt and wanduhr else None,
        # Ohne die Wartezeit auf das erste Token - das ist die Zahl, die MTP
        # ueberhaupt beeinflussen kann.
        "gen_tps_nach_ttft": (round(erzeugt / (wanduhr - ttft), 2)
                              if erzeugt and ttft is not None and wanduhr > ttft else None),
        "server_gen_tps": (round(float(timings["predicted_per_second"]), 2)
                           if timings and timings.get("predicted_per_second") else None),
        "server_prompt_tps": (round(float(timings["prompt_per_second"]), 1)
                              if timings and timings.get("prompt_per_second") else None),
        "rechen_s": round(rechen, 2) if rechen is not None else None,
        "nachladen_s": round(max(0.0, wanduhr - rechen), 2) if rechen is not None else None,
        "ausgabe": ausgabe,
        "denken": denken_roh,
    })
    if not ergebnis["ok"]:
        ergebnis["fehler"] = "keine_token"
    return ergebnis


# --- Ausgabe ----------------------------------------------------------------

CSV_FELDER = ["marke", "modell", "ngl", "mtp", "n_max", "fall", "ladezeit_s",
              "aufwaermen_s",
              "vram_leer_mib", "vram_geladen_mib", "vram_last_mib",
              "prompt_tokens", "antwort_tokens", "ttft_s", "wanduhr_s",
              "gen_tps_wanduhr", "gen_tps_nach_ttft", "server_gen_tps",
              "server_prompt_tps", "nachladen_s", "ok", "fehler", "hinweis"]


def csv_schreiben(zeilen: list[dict], ziel: Path) -> None:
    ziel.parent.mkdir(parents=True, exist_ok=True)
    with ziel.open("w", newline="", encoding="utf-8") as f:
        s = csv.DictWriter(f, fieldnames=CSV_FELDER, extrasaction="ignore")
        s.writeheader()
        s.writerows(zeilen)
    print(f"\nGeschrieben: {ziel}")


def zeile_drucken(z: dict) -> None:
    if not z.get("ok"):
        print(f"    {z['fall']:<14} FEHLGESCHLAGEN: {z.get('fehler')}")
        return
    print(f"    {z['fall']:<14} {z['prompt_tokens']:>6} Prompt-Token  "
          f"TTFT {z['ttft_s']:>6}s  Wanduhr {z['wanduhr_s']:>7}s  "
          f"{z['antwort_tokens']:>4} Token  "
          f"{z['gen_tps_wanduhr']:>6} tok/s (Wanduhr)  "
          f"{z['gen_tps_nach_ttft']:>6} tok/s (Gen)  VRAM {z.get('vram_last_mib')} MiB")


def lauf(konf: Konfiguration, faelle: list[tuple[str, str, int, bool]],
         zeilen: list[dict], bis_zum_ende: bool = False) -> None:
    """Ein Serverstart, danach alle uebergebenen Faelle in Folge."""
    leer = vram_used_mib()
    kopf = (f"[{konf.marke}]  {konf.modell}  -ngl {konf.ngl}  "
            + (f"MTP n={konf.n_max}" if konf.mtp else "ohne MTP"))
    print(f"\n{kopf}")
    grund = {"marke": konf.marke, "modell": konf.modell, "ngl": konf.ngl,
             "mtp": konf.mtp, "n_max": konf.n_max if konf.mtp else "", "vram_leer_mib": leer}
    try:
        with Server(konf) as srv:
            geladen = vram_used_mib()
            print(f"    geladen in {srv.ladezeit}s, VRAM {leer} -> {geladen} MiB "
                  f"(frei: {16376 - (geladen or 0)} MiB)")
            grund["ladezeit_s"] = srv.ladezeit
            grund["vram_geladen_mib"] = geladen

            # Aufwaermanfrage, wird verworfen. Die erste Anfrage nach einem
            # Serverstart hing in der Messreihe vom 19.08.2026 dreimal zwischen
            # 624 und 1430 Sekunden und lief danach sofort normal - dieselben
            # Prompts im zweiten Durchgang in 95 bis 127 s. Ohne diesen Wurf
            # landet der Hänger auf dem ersten echten Messpunkt.
            aufwaermen = messen(konf.marke, "Antworte mit genau einem Wort: bereit.",
                                max_tokens=16, denken=False)
            print(f"    aufgewaermt: {aufwaermen.get('wanduhr_s')}s"
                  + ("" if aufwaermen.get("ok") else
                     f"  ACHTUNG: {aufwaermen.get('fehler')}"))
            grund["aufwaermen_s"] = aufwaermen.get("wanduhr_s")
            for name, prompt, max_tokens, denken in faelle:
                m = messen(konf.marke, prompt, max_tokens, denken, bis_zum_ende)
                z = {**grund, "fall": name, "vram_last_mib": vram_used_mib(), **m}
                zeilen.append(z)
                zeile_drucken(z)
                if not srv.lebt():
                    print("    Server ist gestorben - Rest dieser Konfiguration entfaellt.")
                    break
            stat = srv.entwurfs_statistik()
            if stat:
                print(f"    Entwurfsstatistik aus dem Serverlog: {stat}")
                for z in zeilen:
                    if z.get("marke") == konf.marke and not z.get("hinweis"):
                        z["hinweis"] = json.dumps(stat, ensure_ascii=False)
    except Exception as e:
        print(f"    START FEHLGESCHLAGEN: {e}")
        zeilen.append({**grund, "fall": "start", "ok": False, "fehler": str(e)[:400]})


# --- Die drei Laeufe --------------------------------------------------------

def lauf_tiefe(args) -> list[dict]:
    """Aufgabe 2: --spec-draft-n-max 1/2/3, und 4 einmal, um den Bruch zu sehen."""
    prompt = kontext_prompt(8000)
    faelle = [("8k", prompt, args.tokens, args.denken)]
    zeilen: list[dict] = []
    lauf(Konfiguration("iq4-ohne-mtp", "iq4", args.ngl), faelle, zeilen, True)
    for n in (1, 2, 3, 4):
        lauf(Konfiguration(f"iq4-mtp{n}", "iq4", args.ngl, mtp=True, n_max=n),
             faelle, zeilen, True)
    return zeilen


def lauf_kontext(args) -> list[dict]:
    """Aufgabe 3 und 4: Nettoeffekt bei 8k und 28k, beide Quantisierungen."""
    faelle = [(f"{laenge // 1000}k", kontext_prompt(laenge), args.tokens, args.denken)
              for laenge in args.laengen]
    zeilen: list[dict] = []
    for konf in args.matrix:
        lauf(konf, faelle, zeilen, True)
    return zeilen


def lauf_qualitaet(args) -> list[dict]:
    """Aufgabe 5: dieselbe Frage mit und ohne MTP, temperature 0.

    Jede Konfiguration laeuft **zweimal** ueber dieselben Prompts. Der zweite
    Durchgang ist keine Doppelung, sondern der Massstab: weicht schon der Lauf
    ohne MTP von sich selbst ab, sagt ein Unterschied zwischen mit und ohne MTP
    nichts ueber MTP aus.
    """
    ausgewaehlt = {k: v for k, v in QUALITAETS_PROMPTS.items()
                   if not args.prompts or k in args.prompts}
    faelle = [(f"{name}-{durchgang}", prompt, args.tokens, args.denken)
              for durchgang in range(1, args.durchgaenge + 1)
              for name, prompt in ausgewaehlt.items()]
    zeilen: list[dict] = []
    for konf in args.matrix:
        lauf(konf, faelle, zeilen)

    ziel = LOG_DIR / f"qualitaet-{time.strftime('%Y%m%d-%H%M%S')}"
    ziel.mkdir(parents=True, exist_ok=True)
    for z in zeilen:
        if z.get("ok"):
            (ziel / f"{z['marke']}--{z['fall']}.txt").write_text(
                (z.get("denken") or "") + "\n===ANTWORT===\n" + (z.get("ausgabe") or ""),
                encoding="utf-8")
    print(f"\nAntworttexte: {ziel}")
    vergleich_drucken(zeilen)
    return zeilen


def volltext(z: dict) -> str:
    return (z.get("denken") or "") + (z.get("ausgabe") or "")


def abweichung(a: str, b: str) -> tuple[int, str, str] | None:
    """Erste abweichende Stelle mit Umfeld, oder None bei Gleichheit."""
    if a == b:
        return None
    i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]),
             min(len(a), len(b)))
    return i, a[max(0, i - 70):i + 70], b[max(0, i - 70):i + 70]


def paar_drucken(titel: str, a: dict, b: dict, einzug: str = "   ") -> bool:
    """Vergleicht zwei Antworten. Rueckgabe: True, wenn identisch."""
    ta, tb = volltext(a), volltext(b)
    d = abweichung(ta, tb)
    if d is None:
        print(f"{einzug}{titel:<34} identisch ({len(ta)} Zeichen)")
        return True
    i, ua, ub = d
    anteil = round(100 * i / max(1, min(len(ta), len(tb))), 1)
    print(f"{einzug}{titel:<34} ABWEICHUNG ab Zeichen {i} "
          f"({anteil} % der Antwort)  Laengen {len(ta)}/{len(tb)}")
    print(f"{einzug}   A: ...{ua!r}")
    print(f"{einzug}   B: ...{ub!r}")
    return False


def vergleich_drucken(zeilen: list[dict]) -> None:
    """Zwei Vergleiche, und die Reihenfolge ist der ganze Punkt.

    **Zuerst die Eigenvarianz**: dieselbe Konfiguration, derselbe Prompt,
    zweimal. Weicht die schon von sich selbst ab, ist der Server bei
    `temperature 0` nicht reproduzierbar - und dann sagt ein Unterschied
    zwischen mit und ohne MTP nichts ueber MTP aus, sondern nur ueber
    Gleitkomma-Reduktionsreihenfolgen.

    **Dann der Vergleich zwischen den Konfigurationen**, jeweils Durchgang
    gegen denselben Durchgang. Dabei sind zwei Faelle streng zu trennen:

    * `ohne` gegen `mtp2` - hier aendert sich die Batch-Form, weil mehrere
      Token gemeinsam geprueft werden. Eine Abweichung kann Artefakt sein.
    * `mtp1` gegen `mtp2` gegen `mtp3` - die Entwurfstiefe darf die
      **akzeptierte** Verteilung theoretisch nicht veraendern. Weichen die
      drei untereinander ab, ist damit belegt, dass Abweichungen in dieser
      Messreihe Batch-Artefakte sind und kein Qualitaetsverlust.
    """
    # Die Konfiguration ohne MTP ist der Bezugspunkt, nicht die alphabetisch
    # erste - sonst waere unten "mtp1" der Massstab und die Tabelle unlesbar.
    marken = sorted({z["marke"] for z in zeilen if z.get("ok")},
                    key=lambda m: (0 if m.endswith("ohne") else 1, m))
    prompts = sorted({z["fall"].rsplit("-", 1)[0] for z in zeilen if z.get("ok")})

    def hole(marke: str, prompt: str, durchgang: int) -> dict | None:
        for z in zeilen:
            if z.get("ok") and z["marke"] == marke and z["fall"] == f"{prompt}-{durchgang}":
                return z
        return None

    hat_zweiten = any(z["fall"].endswith("-2") for z in zeilen if z.get("ok"))
    eigen_stabil = True
    if not hat_zweiten:
        print("\n(Nur ein Durchgang gemessen - die Eigenvarianz bleibt damit unbekannt.)")
    else:
        print("\n" + "=" * 78)
        print("1) EIGENVARIANZ - dieselbe Konfiguration zweimal, temperature 0")
        print("=" * 78)
        for marke in marken:
            print(f"\n{marke}")
            for prompt in prompts:
                a, b = hole(marke, prompt, 1), hole(marke, prompt, 2)
                if not (a and b):
                    print(f"   {prompt:<34} unvollstaendig")
                    eigen_stabil = False
                    continue
                if not paar_drucken(f"{prompt}: Durchgang 1 vs 2", a, b):
                    eigen_stabil = False

    print("\n" + "=" * 78)
    print("2) ZWISCHEN DEN KONFIGURATIONEN - je Durchgang 1")
    print("=" * 78)
    if not eigen_stabil:
        print("\nACHTUNG: schon die Eigenvarianz ist nicht null. Alles unten ist")
        print("damit nicht mehr MTP zuzurechnen - siehe Abschnitt 1.\n")
    bezug = marken[0] if marken else None
    for prompt in prompts:
        print(f"\n{prompt}  (Bezug: {bezug})")
        for marke in marken[1:]:
            a, b = hole(bezug, prompt, 1), hole(marke, prompt, 1)
            if a and b:
                paar_drucken(f"{bezug} vs {marke}", a, b)

    tiefen = [m for m in marken if "mtp" in m]
    if len(tiefen) > 1:
        print("\n" + "=" * 78)
        print("3) ENTWURFSTIEFEN UNTEREINANDER - darf theoretisch nichts aendern")
        print("=" * 78)
        for prompt in prompts:
            print(f"\n{prompt}")
            for marke in tiefen[1:]:
                a, b = hole(tiefen[0], prompt, 1), hole(marke, prompt, 1)
                if a and b:
                    paar_drucken(f"{tiefen[0]} vs {marke}", a, b)


def matrix_bauen(text: str) -> list[Konfiguration]:
    """`iq4:54`, `iq4:54:mtp2`, `q5:42:mtp2` - eine Zeile je Serverstart."""
    konfs = []
    for teil in text.split(","):
        stueck = teil.strip().split(":")
        modell, ngl = stueck[0], int(stueck[1])
        if len(stueck) > 2 and stueck[2].startswith("mtp"):
            n = int(stueck[2][3:] or 2)
            konfs.append(Konfiguration(f"{modell}-{ngl}-mtp{n}", modell, ngl, True, n))
        else:
            konfs.append(Konfiguration(f"{modell}-{ngl}-ohne", modell, ngl))
    return konfs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("lauf", choices=["tiefe", "kontext", "qualitaet"])
    ap.add_argument("--ngl", type=int, default=54, help="nur fuer 'tiefe'")
    ap.add_argument("--matrix", default="iq4:54,iq4:54:mtp2",
                    help="Kommaliste, z.B. iq4:54,iq4:54:mtp2,q5:42,q5:42:mtp2")
    ap.add_argument("--laengen", nargs="*", type=int, default=[8000, 28000],
                    help="Prompt-Laengen fuer 'kontext'")
    ap.add_argument("--tokens", type=int, default=256, help="max_tokens je Anfrage")
    ap.add_argument("--prompts", nargs="*", default=None,
                    choices=list(QUALITAETS_PROMPTS),
                    help="nur diese Qualitaets-Prompts (ohne Angabe: alle)")
    ap.add_argument("--durchgaenge", type=int, default=2,
                    help="Wiederholungen je Prompt - 2 misst die Eigenvarianz mit")
    ap.add_argument("--denken", action="store_true",
                    help="Denkmodus an (fuer die Qualitaetspruefung richtig, "
                         "fuer Geschwindigkeit unnoetig teuer)")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    args.matrix = matrix_bauen(args.matrix)

    if not LLAMA_SERVER.exists():
        print(f"Nicht gefunden: {LLAMA_SERVER}", file=sys.stderr)
        return 1

    t0 = time.time()
    with standby_verhindern():
        zeilen = {"tiefe": lauf_tiefe, "kontext": lauf_kontext,
                  "qualitaet": lauf_qualitaet}[args.lauf](args)
    ziel = Path(args.csv) if args.csv else (
        LOG_DIR / f"mtp-{args.lauf}-{time.strftime('%Y%m%d-%H%M%S')}.csv")
    csv_schreiben(zeilen, ziel)
    print(f"Gesamtdauer: {round((time.time() - t0) / 60, 1)} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
