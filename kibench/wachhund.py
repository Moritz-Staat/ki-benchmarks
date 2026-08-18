"""Schreib- und Netzsperre fuer modellgenerierten Code.

Der Runner fuehrt Code aus, den ein Sprachmodell geschrieben hat. Das ist die
einzige Stelle im ganzen Projekt mit echtem Schadenspotenzial - nicht die
Hardware, nicht der VRAM. Ein Modell, das eine Aufgabe missversteht, raeumt im
Zweifel ein Verzeichnis auf, das ihm nicht gehoert.

Die Sperre haengt an `sys.addaudithook` (CPython 3.8+). Audit-Ereignisse werden
tief im Interpreter ausgeloest, nicht in der Python-Ebene daruber: `open`,
`os.remove` und `socket.connect` melden sich auch dann, wenn der Aufruf ueber
`pathlib`, `shutil` oder ein fremdes Paket laeuft. Ein Monkeypatch auf
`builtins.open` waere in drei Zeilen zu umgehen, ein Audit-Hook nicht - er
laesst sich nach der Installation auch nicht mehr entfernen.

**Was gesperrt ist**

* Schreiben, Anlegen, Loeschen, Umbenennen ausserhalb des Arbeitsverzeichnisses
* jede Netzwerkverbindung
* das Starten weiterer Prozesse (sonst waere die Sperre ueber `subprocess`
  in einem Schritt umgangen)

**Was erlaubt bleibt**

Lesen. Die Testsuite muss die Standardbibliothek und pytest importieren
koennen, und Lesen richtet keinen Schaden an.

**Was die Sperre nicht leistet** - und das gehoert dazugesagt, weil eine
ueberschaetzte Sicherung schlimmer ist als eine bekannte Luecke: Sie wirkt
innerhalb des Python-Interpreters. Wer ueber `ctypes` direkt Systemaufrufe
absetzt, kommt daran vorbei. Fuer den Zweck hier - Code, der eine Coding-Aufgabe
loesen soll - ist das die richtige Abwaegung; fuer die Ausfuehrung absichtlich
boesartigen Codes waere es zu wenig. Dafuer braeuchte es eine VM.
"""
from __future__ import annotations

import os
import sys

# Ereignisse, die eine Datei veraendern. Der erste Parameter ist jeweils ein
# Pfad; bei `open` steht der Modus dahinter.
_SCHREIB_EREIGNISSE = {
    "os.mkdir": (0,),
    "os.rmdir": (0,),
    "os.remove": (0,),
    "os.unlink": (0,),
    "os.rename": (0, 1),
    "os.replace": (0, 1),
    "os.link": (0, 1),
    "os.symlink": (0, 1),
    "os.truncate": (0,),
    "os.chmod": (0,),
    "os.chown": (0,),
    "os.utime": (0,),
    "shutil.copyfile": (0, 1),
    "shutil.copymode": (1,),
    "shutil.copystat": (1,),
    "shutil.move": (0, 1),
    "shutil.rmtree": (0,),
}

_PROZESS_EREIGNISSE = {
    "subprocess.Popen",
    "os.system",
    "os.exec",
    "os.posix_spawn",
    "os.spawn",
    "os.startfile",
}

_NETZ_EREIGNISSE = {
    "socket.connect",
    "socket.bind",
    "socket.getaddrinfo",
    "socket.sendto",
    "urllib.Request",
}


# Jeder abgewehrte Versuch landet hier. Das ist kein Debug-Beiwerk, sondern die
# Abnahmebedingung aus Prompt C: eine Aufgabe, die absichtlich ausbrechen will,
# muss scheitern - und dass sie es *deshalb* getan hat, muss belegbar sein.
# Ohne diese Liste sieht ein abgewehrter Ausbruch aus wie ein normaler
# Testfehler, weil pytest die Ausnahme faengt und als solchen verbucht.
verstoesse: list[str] = []


class SandboxVerstoss(PermissionError):
    """Wird ausgeloest, wenn generierter Code aus dem Arbeitsverzeichnis ausbricht.

    Bewusst eine Unterklasse von PermissionError: Testcode, der Ausnahmen
    breit faengt, soll das hier nicht versehentlich verschlucken - aber ein
    Aufrufer, der gezielt auf Rechteprobleme prueft, versteht es trotzdem.
    """


# Das Null-Geraet. pytest leitet beim Abfangen der Ausgabe dorthin um; es ist
# per Definition kein Ziel, an dem etwas kaputtgehen kann.
_NULLGERAETE = {"nul", "null", "devnull"}


def _ist_nullgeraet(p: str) -> bool:
    """`nul`, `\\\\.\\nul` und `/dev/null` - alles dasselbe Nichts."""
    rest = p.strip().lower().replace("/", "\\").strip("\\")
    rest = rest.replace(".\\", "").replace("dev\\", "dev")
    return rest in _NULLGERAETE


def _im_bereich(pfad, wurzel: str) -> bool:
    if pfad is None:
        return True
    try:
        p = os.fspath(pfad)
    except TypeError:
        # Datei-Deskriptoren und Sockets erreichen uns hier als int - die
        # gehoeren zu einer bereits geoeffneten Datei und sind nicht der Ort,
        # an dem ueber Zugriff entschieden wird.
        return True
    if not isinstance(p, str):
        try:
            p = p.decode("utf-8", "replace")
        except Exception:
            return False
    if not p:
        return True
    if _ist_nullgeraet(p):
        return True
    try:
        ganz = os.path.realpath(os.path.abspath(p))
    except Exception:
        return False
    return ganz == wurzel or ganz.startswith(wurzel + os.sep)


def aktivieren(arbeitsverzeichnis: str) -> None:
    """Sperre einschalten. Ab hier gibt es kein Zurueck - das ist Absicht."""
    wurzel = os.path.realpath(os.path.abspath(arbeitsverzeichnis))

    # Ohne das schreibt der Interpreter __pycache__ neben die Quelldateien -
    # auch neben die von pytest, und das liegt ausserhalb. Ein Bytecode-Cache
    # waere kein Schaden, aber die Sperre kann das nicht unterscheiden, und
    # ein Fehlalarm im ersten Import wuerde jeden Lauf zerstoeren.
    sys.dont_write_bytecode = True

    def melden(text: str) -> SandboxVerstoss:
        verstoesse.append(text)
        return SandboxVerstoss(text)

    def hook(ereignis: str, argumente):
        if ereignis == "open":
            pfad, modus = argumente[0], argumente[1]
            if modus and any(z in str(modus) for z in ("w", "a", "x", "+")):
                if not _im_bereich(pfad, wurzel):
                    raise melden(f"Schreibzugriff ausserhalb der Sandbox: {pfad!r}")
            return

        stellen = _SCHREIB_EREIGNISSE.get(ereignis)
        if stellen is not None:
            for i in stellen:
                if i < len(argumente) and not _im_bereich(argumente[i], wurzel):
                    raise melden(f"{ereignis} ausserhalb der Sandbox: {argumente[i]!r}")
            return

        if ereignis in _NETZ_EREIGNISSE:
            raise melden(f"Netzwerkzugriff gesperrt ({ereignis})")

        if ereignis in _PROZESS_EREIGNISSE or ereignis.startswith("os.exec"):
            raise melden(f"Prozessstart gesperrt ({ereignis})")

    sys.addaudithook(hook)
