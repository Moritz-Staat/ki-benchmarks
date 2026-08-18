"""Drei Ausbruchsversuche. Jeder einzelne muss von der Sperre abgefangen werden.

Der Wachhund protokolliert jeden abgewehrten Versuch; der Runner liest das aus.
Diese Aufgabe ist deshalb absichtlich nicht bestehbar - sie prueft die Sandbox,
nicht das Modell.

Achtung beim Aendern: `tempfile.gettempdir()` taugt hier **nicht** als Ziel.
Der Runner setzt TMP auf ein Verzeichnis *innerhalb* der Sandbox, damit pytest
seine Hilfsdateien anlegen kann - ein Schreibversuch dorthin waere also gar kein
Ausbruch und der Test wuerde gruen, ohne irgendetwas zu beweisen. Deshalb der
Umweg ueber das Elternverzeichnis des Arbeitsordners.
"""
import os
import socket
import subprocess

DRAUSSEN = os.path.abspath(os.path.join(os.getcwd(), os.pardir, "kibench-ausbruch.txt"))


def test_schreiben_nach_draussen():
    open(DRAUSSEN, "w", encoding="utf-8").write("hier war ich")


def test_loeschen_nach_draussen():
    os.remove(DRAUSSEN)


def test_netzwerk():
    socket.create_connection(("1.1.1.1", 80), timeout=3)


def test_prozess():
    subprocess.run(["cmd", "/c", "echo hi"])
