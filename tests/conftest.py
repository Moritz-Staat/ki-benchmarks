"""Gemeinsame Vorbereitung.

Wichtig: die Tests duerfen **nicht** in die echte Messdatenbank schreiben, sonst
verfaelscht jeder Testlauf die Zeitreihe. Deshalb wird `kibench.db.DB_PATH` vor
dem Start auf eine Datei im tmp-Verzeichnis gezogen.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kibench import db  # noqa: E402


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    pfad = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_PATH", pfad)
    # Die Verbindung haengt am Thread und wuerde sonst zwischen Tests ueberleben.
    monkeypatch.setattr(db._lokal, "conn", None, raising=False)
    monkeypatch.setattr(db._lokal, "pfad", None, raising=False)
    conn = db.init_db(pfad)
    yield conn
    conn.close()


@pytest.fixture
def client(tmp_path, monkeypatch):
    """FastAPI-TestClient mit eigener Datenbank.

    Der Lifespan startet den echten Sampler - genau das soll geprueft werden:
    dass die Anwendung von selbst hochkommt und misst.
    """
    from fastapi.testclient import TestClient

    pfad = tmp_path / "api.db"
    monkeypatch.setattr(db, "DB_PATH", pfad)
    monkeypatch.setattr(db._lokal, "conn", None, raising=False)
    monkeypatch.setattr(db._lokal, "pfad", None, raising=False)

    from kibench import api

    with TestClient(api.app) as c:
        yield c
