# ki-benchmarks

Messapparat fuer lokale LLM-Inferenz. Siehe `..\SETUP.md` fuer die in Prompt A
ermittelten Offload-Werte, Ports und Modellnamen.

## Stand

**Prompt B, Teil 1-4 gebaut, Abnahme noch offen.** Details in `..\FORTSETZEN-B.md`.

## Einrichten

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Alles laeuft im venv des Repos, **nichts systemweit**. `pywin32` wird nur unter Windows
gebraucht und liefert `win32pdh` fuer den VRAM je Prozess.

## Starten

Doppelklick auf die Desktop-Verknuepfung **KI-Dashboard**, anzulegen mit:

```powershell
.\tools\verknuepfung-anlegen.ps1     # -Entfernen macht es rueckgaengig
```

Oder von Hand:

```powershell
.\.venv\Scripts\python.exe dashboard.py
```

Oeffnet http://127.0.0.1:8420/ - Sampler und Live-Ansicht laufen im selben Prozess.
`--kein-browser` startet ohne Browser, `--port` aendert den Port.

## Aufbau

| Datei | Inhalt |
|---|---|
| `kibench/config.py` | alle Werte aus Prompt A an einer Stelle, Warnschwellen |
| `kibench/schema.sql` | vollstaendiges Schema, auch die Tabellen fuer Prompt C |
| `kibench/db.py` | SQLite mit WAL, ein Schreiber, viele Leser |
| `kibench/gpu.py` | NVML fuer die Karte, Windows-PDH fuer VRAM je Prozess |
| `kibench/sysinfo.py` | CPU, RAM, Pagefile, Disk-I/O ueber psutil |
| `kibench/runtimes.py` | llama-server `/metrics` und `/props`, Ollama `/api/ps` |
| `kibench/sampler.py` | Hintergrund-Thread, eine Messung je Sekunde |
| `kibench/adapter.py` | einheitlicher Modellzugriff ueber beide Runtimes |
| `kibench/api.py` | FastAPI: JSON-Endpunkte und die Live-Seite |
| `kibench/static/` | die Seite, uPlot lokal eingebunden (kein CDN, kein Node) |
| `tools/dauerlauf_pruefen.py` | Abnahme: laeuft der Sampler ohne Speicherleck? |

## Zwei Dinge, die beim Bauen nicht so waren wie erwartet

**NVML liefert auf dieser Karte keinen VRAM je Prozess.** `usedGpuMemory` ist bei
allen Prozessen `None`, in v1, v2 und v3 - die WDDM-Einschraenkung bei GeForce
unter Windows. Prompt B verlangt den Fremd-VRAM aber ausdruecklich. Geloest ueber
den Windows-Leistungsindikator `GPU Process Memory`, dieselbe Quelle wie der
Task-Manager, ueber `win32pdh` und rund 0,03 ms je Abfrage. Siehe `kibench/gpu.py`.

**Ein toter Port kostet unter Windows den vollen Timeout.** Die HTTP-Abfrage an
einen nicht laufenden `llama-server` brauchte 810 ms von 1000 ms Taktbudget.
Vor jeder HTTP-Abfrage steht deshalb ein Prozess-Check (rund 1 ms) und danach
eine Sperre. Damit: 3 ms je Messung statt 810.
