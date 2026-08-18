"""Zentrale Konfiguration.

Alle Werte, die aus Prompt A stammen, stehen hier an einer Stelle - damit die
Warnschwellen des Dashboards nicht an drei Orten gepflegt werden muessen.
Quelle ist ..\\..\\SETUP.md, Abschnitt "Das Ergebnis auf einen Blick".
"""
from __future__ import annotations

from pathlib import Path

# --- Pfade ------------------------------------------------------------------

REPO_DIR = Path(__file__).resolve().parent.parent
PROJEKT_DIR = REPO_DIR.parent                      # ...\localai
DB_PATH = REPO_DIR / "kibench.db"                  # via .gitignore (*.db) ausgeschlossen
STATIC_DIR = Path(__file__).resolve().parent / "static"

# --- Endpunkte --------------------------------------------------------------

LLAMA_BASE = "http://127.0.0.1:8080"
OLLAMA_BASE = "http://127.0.0.1:11434"

# Nur lokal binden - Vorgabe aus Prompt A und B.
HOST = "127.0.0.1"
PORT = 8420

# --- Die sechs Konfigurationen ----------------------------------------------
#
# Fuenf Modelle, aber sechs Konfigurationen: das dense Modell laeuft in zwei
# Quantisierungen, und der Unterschied ist zu gross, um eine davon wegzulassen -
# IQ4_XS ist 2,5-mal schneller als Q5 (18,38 gegen 7,34 tok/s), aber ob die
# Antwortqualitaet mithaelt, ist ungemessen. Genau das beantwortet Prompt C.
#
# `wechsel_ziel` ist das Argument fuer scripts\modell-wechsel.ps1. Nur die
# llama.cpp-Konfigurationen brauchen es; die drei Ollama-Modelle laedt Ollama
# bei Bedarf selbst.

MODELLE = [
    {
        "alias": "qwen-dense",
        "name": "Qwen3.8-27B dense Q5",
        "runtime": "llama.cpp",
        "quant": "UD-Q5_K_XL",
        "offload": "-ngl 46",
        "wechsel_ziel": "dense",
        # Sweep-Werte aus SETUP.md, Basis fuer die tok/s-Warnschwelle.
        # Aus dem Wiederholungs-Sweep vom 18.08.2026, nicht aus Prompt A: der
        # erste Lauf mass mit aktivem Sysmem-Fallback und lieferte fuer -ngl 42
        # aufwaerts Zahlen, die den Treiber beschreiben und nicht das Modell.
        "sweep_gen_tps": 7.34,
        "sweep_prompt_tps": 608.3,
    },
    {
        "alias": "qwen-dense-iq4",
        "name": "Qwen3.8-27B dense IQ4_XS",
        "runtime": "llama.cpp",
        "quant": "IQ4_XS",
        "offload": "-ngl 58",
        "wechsel_ziel": "dense-iq4",
        # Wiederholungs-Sweep 18.08.2026, wanduhrgeprueft.
        "sweep_gen_tps": 18.38,
        "sweep_prompt_tps": 1037.6,
    },
    {
        "alias": "qwen-moe",
        "name": "Qwen3.6-35B-A3B MoE Q4",
        "runtime": "llama.cpp",
        "quant": "UD-Q4_K_XL",
        "offload": "-ngl 99 --n-cpu-moe 20",
        "wechsel_ziel": "moe",
        "sweep_gen_tps": 64.40,
        "sweep_prompt_tps": 377.3,
    },
    {
        "alias": "qwen2.5:14b-instruct-q8_0",
        "name": "Qwen2.5 14B q8",
        "runtime": "ollama",
        "quant": "q8_0",
        "offload": "34% CPU / 66% GPU",
        "sweep_gen_tps": None,
        "sweep_prompt_tps": None,
    },
    {
        "alias": "qwen3:8b",
        "name": "Qwen3 8B",
        "runtime": "ollama",
        "quant": "q4_K_M",
        "offload": None,
        "sweep_gen_tps": None,
        "sweep_prompt_tps": None,
    },
    {
        "alias": "llama3.2:3b",
        "name": "Llama3.2 3B",
        "runtime": "ollama",
        "quant": "q4_K_M",
        "offload": None,
        "sweep_gen_tps": None,
        "sweep_prompt_tps": None,
    },
]

MODELLE_NACH_ALIAS = {m["alias"]: m for m in MODELLE}

# --- Warnschwellen aus Prompt B ---------------------------------------------

VRAM_TOTAL_MIB = 16376
RAM_TOTAL_GIB = 64

SCHWELLE_VRAM_MIB = int(15.2 * 1024)   # 15564 MiB - OOM-Gefahr
SCHWELLE_RAM_GIB = 56.0                # ab hier droht Auslagerung
SCHWELLE_GPU_TEMP_C = 83               # thermisches Throttling

# "tok/s unter der Haelfte des Sweep-Werts" - Faktor an einer Stelle.
SCHWELLE_TPS_ANTEIL = 0.5

# Fremd-VRAM-Schwankung, ab der die Live-Ansicht warnt (Prompt B, Teil 1).
SCHWELLE_FREMD_VRAM_SCHWANKUNG_MIB = 200
# Fenster, ueber das die Schwankung bestimmt wird.
FREMD_VRAM_FENSTER_S = 60

# --- Sampler ----------------------------------------------------------------

SAMPLE_INTERVALL_S = 1.0
# Ollama nach geladenen Modellen fragen ist teurer als NVML - seltener abfragen.
OLLAMA_PS_INTERVALL_S = 5.0

# HTTP-Timeout fuer die Messquellen. Bewusst knapp: ein lokaler Server antwortet
# in rund 2 ms, alles darueber ist ein Problem und darf den Takt nicht aufhalten.
MESS_TIMEOUT_S = 0.25

# Gemessen: ein Verbindungsversuch auf einen toten Port unter Windows kostet den
# vollen Timeout - bei 0,8 s waren das 810 ms von 1000 ms Taktbudget, nur um
# festzustellen, dass llama-server nicht laeuft. Deshalb steht vor jeder HTTP-
# Abfrage ein Prozess-Check (rund 1 ms), und nach einem Fehlschlag eine Sperre.
PROZESS_CACHE_S = 1.0
RUNTIME_BACKOFF_S = 2.0

# Prozesse, deren VRAM als "eigener" Verbrauch gilt. Alles andere ist fremd.
EIGENE_PROZESSE = {"llama-server.exe", "ollama.exe", "ollama app.exe"}
