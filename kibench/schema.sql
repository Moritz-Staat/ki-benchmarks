-- Vollstaendiges Schema, auch die Tabellen, die erst Prompt C fuellt.
-- Vorgabe aus Prompt B, Teil 2: jetzt komplett anlegen, keine Migrationen spaeter.

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

-- ---------------------------------------------------------------------------
-- samples - eine Zeile je Sekunde, ab sofort gefuellt
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS samples (
    id                      INTEGER PRIMARY KEY,
    ts                      REAL    NOT NULL,   -- Unix-Zeit, Sekunden
    ts_iso                  TEXT    NOT NULL,   -- lesbar, lokale Zeit

    -- GPU gesamt (NVML)
    gpu_util_pct            INTEGER,
    gpu_mem_util_pct        INTEGER,
    vram_used_mib           INTEGER,
    vram_free_mib           INTEGER,
    gpu_temp_c              INTEGER,
    gpu_clock_sm_mhz        INTEGER,
    gpu_clock_mem_mhz       INTEGER,
    gpu_power_w             REAL,

    -- VRAM je Prozessgruppe (Windows-PDH, siehe kibench/gpu.py)
    vram_llama_mib          INTEGER,
    vram_ollama_mib         INTEGER,
    vram_fremd_mib          INTEGER,
    vram_prozesse_json      TEXT,               -- [{pid, name, mib}, ...]

    -- CPU / RAM (psutil)
    cpu_pct                 REAL,
    cpu_kerne_json          TEXT,               -- [pct je logischem Kern]
    ram_used_gib            REAL,
    ram_pct                 REAL,
    pagefile_used_gib       REAL,
    pagefile_pct            REAL,
    disk_read_mibs          REAL,
    disk_write_mibs         REAL,

    -- llama-server
    llama_alive             INTEGER NOT NULL DEFAULT 0,
    llama_model             TEXT,               -- Alias aus /props
    llama_gen_tps           REAL,               -- llamacpp:predicted_tokens_seconds
    llama_prompt_tps        REAL,               -- llamacpp:prompt_tokens_seconds
    llama_kv_cache_pct      REAL,
    llama_kv_cache_tokens   INTEGER,
    llama_requests_processing INTEGER,
    llama_requests_deferred INTEGER,
    llama_metrics_json      TEXT,               -- alle uebrigen /metrics-Werte

    -- Ollama
    ollama_alive            INTEGER NOT NULL DEFAULT 0,
    ollama_modelle_json     TEXT                -- [{name, size_mib, bis}, ...]
);

CREATE INDEX IF NOT EXISTS idx_samples_ts ON samples(ts);

-- ---------------------------------------------------------------------------
-- runs - ein Benchmark-Durchlauf. Ab Prompt C.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY,
    modell_alias    TEXT    NOT NULL,
    modell_name     TEXT,
    runtime         TEXT,                       -- 'llama.cpp' | 'ollama'
    quant           TEXT,
    offload         TEXT,                       -- '-ngl 40', '--n-cpu-moe 20', ...
    kontext         INTEGER,
    thinking        INTEGER,                    -- 0/1, beide Modi werden gemessen
    ts_start        REAL    NOT NULL,
    ts_ende         REAL,
    status          TEXT    NOT NULL DEFAULT 'laufend',  -- laufend|fertig|abgebrochen|fehler
    notiz           TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_ts ON runs(ts_start);

-- ---------------------------------------------------------------------------
-- tasks - die Aufgabensammlung. Ab Prompt C.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tasks (
    id              INTEGER PRIMARY KEY,
    schluessel      TEXT    NOT NULL UNIQUE,    -- stabiler Name, z.B. 'readme-zusammenfassen'
    kategorie       TEXT,                       -- 'tool-call' | 'code' | 'zusammenfassung' | ...
    titel           TEXT    NOT NULL,
    prompt          TEXT    NOT NULL,
    erwartung       TEXT,                       -- Beschreibung oder Pruefausdruck
    werkzeuge_json  TEXT,                       -- Tool-Definitionen fuer diese Aufgabe
    max_tokens      INTEGER,
    quelle          TEXT,                       -- z.B. 'NOTIZEN.md'
    aktiv           INTEGER NOT NULL DEFAULT 1
);

-- ---------------------------------------------------------------------------
-- results - ein Ergebnis je Aufgabe, Modell und Durchgang. Ab Prompt C.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS results (
    id                  INTEGER PRIMARY KEY,
    run_id              INTEGER NOT NULL REFERENCES runs(id)  ON DELETE CASCADE,
    task_id             INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    durchgang           INTEGER NOT NULL DEFAULT 1,

    erfolg              INTEGER NOT NULL DEFAULT 0,
    fehlergrund         TEXT,                   -- u.a. 'leere_antwort', siehe adapter.py

    ts_start            REAL    NOT NULL,
    ts_ende             REAL,
    dauer_s             REAL,

    prompt_tokens       INTEGER,
    denk_tokens         INTEGER,                -- getrennt von den Antwort-Tokens
    antwort_tokens      INTEGER,
    gesamt_tokens       INTEGER,
    gen_tps             REAL,
    prompt_tps          REAL,
    finish_reason       TEXT,

    tool_calls_erwartet INTEGER,
    tool_calls_erhalten INTEGER,
    tool_calls_korrekt  INTEGER,
    tool_call_fliesstext INTEGER,               -- Tool-Aufruf kam als Text statt als tool_calls

    -- Kontext der Maschine waehrend des Laufs, aus samples aggregiert
    vram_max_mib        INTEGER,
    vram_fremd_max_mib  INTEGER,
    maschine_ruhig      INTEGER                 -- Fremd-VRAM-Schwankung unter der Schwelle?
);

CREATE INDEX IF NOT EXISTS idx_results_run  ON results(run_id);
CREATE INDEX IF NOT EXISTS idx_results_task ON results(task_id);

-- ---------------------------------------------------------------------------
-- raw_logs - vollstaendige Anfragen und Antworten. Ab Prompt C.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw_logs (
    id          INTEGER PRIMARY KEY,
    result_id   INTEGER REFERENCES results(id) ON DELETE CASCADE,
    run_id      INTEGER REFERENCES runs(id)    ON DELETE CASCADE,
    ts          REAL NOT NULL,
    richtung    TEXT NOT NULL,                  -- 'anfrage' | 'antwort'
    endpunkt    TEXT,
    http_status INTEGER,
    inhalt      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_raw_logs_result ON raw_logs(result_id);

-- ---------------------------------------------------------------------------
-- meta - Schemaversion, damit spaetere Prompts erkennen, was sie vorfinden
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS meta (
    schluessel  TEXT PRIMARY KEY,
    wert        TEXT NOT NULL
);
