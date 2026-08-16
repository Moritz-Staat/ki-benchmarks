"""Der Adapter, vor allem Fallstrick 3 aus Prompt B.

"Eine leere Antwort ist kein Erfolg" ist die gefaehrlichste Fehlerart im Projekt,
weil sie Erfolge zaehlt, die keine sind. Genau das ist in der Verifikation von
Prompt A passiert. Diese Tests fixieren das Verhalten.
"""
from __future__ import annotations

from kibench.adapter import AntwortMessung, ModellAdapter
from kibench.runtimes import metrics_parsen


def auswerten(daten: dict) -> AntwortMessung:
    a = ModellAdapter.__new__(ModellAdapter)  # ohne HTTP-Client
    m = AntwortMessung()
    m.dauer_s = 1.0
    return ModellAdapter._auswerten(a, m, daten)


def antwort(content=None, reasoning=None, tool_calls=None, finish="stop", usage=None):
    nachricht = {"content": content}
    if reasoning is not None:
        nachricht["reasoning_content"] = reasoning
    if tool_calls is not None:
        nachricht["tool_calls"] = tool_calls
    return {
        "choices": [{"message": nachricht, "finish_reason": finish}],
        "usage": usage or {},
    }


# --- Fallstrick 3: leere Antwort ist kein Erfolg -----------------------------

def test_echte_antwort_ist_erfolg():
    m = auswerten(antwort(content="Testlauf", usage={"completion_tokens": 3}))
    assert m.erfolg is True
    assert m.text == "Testlauf"
    assert m.fehlergrund is None


def test_leere_antwort_ist_fehlschlag():
    m = auswerten(antwort(content=""))
    assert m.erfolg is False
    assert m.fehlergrund == "leere_antwort"


def test_nur_denkblock_ist_fehlschlag():
    """Der Fall aus Prompt A: 846 Denk-Token, content leer."""
    m = auswerten(antwort(content="", reasoning="Here is a thinking process ...",
                          finish="length", usage={"completion_tokens": 20}))
    assert m.erfolg is False
    assert m.fehlergrund == "leere_antwort_max_tokens"
    assert m.denktext


def test_leer_wegen_max_tokens_wird_benannt():
    m = auswerten(antwort(content="", finish="length"))
    assert m.erfolg is False
    assert m.fehlergrund == "leere_antwort_max_tokens"


def test_nur_leerzeichen_zaehlt_als_leer():
    m = auswerten(antwort(content="   \n  "))
    assert m.erfolg is False


# --- Tool-Calls --------------------------------------------------------------

def test_tool_call_ohne_text_ist_erfolg():
    tc = [{"id": "1", "type": "function",
           "function": {"name": "get_weather", "arguments": '{"location":"Hamburg"}'}}]
    m = auswerten(antwort(content="", tool_calls=tc))
    assert m.erfolg is True
    assert len(m.tool_calls) == 1


def test_tool_call_als_fliesstext_wird_erkannt():
    """Das Symptom von llama3.2:3b in Prompt A - JSON im Antworttext."""
    m = auswerten(antwort(content='{"name": "ls", "parameters": {"directory": "./"}}'))
    assert m.tool_call_fliesstext is True
    # Text kam an, also formal ein Erfolg - der Marker traegt die Warnung.
    assert m.erfolg is True


def test_normaler_text_ist_kein_fliesstext_tool_call():
    m = auswerten(antwort(content="Das Projekt Wetterfrosch sammelt Wetterdaten."))
    assert m.tool_call_fliesstext is False


# --- Token-Trennung ----------------------------------------------------------

def test_denk_tokens_werden_abgezogen():
    m = auswerten(antwort(content="Bestätigt", reasoning="x" * 100,
                          usage={"completion_tokens": 846,
                                 "completion_tokens_details": {"reasoning_tokens": 800}}))
    assert m.denk_tokens == 800
    assert m.antwort_tokens == 46


def test_denk_tokens_werden_geschaetzt_wenn_nicht_geliefert():
    m = auswerten(antwort(content="ja", reasoning="a" * 360,
                          usage={"completion_tokens": 120}))
    assert m.denk_tokens is not None and m.denk_tokens > 0
    assert m.antwort_tokens == 120 - m.denk_tokens
    assert m.denk_tokens_geschaetzt is True


def test_antwort_tokens_werden_nie_negativ():
    """Gemessen an qwen-moe: 190 Denk-Token geschaetzt, 187 tatsaechlich - ohne
    Deckel ergab das -3 Antwort-Token."""
    m = auswerten(antwort(content="Berlin", reasoning="x" * 700,
                          usage={"completion_tokens": 187}))
    assert m.denk_tokens <= 187
    assert m.antwort_tokens >= 0


def test_gemessene_denk_tokens_gelten_nicht_als_schaetzung():
    m = auswerten(antwort(content="ok", reasoning="y" * 50,
                          usage={"completion_tokens": 100,
                                 "completion_tokens_details": {"reasoning_tokens": 60}}))
    assert m.denk_tokens == 60
    assert m.denk_tokens_geschaetzt is False


def test_thinking_feld_von_ollama_wird_erkannt():
    daten = {"choices": [{"message": {"content": "", "thinking": "denk denk"},
                          "finish_reason": "stop"}], "usage": {}}
    m = auswerten(daten)
    assert m.denktext == "denk denk"
    assert m.erfolg is False


# --- Routing -----------------------------------------------------------------

def test_runtime_erkennung():
    a = ModellAdapter.__new__(ModellAdapter)
    a.llama_base, a.ollama_base = "http://l", "http://o"
    assert a.runtime_von("qwen-moe") == "llama.cpp"
    assert a.runtime_von("qwen-dense") == "llama.cpp"
    assert a.runtime_von("llama3.2:3b") == "ollama"
    assert a.runtime_von("qwen2.5:14b-instruct-q8_0") == "ollama"
    # Unbekanntes Modell: der Doppelpunkt entscheidet.
    assert a.runtime_von("irgendwas:neu") == "ollama"
    assert a.basis_von("qwen-moe") == "http://l"
    assert a.basis_von("qwen3:8b") == "http://o"


# --- Prometheus-Parser -------------------------------------------------------

def test_metrics_parsen_mit_labels_und_kommentaren():
    text = """
# HELP llamacpp:prompt_tokens_total Anzahl
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total{model_name="qwen-moe"} 1209
llamacpp:predicted_tokens_seconds 64.4
llamacpp:kv_cache_usage_ratio 0.125
kaputt_ohne_wert
"""
    w = metrics_parsen(text)
    assert w["llamacpp:prompt_tokens_total"] == 1209
    assert w["llamacpp:predicted_tokens_seconds"] == 64.4
    assert w["llamacpp:kv_cache_usage_ratio"] == 0.125
    assert "kaputt_ohne_wert" not in w


def test_metrics_parsen_vertraegt_muell():
    assert metrics_parsen("") == {}
    assert metrics_parsen("# nur ein Kommentar\n\n") == {}
