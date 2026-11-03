#!/usr/bin/env python3
"""
tests/test_ai.py — AI layer tests (offline): fallback provider, tool
allowlist, guardrails, capability registry, memory.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_fallback_provider_always_available():
    from ai.fallback_provider import FallbackProvider
    p = FallbackProvider()
    assert p.available()
    out = p.chat([{"role": "user", "content": "what is the nodularity?"}],
                 context={"iso945": {"nodularity_pct": 71.4},
                          "detections": {"count": 60},
                          "self_check": {"f1_pct": 100.0}})
    assert "71.4" in out["content"]
    assert out["provider"] == "fallback"


def test_fallback_never_invents_scale():
    from ai.fallback_provider import FallbackProvider
    p = FallbackProvider()
    out = p.chat([{"role": "user",
                   "content": "what is the size in micrometers?"}],
                 context={"calibration": {"available": False}})
    content = out["content"]
    assert "unavailable" in content.lower()
    assert "µm/pixel" not in content        # no fabricated scale factor
    assert "pixels" in content


def test_fallback_capabilities_listing():
    from ai.fallback_provider import FallbackProvider
    p = FallbackProvider()
    out = p.chat([{"role": "user", "content": "what can you do?"}], context={})
    assert "quality_gate" in out["content"]  # from the live registry


# ------------------------------------------------------------ tools ----
def _ctx():
    return {
        "analysis_id": None,
        "contract": lambda aid: {"sample": {"analysis_id": aid},
                                 "iso945": {"nodularity_pct": 85.0}},
        "latest_analysis": lambda: {"id": "ANL-1"},
        "analyze_dataset_image": lambda name: {"analysis": {
            "analysis_id": "ANL-NEW", "nodule_count": 10}},
        "nodule_details": lambda aid, nid: {"nodule": {"nodule_id": nid}},
        "compare": lambda a, b: {"summary": "b higher", "nodularity": {}},
        "batch_stats": lambda: {"samples": 2},
        "spc": lambda m: {"sufficient_data": False},
        "anomalies": lambda: {"anomalies": [], "scanned": 0},
        "decision_support": lambda aid: {"measured": ["ISO 85%"]},
        "validation": lambda kind: {"source": kind.upper()},
        "generate_report": lambda aid: {"id": "RPT-1",
                                        "title": "Technical",
                                        "metrics": {"nodularity_pct": 85.0,
                                                    "nodule_count": 40}},
        "get_report": lambda rid: {"id": rid},
        "search_reference": lambda q: {"results": [{"topic": "iso",
                                                    "text": "x"}]},
    }


def test_tool_allowlist_rejects_unknown():
    from ai.tools import execute_action
    out = execute_action({"tool": "run_shell_command",
                          "args": {"cmd": "rm -rf /"}}, _ctx())
    assert "error" in out and "allowlist" in out["error"]


def test_tool_allowlist_rejects_bad_args():
    from ai.tools import execute_action
    out = execute_action({"tool": "analyze_image",
                          "args": {"dataset_image": "../../etc/passwd"}},
                         _ctx())
    assert "error" in out


def test_tool_valid_call_roundtrip():
    from ai.tools import execute_action
    out = execute_action({"tool": "analyze_image",
                          "args": {"dataset_image": "micrograph_0042.png"}},
                         _ctx())
    assert out["analysis"]["analysis_id"] == "ANL-NEW"


def test_tool_unknown_arg_smuggling_rejected():
    from ai.tools import validate_action, ToolError
    with pytest.raises(ToolError):
        validate_action({"tool": "get_batch_statistics",
                         "args": {"sql": "DROP TABLE x"}})


# ------------------------------------------------------ capabilities ----
def test_registry_dynamic_registration():
    from services.capabilities import (register_capability,
                                       get_capability, get_capabilities)
    register_capability("test_feature", "A test feature",
                        tools=["test_tool"], actions=["show_test"])
    cap = get_capability("test_feature")
    assert cap["description"] == "A test feature"
    assert any(c["name"] == "test_feature" for c in get_capabilities())


def test_capabilities_builtin_set():
    from services.capabilities import get_capability
    for name in ("analysis", "quality_gate", "calibration", "comparison",
                 "batch_analytics", "spc", "anomaly_detection", "validation",
                 "reports"):
        assert get_capability(name), f"missing capability {name}"


# ---------------------------------------------------------- reference ----
def test_reference_search_topics():
    from ai.reference import search_reference
    out = search_reference("how does nodularity work")
    assert out["results"]
    assert out["results"][0]["topic"] in ("nodularity", "iso", "engine")


# ------------------------------------------------------------ memory ----
def test_memory_session_roundtrip(tmp_path, monkeypatch):
    import ai.memory as mem
    monkeypatch.setattr(mem, "SESSIONS_DIR", tmp_path)
    s = mem.load_session("sess1")
    mem.append_message(s, "user", "hello")
    mem.set_referenced_analysis(s, "ANL-1")
    mem.save_session(s)
    s2 = mem.load_session("sess1")
    assert s2["messages"][0]["content"] == "hello"
    assert s2["referenced_analysis_id"] == "ANL-1"


def test_agent_turn_offline_uses_fallback(tmp_path, monkeypatch):
    """Full offline turn: no LLM configured -> fallback answers from data."""
    import ai.memory as mem
    monkeypatch.setattr(mem, "SESSIONS_DIR", tmp_path)
    from ai import agent
    result = agent.handle_turn(
        "what is the nodularity?", "sess-test", _ctx(),
        history=[], app_context={"current_analysis_id": "ANL-1"})
    assert result["reply"]
    assert "85.0" in result["reply"] or "85" in result["reply"]


# ------------------------------------------- intent router (safety net) ----
_CANNED_FALLBACK = ("Measured image quality is 62/100 (REJECT). Checks: . "
                    "The analysis engine ran regardless; quality affects how "
                    "much the measurements can be trusted.")


def _degraded_chain(monkeypatch):
    """Simulate a provider outage the way production sees it: the chain
    degrades to the canned fallback provider output (never raises). With the
    pre-fix agent, this canned template leaked through as the final answer
    even when the intent router had executed the real tool."""
    import ai.provider as prov
    monkeypatch.setattr(prov, "chat", lambda *a, **k:
                        {"content": _CANNED_FALLBACK,
                         "provider": "fallback"})


def test_intent_router_generates_real_report(tmp_path, monkeypatch):
    """'Generate a report' with a dead LLM still creates a REAL report via
    the deterministic router, and the reply is the honest presenter — never
    a canned fallback template (regression: gate c10)."""
    import ai.memory as mem
    monkeypatch.setattr(mem, "SESSIONS_DIR", tmp_path)
    _degraded_chain(monkeypatch)
    from ai import agent
    ctx = _ctx()
    result = agent.handle_turn(
        "Generate a report.", "sess-router-1", ctx,
        history=[], app_context={"current_analysis_id": "ANL-1"})
    assert result["tool_trace"]["requested"]["tool"] == "generate_report"
    assert result["tool_trace"]["result_digest"].get("report_id") == "RPT-1"
    low = result["reply"].lower()
    assert "rpt-1" in low                       # the REAL id, not a fake
    assert "measured image quality is" not in low
    assert not result["reply"].startswith("Done —")


def test_intent_router_compare_presents_summary(tmp_path, monkeypatch):
    """Compare intent executes the real compare tool and presents its actual
    summary (regression: gate c5)."""
    import ai.memory as mem
    monkeypatch.setattr(mem, "SESSIONS_DIR", tmp_path)
    _degraded_chain(monkeypatch)
    from ai import agent
    result = agent.handle_turn(
        "Compare it with the previous sample.", "sess-router-2", _ctx(),
        history=[], app_context={"current_analysis_id": "ANL-1"})
    assert result["tool_trace"]["requested"]["tool"] == "compare_samples"
    assert "b higher" in result["reply"]        # the fake tool's summary


def test_intent_answer_reports_tool_errors_honestly(tmp_path, monkeypatch):
    """When the routed tool errors, the answer says so instead of inventing."""
    import ai.memory as mem
    monkeypatch.setattr(mem, "SESSIONS_DIR", tmp_path)
    _degraded_chain(monkeypatch)
    from ai import agent
    ctx = _ctx()
    ctx["generate_report"] = lambda aid: {"error": "no analysis available"}
    result = agent.handle_turn(
        "Generate a report.", "sess-router-3", ctx,
        history=[], app_context={"current_analysis_id": "ANL-1"})
    low = result["reply"].lower()
    assert "could not complete" in low and "no analysis available" in low
    assert "rpt" not in low


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
