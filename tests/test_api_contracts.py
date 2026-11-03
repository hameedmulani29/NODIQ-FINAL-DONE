"""Deterministic HTTP contract tests for the protected NODIQ-X API."""
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import services.auth as auth_svc
import services.context as context_svc
import services.db as store


@pytest.fixture(scope="module")
def api_env(tmp_path_factory):
    import server
    # Rate limiter is IP-keyed; many signups from the shared test IP would
    # trip it across the module's tests. Module-scoped fixture: save/restore
    # the flag directly (monkeypatch is function-scoped).
    _rl_saved = server.RATE_LIMITS_ENABLED
    server.RATE_LIMITS_ENABLED = False

    root = tmp_path_factory.mktemp("api-contracts")
    # Save and restore every redirected global (Issue 1: the previous
    # teardown closed the temp connection but left store.DATA_DIR /
    # store.DB_PATH / context_svc.FULL_DIR / server.UPLOAD_DIR pointing at
    # the deleted tmp dir, so every module collected AFTER this file — e.g.
    # test_size_filter — inherited a dead DB path unless it re-patched.
    # Tests are now order-independent.
    saved = (store.DATA_DIR, store.DB_PATH, store._CONN,
             context_svc.FULL_DIR, server.UPLOAD_DIR, server.ai_chat)
    store.DATA_DIR = root / "server_data"
    store.DATA_DIR.mkdir()
    store.DB_PATH = store.DATA_DIR / "test.db"
    store._CONN = None
    context_svc.FULL_DIR = store.DATA_DIR / "analyses_full"
    context_svc.FULL_DIR.mkdir()
    server.UPLOAD_DIR = store.DATA_DIR / "uploads"
    server.UPLOAD_DIR.mkdir()
    store.init_db()
    auth_svc._user_conn()

    server.ai_chat = lambda messages: {
        "content": "Measured values are available from the stored analysis.",
        "provider": "test",
    }
    try:
        yield server.app.test_client(), server
    finally:
        server.RATE_LIMITS_ENABLED = _rl_saved
        temp_conn = store._CONN
        (store.DATA_DIR, store.DB_PATH, store._CONN,
         context_svc.FULL_DIR, server.UPLOAD_DIR, server.ai_chat) = saved
        if temp_conn is not None and temp_conn is not store._CONN:
            temp_conn.close()


@pytest.fixture(scope="module")
def account(api_env):
    client, _ = api_env
    email = f"api-contract-{time.time_ns()}@test.local"
    response = client.post("/api/auth/signup", json={
        "email": email,
        "password": "password1",
        "name": "API Contract User",
    })
    assert response.status_code == 201
    payload = response.get_json()
    # The session token lives ONLY in the HttpOnly cookie now — read it
    # from Set-Cookie (never from the JSON body).
    token = response.headers.get("Set-Cookie", "").split(
        "nodiq_session=", 1)[1].split(";", 1)[0]
    return {
        "token": token,
        "headers": {"Authorization": f"Bearer {token}"},
    }


@pytest.fixture(scope="module")
def analysis(api_env, account):
    """One completed analysis: /api/analyze is async (202 + analysis_id, the
    heavy pipeline runs on a worker thread so a slow CPU can never exceed the
    proxy/gateway request timeout), so the fixture polls the detail endpoint
    until the worker stores the full payload — exactly like the frontend."""
    client, _ = api_env
    response = client.post(
        "/api/analyze",
        json={"dataset_image": "micrograph_0001.png"},
        headers=account["headers"],
    )
    assert response.status_code == 202, response.get_json()
    enqueue = response.get_json()
    assert enqueue["analysis_id"]
    assert enqueue["status"] == "processing"

    deadline = time.time() + 60
    detail = None
    status = None
    while time.time() < deadline:
        detail = client.get(f"/api/analyses/{enqueue['analysis_id']}",
                            headers=account["headers"]).get_json()
        # processing polls return {status, progress}; the final (worker done)
        # response is the legacy {record, analysis} detail shape.
        status = detail.get("status") or detail.get("record", {}).get("status")
        assert status in ("processing", "completed", "failed"), detail
        if status != "processing":
            break
        time.sleep(0.2)
    assert status == "completed", detail
    payload = detail["analysis"]
    assert payload["analysis_id"] == enqueue["analysis_id"]
    return payload


def test_authentication_and_analysis_contract(api_env, account, analysis):
    client, _ = api_env
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.get_json()["status"] == "ok"
    # Anonymous check uses a FRESH client: the module client's cookie jar now
    # holds the HttpOnly session cookie after the account fixture logged in.
    import server as _server
    anon = _server.app.test_client()
    assert anon.get("/api/auth/me").status_code == 401
    assert anon.get("/api/analyses").status_code == 401
    assert client.get("/api/auth/me", headers=account["headers"]).status_code == 200
    assert analysis["nodule_count"] == len(analysis["nodules"])
    assert 0 <= analysis["iso_nodularity_pct"] <= 100
    assert 0 <= analysis["jis_nodularity_pct"] <= 100
    assert analysis["source_width"] > 0 and analysis["source_height"] > 0


def test_invalid_analysis_input_fails_without_record(api_env, account):
    client, _ = api_env
    before = client.get("/api/analyses", headers=account["headers"])
    response = client.post(
        "/api/analyze",
        data={"image": (  # Werkzeug accepts this as a multipart upload.
            __import__("io").BytesIO(b"not an image"), "invalid.png")},
        content_type="multipart/form-data",
        headers=account["headers"],
    )
    # 400 = decode failure, 415 = magic-number mismatch (upload guard)
    assert response.status_code in (400, 415)
    assert response.get_json().get("status") != "complete"
    after = client.get("/api/analyses", headers=account["headers"])
    assert len(after.get_json()["analyses"]) == len(before.get_json()["analyses"])


def test_persistence_quality_and_nodule_contract(api_env, account, analysis):
    client, _ = api_env
    aid = analysis["analysis_id"]
    detail = client.get(f"/api/analyses/{aid}", headers=account["headers"])
    assert detail.status_code == 200
    body = detail.get_json()
    assert set(body) == {"analysis", "record"}
    assert body["analysis"]["nodule_count"] == analysis["nodule_count"]
    assert body["record"]["id"] == aid

    quality = client.get(f"/api/quality/{aid}", headers=account["headers"])
    assert quality.status_code == 200
    quality_body = quality.get_json()["quality"]
    assert quality_body["status"] in {"PASS", "WARNING", "REJECT"}
    assert isinstance(quality_body["score"], (int, float))
    assert "metrics" in quality_body

    nodules = client.get(f"/api/analyses/{aid}/nodules",
                         headers=account["headers"])
    rows = nodules.get_json()["nodules"]
    assert nodules.status_code == 200
    assert len(rows) == analysis["nodule_count"]
    assert len({row["nodule_id"] for row in rows}) == len(rows)
    for key in ("area_px", "perimeter_px", "roundness", "solidity"):
        assert key in rows[0]
    one = client.get(f"/api/analyses/{aid}/nodules/{rows[0]['nodule_id']}",
                     headers=account["headers"])
    assert one.status_code == 200
    assert one.get_json()["nodule"]["nodule_id"] == rows[0]["nodule_id"]


def test_analysis_authorization(api_env, account, analysis):
    client, _ = api_env
    # Fresh cookie-less clients for the anonymous and other-user
    # perspectives: the module client's jar holds the account's HttpOnly
    # session cookie after the fixtures logged in.
    import server as _server
    anon = _server.app.test_client()
    other_c = _server.app.test_client()
    r = other_c.post("/api/auth/signup", json={
        "email": f"other-{time.time_ns()}@test.local",
        "password": "password1",
        "name": "Other User",
    })
    assert r.status_code == 201, r.get_json()
    # token from the HttpOnly Set-Cookie header (never in the JSON body)
    other_token = r.headers.get("Set-Cookie", "").split(
        "nodiq_session=", 1)[1].split(";", 1)[0]
    other_headers = {"Authorization": f"Bearer {other_token}"}
    aid = analysis["analysis_id"]
    assert anon.get(f"/api/analyses/{aid}").status_code == 404
    assert other_c.get(f"/api/analyses/{aid}", headers=other_headers).status_code == 404
    assert other_c.delete(f"/api/analyses/{aid}", headers=other_headers).status_code == 404
    assert client.delete("/api/analyses/ANL-NOT-OWNED",
                         headers=account["headers"]).status_code == 404


def test_copilot_health_capabilities_and_context(api_env, account, analysis, monkeypatch):
    client, _ = api_env
    import ai.provider as provider

    captured = {}

    def fake_chat(messages, context=None):
        captured["context"] = context
        captured["messages"] = messages
        return {"content": "Measured values are 34 nodules and ISO 19.09%.",
                "provider": "test"}

    monkeypatch.setattr(provider, "chat", fake_chat)
    health = client.get("/api/copilot/health")
    capabilities = client.get("/api/copilot/capabilities")
    assert health.status_code == 200
    assert capabilities.status_code == 200
    names = capabilities.get_json()["tools"]
    assert "get_current_analysis" in names
    response = client.post("/api/copilot/chat", headers=account["headers"],
                           json={
                               "message": "Explain this analysis.",
                               "session_id": "contract-test",
                               "context": {"current_analysis_id": analysis["analysis_id"]},
                           })
    assert response.status_code == 200
    assert response.get_json()["provider"] == "test"
    context_message = next(
        message["content"] for message in captured["messages"]
        if message["content"].startswith("Current measured context (JSON):")
    )
    measured = json.loads(context_message.split(": ", 1)[1])
    assert measured["iso945"]["nodularity_pct"] == analysis["iso_nodularity_pct"]
    assert measured["detections"]["count"] == analysis["nodule_count"]

    # Phase 3 streaming endpoint regression test
    stream_res = client.post("/api/copilot/stream", headers=account["headers"],
                             json={
                                 "message": "Explain this analysis.",
                                 "session_id": "contract-test",
                                 "context": {"current_analysis_id": analysis["analysis_id"]},
                             })
    assert stream_res.status_code == 200
    assert "text/event-stream" in stream_res.content_type
    assert b"data: " in stream_res.data
    stream_res.close()


def test_copilot_stream_security_and_incremental_delivery(api_env, account):
    client, _server = api_env
    anon = _server.app.test_client()
    # 1. Unauthenticated request must return 401
    anon_res = anon.post("/api/copilot/stream", json={"message": "hello"})
    assert anon_res.status_code == 401

    # 2. Empty message must return 400
    bad_res = client.post("/api/copilot/stream", headers=account["headers"], json={"message": "   "})
    assert bad_res.status_code == 400

    # 3. Genuine incremental token delivery test
    from ai import provider
    simulated_chunks = ["Nodularity ", "is measured ", "at 84.2%. "]
    def mock_chat(messages, context=None, stream=False):
        if stream:
            return {"stream": True, "chunks": iter(simulated_chunks), "provider": "mock-stream", "model": "mock-v1"}
        return {"content": "".join(simulated_chunks), "provider": "mock-static"}

    orig_chat = provider.chat
    provider.chat = mock_chat
    try:
        res = client.post("/api/copilot/stream", headers=account["headers"],
                          json={"message": "What is the nodularity?", "session_id": "sec-stream-test"})
        assert res.status_code == 200
        assert "text/event-stream" in res.content_type

        # Parse SSE lines
        events = []
        for line in res.data.decode("utf-8").split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                events.append(json.loads(line[5:].strip()))

        token_deltas = [e["delta"] for e in events if e.get("type") == "token"]
        assert token_deltas == simulated_chunks
        done_events = [e for e in events if e.get("type") == "done"]
        assert len(done_events) == 1
        assert done_events[0]["reply"] == "".join(simulated_chunks).strip()
        assert done_events[0]["provider"] == "mock-stream"
        res.close()
    finally:
        provider.chat = orig_chat


def test_registered_tools_and_argument_guards(api_env):
    from ai import tools

    expected = {
        "get_current_analysis", "get_analysis", "analyze_image",
        "get_nodule_details", "get_quality_report", "get_calibration",
        "get_batch_statistics", "get_anomalies", "get_decision_support",
        "compare_samples", "get_spc", "get_validation", "generate_report",
        "get_report", "search_reference",
    }
    assert expected.issubset(set(tools.tool_names()))
    assert all(callable(spec["func"]) for spec in tools._TOOLS.values())
    ctx = {"latest_analysis": lambda: None}
    assert "allowlist" in tools.execute_action(
        {"tool": "run_shell", "args": {}}, ctx)["error"]
    assert "error" in tools.execute_action(
        {"tool": "get_analysis", "args": {"analysis_id": "../secret"}}, ctx)
    with pytest.raises(tools.ToolError):
        tools.validate_action({"tool": "get_spc", "args": {"sql": "DROP TABLE"}})


def test_reports_and_download_contract(api_env, account, analysis):
    client, _ = api_env
    response = client.post("/api/reports/generate", headers=account["headers"],
                           json={"analysis_id": analysis["analysis_id"]})
    assert response.status_code == 201
    report = response.get_json()
    assert report["analysis_id"] == analysis["analysis_id"]
    report_id = report["id"]
    assert client.get(f"/api/reports/{report_id}", headers=account["headers"]).status_code == 200
    for fmt, prefix in (("json", b"{"), ("csv", b"section,metric,value"),
                        ("pdf", b"%PDF")):
        download = client.get(
            f"/api/reports/{report_id}/download?format={fmt}",
            headers=account["headers"],
        )
        assert download.status_code == 200
        assert download.data.startswith(prefix)


def test_analysis_scoped_spc_is_documented_history_alias(api_env, account, analysis):
    client, _ = api_env
    scoped = client.get(f"/api/batch/{analysis['analysis_id']}/spc",
                        headers=account["headers"])
    general = client.get("/api/spc?metric=nodularity",
                         headers=account["headers"])
    assert scoped.status_code == 200
    assert scoped.get_json() == general.get_json()
