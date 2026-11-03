#!/usr/bin/env python3
"""
tests/test_input_security.py — production hardening regressions (Phase 25).

Locks in the input-security pass:
- unknown JSON fields rejected on every meaningful route (400)
- identity/privilege fields (user_id/owner_id/role/isAdmin/permissions)
  never accepted from clients
- SQL injection payloads cannot reach the database layer
- oversized uploads rejected (413), malformed images rejected (400/415)
- path traversal on dataset routes rejected
- HSTS only over HTTPS, never over plain HTTP
- security headers present
- structured security events are logged without credential material
"""
import io
import json
import re
import time

import pytest

import services.auth as auth_svc
import services.context as context_svc
import services.seclog as seclog
import services.db as store
import services.validate as validate
import server


@pytest.fixture(scope="module")
def api_env(tmp_path_factory):
    """Same isolation contract as test_api_contracts.api_env: rate limits
    disabled for the module, DB/full-payload/upload globals redirected to a
    tmp dir, and every redirected global restored on teardown so this module
    stays order-independent."""
    _rl_saved = server.RATE_LIMITS_ENABLED
    server.RATE_LIMITS_ENABLED = False
    # The module fires rapid-fire /api/analyze enqueues; the per-account
    # async backpressure limit (a production safeguard) would 429 them.
    _conc_saved = server._MAX_CONCURRENT_ANALYSES
    server._MAX_CONCURRENT_ANALYSES = 10_000
    _global_saved = server._MAX_CONCURRENT_CHILDREN
    server._MAX_CONCURRENT_CHILDREN = 10_000
    root = tmp_path_factory.mktemp("input-security")
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
    server.app.config["TESTING"] = True
    try:
        yield server.app.test_client(), server
    finally:
        server.RATE_LIMITS_ENABLED = _rl_saved
        server._MAX_CONCURRENT_ANALYSES = _conc_saved
        server._MAX_CONCURRENT_CHILDREN = _global_saved
        temp_conn = store._CONN
        (store.DATA_DIR, store.DB_PATH, store._CONN,
         context_svc.FULL_DIR, server.UPLOAD_DIR, server.ai_chat) = saved
        if temp_conn is not None and temp_conn is not store._CONN:
            temp_conn.close()


@pytest.fixture(scope="module")
def client(api_env):
    c, _ = api_env
    return c


@pytest.fixture(scope="module")
def acct(client):
    email = f"inputsec-{time.time_ns()}@test.local"
    r = client.post("/api/auth/signup", json={
        "email": email, "password": "password1", "name": "Input Sec"})
    assert r.status_code == 201, r.get_json()
    token = r.headers.get("Set-Cookie", "").split(
        "nodiq_session=", 1)[1].split(";", 1)[0]
    return {"headers": {"Authorization": f"Bearer {token}"}, "email": email}


def _post(client, path, payload, headers):
    return client.post(path, data=json.dumps(payload),
                       content_type="application/json", headers=headers)


# ------------------------------------------------------------- unknown fields
def test_unknown_fields_rejected(client, acct):
    """Every meaningful route must reject fields outside its schema."""
    cases = [
        ("/api/auth/signup", {"email": "x1@example.com", "password": "password1",
                              "name": "X", "role": "admin"}),
        ("/api/auth/signup", {"email": "x2@example.com", "password": "password1",
                              "name": "X", "isAdmin": True}),
        ("/api/analyses", {"id": "ANL-UNKFIELD", "title": "t",
                           "userId": "USR-anything"}),
        ("/api/analyses", {"id": "ANL-UNKFIELD2", "title": "t",
                           "made_up_field": 1}),
        ("/api/feedback", {"message": "hi", "email": "a@b.co", "injected": "<script>"}),
        ("/api/calibration", {"dataset_image": "micrograph_0001.png",
                              "user_id": "USR-OTHER"}),
        ("/api/decision-support", {"dataset_image": "micrograph_0001.png",
                                   "permissions": "elevated"}),
    ]
    for path, payload in cases:
        r = _post(client, path, payload, acct["headers"])
        assert r.status_code == 400, (path, payload, r.status_code, r.get_json())


def test_forbidden_identity_keys_rejected_globally(client, acct):
    """Identity/privilege keys are rejected even where a schema might evolve."""
    for key in ("user_id", "owner_id", "role", "admin", "permissions"):
        r = _post(client, "/api/feedback", {"message": "m", "email": "a@b.co",
                                            key: "pwn"}, acct["headers"])
        assert r.status_code == 400, (key, r.status_code)


def test_validate_module_forbidden_keys():
    for key in validate.FORBIDDEN_KEYS:
        with pytest.raises(validate.InputError):
            validate.validate_body({key: "x"}, {"name": ("str", {})})
    with pytest.raises(validate.InputError):
        validate.validate_body({"ROLE": "admin"}, {"name": ("str", {})})


# ------------------------------------------------------------- sql injection
def test_sql_injection_payloads_rejected_or_inert(client, acct):
    """Malformed ids / SQL metacharacters never reach the DB unsanitized."""
    payloads = [
        "ANL-' OR '1'='1",
        "ANL-1; DROP TABLE analyses--",
        "ANL-1' UNION SELECT user_id FROM users--",
        "ANL-x'--",
        "ANL-\"; DELETE FROM users; --",
    ]
    for pid in payloads:
        r = client.get(f"/api/analyses/{pid}", headers=acct["headers"])
        # malformed ids are rejected by the id grammar or miss as 404;
        # they must never produce a 500 or return foreign data
        assert r.status_code in (400, 404), (pid, r.status_code)
    # store-level: injection attempts in ids are inert (parameterized SQL)
    assert client.get("/api/analyses/ANL-%27%20OR%20%271",  # noqa: RUF100
                      headers=acct["headers"]).status_code in (400, 404)


# ------------------------------------------------------------- uploads
def _jpeg_bytes():
    """JPEG-magic container (not necessarily decodable — the extension guard
    runs before the engine's decode step, so an engine-level 400 still proves
    the extension spelling was accepted)."""
    return (b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            + b"\x00" * 64)


def _png_bytes(size=(8, 8)):
    """Minimal valid PNG (8x8 grayscale) built by hand."""
    import struct, zlib
    def chunk(typ, data):
        c = struct.pack(">I", len(data)) + typ + data
        return c + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)
    w, h = size
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x80" * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def test_oversized_upload_rejected(client, acct):
    big = b"\x89PNG\r\n\x1a\n" + b"A" * (server.MAX_UPLOAD_BYTES + 1)
    r = client.post("/api/analyze", data={"image": (io.BytesIO(big), "x.png")},
                    headers=acct["headers"],
                    content_type="multipart/form-data")
    assert r.status_code == 413, r.status_code


def test_malformed_image_rejected(client, acct):
    # text garbage with an image extension
    r = client.post("/api/analyze", data={"image": (io.BytesIO(b"not an image"),
                                                    "fake.png")},
                    headers=acct["headers"], content_type="multipart/form-data")
    assert r.status_code in (400, 415), (r.status_code, r.get_json())
    # empty file
    r = client.post("/api/analyze", data={"image": (io.BytesIO(b""), "empty.png")},
                    headers=acct["headers"], content_type="multipart/form-data")
    assert r.status_code in (400, 415), r.status_code


def test_arbitrary_filename_with_valid_image_accepted(client, acct):
    """A valid image under an executable-looking filename is accepted: the
    stored name is a server-side hash with a content-derived extension, so
    the client name has no effect on disk."""
    for name in ("payload.exe", "page.html", "script.php"):
        r = client.post("/api/analyze",
                        data={"image": (io.BytesIO(_png_bytes()), name)},
                        headers=acct["headers"], content_type="multipart/form-data")
        assert r.status_code == 202, (name, r.status_code)


def test_valid_png_accepted_shape(client, acct):
    """Positive control: a real PNG gets past the guards (into analysis)."""
    r = client.post("/api/analyze",
                    data={"image": (io.BytesIO(_png_bytes()), "micro.png")},
                    headers=acct["headers"], content_type="multipart/form-data")
    # guards pass -> the upload is enqueued (async analysis); the
    # microstructure gate may later reject a flat 8x8 asynchronously, which is
    # the scientific layer, not the security layer
    assert r.status_code == 202, (r.status_code, r.get_json())


def test_extension_spellings_match_content(client, acct):
    """Every common spelling of JPEG content passes the guard (regression:
    an earlier guard compared against a platform-dependent
    mimetypes.guess_extension() result and 415'd valid .jpeg uploads on
    Windows, where guess_extension() returns .jpg)."""
    for name in ("micro.jpeg", "micro.jpg", "MICRO.JPG", "micro.jpe", "micro.jfif"):
        r = client.post("/api/analyze",
                        data={"image": (io.BytesIO(_jpeg_bytes()), name)},
                        headers=acct["headers"], content_type="multipart/form-data")
        # anything but 415 proves the spelling passed the guard (the engine
        # may still 400 on the non-decodable body — that is by design)
        assert r.status_code != 415, (name, r.status_code, r.get_json())


def test_mismatched_extension_accepted(client, acct):
    """Content is identified by magic number, never by the client's filename:
    a JPEG misnamed .png (camera exports like 'IMG 4.png' do this constantly)
    must be accepted and stored under a content-derived name, not bounced
    with a 415 rename instruction."""
    for data, name in ((_jpeg_bytes(), "micro.png"), (_jpeg_bytes(), "IMG 4.png"),
                       (_png_bytes(), "micro.jpeg"), (_png_bytes(), "photo.jpg")):
        r = client.post("/api/analyze",
                        data={"image": (io.BytesIO(data), name)},
                        headers=acct["headers"], content_type="multipart/form-data")
        assert r.status_code != 415, (name, r.status_code, r.get_json())


def test_non_image_bytes_still_rejected_regardless_of_name(client, acct):
    """The permissive filename policy must not weaken content validation:
    garbage/executable bytes stay refused under any filename."""
    for data, name in ((b"not an image", "payload.exe"),
                       (b"MZ\x90\x00" + b"\x00" * 64, "page.html"),
                       (b"", "empty.png")):
        r = client.post("/api/analyze",
                        data={"image": (io.BytesIO(data), name)},
                        headers=acct["headers"], content_type="multipart/form-data")
        assert r.status_code in (400, 413, 415), (name, r.status_code)


# ------------------------------------------------------------- traversal
def test_dataset_traversal_rejected(client, acct):
    for bad in ("../server.db", "..\\server.db", "sub/dir/micrograph_0001.png",
                "%2e%2e%2fserver.db", "..%252fserver.db", "micrograph_0001.png%00",
                "/etc/passwd", "C:\\Windows\\win.ini", "micrograph_0001.txt"):
        r = _post(client, "/api/analyze", {"dataset_image": bad},
                  acct["headers"])
        assert r.status_code in (400, 404), (bad, r.status_code)


# ------------------------------------------------------------- HSTS + headers
def test_hsts_https_only(client, monkeypatch):
    """HSTS must be present over HTTPS and absent over plain HTTP."""
    http = client.get("/api/health")
    assert "Strict-Transport-Security" not in http.headers
    # direct TLS termination: wsgi.url_scheme == https
    https = client.get("/api/health",
                       environ_overrides={"wsgi.url_scheme": "https"})
    assert "max-age=31536000" in https.headers.get(
        "Strict-Transport-Security", "")
    # trusted proxy announcing https
    monkeypatch.setattr(server.os, "environ",
                        {**server.os.environ, "TRUST_PROXY": "true"})
    proxied = client.get("/api/health",
                         headers={"X-Forwarded-Proto": "https"})
    assert "max-age=31536000" in proxied.headers.get(
        "Strict-Transport-Security", "")
    # forwarded header WITHOUT trust must stay plain
    monkeypatch.setattr(server.os, "environ",
                        {**server.os.environ, "TRUST_PROXY": ""})
    untrusted = client.get("/api/health",
                           headers={"X-Forwarded-Proto": "https"})
    assert "Strict-Transport-Security" not in untrusted.headers


def test_security_headers_present(client):
    r = client.get("/api/health")
    assert r.headers.get("X-Content-Type-Options") == "nosniff"
    assert r.headers.get("X-Frame-Options") == "DENY"
    assert "strict-origin-when-cross-origin" in r.headers.get(
        "Referrer-Policy", "")


# ------------------------------------------------------------- security log
def test_security_events_logged_without_secrets(client, acct):
    marker = f"seclog-test-{time.time_ns()}"
    # a failed login (wrong password) and a malformed id both log events
    client.post("/api/auth/login", json={
        "email": acct["email"], "password": "definitely-wrong-pass"})
    client.get("/api/analyses/ANL-' OR '1'='1", headers=acct["headers"])
    log_path = seclog._LOG_PATH
    text = log_path.read_text(encoding="utf-8", errors="replace") \
        if log_path.exists() else ""
    assert '"event": "login_failed"' in text or "login_failed" in text
    # no credential material may ever appear
    assert "password1" not in text
    assert acct["headers"]["Authorization"] not in text
    assert "Bearer " not in text


def test_seclog_never_raises_and_sanitizes():
    seclog.log_event("unit_test", route="x" * 500, detail="\x00\x01bad\x7f")
    line = seclog._LOG_PATH.read_text(encoding="utf-8", errors="replace") \
        .strip().splitlines()[-1]
    entry = json.loads(line)
    assert len(entry["route"]) <= 128
    assert "\x00" not in entry["detail"]


# ------------------------------------------------------------- misc contracts
def test_health_is_public(client):
    assert client.get("/api/health").status_code == 200


def test_firebase_idtoken_accepts_real_jwt_shape(client):
    """Regression: the generic token rule (≤512 chars) rejected every real
    Firebase ID token (~1000-1200-char RS256 JWT), so Google/phone logins
    died with 'invalid idToken' before Google was ever consulted. Real-shaped
    JWTs must pass validation (then fail upstream signature check → 401);
    garbage must still be rejected at the door (400)."""
    real_shaped = "eyJhbGciOiJSUzI1NiJ9." + "e30" * 400 + "." + "QUJD" * 400
    r = client.post("/api/auth/firebase", json={"idToken": real_shaped})
    assert r.status_code == 401  # passed our validation, rejected by Google
    assert r.get_json()["error"] == "invalid token"
    for bad in ("abc.def", "x.y.z", "a" * 3000, None, 12345):
        r2 = client.post("/api/auth/firebase", json={"idToken": bad})
        assert r2.status_code == 400, bad
        assert "idToken" in r2.get_json()["error"]
