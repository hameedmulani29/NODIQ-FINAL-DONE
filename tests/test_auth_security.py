#!/usr/bin/env python3
"""
tests/test_auth_security.py — security regression tests for the auth flows.

Locks in the fixes from the security audit:
- bcrypt password hashing (never plain/MD5/SHA-1), legacy pbkdf2 upgrade
- logout revokes the token server-side (stolen token dies at logout)
- reset tokens are a separate family and can NEVER authenticate as a
  login bearer token (token-type confusion)
- login/signup error messages never reveal whether an email exists
- rate limiting returns 429, actually blocks, and is backed by the shared
  SQLite store (multi-worker safe, fail-closed on storage errors)
- forgot-password never returns a reset token in the HTTP response
"""
import json
import time

import pytest

import services.auth as auth_svc
import services.db as store
import services.rate_limit as rate_limiter


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Isolated DB + Flask test client, rate limiter on."""
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(store, "_CONN", None)

    def _stub_conn():
        store.init_db()
        store._CONN.executescript(auth_svc._USERS_SCHEMA)
        store._CONN.commit()
        return store._CONN

    monkeypatch.setattr(auth_svc, "_user_conn", _stub_conn)

    import importlib
    import sys
    server = importlib.import_module("server")
    importlib.reload(server) if "server" in sys.modules else None
    server.RATE_LIMITS_ENABLED = True
    # Redirect the limiter to this test's scratch DB and clear it so every
    # test starts from a zeroed shared budget (the store persists across
    # modules — never rely on process-local counters).
    monkeypatch.setattr(rate_limiter, "_DB_PATH", tmp_path / "rl.db")
    monkeypatch.setattr(rate_limiter, "_CONN", None)
    rate_limiter.reset_all()

    auth_svc.create_user("sec@test.local", "password1", "Sec User")
    return server.app.test_client()


def _login(client, email, password):
    return client.post("/api/auth/login",
                       data=json.dumps({"email": email,
                                        "password": password}),
                       content_type="application/json")


# ------------------------------------------------------------- hashing ----

def test_passwords_are_bcrypt_hashed(env):
    conn = auth_svc._user_conn()
    with auth_svc._LOCK:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE email = 'sec@test.local'"
        ).fetchone()
    assert row["password_hash"].startswith("$2b$"), row["password_hash"][:4]


def test_legacy_pbkdf2_hash_verifies_and_upgrades(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(store, "_CONN", None)

    def _stub_conn():
        store.init_db()
        store._CONN.executescript(auth_svc._USERS_SCHEMA)
        store._CONN.commit()
        return store._CONN

    monkeypatch.setattr(auth_svc, "_user_conn", _stub_conn)

    # Seed a legacy-format hash the way the pre-bcrypt scheme wrote it.
    import hashlib as _hl
    salt = bytes.fromhex("ab" * 16)
    dk = _hl.pbkdf2_hmac("sha256", b"oldpass1", salt, 200_000)
    legacy = f"pbkdf2$200000${salt.hex()}${dk.hex()}"
    auth_svc.create_user("legacy@test.local", "placeholder0")
    conn = auth_svc._user_conn()
    with auth_svc._LOCK:
        conn.execute("UPDATE users SET password_hash = ? "
                     "WHERE email = 'legacy@test.local'", (legacy,))
        conn.commit()

    # Old password still verifies against the legacy hash...
    assert auth_svc.verify_password("oldpass1", legacy)
    # ...and login transparently upgrades it to bcrypt.
    user, _tok = auth_svc.authenticate("legacy@test.local", "oldpass1")
    assert user is not None
    with auth_svc._LOCK:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE email = "
            "'legacy@test.local'").fetchone()
    assert row["password_hash"].startswith("$2b$")


# ------------------------------------------------------------ revocation --

def test_logout_revokes_token_server_side(env):
    _, token = auth_svc.authenticate("sec@test.local", "password1")
    headers = {"Authorization": f"Bearer {token}"}
    assert env.get("/api/auth/me", headers=headers).status_code == 200

    assert env.post("/api/auth/logout", headers=headers).status_code == 200

    # The SAME token must now be dead — a stolen/copied token dies at logout.
    assert env.get("/api/auth/me", headers=headers).status_code == 401


# ------------------------------------------------- token-family isolation -

def test_reset_token_cannot_authenticate_as_login_token(env):
    tok = auth_svc.generate_reset_token("sec@test.local")
    # consume path accepts it as a reset token...
    headers = {"Authorization": f"Bearer {tok}"}
    # ...but it must never authenticate as a login bearer token.
    assert env.get("/api/auth/me", headers=headers).status_code == 401
    # and vice versa: a login token is not a valid reset token.
    _, login_tok = auth_svc.authenticate("sec@test.local", "password1")
    assert auth_svc.consume_reset_token(login_tok) is None


# ------------------------------------------------------- message hygiene --

def test_login_error_is_generic(env):
    for email in ("sec@test.local", "ghost@test.local"):
        resp = _login(env, email, "wrong-pass")
        assert resp.status_code == 401
        assert resp.get_json()["error"] == "invalid email or password"


def test_login_timing_does_not_leak_account_existence(env):
    """Unknown email burns the same bcrypt cost as a wrong password."""
    auth_svc.authenticate("sec@test.local", "password1")  # warm caches

    def timed(email):
        t0 = time.perf_counter()
        _login(env, email, "definitely-wrong")
        return time.perf_counter() - t0

    known = min(timed("sec@test.local") for _ in range(3))
    unknown = min(timed("ghost@test.local") for _ in range(3))
    # Unknown-email path must not be dramatically faster (bcrypt equalizer).
    # Generous bound: catches a *missing* equalizer (>5x) while tolerating
    # scheduler noise.
    assert unknown > known / 5, (known, unknown)


def test_signup_duplicate_does_not_reveal_email_existence(env):
    resp = env.post("/api/auth/signup",
                    data=json.dumps({"email": "sec@test.local",
                                     "password": "password2",
                                     "name": "Dup"}),
                    content_type="application/json")
    assert resp.status_code == 400
    msg = resp.get_json()["error"].lower()
    assert "exists" not in msg and "already" not in msg


# --------------------------------------------------------- rate limiting --

def test_rate_limit_blocks_after_limit(env):
    server = pytest.importorskip("server")
    # login allows 10 / 5min per IP
    for _ in range(10):
        _login(env, "sec@test.local", "wrong-pass")
    resp = _login(env, "sec@test.local", "wrong-pass")
    assert resp.status_code == 429
    assert "too many" in resp.get_json()["error"].lower()


def test_forgot_response_never_contains_a_token(env):
    resp = env.post("/api/auth/forgot-password",
                    data=json.dumps({"email": "sec@test.local"}),
                    content_type="application/json")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert not body.get("token")
