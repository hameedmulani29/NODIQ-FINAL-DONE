"""Shared pytest configuration.

Under the supabase backend the per-fixture DB_PATH redirect (sqlite tmp
files) maps to isolated Postgres schemas, so fixtures get fresh databases
exactly like on sqlite and no cross-test truncation is needed. This hook
only guards the shared public schema: anything left over from earlier
non-test runs is cleared once before each full pytest invocation, so fixed
emails (s1@test.local...) never collide with stale rows. Under the sqlite
backend it is a no-op.

Usage:
    pytest                                # sqlite (default, unchanged)
    NODIQ_BACKEND=supabase DATABASE_URL=... pytest
"""
import os


def pytest_configure(config):
    if os.environ.get("NODIQ_BACKEND", "").strip().lower() != "supabase":
        return
    _truncate_public()


def _truncate_public():
    import services.db as store
    store.init_db()
    tables = ("analyses", "reports", "feedback", "calibrations", "sessions",
              "full_payloads", "users", "revoked_tokens", "resets")
    with store._LOCK:
        conn = store._connect()
        conn.execute("TRUNCATE " + ", ".join(tables))
