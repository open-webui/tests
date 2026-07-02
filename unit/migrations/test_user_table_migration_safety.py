"""Regression for open-webui/open-webui#26403 — upgrading an existing SQLite
DB must not crash or corrupt saved user settings during the "Update user table"
migration (b10670c03dd5).

Fixed in c416c6cad. The migration converts the legacy TEXT `user.info` /
`user.settings` columns (which stored JSON as a serialized string) into real
JSON columns. The buggy version, on a SQLite DB that already had user rows,
had two data-dependent failure modes that a FRESH-DB migration test never
hits (an empty DB upgrades fine on the buggy code):

  * `info` populated   -> the migration CRASHED. `_convert_column_to_json`
    reused a single module/closure-bound `sa.column('..._json')` object across
    both the `info` and `settings` conversions; binding it to the ad-hoc table
    a second time raised `ArgumentError: column object '..._json' already
    assigned to table 'user'`.
  * `settings` populated -> the migration SUCCEEDED but CORRUPTED settings.
    The new column is a real JSON type, yet the buggy UPDATE passed
    `json.dumps(parsed)` (an already-serialized string) as the value, so the
    JSON column serialized it a second time. A dict `{"ui": {...}}` came back
    as the *string* `'{"ui": {...}}'` — the app then reads a str where it
    expects a dict. The buggy `if parsed` guard (falsy, not `is not None`)
    additionally dropped empty/false-y settings to NULL.

The fix stores the parsed object directly (`.values({... : parsed})`) and
stops reusing the bound column object, so a dict stays a dict and `{}` stays
`{}`.

This test seeds a DB at the revision JUST BEFORE b10670c03dd5 with realistic
rows, then upgrades and asserts (a) no crash and (b) each user's settings
round-trip intact through the now-JSON column. It runs the migration in a
subprocess against a throwaway temp SQLite file, so it is deterministic and
fully offline. Discriminates: PASSES on fixed dev, FAILS on v0.10.1 (crash
when `info` is set; corrupted settings otherwise).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression

# The migration under test and the revision immediately before it. Seeding at
# DOWN_REVISION reproduces the "existing DB with the old TEXT columns" state
# that the buggy upgrade mishandled. Read from the migration's own headers
# (revision / down_revision) so this stays correct if the chain is renumbered.
TARGET_REVISION = "b10670c03dd5"
DOWN_REVISION = "2f1211949ecc"

# Realistic saved settings — a nested dict, mirroring what the UI persists to
# user.settings. If double-encoded it comes back as a str instead of a dict.
_RICH_SETTINGS = {
    "ui": {"theme": "dark", "version": 1, "widescreen": True},
    "notifications": True,
    "models": ["gpt-4o", "claude"],
}
# Empty dict: falsy, so the buggy `if parsed`/`if raw` guard dropped it to NULL.
_EMPTY_SETTINGS: dict = {}
# Populated info: forces the buggy code down the second _convert_column_to_json
# call, which is where the reused bound-column object raised ArgumentError.
_RICH_INFO = {"organization": "acme", "seats": 3}


def _run_python(
    backend: Path, db_url: str, data_dir: Path, body: str, *, timeout: int = 180
) -> subprocess.CompletedProcess:
    """Run a Python snippet with `backend/` on sys.path and DB env set.

    Own subprocess per run so open_webui's import-time caching doesn't leak
    between invocations. (Same shape as test_lifecycle.py's helper; kept local
    so this file stays self-contained.)
    """
    preamble = textwrap.dedent(
        f"""
        import json, os, sys
        os.environ["DATABASE_URL"] = {db_url!r}
        os.environ["DATA_DIR"] = {str(data_dir)!r}
        os.environ.setdefault("WEBUI_SECRET_KEY", "test-secret-key-for-unit-tests")
        sys.path.insert(0, {str(backend)!r})
        """
    )
    return subprocess.run(
        [sys.executable, "-c", preamble + body],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


# Body executed in the subprocess: bring a fresh SQLite DB up to the revision
# before the fix, seed rows via the OLD TEXT columns, then upgrade to the
# target and read settings back THROUGH a JSON-typed column (so the check is
# about the logical value, not SQLite's raw storage). Prints a single RESULT:
# line of JSON for the parent to assert on. Any crash in the upgrade surfaces
# as a non-zero exit + traceback on stderr.
_BODY = textwrap.dedent(
    f"""
    import sqlalchemy as sa
    from alembic import command
    from alembic.config import Config as AlembicConfig
    from sqlalchemy import create_engine, text
    from open_webui.env import OPEN_WEBUI_DIR

    cfg = AlembicConfig(OPEN_WEBUI_DIR / "alembic.ini")
    cfg.set_main_option("script_location", str(OPEN_WEBUI_DIR / "migrations"))

    # 1. Existing DB state: everything up to just before the migration.
    command.upgrade(cfg, {DOWN_REVISION!r})

    rich_settings = {_RICH_SETTINGS!r}
    empty_settings = {_EMPTY_SETTINGS!r}
    rich_info = {_RICH_INFO!r}

    engine = create_engine(os.environ["DATABASE_URL"])
    # Seed via the legacy TEXT columns: JSON persisted as a serialized string,
    # exactly how the pre-migration app wrote user.settings / user.info.
    with engine.begin() as c:
        c.execute(
            text("INSERT INTO user (id, name, email, settings, info) "
                 "VALUES (:i, :n, :m, :s, :info)"),
            {{"i": "u_rich", "n": "Alice", "m": "alice@example.com",
              "s": json.dumps(rich_settings), "info": json.dumps(rich_info)}},
        )
        c.execute(
            text("INSERT INTO user (id, name, email, settings) "
                 "VALUES (:i, :n, :m, :s)"),
            {{"i": "u_empty", "n": "Bob", "m": "bob@example.com",
              "s": json.dumps(empty_settings)}},
        )
        c.execute(
            text("INSERT INTO user (id, name, email, settings) "
                 "VALUES (:i, :n, :m, :s)"),
            {{"i": "u_null", "n": "Carol", "m": "carol@example.com", "s": None}},
        )
    engine.dispose()

    # 2. The upgrade under test. On buggy v0.10.1 this raises for u_rich (info
    #    set) -> non-zero exit, caught as a FAIL by the parent.
    command.upgrade(cfg, {TARGET_REVISION!r})

    # 3. Read settings back through the JSON type. On fixed code these are
    #    dicts; on buggy code (had it not crashed) they'd be strings.
    engine = create_engine(os.environ["DATABASE_URL"])
    user_json = sa.table("user", sa.column("id", sa.Text), sa.column("settings", sa.JSON))
    out = {{}}
    with engine.connect() as c:
        for uid in ("u_rich", "u_empty", "u_null"):
            v = c.execute(
                sa.select(user_json.c.settings).where(user_json.c.id == uid)
            ).scalar()
            out[uid] = {{"type": type(v).__name__, "value": v}}
    engine.dispose()

    print("RESULT:" + json.dumps(out))
    """
)


def _upgrade_and_read(backend: Path, tmp_path: Path) -> dict:
    """Seed + upgrade in a subprocess; return the parsed settings-per-user map.

    Fails the test with the subprocess traceback if the migration crashed
    (the buggy-info path) or didn't emit its RESULT line.
    """
    db_path = (tmp_path / "webui.db").resolve()
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    db_url = f"sqlite:///{db_path.as_posix()}"

    result = _run_python(backend, db_url, data_dir, _BODY)

    line = next((ln for ln in result.stdout.splitlines() if ln.startswith("RESULT:")), None)
    if result.returncode != 0 or line is None:
        pytest.fail(
            "Migration to b10670c03dd5 on an existing SQLite DB with user rows "
            "crashed or produced no result (regression #26403).\n"
            f"returncode={result.returncode}\n"
            f"--- stderr (tail) ---\n{result.stderr[-3000:]}\n"
            f"--- stdout (tail) ---\n{result.stdout[-1500:]}"
        )
    return json.loads(line.removeprefix("RESULT:"))


def test_existing_db_settings_survive_user_table_migration(
    open_webui_backend: Path, tmp_path: Path
) -> None:
    """Upgrading an existing SQLite DB keeps user settings intact (not crashed,
    not double-encoded, not dropped)."""
    out = _upgrade_and_read(open_webui_backend, tmp_path)

    # Rich settings must round-trip to the SAME dict — the core anti-corruption
    # assertion. Buggy code returns the str "{...}" here (double-encoded).
    rich = out["u_rich"]
    assert rich["type"] == "dict", (
        f"user.settings came back as {rich['type']}, not dict — double-encoded "
        f"by the migration (regression #26403). value={rich['value']!r}"
    )
    assert rich["value"] == _RICH_SETTINGS, (
        f"user.settings was altered by the migration.\n"
        f"expected: {_RICH_SETTINGS!r}\n"
        f"got:      {rich['value']!r}"
    )

    # Empty dict must stay {} (a dict), not be dropped to NULL by the falsy guard.
    empty = out["u_empty"]
    assert empty["value"] == _EMPTY_SETTINGS and empty["type"] == "dict", (
        f"empty settings {{}} was corrupted/dropped by the migration: "
        f"type={empty['type']} value={empty['value']!r}"
    )

    # A genuinely-NULL settings row stays NULL (sanity: the fix didn't
    # over-correct None into "null"/{}).
    assert out["u_null"]["value"] is None, (
        f"NULL settings should stay NULL, got {out['u_null']['value']!r}"
    )
