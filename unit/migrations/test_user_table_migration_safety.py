"""Regression for open-webui/open-webui#26403: upgrading an existing SQLite
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
    as the *string* `'{"ui": {...}}'`, and the app then reads a str where it
    expects a dict. The buggy `if parsed` guard (falsy, not `is not None`)
    additionally dropped empty/false-y settings to NULL.

The fix stores the parsed object directly (`.values({... : parsed})`) and
stops reusing the bound column object, so a dict stays a dict and `{}` stays
`{}`.

This test seeds a DB at the revision JUST BEFORE b10670c03dd5 with realistic
rows, then upgrades and asserts (a) no crash and (b) each user's settings
round-trip intact through the now-JSON column. It runs the migration in a
fresh interpreter against a throwaway SQLite file. Stays a unit test: the rows
it needs only exist on a database from before that migration.

Discriminates: passes on dev bbfa876af, fails with the migration's
`.values({...: parsed})` back to `json.dumps(parsed)` in a copy of it (the
settings come back a string).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression

# The migration under test and the revision before it. Seeding at DOWN_REVISION reproduces the
# existing database with the old TEXT columns that the buggy upgrade mishandled.
TARGET_REVISION = "b10670c03dd5"
DOWN_REVISION = "2f1211949ecc"

# What the UI persists to user.settings; double-encoded it comes back as a str.
RICH_SETTINGS = {
    "ui": {"theme": "dark", "version": 1, "widescreen": True},
    "notifications": True,
    "models": ["gpt-4o", "claude"],
}
# Falsy, so the buggy `if parsed` guard dropped it to NULL.
EMPTY_SETTINGS: dict = {}
# Populated info sends the buggy code down the second conversion, where it crashed.
RICH_INFO = {"organization": "acme", "seats": 3}

# Seed rows through the legacy TEXT columns, upgrade across the migration, and read settings
# back through a JSON-typed column, so the check reads the logical value.
UPGRADE_SEEDED_USERS = f"""
import sqlalchemy as sa
from sqlalchemy import create_engine, text

command.upgrade(cfg, {DOWN_REVISION!r})
engine = create_engine(os.environ["DATABASE_URL"])
insert = "INSERT INTO user (id, name, email, settings, info) VALUES (:i, :n, :m, :s, :info)"
with engine.begin() as connection:
    connection.execute(text(insert), {{"i": "u_rich", "n": "Alice", "m": "alice@example.com",
        "s": json.dumps({RICH_SETTINGS!r}), "info": json.dumps({RICH_INFO!r})}})
    connection.execute(text(insert), {{"i": "u_empty", "n": "Bob", "m": "bob@example.com",
        "s": json.dumps({EMPTY_SETTINGS!r}), "info": None}})
    connection.execute(text(insert), {{"i": "u_null", "n": "Carol", "m": "carol@example.com",
        "s": None, "info": None}})

command.upgrade(cfg, {TARGET_REVISION!r})

user = sa.table("user", sa.column("id", sa.Text), sa.column("settings", sa.JSON))
settings = {{}}
with engine.connect() as connection:
    for user_id in ("u_rich", "u_empty", "u_null"):
        value = connection.execute(sa.select(user.c.settings).where(user.c.id == user_id)).scalar()
        settings[user_id] = {{"type": type(value).__name__, "value": value}}
print("RESULT:" + json.dumps(settings))
"""


def test_existing_db_settings_survive_user_table_migration(sqlite_database) -> None:
    """Upgrading an existing SQLite DB keeps user settings intact: not crashed, not
    double-encoded, not dropped."""
    settings = sqlite_database.run(
        UPGRADE_SEEDED_USERS, what=f"the upgrade to {TARGET_REVISION} over existing user rows"
    )

    assert settings["u_rich"] == {"type": "dict", "value": RICH_SETTINGS}, (
        f"user.settings came back altered or double-encoded (#26403): {settings['u_rich']}"
    )
    assert settings["u_empty"] == {"type": "dict", "value": EMPTY_SETTINGS}, (
        f"empty settings were corrupted or dropped: {settings['u_empty']}"
    )
    assert settings["u_null"]["value"] is None, f"NULL settings did not stay NULL: {settings}"
