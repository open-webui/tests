"""Regression: `user.oauth` double-encoded by the user-table migration (PR #28107, commit
bd8378f643, issue #28101, shipped in v0.11.1).

The 0.6.41 -> 0.9.6 user-table migration `b10670c03dd5` wrote the oauth mapping as
`json.dumps({provider: {...}})` into a JSON column, so the column held a JSON string instead of
an object. Every provider-and-sub lookup missed and those accounts could not sign in. The fix
stores the object directly and adds migration `6d09d1bf1f23`, which rewrites the rows already
double-encoded. Both run on a database seeded at the revision before them, in a fresh
interpreter, and the rows are read back through a JSON column.

Stays a unit test: reaching these rows through the API needs a database from before v0.11.1.
The two other v0.11.1 repairs of this file, list-shaped default model rows and connection tags
saved as strings, are pinned from outside in `integration/migrations/test_startup_repairs.py`.

Discriminates: passes on dev bbfa876af; with `6d09d1bf1f23`'s upgrade emptied in a copy of it
the double-encoded row stays a string, and with `b10670c03dd5` writing `json.dumps(...)` again
the converted `oauth_sub` row comes back a string.
"""

from __future__ import annotations

import pytest

from .conftest import sqlite_at

pytestmark = pytest.mark.regression

# Head of the chain on v0.11.0 and the parent of the repair on v0.11.1. Seeding here reaches the
# same schema on both refs; upgrading to head afterwards runs 6d09d1bf1f23 where it exists.
PRE_REPAIR_REVISION = "f0bd01a18a3d"
# The user-table migration that writes user.oauth, and the revision before it.
USER_TABLE_REVISION = "b10670c03dd5"
USER_TABLE_DOWN_REVISION = "2f1211949ecc"

OAUTH_OBJECT = {"google": {"sub": "108154321"}}

READ_BACK = """
import sqlalchemy as sa
from sqlalchemy import create_engine

engine = create_engine(os.environ['DATABASE_URL'])
user = sa.table(
    'user',
    sa.column('id', sa.Text),
    sa.column('name', sa.Text),
    sa.column('email', sa.Text),
    sa.column('oauth', sa.JSON),
)


def read_back(user_ids):
    rows = {}
    with engine.connect() as connection:
        for user_id in user_ids:
            value = connection.execute(sa.select(user.c.oauth).where(user.c.id == user_id)).scalar()
            rows[user_id] = {'type': type(value).__name__, 'value': value}
    return rows
"""

REPAIR = (
    READ_BACK
    + f"""
command.upgrade(cfg, {PRE_REPAIR_REVISION!r})
seeds = {{
    'u_double': json.dumps({OAUTH_OBJECT!r}),
    'u_object': {OAUTH_OBJECT!r},
    'u_null': None,
    'u_plain': 'not-json-at-all',
    'u_scalar': '12345',
}}
with engine.begin() as connection:
    for user_id, value in seeds.items():
        connection.execute(
            sa.insert(user).values(id=user_id, name=user_id, email=user_id + '@x.io', oauth=value)
        )
command.upgrade(cfg, 'head')
print('RESULT:' + json.dumps(read_back(seeds)))
"""
)

OAUTH_SUB_CONVERSION = (
    READ_BACK
    + f"""
command.upgrade(cfg, {USER_TABLE_DOWN_REVISION!r})
with engine.begin() as connection:
    connection.execute(
        sa.text("INSERT INTO user (id, name, email, oauth_sub) VALUES (:i, :n, :m, :s)"),
        {{'i': 'u_sub', 'n': 'Dana', 'm': 'dana@example.com', 's': 'google@108154321'}},
    )
command.upgrade(cfg, {USER_TABLE_REVISION!r})
print('RESULT:' + json.dumps(read_back(['u_sub'])))
"""
)


@pytest.fixture(scope="module")
def repaired_rows(open_webui_backend, tmp_path_factory) -> dict:
    database = sqlite_at(open_webui_backend, tmp_path_factory.mktemp("oauth-repair"))
    return database.run(REPAIR, what="upgrading seeded user.oauth rows to head")


def test_double_encoded_oauth_is_repaired_to_an_object(repaired_rows):
    row = repaired_rows["u_double"]
    assert row["type"] == "dict", (
        f"user.oauth is still a {row['type']} after upgrading to head, so provider+sub lookups "
        f"keep missing and the account cannot sign in (#28101): {row['value']!r}"
    )
    assert row["value"] == OAUTH_OBJECT


def test_an_oauth_object_is_left_alone(repaired_rows):
    assert repaired_rows["u_object"] == {"type": "dict", "value": OAUTH_OBJECT}


def test_null_oauth_stays_null(repaired_rows):
    assert repaired_rows["u_null"]["value"] is None


@pytest.mark.parametrize(
    ("user_id", "stored"), [("u_plain", "not-json-at-all"), ("u_scalar", "12345")]
)
def test_strings_that_are_not_an_encoded_object_are_left_alone(repaired_rows, user_id, stored):
    assert repaired_rows[user_id] == {"type": "str", "value": stored}


def test_legacy_oauth_sub_migrates_to_an_object(sqlite_database):
    row = sqlite_database.run(OAUTH_SUB_CONVERSION, what="the oauth_sub conversion")["u_sub"]

    assert row["type"] == "dict", (
        f"the oauth_sub conversion wrote a {row['type']} into the JSON column, so every upgrade "
        f"keeps producing oauth rows no sign-in can match: {row['value']!r}"
    )
    assert row["value"] == OAUTH_OBJECT
