"""Regression: OAuth `sub` lookup matched the wrong account on SQLite.

`17cc56670` (open-webui 0.11.3). `get_user_by_oauth_sub` widened the SQLite query with
`or_(sub_expr == sub, sub_expr == int(sub))` for any sub passing `str.isdecimal()`, so a sub
whose integer form is not its own text ('007', or a non-ASCII decimal digit) resolved to a
DIFFERENT person whose stored sub is that number, and a sub past the signed 64 bit range blew
up in the SQLite driver. The fix only widens when `str(int(sub)) == sub` and the value fits in
a signed 64 bit integer.

Discriminates: passes on v0.11.3, fails on v0.11.1 (the unconditional int() widening matches a
foreign account for zero padded and non-ASCII decimal subs, and overflows on a huge sub).
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.regression

INT64_MAX = 2**63 - 1


@pytest.fixture(scope="module")
def users_module(owui_module):
    """`open_webui.models.users` (get_user_by_oauth_sub)."""
    return owui_module("open_webui.models.users")


@pytest.fixture(scope="module")
def db_module(owui_module):
    """`open_webui.internal.db` (session sharing flag)."""
    return owui_module("open_webui.internal.db")


@asynccontextmanager
async def _user_db(users_module, db_module, rows):
    """A throwaway in-memory SQLite holding just the user table.

    `rows` is an ordered mapping of user name to the raw `oauth` JSON to store. Insertion order
    is the order SQLite scans in, so a test can pin which account an over-wide query would hit.
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    user_table = users_module.User
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(user_table.__table__.create)
        maker = async_sessionmaker(engine, expire_on_commit=False)
        async with maker() as session:
            now = int(time.time())
            for name, oauth in rows.items():
                session.add(
                    user_table(
                        id=str(uuid.uuid4()),
                        name=name,
                        email=f"{name}@example.com",
                        role="user",
                        profile_image_url="",
                        oauth=oauth,
                        created_at=now,
                        updated_at=now,
                        last_active_at=now,
                    )
                )
            await session.commit()
            # get_async_db_context only reuses the passed session when sharing is enabled.
            with patch.object(db_module, "DATABASE_ENABLE_SESSION_SHARING", True):
                yield session
    finally:
        await engine.dispose()


async def _lookup(users_module, session, provider, sub):
    return await users_module.Users.get_user_by_oauth_sub(provider, sub, db=session)


# --------------------------------------------------------------------------- narrow


@pytest.mark.asyncio
async def test_zero_padded_sub_does_not_match_numeric_account(users_module, db_module):
    """Narrow: '007' must not resolve to the account whose stored sub is the number 7."""
    rows = {"seven": {"github": {"sub": 7}}}
    async with _user_db(users_module, db_module, rows) as session:
        assert await _lookup(users_module, session, "github", "007") is None


@pytest.mark.asyncio
async def test_zero_padded_sub_resolves_to_its_own_account(users_module, db_module):
    """Narrow: with both accounts present, '007' gets the '007' account, never the numeric one.

    The numeric account is inserted first so an over-wide query returns it.
    """
    rows = {
        "seven": {"github": {"sub": 7}},
        "double_oh_seven": {"github": {"sub": "007"}},
    }
    async with _user_db(users_module, db_module, rows) as session:
        found = await _lookup(users_module, session, "github", "007")
        assert found is not None and found.name == "double_oh_seven"


@pytest.mark.asyncio
async def test_non_ascii_decimal_sub_does_not_match_numeric_account(users_module, db_module):
    """Narrow: an Arabic-Indic digit is decimal to Python but is not the text of its int value."""
    rows = {"seven": {"github": {"sub": 7}}}
    async with _user_db(users_module, db_module, rows) as session:
        assert await _lookup(users_module, session, "github", "٧") is None


@pytest.mark.asyncio
async def test_non_ascii_decimal_sub_resolves_to_its_own_account(users_module, db_module):
    """Narrow: with both accounts present, '٧' gets the '٧' account, never the numeric one.

    The numeric account is inserted first so an over-wide query returns it.
    """
    rows = {
        "seven": {"github": {"sub": 7}},
        "arabic_seven": {"github": {"sub": "٧"}},
    }
    async with _user_db(users_module, db_module, rows) as session:
        found = await _lookup(users_module, session, "github", "٧")
        assert found is not None and found.name == "arabic_seven"


@pytest.mark.asyncio
async def test_sub_beyond_int64_still_matches_its_own_account(users_module, db_module):
    """Narrow: a sub past the signed 64 bit range is compared as text, not handed to SQLite."""
    huge = str(INT64_MAX + 1)
    rows = {"huge": {"oidc": {"sub": huge}}}
    async with _user_db(users_module, db_module, rows) as session:
        found = await _lookup(users_module, session, "oidc", huge)
        assert found is not None and found.name == "huge"


@pytest.mark.asyncio
async def test_sub_beyond_int64_misses_cleanly_when_absent(users_module, db_module):
    """Narrow: an unknown out-of-range sub returns None instead of erroring."""
    rows = {"seven": {"github": {"sub": 7}}}
    async with _user_db(users_module, db_module, rows) as session:
        assert await _lookup(users_module, session, "github", str(2**70)) is None


# --------------------------------------------------------------------------- broad


@pytest.mark.parametrize("sub", ["007", "0007", "00000007", "٧", "۷"])
@pytest.mark.asyncio
async def test_lookup_never_returns_account_that_merely_coerces_to_same_number(
    users_module, db_module, sub
):
    """Broad: no textual variant of a number may reach the account storing that number."""
    rows = {"seven": {"github": {"sub": 7}}, "seven_text": {"github": {"sub": "7"}}}
    async with _user_db(users_module, db_module, rows) as session:
        assert await _lookup(users_module, session, "github", sub) is None


@pytest.mark.parametrize("stored", [7, "7", 9223372036854775807, "9223372036854775807"])
@pytest.mark.asyncio
async def test_plain_numeric_sub_round_trips(users_module, db_module, stored):
    """Broad: a plain decimal sub still matches whether stored as a JSON number or a string."""
    rows = {"target": {"github": {"sub": stored}}}
    async with _user_db(users_module, db_module, rows) as session:
        found = await _lookup(users_module, session, "github", str(stored))
        assert found is not None and found.name == "target"


@pytest.mark.asyncio
async def test_int64_boundary_sub_is_still_widened(users_module, db_module):
    """Broad: the boundary value itself stays inside the numeric widening."""
    rows = {"boundary": {"github": {"sub": INT64_MAX}}}
    async with _user_db(users_module, db_module, rows) as session:
        found = await _lookup(users_module, session, "github", str(INT64_MAX))
        assert found is not None and found.name == "boundary"


# --------------------------------------------------------------------------- nearby


@pytest.mark.asyncio
async def test_non_numeric_sub_matches_exactly(users_module, db_module):
    """Nearby: ordinary opaque subs are unaffected."""
    rows = {"alice": {"oidc": {"sub": "abc12345"}}, "bob": {"oidc": {"sub": "abc12346"}}}
    async with _user_db(users_module, db_module, rows) as session:
        found = await _lookup(users_module, session, "oidc", "abc12345")
        assert found is not None and found.name == "alice"


@pytest.mark.asyncio
async def test_empty_sub_matches_nobody(users_module, db_module):
    """Nearby: an empty sub is a miss, not a crash."""
    rows = {"alice": {"oidc": {"sub": "abc12345"}}, "seven": {"github": {"sub": 7}}}
    async with _user_db(users_module, db_module, rows) as session:
        assert await _lookup(users_module, session, "oidc", "") is None


@pytest.mark.asyncio
async def test_unknown_sub_and_unknown_provider_match_nobody(users_module, db_module):
    """Nearby: a sub that does not exist, and a known sub under another provider, both miss."""
    rows = {"alice": {"oidc": {"sub": "abc12345"}}}
    async with _user_db(users_module, db_module, rows) as session:
        assert await _lookup(users_module, session, "oidc", "nope") is None
        assert await _lookup(users_module, session, "github", "abc12345") is None


@pytest.mark.asyncio
async def test_account_without_oauth_is_never_matched(users_module, db_module):
    """Nearby: a local account with no oauth entry stays invisible to the sub lookup."""
    rows = {"local": None, "seven": {"github": {"sub": 7}}}
    async with _user_db(users_module, db_module, rows) as session:
        assert await _lookup(users_module, session, "github", "0") is None
        found = await _lookup(users_module, session, "github", "7")
        assert found is not None and found.name == "seven"
