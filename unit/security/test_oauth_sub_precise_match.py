"""Regression: OAuth `sub` lookup matched the wrong account on SQLite.

`17cc56670` (open-webui 0.11.3). `get_user_by_oauth_sub` widened the SQLite query with
`or_(sub_expr == sub, sub_expr == int(sub))` for any sub passing `str.isdecimal()`, so a sub
whose integer form is not its own text ('007', or a non-ASCII decimal digit) resolved to a
DIFFERENT person whose stored sub is that number. The fix only widens when `str(int(sub)) == sub`
and the value fits in a signed 64 bit integer. A stored JSON-number sub only exists in rows
written by older releases, so these cases stay unit tests on the public model function; the
beyond-64-bit sign-in is pinned over HTTP by integration/security/test_oauth_sub_precise_match.py.

Discriminates: passes on dev bbfa876af; with 17cc56670 reverted in a copy the zero padded and
non-ASCII decimal subs resolve to the account storing the number 7.
"""

from __future__ import annotations

import pytest

from unit.security.memory_db import memory_database

pytestmark = pytest.mark.regression

INT64_MAX = 2**63 - 1


@pytest.fixture
def users(owui_module):
    return owui_module("open_webui.models.users")


async def lookup_among(owui_module, users, accounts: dict, provider: str, sub: str) -> str | None:
    """Store `accounts` (name to oauth JSON, in scan order) and return the name `sub` finds."""
    async with memory_database(owui_module, users.User):
        for name, oauth in accounts.items():
            await users.Users.insert_new_user(
                id=name, name=name, email=f"{name}@example.com", oauth=oauth
            )
        found = await users.Users.get_user_by_oauth_sub(provider=provider, sub=sub)
    return found.name if found else None


# --------------------------------------------------------------------------- narrow


@pytest.mark.parametrize("sub", ["007", "٧"], ids=["zero-padded", "arabic-indic"])
@pytest.mark.asyncio
async def test_a_sub_that_only_coerces_to_a_number_does_not_match_it(owui_module, users, sub):
    """Narrow: '007' or '٧' must not resolve to the account whose stored sub is the number 7."""
    accounts = {"seven": {"github": {"sub": 7}}}
    assert await lookup_among(owui_module, users, accounts, "github", sub) is None


@pytest.mark.parametrize("sub", ["007", "٧"], ids=["zero-padded", "arabic-indic"])
@pytest.mark.asyncio
async def test_a_sub_that_only_coerces_to_a_number_finds_its_own_account(owui_module, users, sub):
    """Narrow: with both present, the sub gets its own account; the numeric one scans first."""
    accounts = {"seven": {"github": {"sub": 7}}, "own": {"github": {"sub": sub}}}
    assert await lookup_among(owui_module, users, accounts, "github", sub) == "own"


# --------------------------------------------------------------------------- broad


@pytest.mark.parametrize("sub", ["007", "0007", "00000007", "٧", "۷"])
@pytest.mark.asyncio
async def test_no_textual_variant_of_a_number_reaches_its_account(owui_module, users, sub):
    """Broad: no other spelling of a number may reach the account storing that number."""
    accounts = {"seven": {"github": {"sub": 7}}, "seven_text": {"github": {"sub": "7"}}}
    assert await lookup_among(owui_module, users, accounts, "github", sub) is None


@pytest.mark.parametrize("stored", [7, "7", INT64_MAX, str(INT64_MAX)])
@pytest.mark.asyncio
async def test_a_plain_numeric_sub_matches_as_number_or_text(owui_module, users, stored):
    """Broad: a plain decimal sub, up to the 64 bit boundary, matches either stored form."""
    accounts = {"target": {"github": {"sub": stored}}}
    assert await lookup_among(owui_module, users, accounts, "github", str(stored)) == "target"


# --------------------------------------------------------------------------- nearby


@pytest.mark.parametrize(
    ("sub", "provider", "expected"),
    [
        ("abc12345", "oidc", "alice"),
        ("", "oidc", None),
        ("nope", "oidc", None),
        ("abc12345", "github", None),
        ("0", "github", None),
    ],
    ids=["exact", "empty", "unknown-sub", "other-provider", "local-account"],
)
@pytest.mark.asyncio
async def test_ordinary_hits_and_misses_are_unchanged(owui_module, users, sub, provider, expected):
    """Nearby: opaque subs match exactly; empty, unknown and local-only accounts miss."""
    accounts = {
        "alice": {"oidc": {"sub": "abc12345"}},
        "bob": {"oidc": {"sub": "abc12346"}},
        "local": None,
    }
    assert await lookup_among(owui_module, users, accounts, provider, sub) == expected
