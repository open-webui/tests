"""Dependency contract: pytz (import name ``pytz``).

Open WebUI uses pytz for one focused job: timezone-aware UTC timestamps on
the auth path. ``utils/auth.py`` does ``import pytz`` and ``from pytz import
UTC`` and then stamps JWTs with ``datetime.now(UTC)`` (the ``exp`` and
``iat`` claims) and computes the revocation TTL as ``exp -
int(datetime.now(UTC).timestamp())``. The correctness of token expiry and
revocation therefore depends on ``pytz.UTC`` being a real ``datetime.tzinfo``
with a zero UTC offset that ``datetime.now()`` accepts.

This module pins exactly that contract (plus the small amount of general
pytz surface a future change might reach for: the ``pytz.timezone`` factory
and the ``UnknownTimeZoneError`` exception) so a pytz bump that broke
``UTC`` fails loudly here instead of as a subtly-wrong token-expiry
calculation in production. Pattern mirrors test_requests.py: symbol checks
plus offline behavioural contracts. Everything is pure-Python and
deterministic — no network, no clock dependence beyond a monotonic sanity
check.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone, tzinfo

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "pytz"
DIST_NAME = "pytz"

USED_SYMBOLS = [
    "UTC",  # auth.py: `from pytz import UTC`
    "utc",  # lowercase alias (same object)
    "timezone",  # general factory (pytz.timezone('...'))
    "UnknownTimeZoneError",  # raised for bad zone names
]


# ---------------------------------------------------------------------------
# Import + version
# ---------------------------------------------------------------------------


def test_import(depcheck):
    """`pytz` must import (skip cleanly if absent in this env)."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "pytz"


def test_version_reported(depcheck):
    """The installed distribution version must be resolvable so bump tooling
    and this suite agree on what's under test."""
    depcheck.load(IMPORT_NAME)
    assert depcheck.dist_version(DIST_NAME) is not None


# ---------------------------------------------------------------------------
# Symbol-existence (API surface)
# ---------------------------------------------------------------------------


def test_used_symbols_exist(depcheck):
    """Every pytz symbol the codebase references (and the small general surface)
    must still exist."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_timezone_is_callable(depcheck):
    """pytz.timezone(name) is the general factory; must remain callable."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "timezone")


# ---------------------------------------------------------------------------
# UTC contract — the object auth.py actually uses
# ---------------------------------------------------------------------------


def test_utc_is_tzinfo(depcheck):
    """auth.py passes pytz.UTC straight into datetime.now(UTC). It must be a
    real datetime.tzinfo subclass instance, or datetime would reject it."""
    mod = depcheck.load(IMPORT_NAME)
    assert isinstance(mod.UTC, tzinfo), "pytz.UTC is no longer a datetime.tzinfo"


def test_utc_and_lowercase_alias_are_same(depcheck):
    """`from pytz import UTC` and `pytz.utc` have always referred to the same
    singleton; pin that so either import keeps the identical behaviour."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.UTC is mod.utc, "pytz.UTC and pytz.utc diverged"


def test_utc_has_zero_offset(depcheck):
    """The whole point of stamping with UTC is a zero offset. utcoffset must be
    exactly timedelta(0), otherwise exp/iat claims would be skewed."""
    mod = depcheck.load(IMPORT_NAME)
    # tzinfo.utcoffset takes a datetime argument.
    sample = datetime(2024, 1, 1, 12, 0, 0)
    assert mod.UTC.utcoffset(sample) == timedelta(0), "pytz.UTC offset is no longer zero"


# ---------------------------------------------------------------------------
# Behavioural: the exact auth.py timestamp idioms
# ---------------------------------------------------------------------------


def test_behaviour_datetime_now_utc_is_aware(depcheck):
    """auth.py: payload['iat'] = datetime.now(UTC). The result must be timezone
    aware with pytz.UTC attached (a naive datetime would break exp comparisons)."""
    mod = depcheck.load(IMPORT_NAME)
    now = datetime.now(mod.UTC)
    assert now.tzinfo is not None, "datetime.now(pytz.UTC) produced a naive datetime"
    assert now.utcoffset() == timedelta(0)


def test_behaviour_utc_timestamp_matches_stdlib_utc(depcheck):
    """auth.py computes ttl = exp - int(datetime.now(UTC).timestamp()). The
    POSIX timestamp pytz.UTC yields must agree with the stdlib timezone.utc
    timestamp (same instant), so the TTL maths is correct regardless of which
    UTC object is used."""
    mod = depcheck.load(IMPORT_NAME)
    pytz_ts = datetime.now(mod.UTC).timestamp()
    std_ts = datetime.now(timezone.utc).timestamp()
    # Both sample "now"; allow a small wall-clock delta between the two calls.
    assert abs(pytz_ts - std_ts) < 5.0, (
        "pytz.UTC and stdlib timezone.utc disagree on the current POSIX "
        "timestamp; token TTL/exp computations would be wrong."
    )


def test_behaviour_exp_arithmetic_roundtrip(depcheck):
    """Mirror auth.py end to end: expire = datetime.now(UTC) + expires_delta,
    then ttl = int(expire.timestamp()) - int(datetime.now(UTC).timestamp()).
    For a 1-hour delta the TTL must come out at ~3600s (positive, near the
    delta) — the property the revocation store relies on."""
    mod = depcheck.load(IMPORT_NAME)
    now = datetime.now(mod.UTC)
    expire = now + timedelta(hours=1)
    ttl = int(expire.timestamp()) - int(now.timestamp())
    assert 3590 <= ttl <= 3600, f"exp-now TTL arithmetic drifted: got {ttl}s"
    assert ttl > 0, "TTL must be positive for a future expiry"


def test_behaviour_timezone_factory_returns_utc(depcheck):
    """pytz.timezone('UTC') resolves to a usable tzinfo with zero offset (the
    general factory path, in case config-driven zones are introduced)."""
    mod = depcheck.load(IMPORT_NAME)
    tz = mod.timezone("UTC")
    assert isinstance(tz, tzinfo)
    assert datetime.now(tz).utcoffset() == timedelta(0)


def test_behaviour_unknown_timezone_raises(depcheck):
    """pytz.timezone(<garbage>) raises UnknownTimeZoneError — pin the exception
    type so callers can keep catching it specifically."""
    mod = depcheck.load(IMPORT_NAME)
    with pytest.raises(mod.UnknownTimeZoneError):
        mod.timezone("Not/A_Real_Zone_xyz")
