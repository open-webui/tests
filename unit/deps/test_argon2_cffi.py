"""Dependency contract: argon2-cffi (import name ``argon2``).

argon2-cffi is pinned directly in ``backend/requirements.txt``
(``argon2-cffi==25.1.0``) alongside ``bcrypt``. It provides the Argon2
password-hashing primitive.

IMPORTANT — usage note: at the current HEAD the Open WebUI backend does
NOT import ``argon2`` anywhere. Password hashing in
``open_webui/utils/auth.py`` uses ``bcrypt`` directly
(``bcrypt.hashpw`` / ``bcrypt.checkpw`` / ``bcrypt.gensalt``); the stored
``Auth.password`` column comment even reads "argon2 / bcrypt hash". So
argon2-cffi is a *declared* dependency (kept available for Argon2 hash
support / interop, e.g. via passlib or for verifying pre-existing Argon2
hashes) rather than one the backend's Python imports today.

Because nothing in the source imports it, there are no call sites to pin
keyword arguments against. This module instead pins the *core public
surface* of argon2-cffi — the ``PasswordHasher`` high-level API and the
exception hierarchy — and verifies a real offline hash/verify round-trip,
so that if this dependency is ever wired into the auth path (or a bump
removes/renames the public API) the contract is already guarded. Argon2
is a pure-CPU KDF, so the round-trip is fully offline and deterministic in
outcome (no network, no I/O); we use very low cost parameters to keep it
fast.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "argon2"
DIST_NAME = "argon2-cffi"

# Core public surface of argon2-cffi.
TOP_LEVEL_SYMBOLS = [
    "PasswordHasher",  # high-level hash/verify API
    "Type",  # Argon2 variant enum (Type.ID etc.)
    "exceptions",  # error hierarchy submodule
    "low_level",  # raw hash_secret/verify_secret API
    "exceptions.VerifyMismatchError",
    "exceptions.VerificationError",
    "exceptions.InvalidHashError",  # raised on a malformed encoded hash
    "exceptions.HashingError",
    "exceptions.Argon2Error",  # root of the argon2 error tree
]

PASSWORD_HASHER_METHODS = ["hash", "verify", "check_needs_rehash"]


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "argon2"


def test_version_reported(depcheck):
    assert depcheck.dist_version(DIST_NAME) is not None


def test_top_level_symbols_exist(depcheck):
    """Core argon2-cffi public surface must remain importable."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, TOP_LEVEL_SYMBOLS)


def test_password_hasher_is_class(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert inspect.isclass(mod.PasswordHasher)


def test_password_hasher_methods_exist(depcheck):
    """The hash/verify/check_needs_rehash trio is the high-level API anything
    swapping bcrypt -> argon2 would call. Pin it."""
    mod = depcheck.load(IMPORT_NAME)
    names = set(dir(mod.PasswordHasher))
    missing = [m for m in PASSWORD_HASHER_METHODS if m not in names]
    assert not missing, f"PasswordHasher missing method(s): {missing}"


def test_password_hasher_constructs_with_cost_params(depcheck):
    """PasswordHasher accepts the standard cost knobs (time/memory/parallelism).
    A swap-in would tune these; pin that they remain constructor kwargs."""
    mod = depcheck.load(IMPORT_NAME)
    # Constructor takes keyword cost params; assert the names are accepted.
    depcheck.assert_params(
        mod.PasswordHasher.__init__,
        ["time_cost", "memory_cost", "parallelism"],
    )
    # And that constructing with low cost params works offline.
    ph = mod.PasswordHasher(time_cost=1, memory_cost=8, parallelism=1)
    assert ph is not None


def test_hash_verify_roundtrip_offline(depcheck):
    """A correct password verifies and a wrong one raises VerifyMismatchError —
    the exact contract any password-store integration relies on. Low cost
    params keep this fast; Argon2 is CPU-only so this is fully offline."""
    mod = depcheck.load(IMPORT_NAME)
    ph = mod.PasswordHasher(time_cost=1, memory_cost=8, parallelism=1)
    encoded = ph.hash("correct horse battery staple")
    assert isinstance(encoded, str)
    assert encoded.startswith("$argon2"), f"unexpected hash format: {encoded[:16]!r}"
    # Correct password -> verify returns truthy (True) and does not raise.
    assert ph.verify(encoded, "correct horse battery staple") is True
    # Wrong password -> VerifyMismatchError.
    with pytest.raises(mod.exceptions.VerifyMismatchError):
        ph.verify(encoded, "wrong password")


def test_hash_is_salted_unique(depcheck):
    """Two hashes of the same password must differ (random salt) — the property
    that makes a password store resistant to precomputation."""
    mod = depcheck.load(IMPORT_NAME)
    ph = mod.PasswordHasher(time_cost=1, memory_cost=8, parallelism=1)
    a = ph.hash("samepw")
    b = ph.hash("samepw")
    assert a != b, "argon2 hashes are not salted (identical output for same input)"
    # Both still verify.
    assert ph.verify(a, "samepw") is True
    assert ph.verify(b, "samepw") is True


def test_invalid_hash_raises(depcheck):
    """Verifying against a malformed encoded hash must raise InvalidHashError,
    not silently pass. NOTE: in argon2-cffi 25.x InvalidHashError subclasses
    ValueError (NOT VerificationError), so a wrong-format hash and a
    wrong-password are distinct error families — a swap-in must catch both."""
    mod = depcheck.load(IMPORT_NAME)
    ph = mod.PasswordHasher(time_cost=1, memory_cost=8, parallelism=1)
    with pytest.raises(mod.exceptions.InvalidHashError):
        ph.verify("not-a-real-argon2-hash", "whatever")


def test_exception_hierarchy(depcheck):
    """VerifyMismatchError must subclass VerificationError (and Argon2Error) so
    a broad `except VerificationError` catches a wrong password. InvalidHashError
    sits OUTSIDE that tree (it's a ValueError) — pin both facts, since an auth
    integration's except-clauses depend on the exact placement."""
    mod = depcheck.load(IMPORT_NAME)
    exc = mod.exceptions
    assert issubclass(exc.VerifyMismatchError, exc.VerificationError)
    assert issubclass(exc.VerificationError, exc.Argon2Error)
    # InvalidHashError is deliberately a ValueError, not a VerificationError.
    assert issubclass(exc.InvalidHashError, ValueError)
    assert not issubclass(exc.InvalidHashError, exc.VerificationError)


def test_low_level_api_present(depcheck):
    """argon2.low_level exposes hash_secret / verify_secret + the Type enum —
    the primitives passlib's argon2 handler uses. Pin them so an interop path
    stays available."""
    mod = depcheck.load(IMPORT_NAME)
    low = mod.low_level
    for name in ("hash_secret", "verify_secret"):
        assert hasattr(low, name), f"argon2.low_level.{name} missing"
        assert callable(getattr(low, name))
    # The Type enum (variant selector) must exist with the ID variant.
    assert hasattr(mod, "Type")
    assert hasattr(mod.Type, "ID")


def test_not_imported_by_backend_marker():
    """Documentation guard (no assertion on the dep): records that the backend
    auth path uses bcrypt, not argon2, at this HEAD. If argon2 ever gets wired
    in, the behavioural tests above already pin its contract. This test exists
    purely to make the 'declared, not directly imported' status explicit and
    intentional rather than an oversight."""
    # Intentionally trivial: the real coverage is the offline round-trip above.
    assert True
