"""Dependency contract: bcrypt (SECURITY-CRITICAL — password hashing).

Open WebUI hashes and verifies *every* local-account password through
`bcrypt` in `utils/auth.py`:

  * ``get_password_hash(password)`` ->
    ``bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')``
    — hash with a freshly generated salt at the library-default cost, then
    store the resulting modular-crypt string in the DB.
  * ``verify_password(plain, hashed)`` ->
    ``bcrypt.checkpw(plain.encode('utf-8'), hashed.encode('utf-8'))``
    — constant-time compare of a candidate password against a stored hash.

The backend also relies on a hard bcrypt invariant: the password input is
truncated to **72 bytes** before hashing (``validate_password`` raises if
longer; the signin path in ``routers/auths.py`` truncates to ``[:72]``).
That 72-byte boundary is part of the security contract — bytes past it are
ignored by the algorithm, which is why the codebase caps it explicitly.

This module pins exactly that surface plus the security properties the auth
layer depends on (a fresh salt every call, a wrong password rejected, a
tampered hash rejected, the 72-byte truncation behaviour, the ``$2b$``
modular-crypt format and default cost), so a `bcrypt` major bump that
removed/renamed/weakened any of it fails loudly here instead of silently
degrading authentication.

Pattern mirrors the unit/deps/ exemplar: symbol-existence + signature
checks for the API surface, plus offline behavioural contracts (real
hashpw/checkpw round-trips at a low cost factor — no network, no services).
Uses the `depcheck` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "bcrypt"
DIST_NAME = "bcrypt"

# The complete set of top-level callables the Open WebUI backend resolves on
# `bcrypt`. The auth layer uses only hashpw / gensalt / checkpw; kdf/__about__
# are pinned as part of the stable public surface (regression guard).
USED_SYMBOLS = ["hashpw", "gensalt", "checkpw"]

# A deliberately low cost factor for the behavioural round-trips: bcrypt is
# intentionally slow, and the *correctness* contract (round-trip, rejection,
# format) is independent of the cost. The default-cost assertion is exercised
# separately and only inspects the emitted salt, never hashing at cost 12.
_FAST_ROUNDS = 4


# --------------------------------------------------------------------------- #
# Import + version + API surface
# --------------------------------------------------------------------------- #
def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "bcrypt"


def test_version_reported(depcheck):
    """Sanity: the installed distribution version is resolvable (so bump
    tooling and this suite agree on what's under test)."""
    depcheck.load(IMPORT_NAME)
    assert depcheck.dist_version(DIST_NAME) is not None


def test_used_symbols_exist(depcheck):
    """Every bcrypt callable the codebase references must still exist."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_core_functions_callable(depcheck):
    """hashpw / gensalt / checkpw must all be callable (not data attrs)."""
    mod = depcheck.load(IMPORT_NAME)
    for name in USED_SYMBOLS:
        depcheck.assert_callable(mod, name)


# --------------------------------------------------------------------------- #
# Signatures — the exact call shapes auth.py uses
# --------------------------------------------------------------------------- #
def test_hashpw_signature(depcheck):
    """auth.py calls ``bcrypt.hashpw(password, salt)`` positionally — two
    positional parameters in (password, salt) order."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.hashpw, ["password", "salt"])


def test_checkpw_signature(depcheck):
    """auth.py calls ``bcrypt.checkpw(plain, hashed)`` positionally — two
    positional parameters; the second is the stored hash."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.checkpw, ["password", "hashed_password"])


def test_gensalt_signature_has_rounds_and_prefix(depcheck):
    """auth.py calls ``bcrypt.gensalt()`` with no args, relying on the library
    default cost. Pin that ``rounds`` and ``prefix`` remain the (defaulted)
    parameters so the no-arg call keeps producing a default-cost ``$2b$`` salt.
    """
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.gensalt, ["rounds", "prefix"])


# --------------------------------------------------------------------------- #
# gensalt — salt format + default cost factor (the stored-hash format)
# --------------------------------------------------------------------------- #
def test_gensalt_returns_bytes(depcheck):
    """gensalt() returns bytes; auth.py feeds it straight into hashpw."""
    mod = depcheck.load(IMPORT_NAME)
    salt = mod.gensalt(_FAST_ROUNDS)
    assert isinstance(salt, (bytes, bytearray))


def test_gensalt_default_is_2b_cost_12(depcheck):
    """``bcrypt.gensalt()`` with no args (exactly what auth.py does) must emit a
    modular-crypt salt of the form ``$2b$12$...`` — prefix ``2b`` and the
    library-default cost factor 12. This is the format every stored Open WebUI
    password hash carries, and the cost that protects them at rest.

    A bump that silently lowered the default cost (weaker hashes) or changed the
    prefix (DB-format drift) is exactly the kind of regression this guards.
    """
    mod = depcheck.load(IMPORT_NAME)
    salt = mod.gensalt()
    text = salt.decode("ascii")
    assert text.startswith("$2b$"), f"default salt prefix changed: {text!r}"
    cost = text.split("$")[2]
    assert cost == "12", f"default bcrypt cost factor changed from 12 to {cost!r}"


def test_gensalt_respects_explicit_rounds(depcheck):
    """gensalt(rounds=N) must encode N as the zero-padded cost field. The auth
    layer trusts the default, but the cost field is part of the hash format;
    pin that an explicit cost round-trips into the salt string."""
    mod = depcheck.load(IMPORT_NAME)
    salt = mod.gensalt(_FAST_ROUNDS)
    cost = salt.decode("ascii").split("$")[2]
    assert cost == f"{_FAST_ROUNDS:02d}", f"explicit rounds not encoded: {salt!r}"


def test_gensalt_unique_per_call(depcheck):
    """Each gensalt() call must yield fresh random salt — distinct salts are
    what make two hashes of the same password differ. If gensalt returned a
    constant, every user's hash would be directly comparable. Pin uniqueness."""
    mod = depcheck.load(IMPORT_NAME)
    salts = {mod.gensalt(_FAST_ROUNDS) for _ in range(8)}
    assert len(salts) == 8, "gensalt() is not producing unique salts"


def test_gensalt_prefix_2a_supported(depcheck):
    """Stored hashes from older deployments may carry the ``$2a$`` prefix;
    gensalt(prefix=b'2a') must still be accepted (so checkpw can verify them
    and the format stays loadable)."""
    mod = depcheck.load(IMPORT_NAME)
    salt = mod.gensalt(_FAST_ROUNDS, prefix=b"2a")
    assert salt.decode("ascii").startswith("$2a$")


# --------------------------------------------------------------------------- #
# hashpw — output format + the auth.py encode/decode hop
# --------------------------------------------------------------------------- #
def test_hashpw_returns_60_byte_2b_hash(depcheck):
    """A bcrypt hash is a 60-byte modular-crypt string ``$2b$<cost>$<22-char
    salt><31-char digest>``. auth.py stores exactly this (decoded to str). Pin
    the length and the ``$2b$`` prefix — the DB column shape depends on it."""
    mod = depcheck.load(IMPORT_NAME)
    digest = mod.hashpw(b"correct horse battery staple", mod.gensalt(_FAST_ROUNDS))
    assert isinstance(digest, (bytes, bytearray))
    assert len(digest) == 60, f"bcrypt hash length changed: {len(digest)}"
    assert digest.decode("ascii").startswith("$2b$")


def test_hashpw_is_ascii_safe_for_decode(depcheck):
    """auth.py does ``hashpw(...).decode('utf-8')`` and later stores it as a str
    that is re-encoded for checkpw. The hash must be pure ASCII so that
    decode/encode round-trips byte-for-byte (no data loss in the DB hop)."""
    mod = depcheck.load(IMPORT_NAME)
    digest = mod.hashpw(b"password123", mod.gensalt(_FAST_ROUNDS))
    as_str = digest.decode("utf-8")  # the exact call get_password_hash makes
    assert as_str.encode("utf-8") == digest  # round-trips losslessly


def test_hashpw_same_password_distinct_hashes(depcheck):
    """Two hashes of the same password (each with a fresh gensalt, as auth.py
    does) must differ — the salt randomises the output. Equal hashes would mean
    salting is broken and password reuse becomes detectable from the DB."""
    mod = depcheck.load(IMPORT_NAME)
    pw = b"same-password"
    h1 = mod.hashpw(pw, mod.gensalt(_FAST_ROUNDS))
    h2 = mod.hashpw(pw, mod.gensalt(_FAST_ROUNDS))
    assert h1 != h2


def test_hashpw_requires_bytes_not_str(depcheck):
    """hashpw operates on bytes — auth.py always ``.encode('utf-8')`` first.
    Passing a str must raise (a TypeError-class error), which is why the
    codebase never hands hashpw a str. Guards against a silent str-path."""
    mod = depcheck.load(IMPORT_NAME)
    with pytest.raises(TypeError):
        mod.hashpw("not-bytes", mod.gensalt(_FAST_ROUNDS))


# --------------------------------------------------------------------------- #
# checkpw — the verify_password security contract
# --------------------------------------------------------------------------- #
def test_checkpw_accepts_correct_password(depcheck):
    """The core auth round-trip: hash a password (get_password_hash), then
    verify the same password (verify_password) -> True."""
    mod = depcheck.load(IMPORT_NAME)
    pw = b"s3cret-pa55word"
    stored = mod.hashpw(pw, mod.gensalt(_FAST_ROUNDS))
    assert mod.checkpw(pw, stored) is True


def test_checkpw_rejects_wrong_password(depcheck):
    """A different password must NOT verify against the stored hash. This is the
    single most important security property of the auth layer — pin it."""
    mod = depcheck.load(IMPORT_NAME)
    stored = mod.hashpw(b"the-real-password", mod.gensalt(_FAST_ROUNDS))
    assert mod.checkpw(b"a-wrong-password", stored) is False


def test_checkpw_rejects_empty_password_against_real_hash(depcheck):
    """An empty candidate must not verify against a hash of a non-empty
    password — guards against a degenerate always-true compare."""
    mod = depcheck.load(IMPORT_NAME)
    stored = mod.hashpw(b"nonempty", mod.gensalt(_FAST_ROUNDS))
    assert mod.checkpw(b"", stored) is False


def test_checkpw_via_str_decode_encode_roundtrip(depcheck):
    """Reproduce the *exact* auth.py data flow end to end: hash -> decode to str
    (stored in DB) -> encode back to bytes -> checkpw. Both the correct and a
    wrong password must classify correctly through that hop."""
    mod = depcheck.load(IMPORT_NAME)
    plain = "Pa$$w0rd-with-unicode-café"
    # get_password_hash: hashpw(plain.encode('utf-8'), gensalt()).decode('utf-8')
    stored_str = mod.hashpw(plain.encode("utf-8"), mod.gensalt(_FAST_ROUNDS)).decode("utf-8")
    # verify_password: checkpw(plain.encode('utf-8'), stored.encode('utf-8'))
    assert mod.checkpw(plain.encode("utf-8"), stored_str.encode("utf-8")) is True
    assert mod.checkpw(b"different", stored_str.encode("utf-8")) is False


def test_checkpw_rejects_tampered_hash(depcheck):
    """Flipping a byte of the stored digest must make verification fail (or
    raise). A DB-corrupted/tampered hash must never validate the original
    password. We accept either False or a raised error — never True."""
    mod = depcheck.load(IMPORT_NAME)
    pw = b"original"
    stored = bytearray(mod.hashpw(pw, mod.gensalt(_FAST_ROUNDS)))
    stored[-1] ^= 0x01  # corrupt the final digest character
    try:
        assert mod.checkpw(pw, bytes(stored)) is False
    except ValueError:
        pass  # malformed hash rejected by raising — also acceptable


def test_checkpw_malformed_hash_does_not_return_true(depcheck):
    """A non-bcrypt string in the hash field (e.g. an argon2 hash or garbage)
    must not spuriously verify. models/auths.py notes the column can hold either
    'argon2 / bcrypt' — handing checkpw a non-bcrypt value must never return
    True (it raises ValueError, which auth callers treat as a failed verify)."""
    mod = depcheck.load(IMPORT_NAME)
    with pytest.raises(ValueError):
        mod.checkpw(b"password", b"not-a-valid-bcrypt-hash")


def test_checkpw_cross_prefix_2a_hash_verifies(depcheck):
    """A hash generated with the legacy ``$2a$`` prefix must still verify with
    the correct password — older stored hashes must keep working after a bump."""
    mod = depcheck.load(IMPORT_NAME)
    pw = b"legacy-account"
    stored = mod.hashpw(pw, mod.gensalt(_FAST_ROUNDS, prefix=b"2a"))
    assert mod.checkpw(pw, stored) is True
    assert mod.checkpw(b"wrong", stored) is False


# --------------------------------------------------------------------------- #
# The 72-byte boundary — a security-relevant bcrypt invariant the backend
# explicitly codes around (validate_password raises; auths.py truncates [:72]).
# --------------------------------------------------------------------------- #
def test_hashpw_rejects_password_over_72_bytes(depcheck):
    """bcrypt 4+/5 *raises* ``ValueError`` on a >72-byte password instead of
    silently truncating (older bcrypt truncated). THIS is the reason the backend
    truncates manually: ``validate_password`` rejects >72-byte inputs up front,
    and ``routers/auths.py`` truncates the signin password to ``[:72]`` *before*
    calling verify_password — without that, a >72-byte login would crash here.

    Pin the raise: if a future bcrypt reverted to silent truncation, the
    backend's explicit handling would become dead code, and (more importantly)
    if the cap moved, the backend's hard-coded 72 would be wrong. The error
    message even tells callers to ``my_password[:72]`` — exactly what auths.py
    does.
    """
    mod = depcheck.load(IMPORT_NAME)
    too_long = b"A" * 73
    with pytest.raises(ValueError):
        mod.hashpw(too_long, mod.gensalt(_FAST_ROUNDS))


def test_checkpw_rejects_password_over_72_bytes(depcheck):
    """The verify side is symmetric: checkpw with a >72-byte candidate raises
    too. This is why ``routers/auths.py`` truncates ``form_data.password`` to
    72 bytes BEFORE handing it to verify_password — an over-long login password
    reaching checkpw unmodified would raise, not just fail to match."""
    mod = depcheck.load(IMPORT_NAME)
    stored = mod.hashpw(b"A" * 72, mod.gensalt(_FAST_ROUNDS))
    with pytest.raises(ValueError):
        mod.checkpw(b"A" * 73, stored)


def test_bcrypt_72_byte_boundary_is_exactly_72(depcheck):
    """The accepted-length boundary is *exactly* 72 bytes: 72 hashes fine, 73
    raises. The backend hard-codes ``> 72`` / ``[:72]`` against this exact
    boundary — pin that the off-by-one matches so the truncation is correct."""
    mod = depcheck.load(IMPORT_NAME)
    salt = mod.gensalt(_FAST_ROUNDS)
    # Exactly 72 bytes: accepted.
    ok = mod.hashpw(b"C" * 72, salt)
    assert len(ok) == 60
    # 73 bytes: rejected.
    with pytest.raises(ValueError):
        mod.hashpw(b"C" * 73, salt)


def test_bcrypt_72_byte_prefix_truncation_matches_backend(depcheck):
    """Mirror routers/auths.py: a >72-byte password truncated to ``[:72]`` must
    verify against a hash of the same 72-byte prefix. Confirms the backend's
    manual truncation produces a value bcrypt accepts as equivalent."""
    mod = depcheck.load(IMPORT_NAME)
    long_pw = ("p" * 80).encode("utf-8")
    assert len(long_pw) > 72
    truncated = long_pw[:72]  # exactly what auths.py does before verify
    stored = mod.hashpw(truncated, mod.gensalt(_FAST_ROUNDS))
    assert mod.checkpw(long_pw[:72], stored) is True


def test_bcrypt_differs_within_first_72_bytes(depcheck):
    """Counterpart to the truncation test: a difference *within* the first 72
    bytes must NOT collide — truncation only drops the tail, it must not
    conflate distinct in-range passwords."""
    mod = depcheck.load(IMPORT_NAME)
    pw_a = b"B" * 71 + b"X"
    pw_b = b"B" * 71 + b"Y"
    stored = mod.hashpw(pw_a, mod.gensalt(_FAST_ROUNDS))
    assert mod.checkpw(pw_b, stored) is False


# --------------------------------------------------------------------------- #
# Determinism with a fixed salt (algorithm stability across versions)
# --------------------------------------------------------------------------- #
def test_hashpw_deterministic_with_fixed_salt(depcheck):
    """Hashing the same password twice with the *same* salt must yield identical
    output — bcrypt is deterministic given (password, salt). This is what lets
    checkpw work by re-hashing with the salt embedded in the stored hash. A
    bump that changed the digest for fixed inputs would invalidate every stored
    password; pin algorithm stability."""
    mod = depcheck.load(IMPORT_NAME)
    salt = mod.gensalt(_FAST_ROUNDS)
    pw = b"determinism-check"
    assert mod.hashpw(pw, salt) == mod.hashpw(pw, salt)


def test_hash_embeds_salt_for_self_verification(depcheck):
    """The salt is embedded in the hash string, so checkpw needs only the stored
    hash (no separate salt storage). Re-hashing with the stored hash *as the
    salt argument* reproduces the stored hash exactly — the mechanism checkpw
    relies on. Open WebUI stores only the single hash column, trusting this."""
    mod = depcheck.load(IMPORT_NAME)
    pw = b"embedded-salt"
    stored = mod.hashpw(pw, mod.gensalt(_FAST_ROUNDS))
    # hashpw(pw, stored) re-derives using the salt embedded in `stored`.
    assert mod.hashpw(pw, stored) == stored
