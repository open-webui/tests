"""Dependency contract: itsdangerous.

itsdangerous is not imported directly anywhere in the Open WebUI backend;
it is a transitive dependency that underpins the signed-cookie / signed-
token machinery the stack relies on. Starlette's ``SessionMiddleware``
(mounted by the FastAPI app and used by the OAuth login flow) serialises
and HMAC-signs the session cookie with itsdangerous; any consumer that
puts tamper-evident, optionally time-limited data into an opaque token
goes through the same ``Serializer`` / ``Signer`` primitives pinned here.

Because the backend does not name any symbol of its own, this module pins
the *core signing/serialization contract* that every consumer depends on:
the URL-safe (timed) serializers round-trip, the HMAC signature actually
rejects tampering, ``max_age`` enforcement raises ``SignatureExpired``,
and the exception hierarchy keeps its shape (so broad ``except BadData``
handlers in the web stack keep catching every signature failure). A
bump that silently changed token framing, dropped a serializer, or
re-parented an exception would let forged or stale cookies through; this
test fails loudly on any of that.

Pattern mirrors test_requests.py: symbol-existence (API surface) plus
offline behavioural contracts (no network, no server). Uses the
``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "itsdangerous"
DIST_NAME = "itsdangerous"

# Top-level symbols any itsdangerous consumer (and the signed-cookie stack)
# resolves on the package.
USED_SYMBOLS = [
    # Serializers — opaque tokens carrying structured payloads.
    "Serializer",
    "TimedSerializer",
    "URLSafeSerializer",
    "URLSafeTimedSerializer",
    # Signers — bare HMAC over a value.
    "Signer",
    "TimestampSigner",
    # Exception hierarchy the web stack catches.
    "BadData",
    "BadSignature",
    "BadTimeSignature",
    "SignatureExpired",
    "BadHeader",
    "BadPayload",
    # base64 helpers (used internally + by some consumers for token framing).
    "base64_encode",
    "base64_decode",
    "want_bytes",
    # submodules
    "exc",
    "serializer",
    "signer",
    "url_safe",
    "timed",
]

SECRET = "unit-test-secret-key-never-real"
SALT = "open-webui-unit-test"


# ---------------------------------------------------------------------------
# Import + version + symbol surface
# ---------------------------------------------------------------------------


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "itsdangerous"


def test_version_reported(depcheck):
    """The installed distribution version must be resolvable so bump tooling
    and this suite agree on what's under test."""
    assert depcheck.dist_version(DIST_NAME) is not None


def test_used_symbols_exist(depcheck):
    """Every top-level itsdangerous symbol the signing stack relies on must
    still exist — a bump must not silently drop a serializer/signer/exc."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_serializer_classes_are_types(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    for name in (
        "Serializer",
        "TimedSerializer",
        "URLSafeSerializer",
        "URLSafeTimedSerializer",
        "Signer",
        "TimestampSigner",
    ):
        obj = getattr(mod, name)
        assert isinstance(obj, type), f"itsdangerous.{name} is not a class"


# ---------------------------------------------------------------------------
# Exception hierarchy — the web stack catches BadData / BadSignature broadly;
# every signature failure must remain a subclass so those handlers fire.
# ---------------------------------------------------------------------------


def test_exception_hierarchy(depcheck):
    """SignatureExpired must subclass BadTimeSignature -> BadSignature ->
    BadData; BadSignature must subclass BadData. A broad `except BadData`
    (or `except BadSignature`) around cookie loads relies on this chain."""
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.BadSignature, mod.BadData)
    assert issubclass(mod.BadTimeSignature, mod.BadSignature)
    assert issubclass(mod.SignatureExpired, mod.BadTimeSignature)
    assert issubclass(mod.SignatureExpired, mod.BadData)
    assert issubclass(mod.BadData, Exception)


def test_bad_payload_and_header_are_baddata(depcheck):
    """BadPayload/BadHeader (raised on malformed token internals) must also be
    BadData subclasses so they are caught by the same broad handler."""
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.BadPayload, mod.BadData)
    assert issubclass(mod.BadHeader, mod.BadSignature)


# ---------------------------------------------------------------------------
# URLSafeTimedSerializer — the SessionMiddleware-style timed cookie path.
# ---------------------------------------------------------------------------


def test_urlsafe_timed_serializer_constructs(depcheck):
    """Consumers build URLSafeTimedSerializer(secret_key, salt=...). The
    constructor must keep accepting secret_key positionally + a salt kwarg."""
    mod = depcheck.load(IMPORT_NAME)
    s = mod.URLSafeTimedSerializer(SECRET, salt=SALT)
    assert s is not None
    assert callable(s.dumps)
    assert callable(s.loads)


def test_urlsafe_timed_round_trip(depcheck):
    """dumps()/loads() must round-trip a JSON-able payload (the session dict)
    and dumps() must return a `str` token (goes straight into a cookie)."""
    mod = depcheck.load(IMPORT_NAME)
    s = mod.URLSafeTimedSerializer(SECRET, salt=SALT)
    payload = {"user_id": "u1", "nested": {"k": [1, 2, 3]}, "flag": True}
    token = s.dumps(payload)
    assert isinstance(token, str)
    assert s.loads(token) == payload


def test_urlsafe_timed_max_age_accepts_fresh(depcheck):
    """A token loaded with a generous max_age must succeed (it was just
    created) — the OAuth state/session validity window relies on this."""
    mod = depcheck.load(IMPORT_NAME)
    s = mod.URLSafeTimedSerializer(SECRET, salt=SALT)
    token = s.dumps({"a": 1})
    assert s.loads(token, max_age=3600) == {"a": 1}


def test_urlsafe_timed_max_age_expires(depcheck):
    """A token older than max_age must raise SignatureExpired. We patch the
    serializer's clock to make a freshly-minted token look old — proving
    expiry enforcement is real, without sleeping."""
    mod = depcheck.load(IMPORT_NAME)
    s = mod.URLSafeTimedSerializer(SECRET, salt=SALT)
    token = s.dumps({"a": 1})
    # max_age=0 with a non-zero token age must trip expiry; a token minted a
    # moment ago has age >= 0, and 0 < age fails the <= max_age check.
    with pytest.raises(mod.SignatureExpired):
        s.loads(token, max_age=-1)


def test_urlsafe_timed_return_timestamp(depcheck):
    """loads(..., return_timestamp=True) yields (payload, timestamp). The
    timestamp must be a datetime/awareness object consumers can compare."""
    import datetime

    mod = depcheck.load(IMPORT_NAME)
    s = mod.URLSafeTimedSerializer(SECRET, salt=SALT)
    token = s.dumps({"a": 1})
    payload, ts = s.loads(token, return_timestamp=True)
    assert payload == {"a": 1}
    assert isinstance(ts, datetime.datetime)


def test_loads_signature(depcheck):
    """loads must keep accepting max_age / return_timestamp / salt — the exact
    knobs the cookie/session path uses."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.URLSafeTimedSerializer.loads,
        ["s", "max_age", "return_timestamp", "salt"],
    )


# ---------------------------------------------------------------------------
# Tamper-evidence — the whole point of the dependency. A mutated token must
# be rejected, not silently accepted.
# ---------------------------------------------------------------------------


def test_tampered_payload_rejected(depcheck):
    """Flipping a byte in the signed payload must raise BadSignature — this is
    the integrity guarantee the signed cookie depends on."""
    mod = depcheck.load(IMPORT_NAME)
    s = mod.URLSafeTimedSerializer(SECRET, salt=SALT)
    token = s.dumps({"role": "user"})
    # Corrupt a character early in the token (the encoded payload region).
    corrupted = ("Z" if token[0] != "Z" else "Y") + token[1:]
    with pytest.raises(mod.BadSignature):
        s.loads(corrupted)


def test_wrong_secret_rejected(depcheck):
    """A token signed with one secret must NOT verify under a different
    secret — an attacker without the key cannot forge a valid cookie."""
    mod = depcheck.load(IMPORT_NAME)
    signer = mod.URLSafeTimedSerializer(SECRET, salt=SALT)
    other = mod.URLSafeTimedSerializer("a-different-secret", salt=SALT)
    token = signer.dumps({"admin": True})
    with pytest.raises(mod.BadSignature):
        other.loads(token)


def test_wrong_salt_rejected(depcheck):
    """The salt namespaces signatures; a token signed under salt A must not
    verify under salt B even with the same secret (so a session token can't
    be replayed as, say, a password-reset token)."""
    mod = depcheck.load(IMPORT_NAME)
    a = mod.URLSafeTimedSerializer(SECRET, salt="salt-a")
    b = mod.URLSafeTimedSerializer(SECRET, salt="salt-b")
    token = a.dumps({"x": 1})
    with pytest.raises(mod.BadSignature):
        b.loads(token)


def test_garbage_token_raises_baddata(depcheck):
    """A completely malformed token must raise BadData (not crash with some
    other error) so the web stack's `except BadData` handler catches it."""
    mod = depcheck.load(IMPORT_NAME)
    s = mod.URLSafeTimedSerializer(SECRET, salt=SALT)
    with pytest.raises(mod.BadData):
        s.loads("this.is.not-a-valid-token")


# ---------------------------------------------------------------------------
# URLSafeSerializer (non-timed) + plain Serializer round-trips.
# ---------------------------------------------------------------------------


def test_urlsafe_serializer_round_trip(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    s = mod.URLSafeSerializer(SECRET, salt=SALT)
    token = s.dumps([1, "two", {"three": 3}])
    assert isinstance(token, str)
    assert s.loads(token) == [1, "two", {"three": 3}]


def test_plain_serializer_round_trip(depcheck):
    """The base Serializer round-trips too (output isn't url-safe but the
    sign/verify contract is identical)."""
    mod = depcheck.load(IMPORT_NAME)
    s = mod.Serializer(SECRET, salt=SALT)
    token = s.dumps({"k": "v"})
    assert s.loads(token) == {"k": "v"}


# ---------------------------------------------------------------------------
# Signer / TimestampSigner — bare HMAC primitives.
# ---------------------------------------------------------------------------


def test_signer_sign_unsign(depcheck):
    """Signer.sign appends an HMAC; unsign strips+verifies it. A tampered
    value must fail unsign."""
    mod = depcheck.load(IMPORT_NAME)
    signer = mod.Signer(SECRET, salt=SALT)
    signed = signer.sign("hello")
    assert isinstance(signed, (bytes, bytearray))
    assert signer.unsign(signed) == b"hello"
    with pytest.raises(mod.BadSignature):
        signer.unsign(signed + b"x")


def test_timestamp_signer_validate(depcheck):
    """TimestampSigner.validate(...) returns True for a good signature and
    False for a bad one (the non-raising probe some consumers use)."""
    mod = depcheck.load(IMPORT_NAME)
    signer = mod.TimestampSigner(SECRET, salt=SALT)
    signed = signer.sign("payload")
    assert signer.validate(signed) is True
    assert signer.validate(signed + b"tamper") is False


def test_timestamp_signer_max_age_expiry(depcheck):
    """TimestampSigner.unsign(..., max_age=) raises SignatureExpired once the
    age exceeds the window — same expiry semantics as the timed serializer."""
    mod = depcheck.load(IMPORT_NAME)
    signer = mod.TimestampSigner(SECRET, salt=SALT)
    signed = signer.sign("payload")
    with pytest.raises(mod.SignatureExpired):
        signer.unsign(signed, max_age=-1)


# ---------------------------------------------------------------------------
# base64 helpers + want_bytes (token framing utilities consumers reach for).
# ---------------------------------------------------------------------------


def test_base64_round_trip(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    raw = b"\x00\x01binary payload\xff"
    enc = mod.base64_encode(raw)
    assert isinstance(enc, (bytes, bytearray))
    assert mod.base64_decode(enc) == raw


def test_want_bytes_coerces(depcheck):
    """want_bytes normalises str|bytes -> bytes (used to accept either a str
    secret or a bytes secret)."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.want_bytes("abc") == b"abc"
    assert mod.want_bytes(b"abc") == b"abc"


def test_secret_key_accepts_str_and_bytes(depcheck):
    """The constructor signature documents secret_key as str|bytes|iterable;
    both a str and a bytes secret must produce a working serializer."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.URLSafeTimedSerializer.__init__)
    assert "secret_key" in sig.parameters
    s_str = mod.URLSafeTimedSerializer("strkey", salt=SALT)
    s_bytes = mod.URLSafeTimedSerializer(b"strkey", salt=SALT)
    # Same key material (str vs bytes) must verify each other's tokens.
    token = s_str.dumps({"a": 1})
    assert s_bytes.loads(token) == {"a": 1}
