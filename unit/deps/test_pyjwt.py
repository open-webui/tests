"""Dependency contract: PyJWT (import name ``jwt``).

PyJWT is the JSON Web Token library that underpins Open WebUI's
authentication. The backend uses it to:

  - mint and verify session tokens (``utils/auth.py``:
    ``jwt.encode(payload, SESSION_SECRET, algorithm="HS256")`` /
    ``jwt.decode(token, SESSION_SECRET, algorithms=["HS256"])``);
  - mint signed user-info headers forwarded to upstream backends
    (``utils/headers.py``: ``jwt.encode(..., algorithm="HS256")``);
  - validate OIDC back-channel logout tokens (``utils/oauth.py``):
    peek at unverified claims with ``options={"verify_signature": False}``,
    fetch the signing key via ``jwt.PyJWKClient(...)`` /
    ``get_signing_key_from_jwt(...)``, then ``jwt.decode`` with
    ``algorithms=[RS*/ES*]``, ``audience=``, ``issuer=`` and
    ``options={"require": [...]}``, catching ``jwt.InvalidTokenError``.

This is security-critical code: a PyJWT bump that renames a symbol,
changes a keyword argument, or alters which exception a bad/expired token
raises would silently weaken or break auth. PyJWT was recently bumped
(2.11 -> 2.13), so this module pins the exact slice of the API the
backend relies on plus the behavioural guarantees (roundtrip, expiry,
bad-signature, claim validation), all offline. If any contract breaks,
these tests fail loudly instead of letting an AttributeError or a
swallowed exception surface at a login endpoint.

Exemplar for the unit/deps/ pattern: symbol-existence checks (API
surface) + offline behavioural contracts (no network). Uses the
``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import datetime
import inspect
import warnings

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "jwt"
DIST_NAME = "PyJWT"

# The algorithm Open WebUI signs/verifies its own tokens with.
HS_ALG = "HS256"
# A 64-byte key: comfortably above PyJWT's 32-byte HMAC minimum so newer
# versions don't emit InsecureKeyLengthWarning into the test output.
SECRET = "x" * 64
OTHER_SECRET = "y" * 64

# Asymmetric algorithms the OIDC back-channel-logout path passes to decode().
OIDC_ALGORITHMS = ["RS256", "RS384", "RS512", "ES256", "ES384", "ES512"]

# Top-level symbols the backend references directly on the `jwt` module.
USED_TOP_LEVEL_SYMBOLS = [
    "encode",
    "decode",
    "PyJWKClient",
    # Exception classes the code catches at the top level (e.g.
    # `except pyjwt.InvalidTokenError`).
    "InvalidTokenError",
    "ExpiredSignatureError",
    "InvalidSignatureError",
    "DecodeError",
    "PyJWTError",
    # Exceptions raised by the claim-validation options the code uses
    # (audience=/issuer=/options={"require": [...]}).
    "InvalidAudienceError",
    "InvalidIssuerError",
    "MissingRequiredClaimError",
    "ImmatureSignatureError",
    # Submodule the exceptions also live under.
    "exceptions",
]

# The same exception classes must also be reachable under jwt.exceptions.*.
USED_EXCEPTION_SYMBOLS = [
    "exceptions.PyJWTError",
    "exceptions.InvalidTokenError",
    "exceptions.ExpiredSignatureError",
    "exceptions.InvalidSignatureError",
    "exceptions.DecodeError",
    "exceptions.InvalidAudienceError",
    "exceptions.InvalidIssuerError",
    "exceptions.MissingRequiredClaimError",
    "exceptions.ImmatureSignatureError",
]


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _hs_token(mod, payload: dict, key: str = SECRET) -> str:
    """Mint an HS256 token the way utils/auth.py does, warning-free."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mod.encode(payload, key, algorithm=HS_ALG)


def _hs_decode(mod, token: str, key: str = SECRET, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mod.decode(token, key, algorithms=[HS_ALG], **kwargs)


# --------------------------------------------------------------------------- #
# Import / version
# --------------------------------------------------------------------------- #


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "jwt"


def test_version_reported(depcheck):
    """Sanity: the installed distribution version is resolvable (so bump
    tooling and this suite agree on what PyJWT is under test)."""
    assert depcheck.dist_version(DIST_NAME) is not None


# --------------------------------------------------------------------------- #
# Symbol existence (API surface)
# --------------------------------------------------------------------------- #


def test_used_top_level_symbols_exist(depcheck):
    """Every `jwt.<symbol>` the codebase references must still exist."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_TOP_LEVEL_SYMBOLS)


def test_exception_module_symbols_exist(depcheck):
    """The exception classes the code catches must also resolve under
    jwt.exceptions.* (the canonical location they're defined in)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_EXCEPTION_SYMBOLS)


def test_encode_decode_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "encode")
    depcheck.assert_callable(mod, "decode")


def test_pyjwkclient_is_class(depcheck):
    """oauth.py does `pyjwt.PyJWKClient(jwks_uri)` then calls
    `.get_signing_key_from_jwt(token)` on the instance."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "PyJWKClient")
    assert inspect.isclass(mod.PyJWKClient)
    names = set(dir(mod.PyJWKClient))
    for meth in ("get_signing_key_from_jwt", "get_signing_key"):
        assert meth in names, f"PyJWKClient.{meth} missing"
        assert callable(getattr(mod.PyJWKClient, meth))


def test_top_level_and_module_exceptions_are_same_object(depcheck):
    """jwt.InvalidTokenError and jwt.exceptions.InvalidTokenError must be the
    identical class, so `except jwt.X` catches what jwt.exceptions.X raises."""
    mod = depcheck.load(IMPORT_NAME)
    for name in (
        "PyJWTError",
        "InvalidTokenError",
        "ExpiredSignatureError",
        "InvalidSignatureError",
        "DecodeError",
        "InvalidAudienceError",
        "InvalidIssuerError",
        "MissingRequiredClaimError",
    ):
        assert getattr(mod, name) is getattr(mod.exceptions, name), (
            f"jwt.{name} is not jwt.exceptions.{name}"
        )


# --------------------------------------------------------------------------- #
# Signatures (the kwargs the backend passes)
# --------------------------------------------------------------------------- #


def test_encode_signature_supports_our_kwargs(depcheck):
    """auth.py/headers.py call jwt.encode(payload, key, algorithm=...).
    Pin that `payload`, `key` and `algorithm` remain accepted."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.encode, ["payload", "key", "algorithm"])


def test_decode_signature_supports_our_kwargs(depcheck):
    """The backend calls jwt.decode with key, algorithms=, options=,
    audience=, issuer= and (in tests of the leeway contract) leeway=.
    All of those parameter names must remain on the signature."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.decode,
        ["key", "algorithms", "options", "audience", "issuer", "leeway"],
    )


def test_decode_first_positional_is_the_token(depcheck):
    """The code passes the token positionally as the first argument. Pin that
    the first parameter still exists and isn't keyword-only."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.decode)
    params = list(sig.parameters.values())
    assert params, "jwt.decode has no parameters"
    first = params[0]
    assert first.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ), f"jwt.decode first parameter {first.name!r} is no longer positional"


def test_encode_algorithm_defaults_to_hs256(depcheck):
    """headers.py and auth.py both pass algorithm explicitly, but pin the
    documented default so a change in PyJWT's default signing alg is noticed."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.encode)
    alg = sig.parameters.get("algorithm")
    if alg is None or alg.default is inspect.Parameter.empty:
        pytest.skip("jwt.encode.algorithm has no introspectable default")
    assert alg.default == "HS256"


# --------------------------------------------------------------------------- #
# Behavioural: HS256 encode/decode roundtrip (utils/auth.py core contract)
# --------------------------------------------------------------------------- #


def test_encode_returns_str(depcheck):
    """create_token returns the encoded JWT directly; modern PyJWT returns a
    str (not bytes as in 1.x). Pin that so the token is JSON/header-safe."""
    mod = depcheck.load(IMPORT_NAME)
    token = _hs_token(mod, {"id": "user-1"})
    assert isinstance(token, str)
    assert token.count(".") == 2  # header.payload.signature


def test_hs256_roundtrip_preserves_claims(depcheck):
    """encode then decode with the same secret must return the payload."""
    mod = depcheck.load(IMPORT_NAME)
    payload = {"id": "user-1", "role": "admin", "jti": "abc-123"}
    token = _hs_token(mod, payload)
    decoded = _hs_decode(mod, token)
    for k, v in payload.items():
        assert decoded[k] == v


def test_hs256_roundtrip_with_exp_and_iat(depcheck):
    """auth.create_token sets exp (datetime) and iat (datetime); decode must
    accept the datetime-encoded numeric claims and validate a future exp."""
    mod = depcheck.load(IMPORT_NAME)
    payload = {
        "id": "user-1",
        "iat": _now(),
        "exp": _now() + datetime.timedelta(hours=1),
    }
    token = _hs_token(mod, payload)
    decoded = _hs_decode(mod, token)
    assert decoded["id"] == "user-1"
    # exp/iat come back as integer POSIX timestamps.
    assert isinstance(decoded["exp"], int)
    assert isinstance(decoded["iat"], int)


def test_decode_returns_dict(depcheck):
    """decode_token returns the decoded claims dict to callers."""
    mod = depcheck.load(IMPORT_NAME)
    token = _hs_token(mod, {"id": "user-1"})
    decoded = _hs_decode(mod, token)
    assert isinstance(decoded, dict)


# --------------------------------------------------------------------------- #
# Behavioural: failure modes the backend relies on
# --------------------------------------------------------------------------- #


def test_expired_token_raises_expired_signature_error(depcheck):
    """A token whose exp is in the past must raise ExpiredSignatureError.
    decode_token swallows it via `except Exception` -> returns None, so the
    *type* matters less there, but the OIDC path catches InvalidTokenError and
    ExpiredSignatureError must remain a subclass for that to keep working."""
    mod = depcheck.load(IMPORT_NAME)
    token = _hs_token(mod, {"exp": _now() - datetime.timedelta(hours=1)})
    with pytest.raises(mod.ExpiredSignatureError):
        _hs_decode(mod, token)


def test_expired_is_invalid_token_error(depcheck):
    """ExpiredSignatureError must subclass InvalidTokenError so the OIDC
    handler's `except pyjwt.InvalidTokenError` catches expired logout tokens."""
    mod = depcheck.load(IMPORT_NAME)
    token = _hs_token(mod, {"exp": _now() - datetime.timedelta(hours=1)})
    with pytest.raises(mod.InvalidTokenError):
        _hs_decode(mod, token)


def test_wrong_secret_raises_invalid_signature_error(depcheck):
    """A token verified against the wrong secret must raise
    InvalidSignatureError (this is the core auth guarantee: forged/tampered
    tokens are rejected)."""
    mod = depcheck.load(IMPORT_NAME)
    token = _hs_token(mod, {"id": "user-1"})
    with pytest.raises(mod.InvalidSignatureError):
        _hs_decode(mod, token, key=OTHER_SECRET)


def test_invalid_signature_is_invalid_token_error(depcheck):
    """InvalidSignatureError must be catchable as InvalidTokenError."""
    mod = depcheck.load(IMPORT_NAME)
    token = _hs_token(mod, {"id": "user-1"})
    with pytest.raises(mod.InvalidTokenError):
        _hs_decode(mod, token, key=OTHER_SECRET)


def test_garbage_token_raises_decode_error(depcheck):
    """A structurally malformed token must raise DecodeError (a subclass of
    InvalidTokenError) — mirrors the OIDC `cannot decode logout_token` path."""
    mod = depcheck.load(IMPORT_NAME)
    with pytest.raises(mod.DecodeError):
        _hs_decode(mod, "this.is.not-a-jwt")


def test_garbage_token_is_invalid_token_error(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    with pytest.raises(mod.InvalidTokenError):
        _hs_decode(mod, "not-even-three-segments")


def test_decode_token_swallow_pattern(depcheck):
    """decode_token does `try: jwt.decode(...) except Exception: return None`.
    Replicate it end to end: a tampered token yields None, never a leaked
    exception or a partially-trusted dict."""
    mod = depcheck.load(IMPORT_NAME)

    def decode_token(token: str):
        try:
            return _hs_decode(mod, token)
        except Exception:
            return None

    good = _hs_token(mod, {"id": "user-1"})
    assert decode_token(good)["id"] == "user-1"
    assert decode_token(good + "tamper") is None
    assert decode_token("garbage") is None


# --------------------------------------------------------------------------- #
# Behavioural: decode options the backend uses
# --------------------------------------------------------------------------- #


def test_verify_signature_false_skips_signature_check(depcheck):
    """oauth.py peeks at unverified claims via
    jwt.decode(token, options={"verify_signature": False}) — no key, no
    algorithms. It must return the payload without raising."""
    mod = depcheck.load(IMPORT_NAME)
    token = _hs_token(mod, {"iss": "https://issuer.example", "sub": "u1"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        claims = mod.decode(token, options={"verify_signature": False})
    assert claims["iss"] == "https://issuer.example"
    assert claims["sub"] == "u1"


def test_verify_signature_false_ignores_bad_signature(depcheck):
    """With verify_signature off, even a tampered signature decodes — that's
    exactly why the OIDC code only uses it to read `iss`, then re-decodes with
    full verification."""
    mod = depcheck.load(IMPORT_NAME)
    token = _hs_token(mod, {"iss": "x"})
    tampered = token[:-4] + ("AAAA" if not token.endswith("AAAA") else "BBBB")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        claims = mod.decode(tampered, options={"verify_signature": False})
    assert claims["iss"] == "x"


def test_verify_signature_false_skips_exp(depcheck):
    """Reading unverified claims must not trip exp validation either, so the
    issuer peek works on an already-expired logout token."""
    mod = depcheck.load(IMPORT_NAME)
    token = _hs_token(mod, {"iss": "x", "exp": _now() - datetime.timedelta(days=1)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        claims = mod.decode(token, options={"verify_signature": False})
    assert claims["iss"] == "x"


def test_require_option_enforces_present_claims(depcheck):
    """oauth.py passes options={"require": ["iss","aud","iat","events"]}. When
    all required claims are present (and other checks pass) decode succeeds."""
    mod = depcheck.load(IMPORT_NAME)
    token = _hs_token(
        mod,
        {
            "iss": "https://issuer.example",
            "aud": "client-123",
            "iat": _now(),
            "events": {"http://schemas.openid.net/event/backchannel-logout": {}},
        },
    )
    decoded = _hs_decode(
        mod,
        token,
        audience="client-123",
        issuer="https://issuer.example",
        options={"require": ["iss", "aud", "iat", "events"]},
    )
    assert decoded["aud"] == "client-123"


def test_require_option_rejects_missing_claim(depcheck):
    """A token missing a required claim must raise MissingRequiredClaimError
    (a subclass of InvalidTokenError, which the OIDC handler catches)."""
    mod = depcheck.load(IMPORT_NAME)
    token = _hs_token(mod, {"iss": "https://issuer.example"})  # no events
    with pytest.raises(mod.MissingRequiredClaimError):
        _hs_decode(mod, token, options={"require": ["iss", "events"]})


def test_missing_required_claim_is_invalid_token_error(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    token = _hs_token(mod, {"iss": "x"})
    with pytest.raises(mod.InvalidTokenError):
        _hs_decode(mod, token, options={"require": ["events"]})


# --------------------------------------------------------------------------- #
# Behavioural: audience / issuer validation (OIDC logout token path)
# --------------------------------------------------------------------------- #


def test_audience_validation_accepts_matching_aud(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    token = _hs_token(mod, {"aud": "client-123"})
    decoded = _hs_decode(mod, token, audience="client-123")
    assert decoded["aud"] == "client-123"


def test_audience_validation_rejects_wrong_aud(depcheck):
    """A logout token whose aud doesn't match the configured client_id must
    raise InvalidAudienceError (subclass of InvalidTokenError)."""
    mod = depcheck.load(IMPORT_NAME)
    token = _hs_token(mod, {"aud": "someone-else"})
    with pytest.raises(mod.InvalidAudienceError):
        _hs_decode(mod, token, audience="client-123")
    with pytest.raises(mod.InvalidTokenError):
        _hs_decode(mod, token, audience="client-123")


def test_issuer_validation_accepts_matching_iss(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    token = _hs_token(mod, {"iss": "https://issuer.example"})
    decoded = _hs_decode(mod, token, issuer="https://issuer.example")
    assert decoded["iss"] == "https://issuer.example"


def test_issuer_validation_rejects_wrong_iss(depcheck):
    """A token from an unexpected issuer must raise InvalidIssuerError
    (subclass of InvalidTokenError)."""
    mod = depcheck.load(IMPORT_NAME)
    token = _hs_token(mod, {"iss": "https://evil.example"})
    with pytest.raises(mod.InvalidIssuerError):
        _hs_decode(mod, token, issuer="https://issuer.example")
    with pytest.raises(mod.InvalidTokenError):
        _hs_decode(mod, token, issuer="https://issuer.example")


# --------------------------------------------------------------------------- #
# Behavioural: leeway and algorithm pinning
# --------------------------------------------------------------------------- #


def test_leeway_tolerates_small_clock_skew(depcheck):
    """decode accepts leeway= (timedelta or seconds). A token that expired a
    few seconds ago must still validate within a generous leeway window."""
    mod = depcheck.load(IMPORT_NAME)
    token = _hs_token(mod, {"exp": _now() - datetime.timedelta(seconds=5)})
    decoded = _hs_decode(mod, token, leeway=datetime.timedelta(seconds=60))
    assert isinstance(decoded, dict)
    # And the same expired token must still be rejected with no leeway.
    with pytest.raises(mod.ExpiredSignatureError):
        _hs_decode(mod, token)


def test_algorithm_allowlist_rejects_other_alg(depcheck):
    """auth.py pins algorithms=["HS256"]. A token signed with a *different*
    HMAC alg must be rejected, proving the allowlist is enforced (defends
    against alg-substitution)."""
    mod = depcheck.load(IMPORT_NAME)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        token = mod.encode({"id": "u1"}, SECRET, algorithm="HS512")
    with pytest.raises(mod.InvalidTokenError):
        _hs_decode(mod, token)  # decode allows only HS256


def test_none_alg_token_rejected_under_hs256(depcheck):
    """A defining PyJWT security guarantee: an unsigned ('alg':'none') token
    must NOT be accepted when decoding with a key and an HMAC allowlist."""
    mod = depcheck.load(IMPORT_NAME)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unsigned = mod.encode({"id": "u1"}, key=None, algorithm="none")
    with pytest.raises(mod.InvalidTokenError):
        _hs_decode(mod, unsigned)


# --------------------------------------------------------------------------- #
# Exception hierarchy (so broad `except` handlers stay correct)
# --------------------------------------------------------------------------- #


def test_invalid_token_error_subclasses_pyjwt_error(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.InvalidTokenError, mod.PyJWTError)


def test_exception_subclass_relationships(depcheck):
    """Every specific exception the backend's options can trigger must remain a
    subclass of InvalidTokenError, because oauth.py's only specific handler is
    `except pyjwt.InvalidTokenError`. If any of these stops subclassing it, a
    bad logout token would escape as an unhandled 500 instead of a clean 400."""
    mod = depcheck.load(IMPORT_NAME)
    for name in (
        "ExpiredSignatureError",
        "InvalidSignatureError",
        "DecodeError",
        "InvalidAudienceError",
        "InvalidIssuerError",
        "MissingRequiredClaimError",
        "ImmatureSignatureError",
    ):
        exc = getattr(mod, name)
        assert issubclass(exc, mod.InvalidTokenError), (
            f"{name} no longer subclasses InvalidTokenError"
        )


def test_invalid_signature_subclasses_decode_error(depcheck):
    """InvalidSignatureError -> DecodeError -> InvalidTokenError chain pins the
    layering the `except DecodeError` (malformed/unsigned) path depends on."""
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.InvalidSignatureError, mod.DecodeError)
    assert issubclass(mod.DecodeError, mod.InvalidTokenError)
