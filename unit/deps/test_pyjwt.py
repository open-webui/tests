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
    read the ``kid`` with ``jwt.get_unverified_header(...)``, pick the signing
    key out of ``jwt.PyJWKSet.from_dict(jwks).keys`` by ``key_id`` and
    ``public_key_use``, then ``jwt.decode`` with ``algorithms=[RS*/ES*]``,
    ``audience=``, ``issuer=`` and ``options={"require": [...]}``, catching
    ``jwt.InvalidTokenError``.

This is security-critical code: a PyJWT bump that renames a symbol,
changes a keyword argument, or alters which exception a bad/expired token
raises would silently weaken or break auth. PyJWT was recently bumped
(2.11 -> 2.13), so this module pins the exact slice of the API the
backend relies on plus the behavioural guarantees (roundtrip, expiry,
bad-signature, claim validation), all offline. If any contract breaks,
these tests fail loudly instead of letting an AttributeError or a
swallowed exception surface at a login endpoint.

Which ``jwt.*`` names the backend uses, and the arguments it passes them,
are read from the backend with ``ast``, so the checks follow upstream: a
hand-kept list here still pinned ``PyJWKClient`` after the logout path moved
to ``PyJWKSet``, and a default the backend never relies on once had to be
repaired. Offline, no network. Uses the ``depcheck`` fixture from
unit/deps/conftest.py.

Discriminates: in a backend copy, a ``jwt`` name PyJWT lacks fails the
inventory and a keyword ``jwt.encode`` does not take fails the call check; a
PyJWT without ``get_unverified_header`` or whose ``PyJWKSet.from_dict``
refuses a JWKS (patched in process) fails the inventory and the logout-token
test.
"""

from __future__ import annotations

import ast
import datetime
import inspect
import warnings
from pathlib import Path
from typing import NamedTuple

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
LOGOUT_EVENT = "http://schemas.openid.net/event/backchannel-logout"


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


class JwtUse(NamedTuple):
    where: str
    name: str  # "PyJWKSet.from_dict" for `jwt.PyJWKSet.from_dict`
    call: ast.Call | None  # the call, when the name is called right there


def _dotted(node: ast.AST) -> list[str]:
    """`jwt.PyJWKSet.from_dict` as ["jwt", "PyJWKSet", "from_dict"]; [] for anything else."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.insert(0, node.attr)
        node = node.value
    return [node.id, *parts] if isinstance(node, ast.Name) and parts else []


def _backend_jwt_uses(backend: Path) -> list[JwtUse]:
    """Every `jwt.<...>` the backend writes, where `jwt` is PyJWT."""
    uses = []
    for path in sorted((backend / "open_webui").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "import jwt" not in source:
            continue
        tree = ast.parse(source)
        aliases = {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
            if alias.name == IMPORT_NAME
        }
        calls = {id(node.func): node for node in ast.walk(tree) if isinstance(node, ast.Call)}
        for node in ast.walk(tree):
            chain = _dotted(node)
            if chain and chain[0] in aliases:
                where = f"{path.relative_to(backend)}:{node.lineno}"
                uses.append(JwtUse(where, ".".join(chain[1:]), calls.get(id(node))))
    assert uses, f"nothing under {backend} uses `import jwt`; retarget this contract"
    return uses


def test_every_jwt_name_the_backend_uses_exists(depcheck, open_webui_backend):
    mod = depcheck.load(IMPORT_NAME)
    missing = sorted(
        f"{use.where} jwt.{use.name}"
        for use in _backend_jwt_uses(open_webui_backend)
        if not depcheck.has(mod, use.name)
    )
    assert not missing, f"PyJWT lacks names the backend uses: {missing}"


def test_every_jwt_call_the_backend_makes_binds(depcheck, open_webui_backend):
    """`jwt.encode(payload, key, algorithm=...)`, `jwt.decode(token, key, ...)` and the rest,
    bound the way the backend passes them, so what it passes by position is not pinned by name."""
    mod = depcheck.load(IMPORT_NAME)
    problems = []
    for use in _backend_jwt_uses(open_webui_backend):
        if use.call is None or not depcheck.has(mod, use.name):
            continue
        positional = [None for argument in use.call.args if not isinstance(argument, ast.Starred)]
        keywords = {keyword.arg: None for keyword in use.call.keywords if keyword.arg}
        try:
            inspect.signature(depcheck.resolve(mod, use.name)).bind_partial(*positional, **keywords)
        except TypeError as error:
            problems.append(f"{use.where} jwt.{use.name}(...): {error}")
        except ValueError:
            continue  # no introspectable signature (a builtin exception class)
    assert not problems, f"backend calls PyJWT no longer accepts: {problems}"


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


def _rsa_key():
    rsa = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.rsa")
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _logout_token(mod, signing_key, kid: str) -> str:
    claims = {
        "iss": "https://issuer.example",
        "aud": "client-123",
        "iat": _now(),
        "events": {LOGOUT_EVENT: {}},
    }
    return mod.encode(claims, signing_key, algorithm="RS256", headers={"kid": kid})


def _decode_logout_token(mod, token: str, jwks: dict) -> dict:
    """oauth.py's back-channel logout check, step by step."""
    kid = mod.get_unverified_header(token).get("kid")
    signing_key = next(
        key
        for key in mod.PyJWKSet.from_dict(jwks).keys
        if key.key_id == kid and key.public_key_use in ["sig", None]
    )
    return mod.decode(
        token,
        signing_key.key,
        algorithms=OIDC_ALGORITHMS,
        audience="client-123",
        issuer="https://issuer.example",
        options={"require": ["iss", "aud", "iat", "events"]},
    )


def test_a_logout_token_is_checked_against_the_jwks_key_its_kid_names(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    provider_key = _rsa_key()
    public = mod.algorithms.RSAAlgorithm.to_jwk(provider_key.public_key(), as_dict=True)
    jwks = {
        "keys": [
            {**public, "kid": "encryption", "use": "enc"},
            {**public, "kid": "signing", "use": "sig"},
        ]
    }

    decoded = _decode_logout_token(mod, _logout_token(mod, provider_key, "signing"), jwks)
    assert decoded["events"] == {LOGOUT_EVENT: {}}

    forged = _logout_token(mod, _rsa_key(), "signing")
    with pytest.raises(mod.InvalidTokenError):
        _decode_logout_token(mod, forged, jwks)


# --------------------------------------------------------------------------- #
# Behavioural: algorithm pinning
# --------------------------------------------------------------------------- #


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
