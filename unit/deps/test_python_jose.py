"""Dependency contract: python-jose (import name ``jose``).

python-jose is a JOSE (JWT/JWS/JWE/JWK) implementation. It is a pinned,
first-class dependency of the Open WebUI backend (``python-jose==3.5.0``
in both ``backend/requirements.txt`` and ``pyproject.toml``), but as of
this writing the backend does **not** import ``jose`` anywhere in
``open_webui/`` — the only ``jose`` token in the source is
``from authlib.jose.errors import BadSignatureError`` in
``utils/oauth.py``, which is *authlib's* ``jose`` subpackage, not this
package. (PyJWT is what actually mints/verifies Open WebUI's own session
tokens — see ``test_pyjwt.py``.)

Because it is nonetheless a declared, version-pinned dependency that is
security-critical wherever it *is* used (token signing/verification for
SSO/JWT flows), this module pins the core public surface and the
behavioural guarantees that matter for any JOSE library: a sign/verify
roundtrip, and rejection of tampered tokens, wrong keys, expired tokens,
bad audience/issuer, and algorithm-substitution / ``alg: none`` attacks.
A python-jose bump that renamed a symbol, changed a keyword argument, or
weakened a rejection path would fail these tests loudly rather than
silently shipping a weaker JWT verifier into the dependency tree.

All checks are OFFLINE and deterministic: every key is generated in-process
and no network or external key server is touched.

Exemplar for the unit/deps/ pattern: symbol-existence checks (API
surface) + offline behavioural contracts. Uses the ``depcheck`` fixture
from unit/deps/conftest.py.
"""

from __future__ import annotations

import datetime
import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "jose"
DIST_NAME = "python-jose"

HS_ALG = "HS256"
# 64-byte HMAC secret (well above any minimum-key-length guard).
SECRET = "x" * 64
OTHER_SECRET = "y" * 64

# Top-level symbols re-exported on the `jose` package.
USED_TOP_LEVEL_SYMBOLS = [
    "JOSEError",
    "JWSError",
    "JWTError",
    "ExpiredSignatureError",
    "exceptions",
]

# Submodules a JOSE consumer reaches for.
USED_SUBMODULES = ["jwt", "jws", "jwk", "exceptions", "constants"]

# The jwt submodule API surface.
USED_JWT_SYMBOLS = [
    "encode",
    "decode",
    "get_unverified_claims",
    "get_unverified_header",
    "get_unverified_headers",
    "ALGORITHMS",
    "JWTError",
    "JWTClaimsError",
    "ExpiredSignatureError",
    "JWSError",
]

# The jws submodule API surface.
USED_JWS_SYMBOLS = [
    "sign",
    "verify",
    "get_unverified_claims",
    "get_unverified_header",
    "get_unverified_headers",
    "JWSError",
    "JWSSignatureError",
]

# The jwk submodule API surface (key construction).
USED_JWK_SYMBOLS = [
    "construct",
    "get_key",
    "Key",
    "HMACKey",
    "RSAKey",
    "ECKey",
    "JWKError",
]

# Exceptions that must exist under jose.exceptions.
USED_EXCEPTION_SYMBOLS = [
    "exceptions.JOSEError",
    "exceptions.JWTError",
    "exceptions.JWSError",
    "exceptions.JWKError",
    "exceptions.ExpiredSignatureError",
    "exceptions.JWTClaimsError",
    "exceptions.JWSSignatureError",
]

# Algorithm-constant attributes referenced through jose.constants.ALGORITHMS.
USED_ALGORITHM_CONSTANTS = ["HS256", "HS384", "HS512", "RS256", "ES256"]


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _jwt(depcheck):
    return depcheck.resolve(depcheck.load(IMPORT_NAME), "jwt")


def _jws(depcheck):
    return depcheck.resolve(depcheck.load(IMPORT_NAME), "jws")


def _jwk(depcheck):
    return depcheck.resolve(depcheck.load(IMPORT_NAME), "jwk")


# --------------------------------------------------------------------------- #
# Import / version / not-directly-imported documentation
# --------------------------------------------------------------------------- #


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "jose"


def test_version_reported(depcheck):
    """Sanity: the pinned distribution version is resolvable."""
    assert depcheck.dist_version(DIST_NAME) is not None


def test_submodules_importable(depcheck):
    """jose.jwt / jose.jws / jose.jwk / jose.exceptions / jose.constants must
    all import (some are lazy submodules, not attributes of the package)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SUBMODULES)


# --------------------------------------------------------------------------- #
# Symbol existence (API surface)
# --------------------------------------------------------------------------- #


def test_top_level_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_TOP_LEVEL_SYMBOLS)


def test_jwt_symbols_exist(depcheck):
    depcheck.assert_symbols(_jwt(depcheck), USED_JWT_SYMBOLS)


def test_jws_symbols_exist(depcheck):
    depcheck.assert_symbols(_jws(depcheck), USED_JWS_SYMBOLS)


def test_jwk_symbols_exist(depcheck):
    depcheck.assert_symbols(_jwk(depcheck), USED_JWK_SYMBOLS)


def test_exception_module_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_EXCEPTION_SYMBOLS)


def test_algorithm_constants_exist(depcheck):
    """jose.constants.ALGORITHMS exposes the alg-name constants used to pass
    algorithm= / algorithms= to encode/decode."""
    algs = depcheck.resolve(depcheck.load(IMPORT_NAME), "constants.ALGORITHMS")
    for name in USED_ALGORITHM_CONSTANTS:
        assert hasattr(algs, name), f"jose.constants.ALGORITHMS.{name} missing"
        assert getattr(algs, name) == name


def test_jwt_encode_decode_callable(depcheck):
    jwt = _jwt(depcheck)
    depcheck.assert_callable(jwt, "encode")
    depcheck.assert_callable(jwt, "decode")


def test_jws_sign_verify_callable(depcheck):
    jws = _jws(depcheck)
    depcheck.assert_callable(jws, "sign")
    depcheck.assert_callable(jws, "verify")


# --------------------------------------------------------------------------- #
# Signatures (the kwargs a JOSE consumer passes)
# --------------------------------------------------------------------------- #


def test_jwt_encode_signature(depcheck):
    """jwt.encode(claims, key, algorithm=...). Pin those param names."""
    jwt = _jwt(depcheck)
    depcheck.assert_params(jwt.encode, ["claims", "key", "algorithm"])


def test_jwt_decode_signature(depcheck):
    """jwt.decode(token, key, algorithms=, options=, audience=, issuer=)."""
    jwt = _jwt(depcheck)
    depcheck.assert_params(
        jwt.decode,
        ["token", "key", "algorithms", "options", "audience", "issuer"],
    )


def test_jwt_encode_algorithm_defaults_to_hs256(depcheck):
    """Pin the documented default signing algorithm so a change is noticed."""
    jwt = _jwt(depcheck)
    sig = inspect.signature(jwt.encode)
    alg = sig.parameters.get("algorithm")
    if alg is None or alg.default is inspect.Parameter.empty:
        pytest.skip("jwt.encode.algorithm has no introspectable default")
    assert alg.default == "HS256"


def test_jws_sign_signature(depcheck):
    jws = _jws(depcheck)
    depcheck.assert_params(jws.sign, ["payload", "key", "algorithm"])


def test_jws_verify_signature(depcheck):
    jws = _jws(depcheck)
    depcheck.assert_params(jws.verify, ["token", "key", "algorithms"])


# --------------------------------------------------------------------------- #
# Behavioural: HS256 JWT encode/decode roundtrip
# --------------------------------------------------------------------------- #


def test_jwt_encode_returns_str(depcheck):
    """jose.jwt.encode returns a compact serialization str with 3 segments."""
    jwt = _jwt(depcheck)
    token = jwt.encode({"sub": "user-1"}, SECRET, algorithm=HS_ALG)
    assert isinstance(token, str)
    assert token.count(".") == 2  # header.payload.signature


def test_jwt_hs256_roundtrip_preserves_claims(depcheck):
    jwt = _jwt(depcheck)
    payload = {"sub": "user-1", "role": "admin", "jti": "abc-123"}
    token = jwt.encode(payload, SECRET, algorithm=HS_ALG)
    decoded = jwt.decode(token, SECRET, algorithms=[HS_ALG])
    assert isinstance(decoded, dict)
    for k, v in payload.items():
        assert decoded[k] == v


def test_jwt_roundtrip_with_exp_and_iat(depcheck):
    jwt = _jwt(depcheck)
    payload = {
        "sub": "user-1",
        "iat": _now(),
        "exp": _now() + datetime.timedelta(hours=1),
    }
    token = jwt.encode(payload, SECRET, algorithm=HS_ALG)
    decoded = jwt.decode(token, SECRET, algorithms=[HS_ALG])
    assert decoded["sub"] == "user-1"
    assert isinstance(decoded["exp"], int)
    assert isinstance(decoded["iat"], int)


def test_get_unverified_claims_reads_without_key(depcheck):
    """get_unverified_claims peeks at the payload without a key/verification."""
    jwt = _jwt(depcheck)
    token = jwt.encode({"iss": "https://issuer.example", "sub": "u1"}, SECRET)
    claims = jwt.get_unverified_claims(token)
    assert claims["iss"] == "https://issuer.example"
    assert claims["sub"] == "u1"


def test_get_unverified_header_reads_alg(depcheck):
    jwt = _jwt(depcheck)
    token = jwt.encode({"sub": "u1"}, SECRET, algorithm=HS_ALG)
    header = jwt.get_unverified_header(token)
    assert header.get("alg") == HS_ALG


# --------------------------------------------------------------------------- #
# Behavioural: rejection paths (the security guarantees)
# --------------------------------------------------------------------------- #


def test_wrong_secret_rejected(depcheck):
    """A token verified with the wrong secret must raise JWTError (the core
    auth guarantee: forged/tampered tokens are rejected)."""
    jwt = _jwt(depcheck)
    token = jwt.encode({"sub": "user-1"}, SECRET, algorithm=HS_ALG)
    with pytest.raises(jwt.JWTError):
        jwt.decode(token, OTHER_SECRET, algorithms=[HS_ALG])


def test_tampered_token_rejected(depcheck):
    """Mutating the signature segment must fail verification."""
    jwt = _jwt(depcheck)
    token = jwt.encode({"sub": "user-1"}, SECRET, algorithm=HS_ALG)
    tampered = token[:-4] + ("AAAA" if not token.endswith("AAAA") else "BBBB")
    with pytest.raises(jwt.JWTError):
        jwt.decode(tampered, SECRET, algorithms=[HS_ALG])


def test_garbage_token_rejected(depcheck):
    """A structurally malformed token must raise JWTError."""
    jwt = _jwt(depcheck)
    with pytest.raises(jwt.JWTError):
        jwt.decode("this.is.not-a-jwt", SECRET, algorithms=[HS_ALG])


def test_expired_token_raises_expired_signature_error(depcheck):
    """An exp in the past must raise ExpiredSignatureError."""
    jwt = _jwt(depcheck)
    token = jwt.encode(
        {"sub": "u1", "exp": _now() - datetime.timedelta(hours=1)},
        SECRET,
        algorithm=HS_ALG,
    )
    with pytest.raises(jwt.ExpiredSignatureError):
        jwt.decode(token, SECRET, algorithms=[HS_ALG])


def test_expired_is_jwt_error(depcheck):
    """ExpiredSignatureError must subclass JWTError so a broad
    `except JWTError` catches expired tokens."""
    jwt = _jwt(depcheck)
    token = jwt.encode(
        {"sub": "u1", "exp": _now() - datetime.timedelta(hours=1)},
        SECRET,
        algorithm=HS_ALG,
    )
    with pytest.raises(jwt.JWTError):
        jwt.decode(token, SECRET, algorithms=[HS_ALG])


# --------------------------------------------------------------------------- #
# Behavioural: audience / issuer validation
# --------------------------------------------------------------------------- #


def test_audience_validation_accepts_matching_aud(depcheck):
    jwt = _jwt(depcheck)
    token = jwt.encode({"aud": "client-123"}, SECRET, algorithm=HS_ALG)
    decoded = jwt.decode(token, SECRET, algorithms=[HS_ALG], audience="client-123")
    assert decoded["aud"] == "client-123"


def test_audience_validation_rejects_wrong_aud(depcheck):
    """A token whose aud doesn't match must raise JWTClaimsError (a JWTError)."""
    jwt = _jwt(depcheck)
    token = jwt.encode({"aud": "someone-else"}, SECRET, algorithm=HS_ALG)
    with pytest.raises(jwt.JWTClaimsError):
        jwt.decode(token, SECRET, algorithms=[HS_ALG], audience="client-123")
    with pytest.raises(jwt.JWTError):
        jwt.decode(token, SECRET, algorithms=[HS_ALG], audience="client-123")


def test_issuer_validation_rejects_wrong_iss(depcheck):
    jwt = _jwt(depcheck)
    token = jwt.encode({"iss": "https://evil.example"}, SECRET, algorithm=HS_ALG)
    with pytest.raises(jwt.JWTClaimsError):
        jwt.decode(
            token,
            SECRET,
            algorithms=[HS_ALG],
            issuer="https://issuer.example",
        )


# --------------------------------------------------------------------------- #
# Behavioural: algorithm allowlist / alg-substitution defenses
# --------------------------------------------------------------------------- #


def test_algorithm_allowlist_rejects_other_alg(depcheck):
    """A token signed HS512 must be rejected when decode allows only HS256
    (proves the allowlist is enforced against alg substitution)."""
    jwt = _jwt(depcheck)
    token = jwt.encode({"sub": "u1"}, SECRET, algorithm="HS512")
    with pytest.raises(jwt.JWTError):
        jwt.decode(token, SECRET, algorithms=[HS_ALG])


def test_none_alg_rejected_under_hmac_allowlist(depcheck):
    """A defining JOSE security guarantee: an unsigned ('alg':'none') token
    must NOT verify when decoding with an HMAC algorithm allowlist."""
    jwt = _jwt(depcheck)
    # python-jose rejects 'none' at encode time without an explicit opt-in,
    # so craft the unsigned token directly from header+payload, empty sig.
    import base64
    import json

    def b64(d: bytes) -> str:
        return base64.urlsafe_b64encode(d).rstrip(b"=").decode("ascii")

    header = b64(json.dumps({"alg": "none", "typ": "JWT"}).encode())
    payload = b64(json.dumps({"sub": "u1"}).encode())
    unsigned = f"{header}.{payload}."
    with pytest.raises(jwt.JWTError):
        jwt.decode(unsigned, SECRET, algorithms=[HS_ALG])


def test_decode_swallow_pattern(depcheck):
    """A consumer that does `try: jwt.decode(...) except JWTError: return None`
    yields None for a tampered token, never a partially-trusted dict."""
    jwt = _jwt(depcheck)

    def safe_decode(token: str):
        try:
            return jwt.decode(token, SECRET, algorithms=[HS_ALG])
        except jwt.JWTError:
            return None

    good = jwt.encode({"sub": "user-1"}, SECRET, algorithm=HS_ALG)
    assert safe_decode(good)["sub"] == "user-1"
    assert safe_decode(good + "tamper") is None
    assert safe_decode("garbage") is None


# --------------------------------------------------------------------------- #
# Behavioural: low-level JWS sign/verify roundtrip
# --------------------------------------------------------------------------- #


def test_jws_sign_verify_roundtrip(depcheck):
    """jose.jws.sign produces a compact JWS that jose.jws.verify accepts with
    the same key+alg, returning the original payload bytes."""
    jws = _jws(depcheck)
    payload = b'{"sub":"u1"}'
    signed = jws.sign(payload, SECRET, algorithm=HS_ALG)
    assert isinstance(signed, str)
    assert signed.count(".") == 2
    verified = jws.verify(signed, SECRET, algorithms=[HS_ALG])
    assert verified == payload


def test_jws_verify_wrong_key_rejected(depcheck):
    jws = _jws(depcheck)
    signed = jws.sign(b"data", SECRET, algorithm=HS_ALG)
    with pytest.raises(jws.JWSError):
        jws.verify(signed, OTHER_SECRET, algorithms=[HS_ALG])


# --------------------------------------------------------------------------- #
# Behavioural: JWK key construction
# --------------------------------------------------------------------------- #


def test_jwk_construct_hmac_key(depcheck):
    """jwk.construct builds a Key object from raw HMAC material + alg name;
    the resulting key can sign and self-verify."""
    jwk = _jwk(depcheck)
    key = jwk.construct(SECRET, algorithm=HS_ALG)
    sig = key.sign(b"message")
    assert isinstance(sig, (bytes, bytearray))
    assert key.verify(b"message", sig) is True
    assert key.verify(b"tampered", sig) is False


# --------------------------------------------------------------------------- #
# Exception hierarchy (so broad `except` handlers stay correct)
# --------------------------------------------------------------------------- #


def test_exception_hierarchy(depcheck):
    """Every specific JOSE exception must subclass JOSEError, and the JWT-layer
    ones must subclass JWTError, so broad handlers keep catching them."""
    mod = depcheck.load(IMPORT_NAME)
    exc = mod.exceptions
    assert issubclass(exc.JWTError, exc.JOSEError)
    assert issubclass(exc.JWSError, exc.JOSEError)
    assert issubclass(exc.JWKError, exc.JOSEError)
    assert issubclass(exc.ExpiredSignatureError, exc.JWTError)
    assert issubclass(exc.JWTClaimsError, exc.JWTError)
    assert issubclass(exc.JWSSignatureError, exc.JWSError)


def test_top_level_exceptions_are_canonical(depcheck):
    """jose.JWTError and jose.exceptions.JWTError must be the identical class."""
    mod = depcheck.load(IMPORT_NAME)
    for name in ("JOSEError", "JWSError", "JWTError", "ExpiredSignatureError"):
        assert getattr(mod, name) is getattr(mod.exceptions, name), (
            f"jose.{name} is not jose.exceptions.{name}"
        )


def test_not_directly_imported_by_backend(depcheck, open_webui_backend):
    """Documents the current reality: python-jose is a pinned dependency but
    the backend does not import `jose` directly (the only `jose` token in the
    source is authlib's `authlib.jose.errors`). If this ever changes — i.e.
    the backend starts importing python-jose — this test flips and the import
    surface above should be re-derived from the real call sites.
    """
    src = open_webui_backend / "open_webui"
    if not src.is_dir():
        pytest.skip("backend source not present")
    # rg-free scan: walk .py files looking for a real `jose` import that is
    # NOT authlib.jose.
    hits: list[str] = []
    for path in src.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if "authlib.jose" in stripped:
                continue
            if stripped.startswith(("import jose", "from jose")):
                # `jose` must be a complete module token, not just a prefix:
                # `from joserfc...` (a *different* JOSE library, used by
                # utils/oauth.py) also starts with "from jose" but is not
                # python-jose.
                rest = stripped.split("jose", 1)[1]
                if rest[:1] in ("", " ", ".", ",", ")"):
                    hits.append(f"{path.name}:{lineno}: {stripped}")
    assert not hits, (
        "python-jose is now imported directly by the backend; update this "
        f"contract from the real call sites: {hits}"
    )
