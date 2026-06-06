"""Dependency contract: cryptography.

Open WebUI uses `cryptography` directly for four security-critical jobs:

  * **Fernet symmetric encryption** (`cryptography.fernet.Fernet`) — encrypts
    OAuth session tokens at rest in the DB (`models/oauth_sessions.py`) and the
    OAuth client-info blob (`utils/oauth.py`). Keys are derived as
    ``base64.urlsafe_b64encode(sha256(secret))`` (a 44-char urlsafe-b64 key).
  * **AES-GCM AEAD** (`cryptography.hazmat.primitives.ciphers.aead.AESGCM`) —
    decrypts the enterprise license blob in `utils/auth.py` with a SHA-256
    key, a 12-byte nonce and ``associated_data=None``.
  * **Ed25519 signature verification** — the license public key is loaded with
    ``serialization.load_pem_public_key(...)`` (`env.py`) and the resulting
    public key's ``.verify(signature, data)`` checks the license signature
    (`utils/auth.py`, via the imported `asymmetric.ed25519` module).
  * **PEM key loading** (`cryptography.hazmat.primitives.serialization`) — see
    above; `env.py` wraps the configured base64 body in PEM armor and parses it.

This module pins exactly that slice of the API plus the failure modes the
backend depends on (wrong key / tampered ciphertext / bad signature must
raise), so a `cryptography` major bump (46 -> 48) that removed, renamed or
changed any of it fails loudly here instead of as a runtime error deep in an
auth or OAuth path.

Modelled on the unit/deps/ exemplar: dotted symbol-existence checks (API
surface) + offline behavioural contracts with locally generated keys (no
network, no disk fixtures, no running services). Uses the `depcheck` fixture
from unit/deps/conftest.py.
"""

from __future__ import annotations

import base64
import hashlib
import os

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "cryptography"
DIST_NAME = "cryptography"

# Every dotted symbol the Open WebUI backend resolves on `cryptography`.
# Paths are relative to the top-level `cryptography` package object.
USED_SYMBOLS = [
    # Fernet — OAuth token / client-info encryption at rest.
    "fernet.Fernet",
    "fernet.InvalidToken",
    # AES-GCM AEAD — license blob decryption.
    "hazmat.primitives.ciphers.aead.AESGCM",
    # PEM public-key loading — license public key.
    "hazmat.primitives.serialization",
    "hazmat.primitives.serialization.load_pem_public_key",
    # Ed25519 — license signature verification.
    "hazmat.primitives.asymmetric.ed25519",
    "hazmat.primitives.asymmetric.ed25519.Ed25519PublicKey",
    "hazmat.primitives.asymmetric.ed25519.Ed25519PrivateKey",
]

# Submodules the backend imports directly (must be importable as modules).
USED_SUBMODULES = [
    "cryptography.fernet",
    "cryptography.hazmat.primitives.serialization",
    "cryptography.hazmat.primitives.asymmetric.ed25519",
    "cryptography.hazmat.primitives.ciphers.aead",
    "cryptography.exceptions",
]


# --------------------------------------------------------------------------- #
# Import + API surface
# --------------------------------------------------------------------------- #
def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "cryptography"


def test_version_reported(depcheck):
    """Sanity: the installed distribution version is resolvable (so bump
    tooling and this suite agree on what's under test)."""
    assert depcheck.dist_version(DIST_NAME) is not None


def test_used_symbols_exist(depcheck):
    """Every cryptography symbol the codebase references must still exist."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_used_submodules_importable(depcheck):
    """The exact submodules the backend `from`-imports must import cleanly.

    These are nested hazmat paths that have moved between major versions in the
    past; pin them as importable modules, not just attribute lookups.
    """
    depcheck.load(IMPORT_NAME)  # skip cleanly if cryptography is absent
    for name in USED_SUBMODULES:
        mod = depcheck.try_load(name)
        assert mod is not None, f"submodule {name!r} no longer importable"


def test_top_level_exception_classes_exist(depcheck):
    """`cryptography.exceptions.{InvalidSignature,InvalidTag}` are the failure
    types raised by Ed25519 verify and AES-GCM decrypt; the backend relies on
    those raising (caught by broad `except`)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(
        mod,
        ["exceptions.InvalidSignature", "exceptions.InvalidTag"],
    )


# --------------------------------------------------------------------------- #
# serialization.load_pem_public_key — env.py
# --------------------------------------------------------------------------- #
def test_load_pem_public_key_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "hazmat.primitives.serialization.load_pem_public_key")


def test_load_pem_public_key_accepts_data_param(depcheck):
    """env.py calls ``serialization.load_pem_public_key(<pem bytes>)`` — a
    single positional `data` argument. Pin that the first parameter is `data`
    (the historic `backend=` arg is deprecated but still accepted)."""
    serialization = depcheck.load("cryptography.hazmat.primitives.serialization")
    depcheck.assert_params(serialization.load_pem_public_key, ["data"])


def test_serialization_pem_enums_exist(depcheck):
    """The Encoding/PublicFormat enums the contract uses to *produce* a PEM for
    these tests (and that the ecosystem relies on) must still be present."""
    serialization = depcheck.load("cryptography.hazmat.primitives.serialization")
    names = set(dir(serialization))
    for attr in ("Encoding", "PublicFormat", "PrivateFormat", "NoEncryption"):
        assert attr in names, f"serialization.{attr} missing"
    assert hasattr(serialization.Encoding, "PEM")
    assert hasattr(serialization.PublicFormat, "SubjectPublicKeyInfo")
    assert hasattr(serialization.PublicFormat, "Raw")
    assert hasattr(serialization.PrivateFormat, "Raw")


def test_load_pem_public_key_parses_env_style_armor(depcheck):
    """Reproduce env.py exactly: wrap a base64 body in PEM armor and parse it.

    env.py builds the PEM as f-string armor around ``LICENSE_PUBLIC_KEY`` (the
    raw base64 of a DER SubjectPublicKeyInfo) and encodes to bytes. Generate a
    real Ed25519 key, emit its SPKI body, rebuild that armor, and confirm it
    round-trips back to a usable public key.
    """
    serialization = depcheck.load("cryptography.hazmat.primitives.serialization")
    ed25519 = depcheck.load("cryptography.hazmat.primitives.asymmetric.ed25519")

    priv = ed25519.Ed25519PrivateKey.generate()
    spki_pem = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    # Strip armor to the raw base64 body, mimicking the configured env value.
    body = b"".join(
        line for line in spki_pem.splitlines() if line and not line.startswith(b"-----")
    ).decode()

    armored = f"""
-----BEGIN PUBLIC KEY-----
{body}
-----END PUBLIC KEY-----
""".encode()

    loaded = serialization.load_pem_public_key(armored)
    assert isinstance(loaded, ed25519.Ed25519PublicKey)


def test_load_pem_public_key_rejects_garbage(depcheck):
    """Malformed PEM must raise, not silently return None (env-config safety)."""
    serialization = depcheck.load("cryptography.hazmat.primitives.serialization")
    with pytest.raises(Exception):
        serialization.load_pem_public_key(b"-----BEGIN PUBLIC KEY-----\nnope\n")


# --------------------------------------------------------------------------- #
# Ed25519 — license signature verification (utils/auth.py: pk.verify(...))
# --------------------------------------------------------------------------- #
def test_ed25519_class_surface(depcheck):
    """Pin the Ed25519 public/private class methods the license flow needs:
    private `generate`/`public_key`/`sign`, public `verify` (+ the raw-bytes
    constructors used to build keys without PEM)."""
    ed25519 = depcheck.load("cryptography.hazmat.primitives.asymmetric.ed25519")
    priv_names = set(dir(ed25519.Ed25519PrivateKey))
    for attr in ("generate", "from_private_bytes", "public_key", "sign"):
        assert attr in priv_names, f"Ed25519PrivateKey.{attr} missing"
    pub_names = set(dir(ed25519.Ed25519PublicKey))
    for attr in ("from_public_bytes", "verify", "public_bytes"):
        assert attr in pub_names, f"Ed25519PublicKey.{attr} missing"


def test_ed25519_generate_returns_private_key(depcheck):
    ed25519 = depcheck.load("cryptography.hazmat.primitives.asymmetric.ed25519")
    priv = ed25519.Ed25519PrivateKey.generate()
    assert isinstance(priv, ed25519.Ed25519PrivateKey)
    assert isinstance(priv.public_key(), ed25519.Ed25519PublicKey)


def test_ed25519_verify_param_order(depcheck):
    """utils/auth.py calls ``pk.verify(signature, data)`` — verify takes
    (signature, data) in that order. Pin the parameter names/order."""
    ed25519 = depcheck.load("cryptography.hazmat.primitives.asymmetric.ed25519")
    pub = ed25519.Ed25519PrivateKey.generate().public_key()
    depcheck.assert_params(pub.verify, ["signature", "data"])


def test_ed25519_sign_verify_roundtrip(depcheck):
    """Exercise the real license-verify primitive: a signature from the
    matching private key must verify with no exception (verify returns None)."""
    ed25519 = depcheck.load("cryptography.hazmat.primitives.asymmetric.ed25519")
    priv = ed25519.Ed25519PrivateKey.generate()
    pub = priv.public_key()
    message = b"open-webui-license-payload"
    sig = priv.sign(message)
    assert pub.verify(sig, message) is None  # no raise == valid


def test_ed25519_verify_rejects_tampered_message(depcheck):
    """A signature over different bytes must raise InvalidSignature — this is
    the security property the license check relies on."""
    crypto = depcheck.load(IMPORT_NAME)
    ed25519 = depcheck.load("cryptography.hazmat.primitives.asymmetric.ed25519")
    InvalidSignature = crypto.exceptions.InvalidSignature

    priv = ed25519.Ed25519PrivateKey.generate()
    pub = priv.public_key()
    sig = priv.sign(b"genuine-payload")
    with pytest.raises(InvalidSignature):
        pub.verify(sig, b"forged-payload")


def test_ed25519_verify_rejects_wrong_key(depcheck):
    """A signature verified against a *different* public key must raise — a
    forged license signed by an attacker key must not pass."""
    crypto = depcheck.load(IMPORT_NAME)
    ed25519 = depcheck.load("cryptography.hazmat.primitives.asymmetric.ed25519")
    InvalidSignature = crypto.exceptions.InvalidSignature

    signer = ed25519.Ed25519PrivateKey.generate()
    other_pub = ed25519.Ed25519PrivateKey.generate().public_key()
    sig = signer.sign(b"payload")
    with pytest.raises(InvalidSignature):
        other_pub.verify(sig, b"payload")


def test_ed25519_verify_via_pem_loaded_key(depcheck):
    """End-to-end mirror of env.py + auth.py: load the public key from PEM
    (as env.py does), then verify a signature with it (as auth.py does)."""
    serialization = depcheck.load("cryptography.hazmat.primitives.serialization")
    ed25519 = depcheck.load("cryptography.hazmat.primitives.asymmetric.ed25519")

    priv = ed25519.Ed25519PrivateKey.generate()
    pem = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    pub = serialization.load_pem_public_key(pem)
    sig = priv.sign(b"data")
    assert pub.verify(sig, b"data") is None


# --------------------------------------------------------------------------- #
# AESGCM — license blob decryption (utils/auth.py)
# --------------------------------------------------------------------------- #
def test_aesgcm_class_surface(depcheck):
    """Pin AESGCM's class methods: instances are constructed ``AESGCM(key)``
    and used via ``.encrypt`` / ``.decrypt`` / ``.generate_key``."""
    aead = depcheck.load("cryptography.hazmat.primitives.ciphers.aead")
    names = set(dir(aead.AESGCM))
    for attr in ("encrypt", "decrypt", "generate_key"):
        assert attr in names, f"AESGCM.{attr} missing"
    assert callable(aead.AESGCM)


def test_aesgcm_constructible_with_sha256_key(depcheck):
    """auth.py builds the key as ``sha256(...).digest()`` (32 bytes) and passes
    it positionally to ``AESGCM(kb)``. Confirm a 32-byte key is accepted.

    NOTE: AESGCM is a Rust-backed class whose ``__init__`` signature isn't
    introspectable (shows ``*args, **kwargs``), so this is checked behaviourally
    rather than via signature inspection.
    """
    aead = depcheck.load("cryptography.hazmat.primitives.ciphers.aead")
    key = hashlib.sha256(b"some-license-key").digest()
    assert len(key) == 32
    inst = aead.AESGCM(key)
    assert inst is not None


def test_aesgcm_encrypt_decrypt_roundtrip(depcheck):
    """Mirror auth.py: 12-byte nonce, ``associated_data=None``, decrypt back to
    the original plaintext bytes."""
    aead = depcheck.load("cryptography.hazmat.primitives.ciphers.aead")
    key = hashlib.sha256(b"license-secret").digest()
    aesgcm = aead.AESGCM(key)
    nonce = os.urandom(12)  # auth.py uses a 12-byte (nl=12) nonce prefix
    plaintext = b'{"exp":"2099-01-01","name":"Org"}'

    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    assert ciphertext != plaintext
    assert aesgcm.decrypt(nonce, ciphertext, None) == plaintext


def test_aesgcm_decrypt_signature_positional_args(depcheck):
    """auth.py calls ``aesgcm.decrypt(ln, lt, None)`` — three positionals
    (nonce, data, associated_data). Verify that exact arity works by call,
    since the signature isn't introspectable on the Rust class."""
    aead = depcheck.load("cryptography.hazmat.primitives.ciphers.aead")
    aesgcm = aead.AESGCM(aead.AESGCM.generate_key(bit_length=256))
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, b"payload", None)
    # Positional (nonce, data, associated_data) — the call shape auth.py uses.
    assert aesgcm.decrypt(nonce, ct, None) == b"payload"


def test_aesgcm_decrypt_wrong_key_raises(depcheck):
    """Decrypting with the wrong key (wrong license key) must raise InvalidTag,
    not return garbage — auth.py wraps the whole decrypt in try/except and
    treats any raise as an invalid license."""
    crypto = depcheck.load(IMPORT_NAME)
    aead = depcheck.load("cryptography.hazmat.primitives.ciphers.aead")
    InvalidTag = crypto.exceptions.InvalidTag

    nonce = os.urandom(12)
    ct = aead.AESGCM(hashlib.sha256(b"right").digest()).encrypt(nonce, b"x", None)
    wrong = aead.AESGCM(hashlib.sha256(b"wrong").digest())
    with pytest.raises(InvalidTag):
        wrong.decrypt(nonce, ct, None)


def test_aesgcm_decrypt_tampered_ciphertext_raises(depcheck):
    """Flipping a ciphertext byte must fail the GCM tag check (InvalidTag)."""
    crypto = depcheck.load(IMPORT_NAME)
    aead = depcheck.load("cryptography.hazmat.primitives.ciphers.aead")
    InvalidTag = crypto.exceptions.InvalidTag

    key = hashlib.sha256(b"k").digest()
    aesgcm = aead.AESGCM(key)
    nonce = os.urandom(12)
    ct = bytearray(aesgcm.encrypt(nonce, b"sensitive", None))
    ct[0] ^= 0xFF
    with pytest.raises(InvalidTag):
        aesgcm.decrypt(nonce, bytes(ct), None)


def test_aesgcm_invalid_tag_is_exception_subclass(depcheck):
    """auth.py relies on a bad decrypt being caught by a broad ``except
    Exception``. Pin that InvalidTag subclasses Exception."""
    crypto = depcheck.load(IMPORT_NAME)
    assert issubclass(crypto.exceptions.InvalidTag, Exception)


def test_aesgcm_generate_key_bit_length(depcheck):
    """``AESGCM.generate_key(bit_length=...)`` keyword must keep working."""
    aead = depcheck.load("cryptography.hazmat.primitives.ciphers.aead")
    key = aead.AESGCM.generate_key(bit_length=256)
    assert isinstance(key, (bytes, bytearray))
    assert len(key) == 32


# --------------------------------------------------------------------------- #
# Fernet — OAuth token / client-info encryption at rest
# --------------------------------------------------------------------------- #
def test_fernet_class_surface(depcheck):
    """Pin Fernet's instance API: constructed ``Fernet(key)`` then
    ``.encrypt`` / ``.decrypt`` (+ the ``generate_key`` classmethod)."""
    fernet = depcheck.load("cryptography.fernet")
    names = set(dir(fernet.Fernet))
    for attr in ("encrypt", "decrypt", "generate_key"):
        assert attr in names, f"Fernet.{attr} missing"
    assert callable(fernet.Fernet)


def test_fernet_init_accepts_key_param(depcheck):
    """Both call sites do ``Fernet(<key>)`` positionally; pin that the first
    constructor parameter is `key`."""
    fernet = depcheck.load("cryptography.fernet")
    depcheck.assert_params(fernet.Fernet.__init__, ["key"])


def test_fernet_encrypt_decrypt_param_names(depcheck):
    """``.encrypt(data)`` / ``.decrypt(token)`` are how the backend calls them;
    pin those parameter names."""
    fernet = depcheck.load("cryptography.fernet")
    depcheck.assert_params(fernet.Fernet.encrypt, ["data"])
    depcheck.assert_params(fernet.Fernet.decrypt, ["token"])


def test_fernet_accepts_sha256_derived_key(depcheck):
    """oauth_sessions.py derives the key as
    ``base64.urlsafe_b64encode(sha256(secret).digest())`` — a 44-byte urlsafe
    base64 value. Confirm Fernet accepts exactly that shape."""
    fernet = depcheck.load("cryptography.fernet")
    key_bytes = hashlib.sha256(b"oauth-encryption-secret").digest()
    key = base64.urlsafe_b64encode(key_bytes)
    assert len(key) == 44  # the length check oauth_sessions.py keys off
    inst = fernet.Fernet(key)
    assert inst is not None


def test_fernet_generate_key_shape(depcheck):
    """The 44-char urlsafe-b64 key shape (what oauth_sessions.py special-cases)
    matches ``Fernet.generate_key()`` output length."""
    fernet = depcheck.load("cryptography.fernet")
    key = fernet.Fernet.generate_key()
    assert isinstance(key, bytes)
    assert len(key) == 44


def test_fernet_encrypt_decrypt_roundtrip(depcheck):
    """Mirror _encrypt_token/_decrypt_token: encrypt JSON bytes, decrypt back.

    The backend does ``fernet.encrypt(s.encode()).decode()`` then later
    ``fernet.decrypt(s.encode()).decode()``; the token is str-safe and the
    plaintext round-trips byte-for-byte.
    """
    fernet = depcheck.load("cryptography.fernet")
    key = base64.urlsafe_b64encode(hashlib.sha256(b"k").digest())
    f = fernet.Fernet(key)
    plaintext = b'{"access_token":"abc","refresh_token":"def"}'

    token = f.encrypt(plaintext)
    assert f.decrypt(token) == plaintext
    # Token is ASCII/urlsafe so the backend's .decode()/.encode() hop is safe.
    assert token.decode("ascii").encode("ascii") == token


def test_fernet_wrong_key_raises_invalid_token(depcheck):
    """A token encrypted under one key must NOT decrypt under another — this is
    why oauth_sessions.py deletes sessions on decrypt failure. Pin it raises
    InvalidToken (the backend catches it as a decrypt failure)."""
    fernet = depcheck.load("cryptography.fernet")
    k1 = base64.urlsafe_b64encode(hashlib.sha256(b"key-one").digest())
    k2 = base64.urlsafe_b64encode(hashlib.sha256(b"key-two").digest())

    token = fernet.Fernet(k1).encrypt(b"secret")
    with pytest.raises(fernet.InvalidToken):
        fernet.Fernet(k2).decrypt(token)


def test_fernet_garbage_token_raises_invalid_token(depcheck):
    """Decrypting a non-token (corrupted DB value) raises InvalidToken — the
    failure mode oauth_sessions.py logs and recovers from."""
    fernet = depcheck.load("cryptography.fernet")
    key = base64.urlsafe_b64encode(hashlib.sha256(b"k").digest())
    with pytest.raises(fernet.InvalidToken):
        fernet.Fernet(key).decrypt(b"this-is-not-a-fernet-token")


def test_fernet_invalid_token_is_exception_subclass(depcheck):
    """InvalidToken must subclass Exception so the backend's broad
    ``except Exception`` handlers around decrypt keep catching it."""
    fernet = depcheck.load("cryptography.fernet")
    assert issubclass(fernet.InvalidToken, Exception)


def test_fernet_rejects_malformed_key(depcheck):
    """A key that isn't 32 urlsafe-b64 bytes must raise at construction — the
    backend re-raises this so a misconfigured key fails fast at startup."""
    fernet = depcheck.load("cryptography.fernet")
    with pytest.raises(Exception):
        fernet.Fernet(b"too-short")
