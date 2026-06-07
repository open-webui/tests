"""Dependency contract: Brotli (PyPI ``Brotli``, import name ``brotli``).

Brotli is a **transitive** dependency for Open WebUI: it is *not* imported
anywhere in `open_webui` directly. It is pulled in (alongside its CFFI
sibling ``brotlicffi`` and via ``httpx[...,brotli]``) so that the
``starlette_compress.CompressMiddleware`` registered in ``main.py``
(``from starlette_compress import CompressMiddleware``) can Brotli-compress
HTTP responses when a client sends ``Accept-Encoding: br``.

Because the backend never calls ``brotli`` itself, the meaningful contract
is the surface that the *actual* consumer — ``starlette_compress._brotli`` —
exercises on the module. That responder does exactly:

  * ``import brotli`` (falling back to ``import brotlicffi as brotli``);
  * one-shot path: ``brotli.compress(body, quality=self.quality)``;
  * streaming path: ``c = brotli.Compressor(quality=...)`` then
    ``c.process(chunk)`` / ``c.flush()`` / ``c.finish()``.

So this module pins that compress/Compressor surface plus the round-trip
correctness Brotli must guarantee (compress -> decompress reproduces the
bytes, a non-Brotli payload raises ``brotli.error``), so a `Brotli` bump
that removed/renamed any of it fails loudly here instead of breaking
response compression at runtime. It is decode-agnostic about which backend
(``brotli`` vs ``brotlicffi``) is installed — both expose the same surface,
and starlette_compress treats them interchangeably.

Pattern mirrors the unit/deps/ exemplar: symbol-existence + behavioural
round-trips, fully offline (compression is pure CPU, no network/services).
Uses the `depcheck` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "brotli"
DIST_NAME = "Brotli"

# Module-level names the starlette_compress consumer (and the wider ecosystem)
# resolves on `brotli`. compress/Compressor/error are load-bearing for the
# response-compression path; decompress/Decompressor/version/MODE_* are pinned
# as the stable public surface (and used here to verify round-trips offline).
USED_SYMBOLS = [
    "compress",
    "decompress",
    "Compressor",
    "Decompressor",
    "error",
    "version",
    "MODE_GENERIC",
    "MODE_TEXT",
    "MODE_FONT",
]

# A low quality for behavioural round-trips — correctness is independent of the
# quality level, and lower is faster. starlette_compress itself defaults to 4.
_FAST_QUALITY = 4


# --------------------------------------------------------------------------- #
# Import + version + API surface
# --------------------------------------------------------------------------- #
def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "brotli"


def test_version_reported(depcheck):
    """The installed distribution version must be resolvable (so bump tooling
    and this suite agree on what's under test)."""
    depcheck.load(IMPORT_NAME)
    assert depcheck.dist_version(DIST_NAME) is not None


def test_module_version_attribute(depcheck):
    """``brotli.version`` is a string the ecosystem reads for the native lib
    version. Pin that it stays a non-empty str."""
    mod = depcheck.load(IMPORT_NAME)
    assert isinstance(mod.version, str)
    assert mod.version


def test_used_symbols_exist(depcheck):
    """Every brotli symbol the consumer surface relies on must still exist."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_compress_decompress_callable(depcheck):
    """The one-shot helpers must be callable (the form
    ``brotli.compress(body, quality=...)`` the responder uses)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "compress")
    depcheck.assert_callable(mod, "decompress")


def test_error_is_exception_subclass(depcheck):
    """``brotli.error`` is the failure type a malformed stream raises; callers
    that guard decompression rely on it being an Exception subclass.

    NOTE: the exception is lowercase ``error`` (not ``Error``) on the Google
    ``brotli`` package — pin the exact name so a rename is caught."""
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.error, Exception)


# --------------------------------------------------------------------------- #
# brotli.compress — the one-shot path starlette_compress takes for small bodies
# --------------------------------------------------------------------------- #
def test_compress_accepts_quality_kwarg(depcheck):
    """starlette_compress calls ``brotli.compress(body, quality=self.quality)``.
    The ``quality`` keyword must remain accepted and produce valid output."""
    mod = depcheck.load(IMPORT_NAME)
    data = b"open-webui response body " * 40
    out = mod.compress(data, quality=_FAST_QUALITY)
    assert isinstance(out, bytes)
    assert out  # non-empty


def test_compress_returns_bytes(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    out = mod.compress(b"hello world", quality=_FAST_QUALITY)
    assert isinstance(out, bytes)


def test_compress_decompress_roundtrip(depcheck):
    """The fundamental contract: ``decompress(compress(x)) == x``. This is what
    makes a Brotli-encoded HTTP response decodable by the client."""
    mod = depcheck.load(IMPORT_NAME)
    data = b"The quick brown fox jumps over the lazy dog. " * 64
    compressed = mod.compress(data, quality=_FAST_QUALITY)
    assert mod.decompress(compressed) == data


def test_compress_reduces_repetitive_payload(depcheck):
    """A highly repetitive body must compress smaller than the original — the
    whole point of enabling the middleware. (Tiny/random bodies can grow; a
    large repetitive one must shrink.)"""
    mod = depcheck.load(IMPORT_NAME)
    data = b"A" * 4096
    compressed = mod.compress(data, quality=_FAST_QUALITY)
    assert len(compressed) < len(data)


def test_compress_empty_roundtrips(depcheck):
    """Compressing an empty body and decompressing back yields empty bytes —
    no crash on a zero-length response."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.decompress(mod.compress(b"", quality=_FAST_QUALITY)) == b""


def test_compress_binary_payload_roundtrip(depcheck):
    """Responses are arbitrary bytes (images, JSON, msgpack). A full-byte-range
    payload must round-trip exactly."""
    mod = depcheck.load(IMPORT_NAME)
    data = bytes(range(256)) * 16
    assert mod.decompress(mod.compress(data, quality=_FAST_QUALITY)) == data


# --------------------------------------------------------------------------- #
# brotli.Compressor — the streaming path (large/chunked response bodies)
# --------------------------------------------------------------------------- #
def test_compressor_class_surface(depcheck):
    """The streaming responder uses ``Compressor(quality=...)`` then
    ``.process`` / ``.flush`` / ``.finish``. Pin those instance methods."""
    mod = depcheck.load(IMPORT_NAME)
    comp = mod.Compressor(quality=_FAST_QUALITY)
    for name in ("process", "flush", "finish"):
        assert callable(getattr(comp, name)), f"Compressor.{name} missing"


def test_compressor_accepts_quality_kwarg(depcheck):
    """``Compressor(quality=self.quality)`` is the exact construction
    starlette_compress uses — the ``quality`` keyword must be accepted."""
    mod = depcheck.load(IMPORT_NAME)
    comp = mod.Compressor(quality=_FAST_QUALITY)
    assert comp is not None


def test_compressor_stream_roundtrip(depcheck):
    """Mirror the streaming responder: feed chunks via ``process`` then close
    with ``finish``; the concatenated output must decompress to the original.
    This is how a large chunked response gets Brotli-encoded on the wire."""
    mod = depcheck.load(IMPORT_NAME)
    chunks = [b"chunk-%03d-payload-data;" % i for i in range(50)]
    comp = mod.Compressor(quality=_FAST_QUALITY)
    blob = b""
    for ch in chunks:
        blob += comp.process(ch)
    blob += comp.finish()
    assert mod.decompress(blob) == b"".join(chunks)


def test_compressor_flush_then_finish_roundtrip(depcheck):
    """``flush()`` (used to emit a complete intermediate frame) plus a final
    ``finish()`` must still produce a fully decodable stream."""
    mod = depcheck.load(IMPORT_NAME)
    comp = mod.Compressor(quality=_FAST_QUALITY)
    out = comp.process(b"first-part-" * 10)
    out += comp.flush()
    out += comp.process(b"second-part-" * 10)
    out += comp.finish()
    expected = b"first-part-" * 10 + b"second-part-" * 10
    assert mod.decompress(out) == expected


# --------------------------------------------------------------------------- #
# brotli.Decompressor — streaming decode (used to verify the streamed output)
# --------------------------------------------------------------------------- #
def test_decompressor_class_surface(depcheck):
    """``Decompressor`` exposes streaming-decode methods. On the Google
    ``brotli`` backend these are ``process`` / ``is_finished`` /
    ``can_accept_more_data`` — pin that ``process`` exists and is callable."""
    mod = depcheck.load(IMPORT_NAME)
    dec = mod.Decompressor()
    assert callable(getattr(dec, "process", None)), "Decompressor.process missing"


def test_decompressor_stream_roundtrip(depcheck):
    """A stream produced by ``Compressor`` must decode through
    ``Decompressor.process`` back to the original bytes."""
    mod = depcheck.load(IMPORT_NAME)
    original = b"streaming-decode-roundtrip;" * 32
    comp = mod.Compressor(quality=_FAST_QUALITY)
    blob = comp.process(original) + comp.finish()
    dec = mod.Decompressor()
    assert dec.process(blob) == original


# --------------------------------------------------------------------------- #
# Failure modes — a corrupt/non-Brotli stream must raise, never silently pass
# --------------------------------------------------------------------------- #
def test_decompress_rejects_non_brotli_data(depcheck):
    """Decompressing bytes that are not a valid Brotli stream must raise
    ``brotli.error`` (not return garbage). A client/proxy that mislabels an
    encoding must fail loudly."""
    mod = depcheck.load(IMPORT_NAME)
    with pytest.raises(mod.error):
        mod.decompress(b"this is definitely not a brotli stream at all")


def test_decompress_rejects_truncated_stream(depcheck):
    """A truncated Brotli stream must raise rather than yield a partial/garbage
    body — guards against silently returning corrupted response data."""
    mod = depcheck.load(IMPORT_NAME)
    full = mod.compress(b"complete payload that will be truncated " * 20, quality=_FAST_QUALITY)
    truncated = full[: len(full) // 2]
    with pytest.raises(mod.error):
        mod.decompress(truncated)


def test_compression_modes_are_distinct_ints(depcheck):
    """``MODE_GENERIC`` / ``MODE_TEXT`` / ``MODE_FONT`` are the documented mode
    constants; pin that they exist as distinct integer values (the responder's
    text/generic selection relies on them being usable mode arguments)."""
    mod = depcheck.load(IMPORT_NAME)
    modes = {mod.MODE_GENERIC, mod.MODE_TEXT, mod.MODE_FONT}
    assert len(modes) == 3, "Brotli MODE_* constants collapsed to fewer values"
    for m in (mod.MODE_GENERIC, mod.MODE_TEXT, mod.MODE_FONT):
        assert isinstance(m, int)


def test_compress_with_text_mode_roundtrips(depcheck):
    """Compressing with an explicit ``mode=MODE_TEXT`` (text responses) must
    still round-trip — the mode argument stays a valid optimisation hint."""
    mod = depcheck.load(IMPORT_NAME)
    data = b"<html><body>hello</body></html>" * 32
    out = mod.compress(data, mode=mod.MODE_TEXT, quality=_FAST_QUALITY)
    assert mod.decompress(out) == data
