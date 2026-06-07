"""Dependency contract: brotlicffi.

brotlicffi is the CFFI Brotli (de)compression binding. Open WebUI pins it
directly in ``backend/requirements.txt`` (``brotlicffi==1.2.0.1``,
alongside ``Brotli==1.2.0``) so that Brotli content-encoding works on the
HTTP paths that negotiate it:

  * ``httpx[...,brotli]`` — outbound HTTP client decompresses ``br``
    responses (provider/proxy/retrieval calls go through httpx);
  * ``starlette-compress`` — compresses outbound responses, offering
    Brotli when the client advertises ``Accept-Encoding: br``.

IMPORTANT — usage note: the Open WebUI *application* code does NOT import
``brotlicffi`` directly anywhere. It is a declared/transitive dependency
consumed by httpx and starlette-compress under the hood. There are
therefore no first-party call sites to pin keyword arguments against.

This module instead pins the *core public surface* brotlicffi must keep
for those consumers — the one-shot ``compress`` / ``decompress`` helpers,
the streaming ``Compressor`` / ``Decompressor`` classes, and the ``Error``
type — and verifies a real offline round-trip (Brotli is a pure in-memory
codec: no network, no I/O). brotlicffi is also the drop-in that exposes
the same API as the ``brotli`` package; httpx selects whichever is
installed, so keeping this surface intact is what makes the ``br`` path
work regardless of which backend wins.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "brotlicffi"
DIST_NAME = "brotlicffi"

# The Brotli API surface httpx / starlette-compress rely on.
USED_SYMBOLS = [
    "compress",  # one-shot compress
    "decompress",  # one-shot decompress
    "Compressor",  # streaming compressor (httpx brotli decoder uses the pair)
    "Decompressor",  # streaming decompressor
    "Error",  # raised on corrupt/short input
]


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "brotlicffi"


def test_version_reported(depcheck):
    assert depcheck.dist_version(DIST_NAME) is not None


def test_used_symbols_exist(depcheck):
    """The compress/decompress + Compressor/Decompressor + Error surface must
    remain present — httpx and starlette-compress dispatch onto exactly these."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_compress_decompress_are_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert callable(mod.compress)
    assert callable(mod.decompress)


def test_oneshot_roundtrip(depcheck):
    """compress(data) then decompress(...) must reproduce the original bytes —
    the exact contract starlette-compress (encode) and httpx (decode) form
    across the wire. Fully offline (in-memory codec)."""
    mod = depcheck.load(IMPORT_NAME)
    original = b"Open WebUI Brotli round-trip payload " * 32
    compressed = mod.compress(original)
    assert isinstance(compressed, bytes)
    assert compressed != original
    restored = mod.decompress(compressed)
    assert restored == original


def test_compress_actually_compresses(depcheck):
    """For sufficiently repetitive input, Brotli output must be smaller than the
    input — otherwise advertising `br` would only add overhead."""
    mod = depcheck.load(IMPORT_NAME)
    original = b"A" * 4096
    compressed = mod.compress(original)
    assert len(compressed) < len(original), "Brotli did not compress repetitive data"
    assert mod.decompress(compressed) == original


def test_empty_input_roundtrip(depcheck):
    """Empty payloads must round-trip (a zero-length response body can still be
    br-encoded)."""
    mod = depcheck.load(IMPORT_NAME)
    compressed = mod.compress(b"")
    assert mod.decompress(compressed) == b""


def test_decompress_rejects_garbage(depcheck):
    """Corrupt/garbage input must raise brotlicffi.Error, not return wrong bytes
    or crash the interpreter — httpx surfaces this as a decode failure."""
    mod = depcheck.load(IMPORT_NAME)
    with pytest.raises(mod.Error):
        mod.decompress(b"this is definitely not a valid brotli stream")


def test_streaming_decompressor_roundtrip(depcheck):
    """httpx's brotli decoder feeds response chunks into a Decompressor and
    concatenates .decompress() output (then .finish()/.flush()). Verify the
    streaming pair reconstructs data fed in pieces."""
    mod = depcheck.load(IMPORT_NAME)
    original = b"streamed brotli body chunk-by-chunk " * 64
    compressed = mod.compress(original)

    dec = mod.Decompressor()
    out = bytearray()
    # Feed the compressed stream in small chunks.
    step = 7
    for i in range(0, len(compressed), step):
        out += dec.decompress(compressed[i : i + step])
    # Drain any buffered tail via finish()/flush() if present.
    for closer in ("finish", "flush"):
        fn = getattr(dec, closer, None)
        if callable(fn):
            try:
                tail = fn()
                if tail:
                    out += tail
            except Exception:
                pass
            break
    assert bytes(out) == original


def test_streaming_compressor_then_oneshot_decompress(depcheck):
    """starlette-compress streams response bodies through a Compressor:
    .process()/.compress() per chunk then .finish()/.flush(). The produced
    stream must decompress back to the original via the one-shot decoder."""
    mod = depcheck.load(IMPORT_NAME)
    original = b"compressor streaming path " * 50

    comp = mod.Compressor()
    blob = bytearray()
    # The chunk-feed method is named process() (older) or compress() (newer).
    feed = getattr(comp, "process", None) or getattr(comp, "compress", None)
    assert feed is not None and callable(feed), "Compressor has no chunk-feed method"
    blob += feed(original)
    # Flush the final block.
    for closer in ("finish", "flush"):
        fn = getattr(comp, closer, None)
        if callable(fn):
            tail = fn()
            if tail:
                blob += tail
            break
    assert mod.decompress(bytes(blob)) == original


def test_quality_kwarg_accepted_by_compress(depcheck):
    """Consumers tune Brotli via the `quality` argument. compress(data,
    quality=N) must remain accepted (pin it behaviourally — brotlicffi's
    compress may be C-backed without an introspectable signature)."""
    mod = depcheck.load(IMPORT_NAME)
    original = b"quality knob payload " * 40
    # Low quality (fast) and high quality (small) must both work and round-trip.
    low = mod.compress(original, quality=1)
    high = mod.compress(original, quality=11)
    assert mod.decompress(low) == original
    assert mod.decompress(high) == original
