"""Dependency contract: soundfile (PySoundFile, import name ``soundfile``).

``soundfile`` writes the WAV output of Open WebUI's local
HuggingFace SpeechT5 text-to-speech pipeline. ``routers/audio.py``
imports it as ``sf`` and calls:

    sf.write(str(file_path), wav["audio"], samplerate=wav["sampling_rate"])

to persist the synthesised audio to disk before returning a FileResponse.
A breaking bump (renamed ``write``/``read``, dropped the ``samplerate``
parameter, or a libsndfile ABI break) would break local TTS audio output.

``soundfile`` is a ctypes binding over the native ``libsndfile`` library,
so per the exemplar's guidance for C-backed deps this module pins the
small public surface (``write`` / ``read`` / ``info`` / ``SoundFile``)
plus a *behavioural* WAV round-trip through an in-memory ``BytesIO`` — but
only when ``libsndfile`` is actually available (the binding imports fine
without the native lib yet raises ``OSError`` on first use, so the
behavioural tests skip cleanly there). No files on disk, no network.

Pattern mirrors test_requests.py. Uses ``depcheck`` from conftest.py.
"""

from __future__ import annotations

import inspect
from io import BytesIO

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "soundfile"
DIST_NAME = "soundfile"

TOP_LEVEL_SYMBOLS = [
    "write",  # sf.write(file, data, samplerate, ...) — used by audio.py
    "read",  # sf.read(file, ...) — symmetric read path
    "info",  # sf.info(file) — metadata probe
    "SoundFile",  # streaming handle class
    "available_formats",
    "available_subtypes",
]


# ---------------------------------------------------------------------------
# libsndfile availability probe — the binding imports without the native lib
# but raises OSError on first real call. Behavioural tests use this to skip.
# ---------------------------------------------------------------------------


def _libsndfile_ready(mod) -> bool:
    try:
        mod.available_formats()
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Import + version
# ---------------------------------------------------------------------------


def test_import(depcheck):
    """`soundfile` must import (skip cleanly if absent)."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "soundfile"


def test_version_reported(depcheck):
    """The installed distribution version must resolve."""
    assert depcheck.dist_version(DIST_NAME) is not None


# ---------------------------------------------------------------------------
# Symbol-existence + signature checks (API surface — no native call needed).
# ---------------------------------------------------------------------------


def test_top_level_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, TOP_LEVEL_SYMBOLS)


def test_write_and_read_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert callable(mod.write)
    assert callable(mod.read)


def test_write_signature_has_samplerate(depcheck):
    """audio.py calls sf.write(file, data, samplerate=...). The first two
    positional params (file, data) and the samplerate parameter must remain."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.write)
    params = list(sig.parameters)
    assert params[:2] == ["file", "data"], f"sf.write positional shape changed: {params}"
    assert "samplerate" in sig.parameters, "sf.write dropped the samplerate parameter"


def test_read_signature(depcheck):
    """sf.read(file, ...) takes the source as its first positional arg and a
    dtype kwarg — the symmetric read contract."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.read)
    params = list(sig.parameters)
    assert params[0] == "file"
    assert "dtype" in sig.parameters


# ---------------------------------------------------------------------------
# Behavioural contracts (OFFLINE) — WAV round-trip through libsndfile.
# Skipped cleanly when the native lib or numpy is unavailable.
# ---------------------------------------------------------------------------


def _require_runtime(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    np = depcheck.try_load("numpy")
    if np is None:
        pytest.skip("numpy not installed; soundfile's surface tests still apply")
    if not _libsndfile_ready(mod):
        pytest.skip("libsndfile native library unavailable; surface tests still apply")
    return mod, np


def test_behaviour_write_then_read_wav(depcheck):
    """Mirror audio.py's exact call shape: sf.write(target, audio,
    samplerate=...), then read it back and confirm the sample rate and frame
    count survive. This is the TTS persistence contract."""
    mod, np = _require_runtime(depcheck)
    audio = (np.sin(np.linspace(0, 6.28, 1600)) * 0.5).astype("float32")

    buf = BytesIO()
    mod.write(buf, audio, samplerate=16000, format="WAV")
    buf.seek(0)

    data, sr = mod.read(buf)
    assert sr == 16000
    assert len(data) == 1600
    # Values must be recovered closely (WAV float -> float is near-lossless).
    assert np.allclose(data, audio, atol=1e-3)


def test_behaviour_roundtrip_preserves_samplerate_variants(depcheck):
    """The SpeechT5 pipeline reports its own sampling_rate; verify a couple of
    rates round-trip exactly (the rate is read back from the file header)."""
    mod, np = _require_runtime(depcheck)
    tone = (np.cos(np.linspace(0, 12.56, 800)) * 0.3).astype("float32")
    for rate in (8000, 22050, 44100):
        buf = BytesIO()
        mod.write(buf, tone, samplerate=rate, format="WAV")
        buf.seek(0)
        _, sr = mod.read(buf)
        assert sr == rate, f"sample rate {rate} not preserved (got {sr})"


def test_behaviour_info_reports_metadata(depcheck):
    """sf.info(file) must report frames / samplerate / channels for a written
    WAV — the metadata probe path."""
    mod, np = _require_runtime(depcheck)
    audio = np.zeros(500, dtype="float32")
    buf = BytesIO()
    mod.write(buf, audio, samplerate=16000, format="WAV")
    buf.seek(0)
    info = mod.info(buf)
    assert info.samplerate == 16000
    assert info.frames == 500
    assert info.channels == 1


def test_behaviour_stereo_roundtrip(depcheck):
    """Multi-channel audio must round-trip with channel count intact (a 2D
    (frames, channels) array)."""
    mod, np = _require_runtime(depcheck)
    stereo = np.zeros((300, 2), dtype="float32")
    stereo[:, 0] = 0.1
    stereo[:, 1] = -0.1
    buf = BytesIO()
    mod.write(buf, stereo, samplerate=16000, format="WAV")
    buf.seek(0)
    data, sr = mod.read(buf)
    assert sr == 16000
    assert data.shape == (300, 2)


def test_behaviour_wav_format_available(depcheck):
    """WAV must be among the supported output formats (audio.py writes WAV)."""
    mod, _ = _require_runtime(depcheck)
    formats = mod.available_formats()
    assert "WAV" in formats, f"WAV format unavailable: {list(formats)}"
