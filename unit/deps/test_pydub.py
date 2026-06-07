"""Dependency contract: pydub (import name ``pydub``).

``routers/audio.py`` uses pydub to preprocess audio for STT and to transcode
TTS output. The exact surface it touches:

  - ``from pydub import AudioSegment``;
    ``from pydub.silence import split_on_silence``;
    ``from pydub.utils import mediainfo``.
  - ``AudioSegment.from_file(path_or_BytesIO[, format=...])`` — auto-detect
    load (delegates to ffmpeg for compressed formats).
  - ``AudioSegment.from_raw(BytesIO, sample_width=2, frame_rate=rate,
    channels=channels)`` — raw PCM ingestion for Gemini-style TTS (PURE
    Python, no ffmpeg).
  - ``audio.set_frame_rate(16000).set_channels(1)`` — the whisper-prep
    downmix/resample chain (returns new AudioSegments, chainable).
  - ``audio.export(out_path_or_str, format="mp3", bitrate="32k")`` — encode.

This module pins that surface and exercises the behavioural contract
OFFLINE and deterministically. Audio is synthesized in-process as 16-bit
PCM, so the core tests need NO ffmpeg, NO files, and NO network:

  - ``from_raw`` builds a segment with the requested frame_rate / channels /
    sample_width;
  - the segment exposes ``frame_rate`` / ``channels`` / ``sample_width`` /
    ``duration_seconds`` and ``len()`` in milliseconds;
  - ``set_frame_rate`` / ``set_channels`` return new segments with the new
    parameters and are chainable (the audio.py downmix chain);
  - a WAV export→reload round-trip preserves frame_rate and length (WAV uses
    Python's ``wave`` module, no ffmpeg).

ONE test (mp3 export, the ``export(format="mp3", bitrate=...)`` call) needs
ffmpeg and SKIPS cleanly when ffmpeg isn't on PATH, so the suite stays
runnable in a minimal environment.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import inspect
import io
import math
import struct

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "pydub"
DIST_NAME = "pydub"

USED_SYMBOLS = ["AudioSegment"]
SUBMODULE_SYMBOLS = ["silence.split_on_silence", "utils.mediainfo"]

# AudioSegment methods the backend calls.
AUDIOSEGMENT_METHODS = [
    "from_file",
    "from_raw",
    "export",
    "set_frame_rate",
    "set_channels",
]


def _pcm_bytes(*, rate: int, seconds: float, freq: float = 220.0) -> bytes:
    """Synthesize 16-bit mono PCM (little-endian) — ffmpeg-free test audio."""
    n = int(rate * seconds)
    return b"".join(
        struct.pack("<h", int(3000 * math.sin(2 * math.pi * freq * i / rate))) for i in range(n)
    )


def _raw_segment(mod, *, rate: int = 8000, seconds: float = 0.2, channels: int = 1):
    return mod.AudioSegment.from_raw(
        io.BytesIO(_pcm_bytes(rate=rate, seconds=seconds)),
        sample_width=2,
        frame_rate=rate,
        channels=channels,
    )


def _ffmpeg_available(mod) -> bool:
    try:
        return mod.utils.which("ffmpeg") is not None
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Import / version
# --------------------------------------------------------------------------- #


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "pydub"


def test_version_reported(depcheck):
    assert depcheck.dist_version(DIST_NAME) is not None


# --------------------------------------------------------------------------- #
# Symbol existence (API surface)
# --------------------------------------------------------------------------- #


def test_used_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_submodule_symbols_exist(depcheck):
    """pydub.silence.split_on_silence and pydub.utils.mediainfo must resolve."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, SUBMODULE_SYMBOLS)


def test_audiosegment_methods_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    names = set(dir(mod.AudioSegment))
    for meth in AUDIOSEGMENT_METHODS:
        assert meth in names, f"AudioSegment.{meth} missing"
        assert callable(getattr(mod.AudioSegment, meth))


def test_from_raw_signature(depcheck):
    """audio.py calls from_raw(BytesIO, sample_width=2, frame_rate=, channels=).
    from_raw takes **kwargs, so assert it accepts arbitrary kwargs (or the
    named ones)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.AudioSegment.from_raw,
        ["sample_width", "frame_rate", "channels"],
    )


def test_export_signature(depcheck):
    """export(out_f, format=, bitrate=). Pin those param names remain."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.AudioSegment.export, ["format", "bitrate"])


def test_split_on_silence_signature(depcheck):
    """split_on_silence(audio_segment, min_silence_len=, silence_thresh=, ...)."""
    mod = depcheck.load(IMPORT_NAME)
    fn = depcheck.resolve(mod, "silence.split_on_silence")
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    assert params and params[0].name == "audio_segment"


# --------------------------------------------------------------------------- #
# Behavioural: from_raw (ffmpeg-free PCM ingestion)
# --------------------------------------------------------------------------- #


def test_from_raw_builds_segment_with_params(depcheck):
    """from_raw must honor the sample_width / frame_rate / channels passed —
    the Gemini-TTS raw-PCM path in transcode_audio_to_mp3."""
    mod = depcheck.load(IMPORT_NAME)
    seg = mod.AudioSegment.from_raw(
        io.BytesIO(_pcm_bytes(rate=24000, seconds=0.1)),
        sample_width=2,
        frame_rate=24000,
        channels=1,
    )
    assert seg.frame_rate == 24000
    assert seg.channels == 1
    assert seg.sample_width == 2


def test_segment_len_is_milliseconds(depcheck):
    """len(AudioSegment) is duration in milliseconds; 0.2s -> ~200ms."""
    mod = depcheck.load(IMPORT_NAME)
    seg = _raw_segment(mod, rate=8000, seconds=0.2)
    assert abs(len(seg) - 200) <= 2  # rounding tolerance


def test_segment_duration_seconds(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    seg = _raw_segment(mod, rate=8000, seconds=0.2)
    assert seg.duration_seconds == pytest.approx(0.2, abs=0.01)


# --------------------------------------------------------------------------- #
# Behavioural: the whisper-prep downmix/resample chain
# --------------------------------------------------------------------------- #


def test_set_frame_rate_returns_resampled_segment(depcheck):
    """audio.set_frame_rate(16000) returns a NEW segment at 16kHz; the original
    is unchanged (pydub segments are immutable)."""
    mod = depcheck.load(IMPORT_NAME)
    seg = _raw_segment(mod, rate=8000, seconds=0.2)
    out = seg.set_frame_rate(16000)
    assert out.frame_rate == 16000
    assert seg.frame_rate == 8000  # original untouched


def test_set_channels_to_mono(depcheck):
    """set_channels(1) downmixes to mono."""
    mod = depcheck.load(IMPORT_NAME)
    stereo = _raw_segment(mod, rate=8000, seconds=0.2, channels=2)
    assert stereo.channels == 2
    mono = stereo.set_channels(1)
    assert mono.channels == 1


def test_downmix_chain_is_chainable(depcheck):
    """The exact audio.py call: set_frame_rate(16000).set_channels(1)."""
    mod = depcheck.load(IMPORT_NAME)
    seg = _raw_segment(mod, rate=8000, seconds=0.2, channels=2)
    out = seg.set_frame_rate(16000).set_channels(1)
    assert out.frame_rate == 16000
    assert out.channels == 1


# --------------------------------------------------------------------------- #
# Behavioural: export/reload round-trip (WAV, ffmpeg-free)
# --------------------------------------------------------------------------- #


def test_wav_export_reload_roundtrip(depcheck):
    """export(format='wav') then from_file(..., format='wav') preserves
    frame_rate and length. WAV uses Python's wave module, so this needs no
    ffmpeg and pins the export→reload contract end to end."""
    mod = depcheck.load(IMPORT_NAME)
    seg = _raw_segment(mod, rate=8000, seconds=0.2)
    buf = io.BytesIO()
    seg.export(buf, format="wav")
    buf.seek(0)
    reloaded = mod.AudioSegment.from_file(buf, format="wav")
    assert reloaded.frame_rate == 8000
    assert abs(len(reloaded) - len(seg)) <= 2


def test_export_returns_filelike(depcheck):
    """export returns the output file object so callers can read the bytes."""
    mod = depcheck.load(IMPORT_NAME)
    seg = _raw_segment(mod, rate=8000, seconds=0.1)
    out = io.BytesIO()
    result = seg.export(out, format="wav")
    assert result is not None
    out.seek(0)
    assert len(out.getvalue()) > 0


# --------------------------------------------------------------------------- #
# Behavioural: mp3 export (needs ffmpeg — skips cleanly without it)
# --------------------------------------------------------------------------- #


def test_mp3_export_with_bitrate(depcheck):
    """compress_audio does export(path, format='mp3', bitrate='32k'). This path
    requires ffmpeg; skip when it's unavailable so the suite stays portable."""
    mod = depcheck.load(IMPORT_NAME)
    if not _ffmpeg_available(mod):
        pytest.skip("ffmpeg not on PATH; skipping mp3 export (needs encoder)")
    seg = _raw_segment(mod, rate=8000, seconds=0.2)
    out = io.BytesIO()
    seg.export(out, format="mp3", bitrate="32k")
    out.seek(0)
    data = out.getvalue()
    assert len(data) > 0
    # An MP3 stream starts with an ID3 tag or a frame sync (0xFF 0xEx/0xFx).
    assert data[:3] == b"ID3" or (data[0] == 0xFF and (data[1] & 0xE0) == 0xE0)
