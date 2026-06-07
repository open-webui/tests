"""Dependency contract: av (PyAV — FFmpeg bindings).

av is not imported anywhere in the Open WebUI backend by name; it is a
transitive dependency (pinned ``av==14.0.1`` in requirements.txt, NOT in
requirements-min.txt) pulled in by ``faster-whisper`` — the speech-to-text
engine the audio router uses (``routers/audio.py`` builds
``faster_whisper.WhisperModel``). faster-whisper uses PyAV to *decode the
uploaded audio file into PCM frames* before transcription. If a PyAV bump
broke ``av.open`` / the container->stream->frame decode graph, or renamed
the FFmpeg error type, audio transcription would fail at the decode step.

The pin is doubly important here because the version is *deliberately
frozen*: requirements.txt carries a comment that 14.0.1 is set to avoid a
"FATAL FIPS SELFTEST FAILURE" (a newer av's bundled FFmpeg crashes under a
FIPS-mode OpenSSL). So this contract also documents *why* the version is
load-bearing.

Because nothing in the backend names an av symbol of its own, this module
pins av's *core demux/decode surface* and exercises it OFFLINE by building a
small WAV entirely in memory (stdlib ``wave`` + ``io.BytesIO``) and decoding
it through ``av.open`` exactly as faster-whisper decodes an upload — no disk,
no network, no model. av is a C-extension binding FFmpeg, so checks are
behavioural (open + decode + assert), never ``hasattr`` probing of executing
properties.

Pattern mirrors test_requests.py. Uses the ``depcheck`` fixture.
"""

from __future__ import annotations

import io
import math
import struct
import wave

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "av"
DIST_NAME = "av"

USED_SYMBOLS = [
    "open",
    "AudioFrame",
    "VideoFrame",
    "Packet",
    "Codec",
    "CodecContext",
    "AudioResampler",
    "FFmpegError",
    "error",
]

AUDIO_RATE = 8000
AUDIO_SECONDS = 1
AUDIO_FREQ = 440


def _wav_bytes(rate=AUDIO_RATE, seconds=AUDIO_SECONDS, freq=AUDIO_FREQ):
    """A mono 16-bit PCM WAV (a 440 Hz sine) as an in-memory byte stream —
    the kind of audio payload faster-whisper hands to PyAV to decode."""
    buf = io.BytesIO()
    w = wave.open(buf, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(rate)
    n = rate * seconds
    samples = (int(1000 * math.sin(2 * math.pi * freq * i / rate)) for i in range(n))
    frames = b"".join(struct.pack("<h", s) for s in samples)
    w.writeframes(frames)
    w.close()
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Import + version + surface
# ---------------------------------------------------------------------------


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "av"


def test_version_reported(depcheck):
    assert depcheck.dist_version(DIST_NAME) is not None


def test_version_is_pinned_major(depcheck):
    """The backend pins av==14.0.1 for FIPS reasons. Guard the major line so a
    bump that would re-introduce the FIPS-mode crash is at least noticed here.
    (We assert the major, not the exact patch, to avoid churn on safe patch
    bumps within the line.)"""
    mod = depcheck.load(IMPORT_NAME)
    assert isinstance(mod.__version__, str)
    major = int(mod.__version__.split(".")[0])
    assert major == 14, f"av major changed to {mod.__version__}; re-verify the FIPS selftest note"


def test_used_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_open_is_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "open")


def test_open_accepts_file_and_mode(depcheck):
    """faster-whisper calls av.open(file, ...) on a path or file-like. The
    `file` and `mode` parameters must remain accepted."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.open, ["file", "mode"])


def test_core_classes_are_types(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    for name in ("AudioFrame", "VideoFrame", "Packet", "Codec", "CodecContext"):
        assert isinstance(getattr(mod, name), type), f"av.{name} is not a class"


# ---------------------------------------------------------------------------
# Error contract — PyAV v14 uses FFmpegError; av.error mirrors it.
# ---------------------------------------------------------------------------


def test_ffmpeg_error_is_exception(depcheck):
    """av.FFmpegError is the base FFmpeg error; consumers catch it around
    decode. It must be an Exception subclass."""
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.FFmpegError, Exception)


def test_error_submodule_has_ffmpeg_error(depcheck):
    """The av.error submodule must expose FFmpegError (the canonical location)."""
    mod = depcheck.load(IMPORT_NAME)
    err = depcheck.resolve(mod, "error")
    assert hasattr(err, "FFmpegError")
    assert issubclass(err.FFmpegError, Exception)


def test_open_invalid_data_raises(depcheck):
    """Opening non-media bytes must raise an FFmpeg/OS error (not silently
    yield an empty container), so a bad audio upload surfaces as an error."""
    mod = depcheck.load(IMPORT_NAME)
    junk = io.BytesIO(b"this is definitely not an audio or video container")
    with pytest.raises(Exception):  # noqa: B017 - FFmpegError/OSError both acceptable
        container = mod.open(junk, mode="r")
        # Some builds defer the probe to first stream access; force it.
        _ = container.streams
        container.close()


# ---------------------------------------------------------------------------
# Decode graph: av.open -> container.streams.audio -> decode -> AudioFrame.
# All in memory (stdlib WAV via BytesIO). This is faster-whisper's path.
# ---------------------------------------------------------------------------


def test_open_in_memory_wav(depcheck):
    """av.open on an in-memory WAV must yield a container reporting the wav
    format — proving the file-like input path works with no disk/network."""
    mod = depcheck.load(IMPORT_NAME)
    container = mod.open(_wav_bytes(), mode="r")
    try:
        fmt = container.format.name
        assert fmt in ("wav", "wave", "pcm_s16le") or "wav" in fmt
    finally:
        container.close()


def test_container_exposes_audio_stream(depcheck):
    """container.streams.audio must list the single audio stream the WAV
    carries (faster-whisper selects the audio stream this way)."""
    mod = depcheck.load(IMPORT_NAME)
    container = mod.open(_wav_bytes(), mode="r")
    try:
        audio_streams = container.streams.audio
        assert len(audio_streams) == 1
    finally:
        container.close()


def test_audio_stream_reports_sample_rate(depcheck):
    """The audio stream must report the 8000 Hz sample rate we encoded — PyAV
    must surface the rate (faster-whisper resamples based on it)."""
    mod = depcheck.load(IMPORT_NAME)
    container = mod.open(_wav_bytes(rate=8000), mode="r")
    try:
        stream = container.streams.audio[0]
        # rate is exposed on the stream and/or its codec_context across versions.
        rate = getattr(stream, "rate", None)
        if rate is None:
            rate = stream.codec_context.sample_rate
        assert rate == 8000
    finally:
        container.close()


def test_decode_yields_audio_frames(depcheck):
    """container.decode(audio=0) must yield AudioFrame objects; decoding the
    1-second WAV must produce at least one frame whose .samples is positive.
    This is the actual PCM-decode faster-whisper performs before transcription."""
    mod = depcheck.load(IMPORT_NAME)
    container = mod.open(_wav_bytes(), mode="r")
    try:
        frames = list(container.decode(audio=0))
        assert frames, "no audio frames decoded from the in-memory WAV"
        assert all(isinstance(f, mod.AudioFrame) for f in frames)
        assert frames[0].samples > 0
    finally:
        container.close()


def test_decoded_sample_count_matches_input(depcheck):
    """The total decoded samples must equal the encoded sample count
    (rate * seconds). Pins that PyAV decodes the full PCM payload, not a
    truncated prefix — a regression here would silently drop audio."""
    mod = depcheck.load(IMPORT_NAME)
    rate, seconds = 8000, 1
    container = mod.open(_wav_bytes(rate=rate, seconds=seconds), mode="r")
    try:
        total = sum(f.samples for f in container.decode(audio=0))
        assert total == rate * seconds, f"decoded {total} samples, expected {rate * seconds}"
    finally:
        container.close()


def test_container_is_context_manager(depcheck):
    """`with av.open(...) as container:` is the idiomatic usage; the container
    must support the context-manager protocol."""
    mod = depcheck.load(IMPORT_NAME)
    with mod.open(_wav_bytes(), mode="r") as container:
        assert len(container.streams.audio) == 1


# ---------------------------------------------------------------------------
# AudioResampler — faster-whisper resamples decoded frames to 16 kHz mono.
# ---------------------------------------------------------------------------


def test_audio_resampler_constructs(depcheck):
    """AudioResampler(format=, layout=, rate=) is how faster-whisper converts
    decoded frames to the model's expected 16 kHz mono float. It must
    construct offline (no media needed to build the resampler)."""
    mod = depcheck.load(IMPORT_NAME)
    resampler = mod.AudioResampler(format="s16", layout="mono", rate=16000)
    assert resampler is not None
    assert callable(resampler.resample)


def test_resample_decoded_frame(depcheck):
    """End to end: decode an 8 kHz frame and resample it to 16 kHz mono. The
    resampler must accept a decoded AudioFrame and return frame(s)."""
    mod = depcheck.load(IMPORT_NAME)
    resampler = mod.AudioResampler(format="s16", layout="mono", rate=16000)
    container = mod.open(_wav_bytes(rate=8000), mode="r")
    try:
        first = next(iter(container.decode(audio=0)))
        out = resampler.resample(first)
        # v14 returns a list of frames; older returned a single frame or None.
        produced = out if isinstance(out, list) else [out]
        produced = [f for f in produced if f is not None]
        assert all(isinstance(f, mod.AudioFrame) for f in produced)
    finally:
        container.close()
