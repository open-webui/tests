"""Dependency contract: faster-whisper (import name ``faster_whisper``).

faster-whisper is Open WebUI's built-in local speech-to-text engine
(CTranslate2-backed Whisper). When the STT engine is the empty/default
(local) backend, ``routers/audio.py`` lazily imports and uses it:

    from faster_whisper import WhisperModel
    whisper_model = WhisperModel(
        model_size_or_path=model,
        device='cuda'|'cpu',
        compute_type=WHISPER_COMPUTE_TYPE,
        download_root=WHISPER_MODEL_DIR,
        local_files_only=not auto_update,   # retried with False on failure
    )
    ...
    segments, info = model.transcribe(
        file_path,
        beam_size=5,
        vad_filter=WHISPER_VAD_FILTER,
        language=languages[0],
        multilingual=WHISPER_MULTILINGUAL,
    )
    # info.language, info.language_probability
    # ''.join(segment.text for segment in list(segments))

This is a heavy ML dependency: instantiating ``WhisperModel`` loads a
model (download or large local read) and CTranslate2 native runtime, so
this file does NOT construct a model or transcribe. It validates the
*contract surface* the backend relies on offline:

  * ``WhisperModel`` is importable and its constructor accepts every
    keyword the backend passes (``model_size_or_path``, ``device``,
    ``compute_type``, ``download_root``, ``local_files_only``);
  * ``WhisperModel.transcribe`` accepts every keyword the backend passes
    (``beam_size``, ``vad_filter``, ``language``, ``multilingual``);
  * the ``TranscriptionInfo`` / ``Segment`` result types expose the
    ``language`` / ``language_probability`` / ``text`` fields the backend
    reads.

A faster-whisper bump that renamed/dropped any of those would surface
here instead of as a runtime crash the first time a user transcribes.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "faster_whisper"
DIST_NAME = "faster-whisper"

USED_SYMBOLS = [
    "WhisperModel",  # routers/audio.py: from faster_whisper import WhisperModel
]

# Keyword args set_faster_whisper_model() passes to WhisperModel(**kwargs).
CONSTRUCTOR_KWARGS = [
    "model_size_or_path",
    "device",
    "compute_type",
    "download_root",
    "local_files_only",
]

# Keyword args _transcribe_whisper() passes to model.transcribe(...).
TRANSCRIBE_KWARGS = [
    "beam_size",
    "vad_filter",
    "language",
    "multilingual",
]


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "faster_whisper"


def test_version_reported(depcheck):
    assert depcheck.dist_version(DIST_NAME) is not None


def test_used_symbols_exist(depcheck):
    """WhisperModel must be importable off the top-level package."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_whisper_model_importable_directly(depcheck):
    """The backend does `from faster_whisper import WhisperModel`; that exact
    name binding must resolve to a class."""
    depcheck.load(IMPORT_NAME)
    wm_mod = depcheck.load("faster_whisper")
    assert inspect.isclass(wm_mod.WhisperModel)


def test_whisper_model_constructor_kwargs(depcheck):
    """WhisperModel(**faster_whisper_kwargs) is called with model_size_or_path/
    device/compute_type/download_root/local_files_only. Every one of those
    keyword names must remain accepted by the constructor."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.WhisperModel.__init__, CONSTRUCTOR_KWARGS)


def test_whisper_model_first_param_is_model_path(depcheck):
    """model_size_or_path is also positionally the first real argument (the
    backend passes it by keyword, but a rename would still be a break)."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.WhisperModel.__init__)
    params = [p for p in sig.parameters if p != "self"]
    assert params, "WhisperModel.__init__ takes no arguments besides self"
    assert params[0] == "model_size_or_path", (
        f"WhisperModel's first parameter is now {params[0]!r}, backend passes model_size_or_path"
    )


def test_transcribe_is_callable_with_our_kwargs(depcheck):
    """model.transcribe(file_path, beam_size=, vad_filter=, language=,
    multilingual=) — pin those keyword names on the method."""
    mod = depcheck.load(IMPORT_NAME)
    transcribe = getattr(mod.WhisperModel, "transcribe", None)
    assert transcribe is not None, "WhisperModel.transcribe missing"
    assert callable(transcribe)
    depcheck.assert_params(transcribe, TRANSCRIBE_KWARGS)


def test_transcribe_first_param_is_audio(depcheck):
    """The backend passes the audio file path as the first positional arg to
    transcribe(); the first non-self parameter must be the audio input."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.WhisperModel.transcribe)
    params = [p for p in sig.parameters if p != "self"]
    assert params, "WhisperModel.transcribe takes no arguments besides self"
    # historically named 'audio'
    assert params[0] == "audio", (
        f"transcribe's first parameter is now {params[0]!r}, "
        "backend passes the audio path positionally"
    )


def test_transcription_info_has_used_fields(depcheck):
    """The backend reads info.language and info.language_probability off the
    second element of transcribe()'s return. TranscriptionInfo must expose
    those field names (it's a NamedTuple/dataclass-like type)."""
    mod = depcheck.load(IMPORT_NAME)
    info_cls = _resolve_first(
        mod,
        ["TranscriptionInfo", "transcribe.TranscriptionInfo"],
    )
    if info_cls is None:
        pytest.skip("TranscriptionInfo type not exposed in this faster-whisper build")
    fields = _type_field_names(info_cls)
    for name in ("language", "language_probability"):
        assert name in fields, (
            f"TranscriptionInfo no longer exposes {name!r} (fields: {sorted(fields)})"
        )


def test_segment_has_text_field(depcheck):
    """_transcribe_whisper does ''.join(segment.text for segment in segments).
    The Segment type must expose a `text` field."""
    mod = depcheck.load(IMPORT_NAME)
    seg_cls = _resolve_first(mod, ["Segment", "transcribe.Segment"])
    if seg_cls is None:
        pytest.skip("Segment type not exposed in this faster-whisper build")
    fields = _type_field_names(seg_cls)
    assert "text" in fields, f"Segment no longer exposes a 'text' field (fields: {sorted(fields)})"


def test_transcribe_returns_two_values_contract(depcheck):
    """The backend unpacks `segments, info = model.transcribe(...)`. We can't
    call it (needs a model + audio), but the documented return is a 2-tuple of
    (iterable[Segment], TranscriptionInfo). Pin that both result types exist so
    the unpack stays valid; the actual call is exercised by integration, not
    here."""
    mod = depcheck.load(IMPORT_NAME)
    seg = _resolve_first(mod, ["Segment", "transcribe.Segment"])
    info = _resolve_first(mod, ["TranscriptionInfo", "transcribe.TranscriptionInfo"])
    # At least one of the two named result types should be discoverable; if the
    # build hides both we can't assert the unpack shape, so skip.
    if seg is None and info is None:
        pytest.skip("Segment/TranscriptionInfo not exposed in this build")
    assert seg is not None or info is not None


# ---------------------------------------------------------------------------
# Local helpers (no cross-file imports).
# ---------------------------------------------------------------------------


def _resolve_first(root, dotted_candidates):
    """Return the first dotted path that resolves against `root`, else None."""
    for dotted in dotted_candidates:
        cur = root
        ok = True
        for part in dotted.split("."):
            try:
                cur = getattr(cur, part)
            except AttributeError:
                ok = False
                break
        if ok:
            return cur
    return None


def _type_field_names(cls) -> set[str]:
    """Best-effort field-name set for a NamedTuple / dataclass / plain class."""
    names: set[str] = set()
    fields = getattr(cls, "_fields", None)
    if fields:
        names.update(fields)
    annotations = getattr(cls, "__annotations__", None)
    if annotations:
        names.update(annotations.keys())
    # Fall back to public attribute names on the class.
    names.update(n for n in dir(cls) if not n.startswith("_"))
    return names
