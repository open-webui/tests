"""Dependency contract: transformers (HuggingFace, import name ``transformers``).

Open WebUI uses transformers narrowly and lazily: ``routers/audio.py``'s
``load_speech_pipeline`` does ``from transformers import pipeline`` and then
``pipeline("text-to-speech", "microsoft/speecht5_tts")`` to build a local
TTS pipeline. That is the only *direct* transformers call in the backend
(``sentence_transformers`` in ``retrieval.py`` is a different package).

transformers is enormous and pulls heavy ML backends; downloading a model
is slow, networked, and non-deterministic. This contract therefore stays
deliberately light and fully OFFLINE:

  - assert the package imports and reports a version;
  - pin the ``pipeline`` factory exists, is callable, and keeps its
    positional contract (``task`` first, ``model`` second) so
    ``pipeline("text-to-speech", "...model...")`` keeps resolving;
  - pin the small set of stable top-level symbols any transformers consumer
    relies on (the ``Auto*`` classes, ``PreTrainedTokenizerBase``,
    ``pipeline``) so an import-surface regression is caught;
  - a single light *tokenizer smoke* that builds an in-memory
    ``PreTrainedTokenizerFast`` from a tiny hand-made vocab (via the
    ``tokenizers`` lib) and round-trips encode/decode — exercising the
    tokenizer machinery with NO model download and NO network. If
    ``tokenizers`` isn't importable, that test skips cleanly.

NOTHING here downloads a model or hits the Hub. A transformers bump that
removed ``pipeline`` or reshuffled its signature would fail loudly instead
of surfacing as a TTS error at runtime.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "transformers"
DIST_NAME = "transformers"

# The only symbol the backend imports directly, plus the stable core surface
# any transformers consumer expects to remain present.
USED_SYMBOLS = ["pipeline"]

CORE_SURFACE = [
    "pipeline",
    "AutoTokenizer",
    "AutoConfig",
    "AutoModel",
    "PreTrainedTokenizerBase",
    "PreTrainedTokenizerFast",
]


# --------------------------------------------------------------------------- #
# Import / version
# --------------------------------------------------------------------------- #


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "transformers"


def test_version_reported(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    # transformers exposes __version__ directly; the dist name matches.
    assert getattr(mod, "__version__", None) is not None
    assert depcheck.dist_version(DIST_NAME) is not None


# --------------------------------------------------------------------------- #
# Symbol existence (API surface)
# --------------------------------------------------------------------------- #


def test_used_symbols_exist(depcheck):
    """The symbol the backend imports must exist."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_core_surface_exists(depcheck):
    """The stable top-level classes/factories. transformers uses a lazy
    module loader, so these resolve as module attributes on access."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, CORE_SURFACE)


def test_pipeline_is_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "pipeline")


# --------------------------------------------------------------------------- #
# pipeline() signature (the positional contract audio.py relies on)
# --------------------------------------------------------------------------- #


def test_pipeline_task_is_first_param(depcheck):
    """audio.py calls pipeline("text-to-speech", model). The first parameter
    must be `task` and be passable positionally."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.pipeline)
    params = list(sig.parameters.values())
    assert params, "pipeline has no parameters"
    first = params[0]
    assert first.name == "task"
    assert first.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )


def test_pipeline_model_is_second_param(depcheck):
    """The model id is passed positionally as the second argument."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.pipeline)
    params = list(sig.parameters.values())
    assert len(params) >= 2
    second = params[1]
    assert second.name == "model"
    assert second.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )


def test_pipeline_accepts_model_keyword(depcheck):
    """Also pin `task`/`model` as accepted keyword names (some callers pass
    model= explicitly)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.pipeline, ["task", "model"])


# --------------------------------------------------------------------------- #
# Auto* classes are classes with from_pretrained factories
# --------------------------------------------------------------------------- #


def test_auto_classes_have_from_pretrained(depcheck):
    """AutoTokenizer / AutoConfig / AutoModel are the documented entry points;
    each must expose a from_pretrained classmethod (we do NOT call it — no
    download)."""
    mod = depcheck.load(IMPORT_NAME)
    for name in ("AutoTokenizer", "AutoConfig", "AutoModel"):
        cls = getattr(mod, name)
        names = set(dir(cls))
        assert "from_pretrained" in names, f"{name}.from_pretrained missing"
        assert callable(getattr(cls, "from_pretrained"))


def test_pretrained_tokenizer_base_methods(depcheck):
    """PreTrainedTokenizerBase is the interface backend-adjacent code relies on
    (encode/decode/__call__). Pin the method names exist on the class without
    instantiating anything."""
    mod = depcheck.load(IMPORT_NAME)
    base = mod.PreTrainedTokenizerBase
    names = set(dir(base))
    for meth in ("encode", "decode", "__call__", "from_pretrained"):
        assert meth in names, f"PreTrainedTokenizerBase.{meth} missing"


# --------------------------------------------------------------------------- #
# Light tokenizer smoke — fully offline, NO model download
# --------------------------------------------------------------------------- #


def test_offline_fast_tokenizer_roundtrip(depcheck):
    """Build a PreTrainedTokenizerFast from a tiny in-memory vocab via the
    `tokenizers` lib and round-trip encode/decode. This exercises the
    tokenizer wrapper machinery with NO network and NO model download.

    Skips cleanly if `tokenizers` isn't importable.
    """
    mod = depcheck.load(IMPORT_NAME)
    tk_lib = depcheck.try_load("tokenizers")
    if tk_lib is None:
        pytest.skip("`tokenizers` not importable; skipping offline tokenizer smoke")

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab = {"[UNK]": 0, "hello": 1, "world": 2, "open": 3, "webui": 4}
    inner = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    inner.pre_tokenizer = Whitespace()

    fast = mod.PreTrainedTokenizerFast(tokenizer_object=inner, unk_token="[UNK]")

    enc = fast("hello world")
    assert "input_ids" in enc
    assert enc["input_ids"] == [1, 2]

    # Unknown token maps to the UNK id.
    enc2 = fast("hello zzz")
    assert enc2["input_ids"][0] == 1
    assert enc2["input_ids"][1] == 0  # [UNK]

    # Decode round-trips the known tokens.
    decoded = fast.decode([1, 2])
    assert "hello" in decoded and "world" in decoded


def test_offline_tokenizer_is_pretrained_base(depcheck):
    """The fast tokenizer we built must be an instance of the base interface
    the rest of the ecosystem programs against."""
    mod = depcheck.load(IMPORT_NAME)
    if depcheck.try_load("tokenizers") is None:
        pytest.skip("`tokenizers` not importable")

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    inner = Tokenizer(WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]"))
    fast = mod.PreTrainedTokenizerFast(tokenizer_object=inner, unk_token="[UNK]")
    assert isinstance(fast, mod.PreTrainedTokenizerBase)
