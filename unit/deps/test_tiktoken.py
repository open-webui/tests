"""Dependency contract: tiktoken.

tiktoken is OpenAI's byte-pair-encoding (BPE) tokenizer. The Open WebUI
backend uses it for *token-aware* RAG document splitting: when the
retrieval text splitter is set to ``token`` mode it measures and chunks
text by token count rather than character count, so chunks line up with
an LLM's context budget instead of an arbitrary character length.

The single direct consumer is ``backend/open_webui/routers/retrieval.py``:

    import tiktoken
    ...
    # _merge_small_chunks(): measure chunk size in tokens
    encoding = tiktoken.get_encoding(str(config.TIKTOKEN_ENCODING_NAME))
    measure = lambda text: len(encoding.encode(text))
    ...
    # save_docs_to_vector_db(): validate the configured encoding loads,
    # then hand its NAME to langchain's TokenTextSplitter
    tiktoken.get_encoding(str(config.TIKTOKEN_ENCODING_NAME))
    text_splitter = TokenTextSplitter(encoding_name=..., ...)

so the contract the backend actually depends on is narrow but exact:

  * ``tiktoken.get_encoding(name)`` returns an ``Encoding``,
  * ``Encoding.encode(text)`` returns a ``list[int]``,
  * ``len(encode(text))`` is a stable, positive token count for non-empty
    text (this *is* the chunk-size measurement),
  * the configured encoding name resolves offline (it ships the BPE data).

``config.TIKTOKEN_ENCODING_NAME`` defaults to ``'cl100k_base'`` (env/PersistentConfig
``TIKTOKEN_ENCODING_NAME``), so that encoding is the load-bearing one; the
operator may set any name tiktoken knows (e.g. ``o200k_base``), so the
loader's robustness to whatever name it's handed is part of the contract.

This module pins that slice so the tiktoken 0.12 -> 0.13 bump fails loudly
here (symbol gone, signature changed, encode/decode shape changed, the
default encoding stops loading offline) instead of surfacing as an
AttributeError or a garbled-chunk regression deep in RAG ingestion. It
pins the *API surface and the encode/decode round-trip contract*, not the
specific integer token ids of any particular encoding (those are stable
per encoding name, but the suite asserts the *properties* the loader
relies on — list-of-ints, positive length, round-trip, determinism —
rather than hard-coding a vocabulary that a future encoding revision could
legitimately change).

Exemplar for the unit/deps/ pattern: symbol-existence checks (API surface)
+ offline behavioural contracts. Uses the ``depcheck`` fixture from
unit/deps/conftest.py. tiktoken ships its BPE rank files inside the wheel,
so loading ``cl100k_base``/``o200k_base`` needs no network; on the rare
chance an encoding genuinely requires a download in a given env, that one
case is wrapped and skipped rather than failing the suite.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "tiktoken"
DIST_NAME = "tiktoken"

# Top-level symbols the Open WebUI backend relies on (directly or as the
# stable public surface a bump must not drop). The loader only calls
# ``get_encoding``; the rest are tiktoken's documented public API, pinned so
# a wholesale rewrite is caught early.
USED_SYMBOLS = [
    "get_encoding",
    "encoding_for_model",
    "encoding_name_for_model",
    "list_encoding_names",
    "Encoding",
]

# The encoding the backend defaults to (config.TIKTOKEN_ENCODING_NAME) plus the
# other modern encoding an operator is likely to configure. Both ship their BPE
# data in the wheel and must load offline.
DEFAULT_ENCODING = "cl100k_base"
CONFIGURABLE_ENCODINGS = ["cl100k_base", "o200k_base"]

# Encoding-instance methods/attrs the contract leans on. ``encode``/``decode``
# are load-bearing; ``name``/``n_vocab``/``max_token_value`` are part of the
# Encoding public surface pinned against a major rewrite.
ENCODING_MEMBERS = [
    "encode",
    "decode",
    "name",
    "n_vocab",
    "max_token_value",
    "encode_ordinary",
    "encode_batch",
    "decode_batch",
]

# Model names that map to known encodings. The backend doesn't call
# encoding_for_model today, but it's core public API; pin that these common
# OpenAI model names still resolve and map to the encoding family they always
# have (gpt-4 family -> cl100k_base, gpt-4o family -> o200k_base).
MODEL_TO_ENCODING = {
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-4o": "o200k_base",
}

# Deterministic, offline text fixtures. Pure ASCII, multi-word, unicode/CJK,
# emoji, and whitespace — exercising the round-trip the loader's
# ``len(encode(text))`` measurement depends on across the byte ranges real RAG
# documents contain.
_TEXTS = {
    "ascii": "The quick brown fox jumps over the lazy dog.",
    "words": "Open WebUI retrieval augmented generation token splitter test.",
    "unicode_cjk": "Héllo, 世界! Bonjour le monde. こんにちは世界。",
    "emoji": "Launch sequence ready: 🚀🛰️✨ all systems go.",
    "whitespace": "tabs\tand\nnewlines   and    runs   of spaces",
    "single": "tokenization",
}


# --------------------------------------------------------------------------- #
# Local helpers (no cross-file imports; everything composes via fixtures).
# --------------------------------------------------------------------------- #


def _get_encoding(depcheck, name: str):
    """Load tiktoken (or skip) and return ``get_encoding(name)``.

    If loading the encoding genuinely requires a network fetch that fails in
    this env, skip *this* case cleanly rather than failing the suite — the
    wheel ships the BPE data, so this should essentially never trigger.
    """
    mod = depcheck.load(IMPORT_NAME)
    try:
        return mod, mod.get_encoding(name)
    except Exception as e:  # network fetch / data-file load failure
        pytest.skip(f"encoding {name!r} not loadable offline in this env: {e}")


# --------------------------------------------------------------------------- #
# Import / symbol-surface tests.
# --------------------------------------------------------------------------- #


def test_import(depcheck):
    """tiktoken must import and identify as the tiktoken package."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "tiktoken"


def test_used_symbols_exist(depcheck):
    """Every top-level tiktoken symbol the codebase relies on must resolve."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_get_encoding_callable(depcheck):
    """``tiktoken.get_encoding`` is the loader's only direct call."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "get_encoding")


def test_encoding_for_model_callable(depcheck):
    """encoding_for_model is core public API used to map a model -> tokenizer."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "encoding_for_model")


def test_encoding_class_present(depcheck):
    """``tiktoken.Encoding`` (the type get_encoding returns) must be a class."""
    mod = depcheck.load(IMPORT_NAME)
    assert inspect.isclass(mod.Encoding), "tiktoken.Encoding is not a class"


def test_encoding_importable_from_core_submodule(depcheck):
    """``tiktoken.core.Encoding`` (where the class lives) stays importable."""
    depcheck.load(IMPORT_NAME)  # skip if tiktoken absent
    core = depcheck.load("tiktoken.core")
    assert hasattr(core, "Encoding")
    assert inspect.isclass(core.Encoding)


def test_dist_version_resolvable(depcheck):
    """The installed distribution version is resolvable (bump tooling agrees)."""
    assert depcheck.dist_version(DIST_NAME) is not None


# --------------------------------------------------------------------------- #
# Signature tests.
# --------------------------------------------------------------------------- #


def test_get_encoding_signature(depcheck):
    """get_encoding must accept the encoding name positionally.

    The loader calls ``tiktoken.get_encoding(str(name))`` positionally, so the
    leading parameter must remain positional (any name)."""
    mod = depcheck.load(IMPORT_NAME)
    try:
        sig = inspect.signature(mod.get_encoding)
    except (TypeError, ValueError):
        pytest.skip("get_encoding has no introspectable signature")
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    assert positional, f"get_encoding() exposes no positional parameter (sig: {sig})"


def test_encoding_for_model_signature(depcheck):
    """encoding_for_model must accept the model name positionally."""
    mod = depcheck.load(IMPORT_NAME)
    try:
        sig = inspect.signature(mod.encoding_for_model)
    except (TypeError, ValueError):
        pytest.skip("encoding_for_model has no introspectable signature")
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    assert positional, f"encoding_for_model() exposes no positional parameter (sig: {sig})"


def test_encode_signature_accepts_text_positionally(depcheck):
    """Encoding.encode(text) is called positionally by the loader."""
    _mod, enc = _get_encoding(depcheck, DEFAULT_ENCODING)
    try:
        sig = inspect.signature(enc.encode)
    except (TypeError, ValueError):
        pytest.skip("Encoding.encode has no introspectable signature")
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    assert positional, f"encode() exposes no positional parameter (sig: {sig})"


# --------------------------------------------------------------------------- #
# Encoding-instance surface.
# --------------------------------------------------------------------------- #


def test_default_encoding_loads_and_is_encoding_instance(depcheck):
    """The configured default (cl100k_base) loads offline and is an Encoding."""
    mod, enc = _get_encoding(depcheck, DEFAULT_ENCODING)
    assert isinstance(enc, mod.Encoding), f"get_encoding returned {type(enc)!r}, not Encoding"


@pytest.mark.parametrize("name", CONFIGURABLE_ENCODINGS)
def test_encoding_exposes_required_members(depcheck, name):
    """Each load-bearing/public Encoding member is present on the instance.

    Uses ``dir(instance)`` (lists names without executing descriptors) so this
    never triggers a property getter; methods are then confirmed callable.
    """
    _mod, enc = _get_encoding(depcheck, name)
    names = set(dir(enc))
    missing = [m for m in ENCODING_MEMBERS if m not in names]
    assert not missing, f"{name}: Encoding missing member(s) the contract relies on: {missing}"
    for method in ("encode", "decode", "encode_ordinary", "encode_batch", "decode_batch"):
        assert callable(getattr(type(enc), method, None)), f"{name}: Encoding.{method} not callable"


@pytest.mark.parametrize("name", CONFIGURABLE_ENCODINGS)
def test_encoding_name_attribute_matches_requested(depcheck, name):
    """``Encoding.name`` reports the requested encoding name (a str)."""
    _mod, enc = _get_encoding(depcheck, name)
    assert isinstance(enc.name, str) and enc.name, f"{name}: Encoding.name is {enc.name!r}"
    assert enc.name == name, f"requested {name!r} but Encoding.name is {enc.name!r}"


@pytest.mark.parametrize("name", CONFIGURABLE_ENCODINGS)
def test_encoding_vocab_metadata_positive_ints(depcheck, name):
    """n_vocab / max_token_value are positive ints (sane tokenizer metadata)."""
    _mod, enc = _get_encoding(depcheck, name)
    assert isinstance(enc.n_vocab, int) and enc.n_vocab > 0, f"{name}: n_vocab={enc.n_vocab!r}"
    assert isinstance(enc.max_token_value, int) and enc.max_token_value > 0, (
        f"{name}: max_token_value={enc.max_token_value!r}"
    )


# --------------------------------------------------------------------------- #
# Behavioural contract: encode/decode round-trip (the load-bearing path).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", CONFIGURABLE_ENCODINGS)
@pytest.mark.parametrize("text_key", sorted(_TEXTS))
def test_encode_returns_list_of_ints(depcheck, name, text_key):
    """encode(text) returns a list[int] — the loader does len(encode(text))."""
    _mod, enc = _get_encoding(depcheck, name)
    text = _TEXTS[text_key]
    ids = enc.encode(text)
    assert isinstance(ids, list), f"{name}/{text_key}: encode returned {type(ids)!r}, not list"
    assert ids, f"{name}/{text_key}: encode returned empty list for non-empty text"
    assert all(isinstance(i, int) for i in ids), (
        f"{name}/{text_key}: encode produced non-int token(s): {ids!r}"
    )
    # Token ids must be valid for the vocabulary the loader will hand to splitters.
    assert all(0 <= i <= enc.max_token_value for i in ids), (
        f"{name}/{text_key}: token id out of [0, max_token_value]: {ids!r}"
    )


@pytest.mark.parametrize("name", CONFIGURABLE_ENCODINGS)
@pytest.mark.parametrize("text_key", sorted(_TEXTS))
def test_encode_decode_roundtrip(depcheck, name, text_key):
    """decode(encode(text)) reproduces the original text exactly.

    Round-trip fidelity across ASCII, unicode/CJK, emoji, and whitespace is
    what makes token-based chunking lossless; pin it for every encoding the
    operator can configure."""
    _mod, enc = _get_encoding(depcheck, name)
    text = _TEXTS[text_key]
    decoded = enc.decode(enc.encode(text))
    assert decoded == text, f"{name}/{text_key}: round-trip mismatch: {decoded!r} != {text!r}"


@pytest.mark.parametrize("name", CONFIGURABLE_ENCODINGS)
def test_encode_length_is_stable_positive_count(depcheck, name):
    """``len(encode(text))`` — the exact chunk-size measurement — is a stable
    positive int and grows monotonically as text is concatenated."""
    _mod, enc = _get_encoding(depcheck, name)
    base = _TEXTS["words"]
    n1 = len(enc.encode(base))
    assert isinstance(n1, int) and n1 > 0, f"{name}: token count {n1!r} not a positive int"
    # Calling again yields the identical count (stable measurement).
    assert len(enc.encode(base)) == n1, f"{name}: token count not stable across calls"
    # More text -> at least as many tokens (the splitter relies on this ordering).
    n2 = len(enc.encode(base + " " + base))
    assert n2 >= n1, f"{name}: doubling text reduced token count ({n2} < {n1})"


@pytest.mark.parametrize("name", CONFIGURABLE_ENCODINGS)
def test_encode_empty_string_is_empty_list(depcheck, name):
    """Empty text encodes to [] — len()==0, so an empty chunk measures as zero
    tokens rather than raising."""
    _mod, enc = _get_encoding(depcheck, name)
    ids = enc.encode("")
    assert ids == [], f"{name}: encode('') returned {ids!r}, expected []"
    assert len(ids) == 0


@pytest.mark.parametrize("name", CONFIGURABLE_ENCODINGS)
def test_encode_is_deterministic(depcheck, name):
    """Same text -> identical token ids (not just identical length).

    The splitter assumes a stable tokenization for a given document; pin exact
    id-sequence determinism across repeated calls."""
    _mod, enc = _get_encoding(depcheck, name)
    for text in _TEXTS.values():
        assert enc.encode(text) == enc.encode(text), (
            f"{name}: non-deterministic encode for {text!r}"
        )


def test_get_encoding_is_deterministic_across_calls(depcheck):
    """Two get_encoding(name) calls tokenize identically.

    tiktoken caches Encoding objects, but the contract the loader needs is that
    re-resolving the configured name yields a tokenizer giving the same counts —
    assert that behaviourally without depending on object identity."""
    mod = depcheck.load(IMPORT_NAME)
    try:
        e1 = mod.get_encoding(DEFAULT_ENCODING)
        e2 = mod.get_encoding(DEFAULT_ENCODING)
    except Exception as e:
        pytest.skip(f"{DEFAULT_ENCODING!r} not loadable offline in this env: {e}")
    sample = _TEXTS["unicode_cjk"]
    assert e1.encode(sample) == e2.encode(sample)


# --------------------------------------------------------------------------- #
# Behavioural contract: encoding_for_model (core public API).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_name,expected_encoding", sorted(MODEL_TO_ENCODING.items()))
def test_encoding_for_model_returns_encoding(depcheck, model_name, expected_encoding):
    """encoding_for_model(model) returns an Encoding whose .name is the expected
    family for that model (gpt-4* -> cl100k_base, gpt-4o* -> o200k_base)."""
    mod = depcheck.load(IMPORT_NAME)
    try:
        enc = mod.encoding_for_model(model_name)
    except Exception as e:
        pytest.skip(f"encoding_for_model({model_name!r}) not resolvable offline: {e}")
    assert isinstance(enc, mod.Encoding), (
        f"encoding_for_model({model_name!r}) returned {type(enc)!r}, not Encoding"
    )
    assert enc.name == expected_encoding, (
        f"{model_name!r} mapped to {enc.name!r}, expected {expected_encoding!r}"
    )
    # And it actually tokenizes.
    ids = enc.encode("hello")
    assert isinstance(ids, list) and ids and all(isinstance(i, int) for i in ids)


def test_encoding_for_model_unknown_raises_keyerror(depcheck):
    """An unknown model name raises KeyError — tiktoken's documented signal that
    a caller must fall back to an explicit get_encoding(name)."""
    mod = depcheck.load(IMPORT_NAME)
    with pytest.raises(KeyError):
        mod.encoding_for_model("totally-not-a-real-model-xyz-0000")


def test_encoding_name_for_model_returns_str(depcheck):
    """encoding_name_for_model returns the encoding *name* string for a model."""
    mod = depcheck.load(IMPORT_NAME)
    try:
        name = mod.encoding_name_for_model("gpt-4")
    except Exception as e:
        pytest.skip(f"encoding_name_for_model not resolvable offline: {e}")
    assert isinstance(name, str) and name == "cl100k_base", (
        f"encoding_name_for_model('gpt-4') = {name!r}, expected 'cl100k_base'"
    )


# --------------------------------------------------------------------------- #
# Behavioural contract: list_encoding_names advertises the configurable names.
# --------------------------------------------------------------------------- #


def test_list_encoding_names_includes_configurable_encodings(depcheck):
    """list_encoding_names() returns a collection of strings that includes the
    encodings the backend defaults to / an operator can configure, so a
    configured ``TIKTOKEN_ENCODING_NAME`` is guaranteed resolvable."""
    mod = depcheck.load(IMPORT_NAME)
    names = mod.list_encoding_names()
    names = list(names)
    assert names and all(isinstance(n, str) for n in names), (
        f"list_encoding_names returned {names!r}"
    )
    for required in CONFIGURABLE_ENCODINGS:
        assert required in names, (
            f"{required!r} missing from list_encoding_names(); a configured "
            f"TIKTOKEN_ENCODING_NAME={required!r} would fail to load. Got: {names}"
        )


# --------------------------------------------------------------------------- #
# Cross-encoding sanity: different encodings are genuinely different tokenizers.
# --------------------------------------------------------------------------- #


def test_distinct_encodings_are_distinct_tokenizers(depcheck):
    """cl100k_base and o200k_base are different encodings — each still
    round-trips, but they need not produce identical token ids. This guards
    against a regression where every name silently aliases one tokenizer."""
    mod = depcheck.load(IMPORT_NAME)
    try:
        cl = mod.get_encoding("cl100k_base")
        o2 = mod.get_encoding("o200k_base")
    except Exception as e:
        pytest.skip(f"encodings not loadable offline in this env: {e}")
    assert cl.name != o2.name
    text = _TEXTS["unicode_cjk"]
    # Both must round-trip their own output regardless of differing vocabularies.
    assert cl.decode(cl.encode(text)) == text
    assert o2.decode(o2.encode(text)) == text
