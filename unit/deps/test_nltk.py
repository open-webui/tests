"""Dependency contract: nltk.

nltk's role in Open WebUI is narrow but load-bearing for RAG document
ingestion. The backend's only *direct* reference to nltk lives in the
launchers, not the Python source: ``start.sh``, ``backend/start_windows.bat``
and the ``Dockerfile`` all run

    python -c "import nltk; nltk.download('punkt_tab')"

to pre-fetch the Punkt sentence-tokenizer data into the image/host before the
server starts. That download exists so document extraction works reliably in
airgapped environments after container restarts (the tokenizer data is bundled
rather than fetched lazily on first use; see CHANGELOG / PR #21165, issue
#21150). The *consumer* of that data is sentence/word tokenization
(``nltk.sent_tokenize`` / ``nltk.word_tokenize`` and the underlying
``nltk.tokenize.PunktSentenceTokenizer``), which langchain's text splitters can
invoke while chunking uploaded documents during ingestion.

So the slice this module pins is exactly:

  * ``nltk.download`` — the symbol the launchers call, and that it still
    accepts a package id as its first positional argument (``'punkt_tab'``).
  * ``nltk.data`` — ``find`` / ``load`` / ``path``, the lookup machinery the
    bundled data is resolved through.
  * ``nltk.sent_tokenize`` / ``nltk.word_tokenize`` and
    ``nltk.tokenize.PunktSentenceTokenizer`` — the tokenizer surface the
    downloaded ``punkt_tab`` data is *for*.

The behavioural contracts are split in two so the suite is deterministic and
fully offline:

  * Symbol/signature checks always run (they need no data files and no
    network).
  * Tokenization behaviour runs only if the required Punkt data is *already*
    present in this environment (probed via ``nltk.data.find`` in a
    try/except). If it is absent the behavioural cases SKIP cleanly — they
    must never trigger ``nltk.download``, which would hit the network and make
    the suite non-deterministic.

A nltk bump (here 3.9.3 -> 3.9.4) that removed/renamed any of this, or broke
the ``download('punkt_tab')`` call shape or the ``list[str]`` tokenizer output,
fails loudly here instead of surfacing as a broken container start or garbled
RAG chunking deep in the ingestion path.

Exemplar for the unit/deps/ pattern: symbol-existence checks (API surface) +
offline behavioural contracts. Uses the ``depcheck`` fixture from
unit/deps/conftest.py.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "nltk"
DIST_NAME = "nltk"

# Symbols the Open WebUI backend (via the launchers + the documented tokenizer
# API the downloaded data feeds) relies on. ``download`` is the only call the
# repo makes directly; the tokenize/data surface is what that download exists to
# enable and is part of nltk's stable public API a bump must not silently drop.
USED_SYMBOLS = [
    "download",
    "sent_tokenize",
    "word_tokenize",
    "data",
    "data.find",
    "data.load",
    "data.path",
    "tokenize",
    "tokenize.sent_tokenize",
    "tokenize.word_tokenize",
    "tokenize.PunktSentenceTokenizer",
]

# The data package the launchers fetch and the exact resource the English
# sentence tokenizer resolves in nltk 3.9.x. ``sent_tokenize`` loads
# ``tokenizers/punkt_tab/<language>/`` (the new tabular Punkt format) rather
# than the legacy ``tokenizers/punkt`` pickle, so the punkt_tab resource is the
# one that actually gates tokenization behaviour.
PUNKT_TAB_RESOURCE = "tokenizers/punkt_tab/english/"
PUNKT_LEGACY_RESOURCE = "tokenizers/punkt"

# Deterministic, offline sample text. Four unambiguous sentence terminators so
# a correct Punkt tokenizer must yield exactly four sentences.
_SAMPLE_TEXT = "Hello world. This is a test! Is it working? Yes."
_EXPECTED_SENTENCES = [
    "Hello world.",
    "This is a test!",
    "Is it working?",
    "Yes.",
]
_WORD_SAMPLE = "Hello, world! It works."


# --------------------------------------------------------------------------- #
# Data-availability probe (no network, no download).
# --------------------------------------------------------------------------- #


def _punkt_data_available(mod, resource: str) -> bool:
    """True iff `resource` is already on disk for this env's nltk data path.

    Pure lookup: ``nltk.data.find`` searches ``nltk.data.path`` and raises
    ``LookupError`` if the package is not installed locally. It never downloads.
    Any other exception is also treated as "absent" so the behavioural tests
    skip rather than error on an odd environment.
    """
    try:
        mod.data.find(resource)
        return True
    except LookupError:
        return False
    except Exception:
        return False


def _require_punkt(depcheck):
    """Load nltk and return it, skipping if Punkt sentence data is absent.

    Skips (does not fail, does not download) when neither the punkt_tab nor the
    legacy punkt data is present locally, so data-dependent behaviour is only
    asserted where it can run fully offline.
    """
    mod = depcheck.load(IMPORT_NAME)
    if _punkt_data_available(mod, PUNKT_TAB_RESOURCE) or _punkt_data_available(
        mod, PUNKT_LEGACY_RESOURCE
    ):
        return mod
    pytest.skip(
        "Punkt tokenizer data not present locally (tokenizers/punkt_tab or "
        "tokenizers/punkt); skipping tokenize behaviour to stay offline "
        "(refusing to call nltk.download, which hits the network)."
    )


# --------------------------------------------------------------------------- #
# Import / symbol-surface tests (always run; no data files, no network).
# --------------------------------------------------------------------------- #


def test_import(depcheck):
    """nltk must import and identify as the nltk package."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "nltk"


def test_used_symbols_exist(depcheck):
    """Every nltk symbol the codebase relies on must still resolve.

    Covers the launcher's ``nltk.download`` plus the tokenizer/data surface the
    downloaded punkt_tab data feeds. Reported all-at-once so a bump that drops
    several is one clear failure.
    """
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_download_is_callable(depcheck):
    """The launchers' only nltk call is ``nltk.download('punkt_tab')``."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "download")


def test_sent_tokenize_is_callable(depcheck):
    """sent_tokenize must remain a top-level callable."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "sent_tokenize")


def test_word_tokenize_is_callable(depcheck):
    """word_tokenize must remain a top-level callable."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "word_tokenize")


def test_data_find_and_load_callable(depcheck):
    """nltk.data.find / nltk.data.load — the lookup machinery the bundled data
    is resolved through — must stay callable."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "data.find")
    depcheck.assert_callable(mod, "data.load")


def test_punkt_sentence_tokenizer_is_class(depcheck):
    """nltk.tokenize.PunktSentenceTokenizer (what sent_tokenize uses under the
    hood) must remain an instantiable class."""
    mod = depcheck.load(IMPORT_NAME)
    cls = depcheck.resolve(mod, "tokenize.PunktSentenceTokenizer")
    assert inspect.isclass(cls), "tokenize.PunktSentenceTokenizer is not a class"


def test_tokenize_submodule_reexports(depcheck):
    """nltk.tokenize must re-export sent_tokenize/word_tokenize (the canonical
    import location ``from nltk.tokenize import sent_tokenize``)."""
    depcheck.load(IMPORT_NAME)  # skip if nltk absent
    sub = depcheck.load("nltk.tokenize")
    for name in ("sent_tokenize", "word_tokenize", "PunktSentenceTokenizer"):
        assert hasattr(sub, name), f"nltk.tokenize.{name} missing"


def test_data_path_is_list_of_str(depcheck):
    """nltk.data.path is the search-path list nltk.download writes into and
    nltk.data.find reads from; it must be a list of strings."""
    mod = depcheck.load(IMPORT_NAME)
    paths = mod.data.path
    assert isinstance(paths, list), f"nltk.data.path is {type(paths)!r}, not a list"
    assert all(isinstance(p, str) for p in paths), (
        f"nltk.data.path contains non-str entries: {paths!r}"
    )


def test_version_attribute_is_string(depcheck):
    """nltk.__version__ is a non-empty string."""
    mod = depcheck.load(IMPORT_NAME)
    assert isinstance(mod.__version__, str) and mod.__version__


def test_dist_version_resolvable(depcheck):
    """The installed distribution version is resolvable (bump tooling agrees)."""
    assert depcheck.dist_version(DIST_NAME) is not None


# --------------------------------------------------------------------------- #
# Signature tests (always run; introspection only).
# --------------------------------------------------------------------------- #


def test_download_accepts_package_id_positionally(depcheck):
    """``nltk.download('punkt_tab')`` is the exact launcher call: download must
    accept the package id as its first positional argument.

    We don't pin the parameter *name* (it has been ``info_or_id``), only that a
    leading positional parameter exists so the positional call keeps working. A
    ``download_dir`` / ``quiet`` keyword is also part of the stable surface and
    asserted softly when introspectable.
    """
    mod = depcheck.load(IMPORT_NAME)
    try:
        sig = inspect.signature(mod.download)
    except (TypeError, ValueError):
        pytest.skip("nltk.download has no introspectable signature")
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_var_pos = any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
    assert positional or has_var_pos, (
        f"nltk.download exposes no positional parameter for the package id (sig: {sig})"
    )


def test_sent_tokenize_signature(depcheck):
    """sent_tokenize(text, language='english', ...): the leading ``text``
    positional and the ``language`` knob langchain/NLTK rely on must remain."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.sent_tokenize, ["language"])
    try:
        sig = inspect.signature(mod.sent_tokenize)
    except (TypeError, ValueError):
        pytest.skip("nltk.sent_tokenize has no introspectable signature")
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    assert positional, f"sent_tokenize exposes no positional text parameter (sig: {sig})"


def test_word_tokenize_signature(depcheck):
    """word_tokenize(text, language='english', preserve_line=False): the
    ``language`` and ``preserve_line`` knobs must remain accepted."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.word_tokenize, ["language", "preserve_line"])


def test_data_find_signature(depcheck):
    """nltk.data.find(resource_name, paths=None) — the resource-name positional
    must remain so ``find('tokenizers/punkt_tab/...')`` keeps working."""
    mod = depcheck.load(IMPORT_NAME)
    try:
        sig = inspect.signature(mod.data.find)
    except (TypeError, ValueError):
        pytest.skip("nltk.data.find has no introspectable signature")
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    assert positional, f"nltk.data.find exposes no positional parameter (sig: {sig})"


# --------------------------------------------------------------------------- #
# Behavioural contract: the data-lookup machinery itself (offline, no download).
# --------------------------------------------------------------------------- #


def test_data_find_raises_lookuperror_for_missing_resource(depcheck):
    """nltk.data.find must raise LookupError (not some other type) for a
    resource that is not installed.

    This is the exact exception the lazy-load path and our ``_punkt_data_available``
    probe depend on to detect missing data without triggering a download. Uses a
    deliberately bogus resource name so the assertion holds regardless of which
    real packages happen to be present.
    """
    mod = depcheck.load(IMPORT_NAME)
    bogus = "tokenizers/__open_webui_nonexistent_tokenizer__/english/"
    with pytest.raises(LookupError):
        mod.data.find(bogus)


def test_data_find_does_not_hit_network_for_present_resource(depcheck):
    """When the punkt data *is* present, nltk.data.find resolves it to a path
    object/string purely from disk — no download. (Skips if data absent.)"""
    mod = _require_punkt(depcheck)
    resource = (
        PUNKT_TAB_RESOURCE
        if _punkt_data_available(mod, PUNKT_TAB_RESOURCE)
        else PUNKT_LEGACY_RESOURCE
    )
    found = mod.data.find(resource)
    assert found is not None
    # FileSystemPathPointer / ZipFilePathPointer both stringify to a usable path.
    assert str(found), f"nltk.data.find({resource!r}) returned an empty path: {found!r}"


# --------------------------------------------------------------------------- #
# Behavioural contract: tokenization output shape (offline; skip if no data).
#
# These are the contracts the bundled punkt_tab download exists to satisfy. They
# run ONLY when the data is already on disk; they never download.
# --------------------------------------------------------------------------- #


def test_sent_tokenize_returns_list_of_str(depcheck):
    """sent_tokenize(text) -> list[str]. langchain's NLTK-backed splitter joins
    these back together, so the list[str] shape is the load-bearing contract."""
    mod = _require_punkt(depcheck)
    out = mod.sent_tokenize(_SAMPLE_TEXT)
    assert isinstance(out, list), f"sent_tokenize returned {type(out)!r}, not list"
    assert out, "sent_tokenize returned an empty list for non-empty text"
    assert all(isinstance(s, str) for s in out), f"sent_tokenize returned non-str elements: {out!r}"


def test_sent_tokenize_splits_on_terminators(depcheck):
    """The four-sentence sample splits into exactly four sentences.

    Pins sane sentence segmentation (the behaviour RAG chunking relies on)
    without asserting Punkt's exact internal handling of edge cases — the sample
    uses unambiguous . ! ? terminators.
    """
    mod = _require_punkt(depcheck)
    out = mod.sent_tokenize(_SAMPLE_TEXT)
    assert out == _EXPECTED_SENTENCES, (
        f"sent_tokenize produced {out!r}, expected {_EXPECTED_SENTENCES!r}"
    )


def test_sent_tokenize_single_sentence(depcheck):
    """A single sentence with no internal terminator returns a one-element list
    carrying the (stripped) text — no over-splitting."""
    mod = _require_punkt(depcheck)
    out = mod.sent_tokenize("This is one whole sentence with no break")
    assert isinstance(out, list) and len(out) == 1, f"expected 1 sentence, got {out!r}"
    assert "one whole sentence" in out[0]


def test_sent_tokenize_empty_string_returns_empty_list(depcheck):
    """Empty input -> empty list (never raises) — ingestion may hand it empty
    page_content for blank pages."""
    mod = _require_punkt(depcheck)
    out = mod.sent_tokenize("")
    assert out == [], f"sent_tokenize('') returned {out!r}, expected []"


def test_word_tokenize_returns_list_of_str(depcheck):
    """word_tokenize(text) -> list[str] with punctuation split off as its own
    tokens (the classic Penn-Treebank-style behaviour)."""
    mod = _require_punkt(depcheck)
    out = mod.word_tokenize(_WORD_SAMPLE)
    assert isinstance(out, list) and out, f"word_tokenize returned {out!r}"
    assert all(isinstance(w, str) for w in out), f"word_tokenize returned non-str elements: {out!r}"
    # The alphabetic words must all be present as standalone tokens.
    for word in ("Hello", "world", "It", "works"):
        assert word in out, f"expected token {word!r} in {out!r}"


def test_word_tokenize_separates_punctuation(depcheck):
    """Punctuation is emitted as separate tokens, not glued to words — the
    property that makes downstream token counting meaningful."""
    mod = _require_punkt(depcheck)
    out = mod.word_tokenize(_WORD_SAMPLE)
    assert "," in out and "!" in out and "." in out, (
        f"word_tokenize did not split punctuation into standalone tokens: {out!r}"
    )


def test_sent_tokenize_is_deterministic(depcheck):
    """Tokenizing identical text twice yields identical output — RAG chunking
    assumes a stable segmentation for a given document."""
    mod = _require_punkt(depcheck)
    assert mod.sent_tokenize(_SAMPLE_TEXT) == mod.sent_tokenize(_SAMPLE_TEXT)


def test_punkt_sentence_tokenizer_instance_tokenize(depcheck):
    """A bare PunktSentenceTokenizer instance (no training) exposes .tokenize()
    returning list[str]. This is the underlying object sent_tokenize wraps; pin
    its minimal lifecycle (instantiate -> tokenize -> list[str])."""
    mod = depcheck.load(IMPORT_NAME)
    cls = depcheck.resolve(mod, "tokenize.PunktSentenceTokenizer")
    tokenizer = cls()
    assert callable(getattr(tokenizer, "tokenize", None)), (
        "PunktSentenceTokenizer instance has no callable .tokenize"
    )
    out = tokenizer.tokenize("Alpha beta gamma. Delta epsilon.")
    assert isinstance(out, list) and all(isinstance(s, str) for s in out), (
        f"PunktSentenceTokenizer().tokenize returned {out!r}"
    )
