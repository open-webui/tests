"""Dependency contract: rank-bm25 (import name ``rank_bm25``).

``rank-bm25`` provides the BM25 lexical-ranking implementation behind Open
WebUI's hybrid retrieval. The backend does not import it directly; it
reaches it through LangChain's ``BM25Retriever``
(``from langchain_community.retrievers import BM25Retriever`` in
``retrieval/utils.py``), constructed via ``BM25Retriever.from_texts(...)``
and combined with vector search in an ``EnsembleRetriever`` weighted by
``HYBRID_BM25_WEIGHT``. ``BM25Retriever`` instantiates ``rank_bm25.BM25Okapi``
internally and uses ``get_scores`` / ``get_top_n`` to rank documents for a
query.

So the contract the backend relies on is ``BM25Okapi``'s ranking
behaviour. This module pins the API surface and exercises the *actual
ranking semantics* OFFLINE and deterministically (no network, no model):
construction from a tokenized corpus, ``get_scores`` returning a per-doc
score array that ranks a matching document above non-matching ones, term
specificity (a rarer term scores higher), and ``get_top_n`` returning the
best documents in rank order.

A rank-bm25 bump that renamed ``BM25Okapi``/``get_scores``/``get_top_n`` or
changed the scoring so a relevant document no longer outranks an irrelevant
one would fail here instead of silently degrading hybrid-search quality.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "rank_bm25"
DIST_NAME = "rank-bm25"

USED_SYMBOLS = ["BM25Okapi", "BM25"]
# The full family the library ships (langchain selects BM25Okapi by default).
VARIANTS = ["BM25Okapi", "BM25L", "BM25Plus"]

# A small, unambiguous tokenized corpus for ranking assertions.
CORPUS = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "in", "the", "park"],
    ["cats", "and", "dogs", "are", "common", "pets"],
    ["quantum", "entanglement", "is", "a", "physics", "phenomenon"],
]


def _BM25Okapi(depcheck):
    return depcheck.resolve(depcheck.load(IMPORT_NAME), "BM25Okapi")


def _scores_as_list(scores):
    """Coerce a numpy array or list of scores to a plain Python list."""
    return [float(x) for x in scores]


# --------------------------------------------------------------------------- #
# Import / version
# --------------------------------------------------------------------------- #


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "rank_bm25"


def test_version_reported(depcheck):
    assert depcheck.dist_version(DIST_NAME) is not None


# --------------------------------------------------------------------------- #
# Symbol existence (API surface)
# --------------------------------------------------------------------------- #


def test_used_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_variants_exist(depcheck):
    """All BM25 variants the library documents must remain importable, so a
    consumer that selects a non-default scorer keeps working."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, VARIANTS)


def test_bm25okapi_is_class(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert inspect.isclass(mod.BM25Okapi)


def test_bm25okapi_methods_exist(depcheck):
    """BM25Retriever calls get_scores and get_top_n. Pin both exist + callable."""
    mod = depcheck.load(IMPORT_NAME)
    names = set(dir(mod.BM25Okapi))
    for meth in ("get_scores", "get_top_n"):
        assert meth in names, f"BM25Okapi.{meth} missing"
        assert callable(getattr(mod.BM25Okapi, meth))


def test_constructor_signature(depcheck):
    """BM25Okapi(corpus, tokenizer=None, k1=, b=, epsilon=). The corpus is the
    first positional; the tuning params are keyword. Pin those names."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.BM25Okapi.__init__)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    assert params, "BM25Okapi.__init__ takes no params besides self"
    assert params[0].name == "corpus"
    depcheck.assert_params(mod.BM25Okapi.__init__, ["corpus", "k1", "b"])


def test_get_top_n_signature(depcheck):
    """get_top_n(query, documents, n=5). Pin the param names BM25Retriever
    passes (it calls get_top_n(tokenized_query, docs, n=k))."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.BM25Okapi.get_top_n, ["query", "documents", "n"])


# --------------------------------------------------------------------------- #
# Behavioural: construction
# --------------------------------------------------------------------------- #


def test_construct_from_tokenized_corpus(depcheck):
    """BM25Okapi takes a list of tokenized documents (list[list[str]]) — the
    shape BM25Retriever feeds it after tokenizing texts."""
    BM25Okapi = _BM25Okapi(depcheck)
    bm = BM25Okapi(CORPUS)
    assert bm is not None
    # corpus_size reflects the number of documents indexed.
    assert getattr(bm, "corpus_size", len(CORPUS)) == len(CORPUS)


# --------------------------------------------------------------------------- #
# Behavioural: ranking semantics (the contract that actually matters)
# --------------------------------------------------------------------------- #


def test_get_scores_length_matches_corpus(depcheck):
    """get_scores returns one score per document."""
    BM25Okapi = _BM25Okapi(depcheck)
    bm = BM25Okapi(CORPUS)
    scores = _scores_as_list(bm.get_scores(["cat"]))
    assert len(scores) == len(CORPUS)


def test_matching_document_scores_highest(depcheck):
    """A query term present in exactly one document must give that document
    the top score and a non-matching document a score of (near) zero — the
    core relevance guarantee hybrid search relies on."""
    BM25Okapi = _BM25Okapi(depcheck)
    bm = BM25Okapi(CORPUS)
    scores = _scores_as_list(bm.get_scores(["cat"]))
    # Doc 0 contains "cat"; doc 3 (quantum physics) does not.
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    assert best_idx == 0, f"expected doc 0 to rank top for 'cat', got {scores}"
    assert scores[0] > 0
    assert scores[3] == pytest.approx(0.0, abs=1e-9)


def test_unrelated_query_scores_all_low(depcheck):
    """A query term absent from every document yields all-zero scores (no
    spurious matches)."""
    BM25Okapi = _BM25Okapi(depcheck)
    bm = BM25Okapi(CORPUS)
    scores = _scores_as_list(bm.get_scores(["nonexistentterm"]))
    assert all(s == pytest.approx(0.0, abs=1e-9) for s in scores)


def test_rarer_term_ranks_its_document_above_common_term(depcheck):
    """BM25 weights rarer terms more (IDF). A query with a term unique to one
    document must rank that document above documents matched only by a common
    term. Here 'quantum' is unique to doc 3; 'the' is common to docs 0 and 1.
    """
    BM25Okapi = _BM25Okapi(depcheck)
    bm = BM25Okapi(CORPUS)
    scores = _scores_as_list(bm.get_scores(["quantum"]))
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    assert best_idx == 3
    # And the unique-term match outscores any 'the'-only document.
    the_scores = _scores_as_list(bm.get_scores(["the"]))
    assert scores[3] > max(the_scores)


def test_get_top_n_returns_ranked_documents(depcheck):
    """get_top_n returns the n best original documents in descending relevance
    — exactly how BM25Retriever produces its result set."""
    BM25Okapi = _BM25Okapi(depcheck)
    bm = BM25Okapi(CORPUS)
    top = bm.get_top_n(["cat"], CORPUS, n=2)
    assert isinstance(top, list)
    assert len(top) == 2
    # The cat document must be the first result.
    assert top[0] == CORPUS[0]


def test_get_top_n_respects_n(depcheck):
    """n caps the number of returned documents."""
    BM25Okapi = _BM25Okapi(depcheck)
    bm = BM25Okapi(CORPUS)
    assert len(bm.get_top_n(["the"], CORPUS, n=1)) == 1
    assert len(bm.get_top_n(["the"], CORPUS, n=3)) == 3


def test_multi_term_query_aggregates_scores(depcheck):
    """A multi-term query sums per-term contributions. A document matching
    multiple query terms should rank at or above one matching fewer. Doc 2
    ('cats and dogs are common pets') is the natural top for ['cats','dogs',
    'pets']."""
    BM25Okapi = _BM25Okapi(depcheck)
    bm = BM25Okapi(CORPUS)
    scores = _scores_as_list(bm.get_scores(["cats", "dogs", "pets"]))
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    assert best_idx == 2
    assert scores[2] > 0
