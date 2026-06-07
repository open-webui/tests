"""Dependency contract: sentence-transformers (import ``sentence_transformers``).

Open WebUI uses `sentence_transformers` for local (in-process) embeddings and
reranking in the retrieval stack:

  * **Embeddings** — ``routers/retrieval.py::get_ef`` and
    ``routers/evaluations.py``:
    ``from sentence_transformers import SentenceTransformer`` then
    ``SentenceTransformer(model_path, device=, trust_remote_code=, backend=,
    model_kwargs=)``; the model is later called to ``.encode`` text.
  * **Reranking (cross-encoder)** — ``routers/retrieval.py::get_rf``:
    ``import sentence_transformers`` then
    ``sentence_transformers.CrossEncoder(model_path, device=,
    trust_remote_code=, backend=, model_kwargs=, activation_fn=)``; the result's
    ``.model.config`` is poked for ``pad_token_id`` and the encoder is called to
    ``.predict`` query/document pairs.
  * **Cosine similarity reranking** — ``retrieval/utils.py``:
    ``from sentence_transformers import util as st_util`` then
    ``st_util.cos_sim(query_embedding, document_embedding)[0]`` to score docs
    when no dedicated reranker is configured.

Loading an actual model requires a multi-hundred-MB download and a GPU/CPU
inference stack, which this offline suite must NOT do. So this module pins the
*import + class/util surface + constructor signatures* (the call shapes the
backend uses), and exercises the ONE piece that needs no model — ``util.cos_sim``
— against hand-built tensors, reproducing the exact reranking math
(``cos_sim(q, docs)[0]`` -> ``.tolist()``). A `sentence_transformers` bump that
removed/renamed any of it, or changed a constructor keyword the backend passes,
fails loudly here instead of at first-RAG-query time.

Pattern mirrors the unit/deps/ exemplar: symbol-existence + signature checks,
plus a model-free behavioural contract for cos_sim. No model downloads, no
network. Uses the `depcheck` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "sentence_transformers"
DIST_NAME = "sentence-transformers"

# Top-level names the backend resolves on the package.
USED_SYMBOLS = ["SentenceTransformer", "CrossEncoder", "util"]

# Constructor keywords get_ef passes to SentenceTransformer(...).
ST_INIT_KWARGS = ["device", "trust_remote_code", "backend", "model_kwargs"]

# Constructor keywords get_rf passes to CrossEncoder(...).
CE_INIT_KWARGS = ["device", "trust_remote_code", "backend", "model_kwargs", "activation_fn"]


# --------------------------------------------------------------------------- #
# Import + version + API surface
# --------------------------------------------------------------------------- #
def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "sentence_transformers"


def test_version_reported(depcheck):
    """The installed distribution version must be resolvable so bump tooling
    and this suite agree on what's under test."""
    depcheck.load(IMPORT_NAME)
    assert depcheck.dist_version(DIST_NAME) is not None


def test_used_symbols_exist(depcheck):
    """SentenceTransformer / CrossEncoder / util must all resolve on the package
    — the three names the backend imports."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_classes_are_callable(depcheck):
    """SentenceTransformer and CrossEncoder are constructed directly; both must
    be callable classes."""
    mod = depcheck.load(IMPORT_NAME)
    assert callable(mod.SentenceTransformer)
    assert callable(mod.CrossEncoder)


# --------------------------------------------------------------------------- #
# SentenceTransformer — constructor surface get_ef relies on
# --------------------------------------------------------------------------- #
def test_sentence_transformer_first_param_is_model_path(depcheck):
    """get_ef passes the model path positionally as the first arg. Pin that the
    first constructor parameter is the model name/path (``model_name_or_path``).
    """
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.SentenceTransformer.__init__, ["model_name_or_path"])


def test_sentence_transformer_accepts_get_ef_kwargs(depcheck):
    """get_ef constructs with device=, trust_remote_code=, backend=,
    model_kwargs=. Every one of those keywords must remain accepted, or RAG
    embedding setup breaks at load time."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.SentenceTransformer.__init__, ST_INIT_KWARGS)


def test_sentence_transformer_has_encode(depcheck):
    """The constructed model is used to embed text via ``.encode(...)``. Pin
    that ``encode`` exists and is callable on the class (no model loaded)."""
    mod = depcheck.load(IMPORT_NAME)
    assert callable(getattr(mod.SentenceTransformer, "encode", None)), (
        "SentenceTransformer.encode missing"
    )


# --------------------------------------------------------------------------- #
# CrossEncoder — constructor surface get_rf relies on
# --------------------------------------------------------------------------- #
def test_cross_encoder_first_param_is_model_path(depcheck):
    """get_rf passes the reranker model path positionally first. Pin the first
    constructor parameter is the model name/path."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.CrossEncoder.__init__, ["model_name_or_path"])


def test_cross_encoder_accepts_get_rf_kwargs(depcheck):
    """get_rf constructs with device=, trust_remote_code=, backend=,
    model_kwargs= and crucially ``activation_fn=`` (set to torch.nn.Sigmoid()
    when sigmoid normalisation is enabled). All must remain accepted."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.CrossEncoder.__init__, CE_INIT_KWARGS)


def test_cross_encoder_has_predict(depcheck):
    """The reranker scores query/document pairs via ``.predict(...)``. Pin that
    ``predict`` exists and is callable on the class."""
    mod = depcheck.load(IMPORT_NAME)
    assert callable(getattr(mod.CrossEncoder, "predict", None)), "CrossEncoder.predict missing"


def test_cross_encoder_exposes_model_attr_name(depcheck):
    """get_rf reaches into ``rf.model.config`` to patch a missing
    ``pad_token_id``. The ``model`` attribute is an instance attribute set in
    __init__ (so it won't appear on the bare class), but the access pattern is
    guarded with getattr(rf, 'model', None) — pin only that CrossEncoder defines
    an __init__ that the access depends on (instance built at runtime)."""
    mod = depcheck.load(IMPORT_NAME)
    # We don't instantiate (needs a model); just confirm the class is real and
    # the guarded attribute access in get_rf is against a normal object.
    assert isinstance(mod.CrossEncoder, type)


# --------------------------------------------------------------------------- #
# util — the model-free piece the reranker uses (cos_sim). Exercised for real.
# --------------------------------------------------------------------------- #
def test_util_submodule_importable(depcheck):
    """retrieval/utils.py does ``from sentence_transformers import util``; pin
    the submodule imports and exposes cos_sim."""
    depcheck.load(IMPORT_NAME)
    util = depcheck.load("sentence_transformers.util")
    assert hasattr(util, "cos_sim"), "sentence_transformers.util.cos_sim missing"


def test_cos_sim_callable(depcheck):
    depcheck.load(IMPORT_NAME)
    util = depcheck.load("sentence_transformers.util")
    assert callable(util.cos_sim)


def test_cos_sim_known_values(depcheck):
    """cos_sim is pure tensor math (no model). Verify the cosine values the
    reranker relies on: identical vectors -> 1.0, orthogonal -> 0.0, a 45-degree
    pair -> ~0.7071. A bump that broke the metric would silently mis-rank RAG
    results; pin the numbers.
    """
    depcheck.load(IMPORT_NAME)
    util = depcheck.load("sentence_transformers.util")
    torch = depcheck.load("torch")

    query = torch.tensor([[1.0, 0.0, 0.0]])
    docs = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # identical -> cos 1.0
            [0.0, 1.0, 0.0],  # orthogonal -> cos 0.0
            [1.0, 1.0, 0.0],  # 45 degrees -> cos ~0.7071
        ]
    )
    sims = util.cos_sim(query, docs)
    # Shape is (1, 3): one query row, three doc columns.
    assert tuple(sims.shape) == (1, 3)
    row = sims[0].tolist()
    assert abs(row[0] - 1.0) < 1e-5
    assert abs(row[1] - 0.0) < 1e-5
    assert abs(row[2] - 0.70710678) < 1e-4


def test_cos_sim_reranking_codepath(depcheck):
    """Reproduce retrieval/utils.py's exact use:
    ``scores = util.cos_sim(query_embedding, document_embedding)[0]`` then
    ``scores.tolist()``. The ``[0]`` selects the single-query row, and the
    backend sorts docs by these scores descending. Pin that the indexed row is a
    1-D score vector, one score per document, ordered to match the docs.
    """
    depcheck.load(IMPORT_NAME)
    util = depcheck.load("sentence_transformers.util")
    torch = depcheck.load("torch")

    query_embedding = torch.tensor([[2.0, 0.0]])  # direction along x
    # doc 0 aligned with query (high score), doc 1 anti-aligned (low score).
    document_embedding = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])

    scores = util.cos_sim(query_embedding, document_embedding)[0]  # the [0] hop
    score_list = scores.tolist()  # the .tolist() hop
    assert len(score_list) == 2
    # Aligned doc must outrank the anti-aligned doc — the ranking invariant.
    assert score_list[0] > score_list[1]
    assert abs(score_list[0] - 1.0) < 1e-5
    assert abs(score_list[1] - (-1.0)) < 1e-5


def test_cos_sim_accepts_plain_python_lists(depcheck):
    """cos_sim's signature accepts ``list | np.ndarray | Tensor``. The backend
    feeds it embeddings that may arrive as lists; pin that a list-of-lists input
    is accepted and produces a Tensor result."""
    depcheck.load(IMPORT_NAME)
    util = depcheck.load("sentence_transformers.util")
    result = util.cos_sim([[1.0, 0.0]], [[1.0, 0.0]])
    # Result supports .tolist() (it is a Tensor), matching the backend's usage.
    assert hasattr(result, "tolist")
    assert abs(result.tolist()[0][0] - 1.0) < 1e-5
