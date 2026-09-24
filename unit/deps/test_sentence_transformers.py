"""Dependency contract: sentence-transformers (import ``sentence_transformers``).

Open WebUI runs local embeddings and reranking through sentence-transformers:

  * **Embeddings**: ``routers/retrieval.py::get_ef`` builds
    ``SentenceTransformer(model_path, device=, trust_remote_code=, backend=, model_kwargs=)``
    and ``retrieval/utils.py`` embeds with
    ``ef.encode(texts, batch_size=, prompt=prefix).tolist()`` (the prompt carries
    ``RAG_EMBEDDING_QUERY_PREFIX`` / ``RAG_EMBEDDING_CONTENT_PREFIX``). The "token_transformers"
    splitter measures chunks with ``ef.tokenizer`` when no tokenizer model is set, and
    ``routers/evaluations.py`` builds ``SentenceTransformer(name)`` and calls ``encode(texts)``.
  * **Reranking**: ``get_rf`` builds ``sentence_transformers.CrossEncoder(model_path, device=,
    trust_remote_code=, backend=, model_kwargs=, activation_fn=)`` and ``retrieval/utils.py``
    scores with ``rf.predict([(query, text), ...], batch_size=)``.

Without a reranker the backend scores with its own numpy ``cosine_similarity``;
``sentence_transformers.util`` is not used any more. A real model is a download, so the
constructors and ``predict`` are pinned by signature, and ``encode`` is exercised for real on a
model-free ``SentenceTransformer`` built from one hand-written module.

Discriminates: with ``encode`` dropping its ``prompt`` (a pytest plugin patching the installed
library), the prompt test goes red.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "sentence_transformers"
DIST_NAME = "sentence-transformers"

# Constructor keywords get_ef passes to SentenceTransformer(...).
ST_INIT_KWARGS = ["device", "trust_remote_code", "backend", "model_kwargs"]

# Constructor keywords get_rf passes to CrossEncoder(...).
CE_INIT_KWARGS = ["device", "trust_remote_code", "backend", "model_kwargs", "activation_fn"]


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "sentence_transformers"


def test_version_reported(depcheck):
    depcheck.load(IMPORT_NAME)
    assert depcheck.dist_version(DIST_NAME) is not None


def test_used_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, ["SentenceTransformer", "CrossEncoder"])


# --------------------------------------------------------------------------- #
# SentenceTransformer: constructor, encode and tokenizer
# --------------------------------------------------------------------------- #
def test_sentence_transformer_takes_the_model_path_and_get_ef_kwargs(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.SentenceTransformer.__init__, ["model_name_or_path"])
    depcheck.assert_params(mod.SentenceTransformer.__init__, ST_INIT_KWARGS)


def test_the_tokenizer_is_a_property_of_the_model(depcheck):
    """The token_transformers splitter reads `ef.tokenizer` when no tokenizer model is set."""
    mod = depcheck.load(IMPORT_NAME)
    assert isinstance(getattr(mod.SentenceTransformer, "tokenizer", None), property)


def _model_free_encoder(depcheck):
    """A SentenceTransformer whose one module embeds a text as (length, count of 'a')."""
    mod = depcheck.load(IMPORT_NAME)
    torch = depcheck.load("torch")

    class CharacterCounts(torch.nn.Module):
        def tokenize(self, texts, **kwargs):
            counts = [[len(text), text.count("a")] for text in texts]
            return {"input_ids": torch.tensor(counts, dtype=torch.float)}

        def forward(self, features):
            return {**features, "sentence_embedding": features["input_ids"]}

    return mod.SentenceTransformer(modules=[CharacterCounts()], device="cpu")


def test_encode_prepends_the_prompt_and_returns_rows_with_tolist(depcheck):
    """retrieval/utils.py: `ef.encode(texts, batch_size=, prompt=prefix).tolist()`.

    `encode` takes **kwargs, so only a call shows that `prompt` still means the prefix.
    """
    encoder = _model_free_encoder(depcheck)

    embeddings = encoder.encode(["abc", "aaaa"], batch_size=2, prompt="query: ").tolist()

    assert embeddings == [[10.0, 1.0], [11.0, 4.0]]


def test_encode_of_one_text_returns_one_row(depcheck):
    """A single query string embeds to one flat vector, as the query path expects."""
    encoder = _model_free_encoder(depcheck)

    assert encoder.encode("abc").tolist() == [3.0, 1.0]


# --------------------------------------------------------------------------- #
# CrossEncoder: constructor and predict
# --------------------------------------------------------------------------- #
def test_cross_encoder_takes_the_model_path_and_get_rf_kwargs(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.CrossEncoder.__init__, ["model_name_or_path"])
    depcheck.assert_params(mod.CrossEncoder.__init__, CE_INIT_KWARGS)


def test_predict_takes_the_pairs_first_and_a_batch_size(depcheck):
    """`rf.predict([(query, text), ...], batch_size=)`; **kwargs would swallow a dropped name."""
    mod = depcheck.load(IMPORT_NAME)
    parameters = list(inspect.signature(mod.CrossEncoder.predict).parameters.values())

    assert parameters[1].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert "batch_size" in [parameter.name for parameter in parameters]
