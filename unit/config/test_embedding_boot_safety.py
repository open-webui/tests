"""Tests for get_embedding_function boot-safety.

Regression for open-webui/open-webui#25634 (and #25165).

get_embedding_function is called at import time from main.py. In 0.9.6
(commit 55ca719b) an empty embedding_engine with no loaded local model
raised ValueError *at construction*, so a blank embedding model set via
the Documents settings UI made the whole instance unbootable — and since
you couldn't boot, you couldn't get back into settings to undo it. The
only escape was hand-editing the config row in the DB.

Fix (PR #25683): the missing-local-model check is deferred into the
returned coroutine, so building the embedding function always succeeds;
the error now fires only when something actually tries to embed, as a
clear request-time error instead of a boot crash.

Invariant under test: constructing an embedding function for any valid
engine selection (including the empty/local one) must never raise —
configuration mistakes degrade RAG at use-time, they don't brick boot.
An genuinely unknown engine string is still a programmer error and may
raise at construction.
"""

from __future__ import annotations

from types import ModuleType
from unittest.mock import MagicMock

import pytest


def _gef(module: ModuleType):
    return module.get_embedding_function


# =============================================================================
# Specific — open-webui#25634: empty engine + no local model
# =============================================================================


@pytest.mark.regression
def test_empty_engine_no_model_constructs_without_raising(
    retrieval_utils_module: ModuleType,
) -> None:
    """Regression for open-webui/open-webui#25634.

    get_embedding_function('', model, None, ...) is what runs at boot when
    the embedding model is blank. It must return a callable WITHOUT
    raising — otherwise the instance can't start.
    """
    fn = _gef(retrieval_utils_module)(
        "",  # embedding_engine (local sentence-transformers)
        "",  # embedding_model
        None,  # embedding_function — nothing loaded
        "",  # url
        "",  # key
        1,  # embedding_batch_size
    )
    assert callable(fn), "construction must yield a callable, not raise"


@pytest.mark.regression
@pytest.mark.asyncio
async def test_empty_engine_no_model_defers_error_to_use(
    retrieval_utils_module: ModuleType,
) -> None:
    """The diagnostic isn't lost — awaiting the constructed function with no
    model still raises the clear 'No embedding model is loaded' ValueError,
    just at use-time instead of boot-time."""
    fn = _gef(retrieval_utils_module)("", "", None, "", "", 1)
    with pytest.raises(ValueError, match="No embedding model is loaded"):
        await fn("some text")


@pytest.mark.regression
@pytest.mark.asyncio
async def test_empty_engine_with_model_still_embeds(
    retrieval_utils_module: ModuleType,
) -> None:
    """Happy path must be untouched: with a real local model loaded, the
    constructed function encodes and returns the vector."""
    model = MagicMock()
    model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3]

    fn = _gef(retrieval_utils_module)("", "all-MiniLM", model, "", "", 1)
    result = await fn("hello")
    assert result == [0.1, 0.2, 0.3]
    model.encode.assert_called_once()


# =============================================================================
# Broad — construction is boot-safe for every valid engine selection
# =============================================================================


@pytest.mark.regression
@pytest.mark.parametrize("engine", ["ollama", "openai", "azure_openai"])
def test_external_engine_constructs_without_raising_when_unconfigured(
    retrieval_utils_module: ModuleType, engine: str
) -> None:
    """External engines with empty url/key/model must also construct without
    raising — they defer to the actual API call. A misconfigured external
    provider degrades RAG at use-time, it doesn't crash boot."""
    fn = _gef(retrieval_utils_module)(
        engine,
        "",  # model
        None,  # embedding_function (external engines build their own)
        "",  # url
        "",  # key
        1,  # batch size
    )
    assert callable(fn)


@pytest.mark.regression
@pytest.mark.parametrize("engine", ["", "ollama", "openai", "azure_openai"])
def test_valid_engine_construction_never_raises(
    retrieval_utils_module: ModuleType, engine: str
) -> None:
    """The umbrella invariant the #25634 fix establishes: for any engine
    value the UI can produce, building the embedding function is total —
    it returns a callable and never raises at construction, regardless of
    how incompletely it's configured."""
    fn = _gef(retrieval_utils_module)(engine, "", None, "", "", 1)
    assert callable(fn)


@pytest.mark.regression
def test_unknown_engine_still_raises_at_construction(
    retrieval_utils_module: ModuleType,
) -> None:
    """Boundary: a genuinely unknown engine string is a programmer/config
    error, not a UI-reachable state, and should still fail loudly at
    construction. The boot-safety fix must not swallow this."""
    with pytest.raises(ValueError, match="Unknown embedding engine"):
        _gef(retrieval_utils_module)("not-a-real-engine", "", None, "", "", 1)
