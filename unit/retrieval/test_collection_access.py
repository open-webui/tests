"""Tests for retrieval collection access control + source injection.

Two layers, both in `open_webui.retrieval.utils`:

  filter_accessible_collections(collection_names, user)  [broad]
    The chokepoint that decides which vector collections a non-admin
    may read. Governs EVERY retrieval read: knowledge bases, uploaded
    files, per-user memory, ephemeral web-search collections, and the
    legacy/unscoped namespace gated by ENABLE_RETRIEVAL_UNSCOPED_COLLECTIONS.

  get_sources_from_items(...)  [specific — open-webui#25585]
    Builds the per-item collection list that is then access-filtered and
    vector-queried. In v0.9.6 a `web_search`-typed item carrying a
    server-generated `collection_name` had NO matching dispatch branch,
    so it fell through to the generic `elif item.get('collection_name')`
    gate — which, with BYPASS_RETRIEVAL_ACCESS_CONTROL off (default),
    silently drops the collection as "untrusted direct collection_name".
    Web-search results were fetched, embedded, stored… then never
    queried, so the model answered with no web context.

The broad suite pins the whole access matrix (so a future refactor
can't quietly widen or break it); the specific test reproduces #25585
and asserts the web-search collection actually reaches the query.
"""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

# -----------------------------------------------------------------------------
# Fakes
# -----------------------------------------------------------------------------


def _user(uid: str = "u1", role: str = "user") -> SimpleNamespace:
    """filter_accessible_collections only touches user.id and user.role."""
    return SimpleNamespace(id=uid, role=role)


@pytest.fixture
def fac(retrieval_utils_module: ModuleType):
    """Shorthand for the function under test."""
    return retrieval_utils_module.filter_accessible_collections


# =============================================================================
# Broad — filter_accessible_collections access matrix
# =============================================================================


@pytest.mark.regression
@pytest.mark.asyncio
async def test_admin_bypasses_access_control(retrieval_utils_module, fac) -> None:
    """Admins read any well-formed collection without per-name checks."""
    names = {"file-1", "some-kb", "web-search-abc", "user-memory-zzz"}
    result = await fac(names, _user(role="admin"))
    assert result == names


@pytest.mark.regression
@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["admin", "user"])
async def test_unsafe_names_always_rejected(retrieval_utils_module, fac, role) -> None:
    """Malformed names (path traversal, separators, quotes) never reach
    the vector store — rejected even for admins, before the bypass."""
    unsafe = {
        "../etc/passwd",
        "a/b",
        "a;b",
        "a b",
        "a'b",
        'a"b',
        "a.b",
        "",
    }
    safe = {"file-1", "web-search-abc"}
    # KB checks pass so 'safe' non-special names survive for the user role.
    with patch.object(retrieval_utils_module, "has_access_to_file", AsyncMock(return_value=True)):
        result = await fac(unsafe | safe, _user(role=role))
    assert not (result & unsafe), f"unsafe names leaked: {result & unsafe}"


@pytest.mark.regression
@pytest.mark.asyncio
async def test_web_search_collections_always_allowed(retrieval_utils_module, fac) -> None:
    """open-webui#25585 invariant: ephemeral web-search-* collections are
    always readable by the requesting non-admin user. The whole web-search
    feature depends on this being a no-questions-asked allow."""
    names = {"web-search-e51138a673bc", "web-search-deadbeef"}
    result = await fac(names, _user())
    assert result == names


@pytest.mark.regression
@pytest.mark.asyncio
async def test_file_collection_requires_file_access(retrieval_utils_module, fac) -> None:
    """file-<id> is allowed iff has_access_to_file grants it."""
    granted = AsyncMock(return_value=True)
    with patch.object(retrieval_utils_module, "has_access_to_file", granted):
        assert await fac({"file-99"}, _user()) == {"file-99"}

    denied = AsyncMock(return_value=False)
    with patch.object(retrieval_utils_module, "has_access_to_file", denied):
        assert await fac({"file-99"}, _user()) == set()


@pytest.mark.regression
@pytest.mark.asyncio
async def test_user_memory_scoped_to_owner(retrieval_utils_module, fac) -> None:
    """user-memory-<uid> is readable only by its own user."""
    assert await fac({"user-memory-u1"}, _user(uid="u1")) == {"user-memory-u1"}
    assert await fac({"user-memory-u1"}, _user(uid="u2")) == set()


@pytest.mark.regression
@pytest.mark.asyncio
async def test_knowledge_bases_meta_collection_denied(retrieval_utils_module, fac) -> None:
    """The `knowledge-bases` system meta-collection is never exposed to
    a non-admin."""
    assert await fac({"knowledge-bases"}, _user()) == set()


@pytest.mark.regression
@pytest.mark.asyncio
async def test_knowledge_base_requires_access_grant(retrieval_utils_module, fac) -> None:
    """A name that resolves to a real KB is allowed iff the user passes
    Knowledges.check_access_by_user_id."""
    kb_allow = SimpleNamespace(
        check_access_by_user_id=AsyncMock(return_value=True),
        get_knowledge_by_id=AsyncMock(return_value=SimpleNamespace(id="kb1")),
    )
    with patch.object(retrieval_utils_module, "Knowledges", kb_allow):
        assert await fac({"kb1"}, _user()) == {"kb1"}

    kb_deny = SimpleNamespace(
        check_access_by_user_id=AsyncMock(return_value=False),
        get_knowledge_by_id=AsyncMock(return_value=SimpleNamespace(id="kb1")),
    )
    with patch.object(retrieval_utils_module, "Knowledges", kb_deny):
        # A real KB the user can't access stays denied even if the
        # unscoped escape hatch is on.
        with patch.object(retrieval_utils_module, "ENABLE_RETRIEVAL_UNSCOPED_COLLECTIONS", True):
            assert await fac({"kb1"}, _user()) == set()


@pytest.mark.regression
@pytest.mark.asyncio
async def test_unknown_collection_denied_by_default(retrieval_utils_module, fac) -> None:
    """A name that is neither a KB nor a recognised prefix is denied when
    ENABLE_RETRIEVAL_UNSCOPED_COLLECTIONS is off (the 0.9.6 default that
    closed the unscoped namespace)."""
    kb = SimpleNamespace(
        check_access_by_user_id=AsyncMock(return_value=False),
        get_knowledge_by_id=AsyncMock(return_value=None),
    )
    with patch.object(retrieval_utils_module, "Knowledges", kb):
        with patch.object(retrieval_utils_module, "ENABLE_RETRIEVAL_UNSCOPED_COLLECTIONS", False):
            assert await fac({"legacy-coll"}, _user()) == set()


@pytest.mark.regression
@pytest.mark.asyncio
async def test_unknown_collection_allowed_when_unscoped_enabled(
    retrieval_utils_module, fac
) -> None:
    """The ENABLE_RETRIEVAL_UNSCOPED_COLLECTIONS escape hatch restores
    legacy behaviour: a non-KB, non-prefixed name is allowed."""
    kb = SimpleNamespace(
        check_access_by_user_id=AsyncMock(return_value=False),
        get_knowledge_by_id=AsyncMock(return_value=None),
    )
    with patch.object(retrieval_utils_module, "Knowledges", kb):
        with patch.object(retrieval_utils_module, "ENABLE_RETRIEVAL_UNSCOPED_COLLECTIONS", True):
            assert await fac({"legacy-coll"}, _user()) == {"legacy-coll"}


@pytest.mark.regression
@pytest.mark.asyncio
async def test_mixed_batch_partitions_correctly(retrieval_utils_module, fac) -> None:
    """A realistic mixed item set: own memory + accessible file +
    web-search + foreign memory + denied KB → only the first three pass."""
    kb = SimpleNamespace(
        check_access_by_user_id=AsyncMock(return_value=False),
        get_knowledge_by_id=AsyncMock(return_value=SimpleNamespace(id="kbX")),
    )
    with (
        patch.object(retrieval_utils_module, "Knowledges", kb),
        patch.object(retrieval_utils_module, "has_access_to_file", AsyncMock(return_value=True)),
    ):
        result = await fac(
            {
                "user-memory-u1",
                "file-7",
                "web-search-abc",
                "user-memory-someone-else",
                "kbX",
            },
            _user(uid="u1"),
        )
    assert result == {"user-memory-u1", "file-7", "web-search-abc"}


# =============================================================================
# Specific — open-webui#25585: web_search item must reach the query
# =============================================================================


def _web_search_item(collection_name: str = "web-search-e51138a673bc") -> dict:
    """Shape produced by chat_web_search_handler when embedding & retrieval
    are enabled (the legacy path)."""
    return {
        "collection_name": collection_name,
        "name": "weather milano",
        "type": "web_search",
        "urls": ["https://example.com/meteo"],
        "queries": ["weather milano"],
    }


async def _run_get_sources(module, items, user, query_result=None):
    """Drive get_sources_from_items with query_collection mocked. Returns
    (sources, query_collection_mock)."""
    if query_result is None:
        query_result = {
            "documents": [["Milan: 28°C, sunny."]],
            "metadatas": [[{"name": "weather milano", "source": "https://example.com/meteo"}]],
        }
    qc = AsyncMock(return_value=query_result)
    with patch.object(module, "query_collection", qc):
        sources = await module.get_sources_from_items(
            request=SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace())),
            items=items,
            queries=["weather milano"],
            embedding_function=AsyncMock(return_value=[[0.0]]),
            k=3,
            reranking_function=None,
            k_reranker=None,
            r=None,
            hybrid_bm25_weight=0.5,
            hybrid_search=False,
            full_context=False,
            user=user,
        )
    return sources, qc


@pytest.mark.regression
@pytest.mark.asyncio
async def test_web_search_item_collection_reaches_query(retrieval_utils_module) -> None:
    """Regression for open-webui/open-webui#25585.

    A web_search item carrying a server-generated collection_name must be
    vector-queried and produce a source. Before the fix the collection was
    dropped at dispatch (no `type == 'web_search'` branch), query_collection
    was never called, and the model got no web context.
    """
    item = _web_search_item()
    sources, qc = await _run_get_sources(retrieval_utils_module, [item], _user())

    qc.assert_awaited_once()
    # The web-search collection must be the one queried.
    _, kwargs = qc.call_args
    assert item["collection_name"] in set(kwargs.get("collection_names", set())), (
        f"Regression of #25585: web_search collection not queried; "
        f"query_collection got collection_names={kwargs.get('collection_names')!r}"
    )
    assert sources, "Regression of #25585: web_search produced no sources"
    assert sources[0]["document"] == ["Milan: 28°C, sunny."]


@pytest.mark.regression
@pytest.mark.asyncio
async def test_untyped_collection_name_item_is_dropped_without_bypass(
    retrieval_utils_module,
) -> None:
    """Trust boundary the #25585 fix must NOT widen: an item carrying a
    bare collection_name but no recognised type is still dropped when
    BYPASS_RETRIEVAL_ACCESS_CONTROL is off. Only `web_search`-typed
    items are trusted, not arbitrary collection_name passthrough."""
    item = {"collection_name": "some-knowledge-base-id", "name": "x"}
    with patch.object(retrieval_utils_module, "BYPASS_RETRIEVAL_ACCESS_CONTROL", False):
        sources, qc = await _run_get_sources(retrieval_utils_module, [item], _user())
    qc.assert_not_awaited()
    assert sources == []


@pytest.mark.regression
@pytest.mark.asyncio
async def test_untyped_collection_name_item_allowed_with_bypass(
    retrieval_utils_module,
) -> None:
    """The BYPASS_RETRIEVAL_ACCESS_CONTROL escape hatch still works for
    bare collection_name items (admin-style trust-everything mode)."""
    item = {"collection_name": "web-search-trusted", "name": "x"}
    with patch.object(retrieval_utils_module, "BYPASS_RETRIEVAL_ACCESS_CONTROL", True):
        sources, qc = await _run_get_sources(retrieval_utils_module, [item], _user())
    qc.assert_awaited_once()
    assert sources


@pytest.mark.regression
@pytest.mark.asyncio
async def test_web_search_with_no_results_yields_no_source(
    retrieval_utils_module,
) -> None:
    """A web_search collection that the vector query returns empty for
    must not crash or fabricate a source — just yields nothing."""
    item = _web_search_item()
    empty = {"documents": [[]], "metadatas": [[]]}
    sources, qc = await _run_get_sources(
        retrieval_utils_module, [item], _user(), query_result=empty
    )
    qc.assert_awaited_once()
    # documents[0] is empty → no source row appended.
    assert sources == [] or all(s["document"] == [] for s in sources)
