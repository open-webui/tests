"""Regression: the ephemeral RAG collection holding a web search's fetched
pages must belong to the user who ran the search.

open-webui 0.11.0 fix `6d4c02a89` (PR #26706): `web-search-*` was the one
collection namespace `filter_accessible_collections` admitted unconditionally
for any non-admin, on both read and write, while `file-*`, `user-memory-*` and
knowledge bases were owner-scoped. `process_web_search` minted the collection as
`web-search-<sha256(queries)>`, a name derived only from the query text, so two
users running the same search landed in the same collection and either could read
or overwrite the pages the other had fetched. The fix mints
`web-search-{user.id}-<hash>` and the access helper admits only names starting
with `web-search-{requester.id}-`.

Discriminates: passes on v0.11.0, fails on v0.10.2 (identical queries mint one
shared collection, and any non-admin is admitted to it).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.regression

ALICE = "11111111-1111-1111-1111-111111111111"
BOB = "22222222-2222-2222-2222-222222222222"


def _user(user_id: str, role: str = "user") -> SimpleNamespace:
    return SimpleNamespace(id=user_id, role=role)


@pytest.fixture
def retrieval_utils(owui_module):
    return owui_module("open_webui.retrieval.utils")


@pytest.fixture
def filter_collections(retrieval_utils):
    return retrieval_utils.filter_accessible_collections


@pytest.fixture
def mint_collection(owui_module):
    """Drive the real process_web_search and hand back the collection it minted.

    Only the I/O boundary is stubbed: config, the permission check, the search
    engine and the vector-DB write.
    """
    router = owui_module("open_webui.routers.retrieval")
    search_result = owui_module("open_webui.retrieval.web.main").SearchResult

    config = SimpleNamespace(
        ENABLE_WEB_SEARCH=True,
        USER_PERMISSIONS={},
        WEB_SEARCH_ENGINE="stub",
        WEB_SEARCH_CONCURRENT_REQUESTS=0,
        BYPASS_WEB_SEARCH_WEB_LOADER=True,
        BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL=False,
    )
    hits = [
        search_result(
            link="https://example.com/page",
            title="Example page",
            snippet="fetched page body",
        )
    ]

    async def _mint(user: SimpleNamespace, queries: list[str]) -> str:
        saved = {}

        def capture_save(request, docs, collection_name, *args, **kwargs):
            saved["collection_name"] = collection_name

        with (
            patch.object(router, "get_retrieval_config", AsyncMock(return_value=config)),
            patch.object(router, "has_permission", AsyncMock(return_value=True)),
            patch.object(router, "search_web", AsyncMock(return_value=hits)),
            patch.object(router, "save_docs_to_vector_db", capture_save),
        ):
            response = await router.process_web_search(
                request=SimpleNamespace(),
                form_data=router.SearchForm(queries=queries),
                user=user,
            )

        assert response["collection_names"] == [saved["collection_name"]], (
            "process_web_search told the caller to query a different collection than "
            "the one it stored the fetched pages in"
        )
        return saved["collection_name"]

    return _mint


# =============================================================================
# Narrow: the ephemeral collection identity is per-owner, and the access
# helper enforces it
# =============================================================================


@pytest.mark.asyncio
async def test_identical_searches_by_two_users_mint_different_collections(mint_collection):
    """The bug: the collection name was a pure hash of the queries."""
    queries = ["open webui release notes"]
    alice_collection = await mint_collection(_user(ALICE), queries)
    bob_collection = await mint_collection(_user(BOB), queries)

    assert alice_collection != bob_collection, (
        "two users running the same web search share one ephemeral collection, so each "
        "can read and overwrite the pages the other fetched (#26706)"
    )
    assert alice_collection.startswith(f"web-search-{ALICE}-"), (
        f"the ephemeral collection is not owner-bound: {alice_collection!r} (#26706)"
    )


@pytest.mark.asyncio
async def test_other_user_cannot_reach_a_minted_web_search_collection(
    mint_collection, filter_collections
):
    """The path the fix closes: the name minted for Alice, handed to the access
    helper as Bob."""
    collection = await mint_collection(_user(ALICE), ["quarterly figures"])
    allowed = await filter_collections({collection}, _user(BOB))

    assert allowed == set(), (
        f"a second user was granted access to {collection!r}, the collection holding the "
        "pages someone else's web search fetched (#26706)"
    )


@pytest.mark.asyncio
async def test_other_user_cannot_write_a_minted_web_search_collection(
    mint_collection, filter_collections
):
    """Write was admitted unconditionally too, so a foreign user could overwrite
    the pages the owner's next answer is grounded in."""
    collection = await mint_collection(_user(ALICE), ["quarterly figures"])
    allowed = await filter_collections({collection}, _user(BOB), access_type="write")

    assert allowed == set(), (
        f"a second user may overwrite {collection!r}, letting them substitute the web "
        "pages someone else's search fetched (#26706)"
    )


# =============================================================================
# Broad: every web-search collection name is owner-bound, at every site
# =============================================================================


def _web_search_source_lines(backend: Path) -> list[tuple[str, int, str]]:
    """Every backend source line that mentions the web-search namespace."""
    found = []
    for path in (backend / "open_webui").rglob("*.py"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "web-search-" in line and not line.lstrip().startswith("#"):
                found.append((str(path), lineno, line.strip()))
    return found


def test_every_web_search_collection_name_is_built_with_the_owner_id(open_webui_backend):
    """The owner binding lives in one inline f-string, not a shared helper, so a
    new call site that forgets it is exactly how this regresses."""
    mint_sites = [
        entry
        for entry in _web_search_source_lines(open_webui_backend)
        if "f'web-search-" in entry[2] or 'f"web-search-' in entry[2]
    ]
    assert mint_sites, "no web-search collection name construction found; the scan is stale"

    unbound = [entry for entry in mint_sites if "{user.id}" not in entry[2]]
    assert unbound == [], (
        "these sites build a web-search collection name without the owner id, so the "
        f"collection they create is reachable by every user (#26706): {unbound}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("access_type", ["read", "write"])
@pytest.mark.parametrize(
    "foreign_name",
    [
        "web-search-abc123def456",
        f"web-search-{ALICE}-abc123def456",
        f"web-search-{ALICE}",
        f"web-search-{BOB}x-abc123",
    ],
)
async def test_no_foreign_web_search_name_is_admitted(
    filter_collections, foreign_name, access_type
):
    allowed = await filter_collections({foreign_name}, _user(BOB), access_type=access_type)
    assert allowed == set(), (
        f"{foreign_name!r} was admitted for {access_type} to a user who does not own it, "
        "so the web-search namespace is still not owner-scoped (#26706)"
    )


@pytest.mark.asyncio
async def test_web_search_is_scoped_like_the_other_owner_bound_namespaces(filter_collections):
    """The invariant the fix restores: web-search behaves like user-memory, the
    namespace it was the lone exception to."""
    foreign = {f"web-search-{ALICE}-hash", f"user-memory-{ALICE}"}
    assert await filter_collections(foreign, _user(BOB)) == set()

    own = {f"web-search-{BOB}-hash", f"user-memory-{BOB}"}
    assert await filter_collections(own, _user(BOB)) == own


# =============================================================================
# Nearby: scoping must not break caching, the owner's own access, the admin
# bypass or name validity
# =============================================================================


@pytest.mark.asyncio
async def test_same_user_repeating_a_search_reuses_one_collection(mint_collection):
    """Owner binding must not defeat the per-query cache, or every repeat of a
    search re-embeds the same pages."""
    queries = ["milan weather", "milan forecast"]
    first = await mint_collection(_user(ALICE), queries)
    second = await mint_collection(_user(ALICE), queries)

    assert first == second, "the same user's repeated identical search minted a new collection"


@pytest.mark.asyncio
async def test_different_queries_by_one_user_get_different_collections(mint_collection):
    """The query hash stays part of the identity, so one user's searches do not
    overwrite each other."""
    alice = _user(ALICE)
    first = await mint_collection(alice, ["milan weather"])
    second = await mint_collection(alice, ["rome weather"])

    assert first != second, "two different searches by one user share a collection"


@pytest.mark.asyncio
async def test_owner_keeps_read_and_write_access_to_its_own_collection(
    mint_collection, filter_collections
):
    """Over-correction check: the user who ran the search must still reach the
    pages it fetched, or web search contributes no context at all."""
    alice = _user(ALICE)
    collection = await mint_collection(alice, ["milan weather"])

    assert await filter_collections({collection}, alice) == {collection}
    assert await filter_collections({collection}, alice, access_type="write") == {collection}


@pytest.mark.asyncio
async def test_admin_still_bypasses_web_search_scoping(mint_collection, filter_collections):
    collection = await mint_collection(_user(ALICE), ["milan weather"])
    admin = _user("admin-user", role="admin")

    assert await filter_collections({collection}, admin) == {collection}


@pytest.mark.asyncio
async def test_minted_name_stays_a_valid_collection_name(mint_collection, retrieval_utils):
    """The owner id lengthens the name, and vector stores cap it at 63 chars."""
    collection = await mint_collection(_user(ALICE), ["milan weather"])

    assert len(collection) <= 63, f"minted name exceeds the 63-char limit: {collection!r}"
    assert retrieval_utils._is_safe_collection_name(collection), (
        f"minted name {collection!r} is rejected by the collection-name filter, so the "
        "fetched pages are unreachable"
    )
