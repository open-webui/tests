"""Regression: a web search's RAG collection must belong to the user who ran the search.

open-webui 0.11.0 fix `6d4c02a89` (PR #26706): `filter_accessible_collections` admitted every
`web-search-*` name to any non-admin, for read and write, and `process_web_search` named the
collection after a hash of the queries alone. Two users running the same search shared one
collection, and anyone holding the name could read or overwrite the pages it held. The fix
mints `web-search-{user.id}-<hash>` and admits only the requester's own prefix.

The search provider is a SearXNG listener; the collections are minted by real searches.

Twin of unit/security/test_websearch_collection_owner_scope.py.

Discriminates: passes on dev bbfa876af, fails with `6d4c02a89` reverted (identical searches
mint one shared name, and a second user reads and overwrites it with HTTP 200).
"""

from __future__ import annotations

import json

import pytest

from harness.listener import json_answer

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

RETRIEVAL_CONFIG = ("/api/v1/retrieval/config", "/api/v1/retrieval/config/update")
EMBEDDING_CONFIG = ("/api/v1/retrieval/embedding", "/api/v1/retrieval/embedding/update")
PAGE_TEXT = "the page the owner's search fetched"
SEARCH_HITS = {
    "results": [{"url": "https://example.com/page", "title": "Page", "content": PAGE_TEXT}]
}


@pytest.fixture
def mock_embeddings(admin, preserve, upstream):
    """The instance's embedding URL defaults to api.openai.com; the mock serves /embeddings."""
    preserve(EMBEDDING_CONFIG)
    with admin.client() as client:
        embedding = client.get(EMBEDDING_CONFIG[0]).json()
        embedding["openai_config"] = {"url": upstream.base_url, "key": "sk-mock"}
        client.post(EMBEDDING_CONFIG[1], json=embedding).raise_for_status()


@pytest.fixture
def mint(admin, listener, preserve, mock_embeddings):
    """`mint(actor, query)` runs a real web search and returns the collection it stored."""
    preserve(RETRIEVAL_CONFIG)
    listener.route("GET", "/search", json_answer(SEARCH_HITS))
    with admin.client() as client:
        web = client.get(RETRIEVAL_CONFIG[0]).json()["web"]
        web.update(
            ENABLE_WEB_SEARCH=True,
            WEB_SEARCH_ENGINE="searxng",
            SEARXNG_QUERY_URL=f"{listener.base_url}/search",
            BYPASS_WEB_SEARCH_WEB_LOADER=True,
            BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL=False,
        )
        client.post(RETRIEVAL_CONFIG[1], json={"web": web}).raise_for_status()

    def search(actor, query: str) -> str:
        with actor.client() as client:
            searched = client.post(
                "/api/v1/retrieval/process/web/search", json={"queries": [query]}
            )
        assert searched.status_code == 200, searched.text
        return searched.json()["collection_names"][0]

    return search


def _read(actor, collection: str):
    with actor.client() as client:
        return client.post(
            "/api/v1/retrieval/query/collection",
            json={"collection_names": [collection], "query": "page"},
        )


def _write(actor, collection: str):
    with actor.client() as client:
        return client.post(
            "/api/v1/retrieval/process/text",
            json={"name": "planted", "content": "planted text", "collection_name": collection},
        )


def test_identical_searches_by_two_users_mint_different_collections(mint, make_user):
    owner, other = make_user(), make_user()

    owners_collection = mint(owner, "open webui release notes")
    others_collection = mint(other, "open webui release notes")

    assert owners_collection != others_collection, (
        "two users running the same web search share one collection, so each can read and "
        "overwrite the pages the other fetched (#26706)"
    )
    assert owners_collection.startswith(f"web-search-{owner.id}-")


def test_another_user_cannot_read_a_minted_collection(mint, make_user):
    owner, other = make_user(), make_user()
    collection = mint(owner, "quarterly figures")

    assert _read(other, collection).status_code == 403, (
        f"a second user read {collection!r}, the pages someone else's search fetched (#26706)"
    )


def test_another_user_cannot_write_a_minted_collection(mint, make_user):
    owner, other = make_user(), make_user()
    collection = mint(owner, "quarterly figures")

    assert _write(other, collection).status_code == 403, (
        f"a second user wrote into {collection!r}, replacing the pages the owner's next answer "
        "is grounded in (#26706)"
    )


@pytest.mark.parametrize("access", [_read, _write], ids=["read", "write"])
@pytest.mark.parametrize(
    "foreign_name",
    [
        "web-search-abc123def456",
        "web-search-{owner}-abc123def456",
        "web-search-{owner}",
        "web-search-{other}x-abc123",
    ],
)
def test_no_foreign_web_search_name_is_admitted(mock_embeddings, make_user, access, foreign_name):
    owner, other = make_user(), make_user()
    name = foreign_name.format(owner=owner.id, other=other.id)

    assert access(other, name).status_code == 403, (
        f"{name!r} was admitted to a user who does not own it, so the web-search namespace is "
        "still not owner-scoped (#26706)"
    )


def test_the_owner_reads_and_writes_its_own_collection(mint, make_user):
    owner = make_user()
    collection = mint(owner, "milan weather")

    read = _read(owner, collection)
    assert read.status_code == 200, read.text
    assert PAGE_TEXT in json.dumps(read.json())
    assert _write(owner, collection).status_code == 200


def test_an_admin_still_reads_any_collection(mint, make_user, admin):
    collection = mint(make_user(), "milan weather")

    assert _read(admin, collection).status_code == 200


def test_repeating_a_search_reuses_one_collection(mint, make_user):
    owner = make_user()

    first = mint(owner, "milan weather")

    assert mint(owner, "milan weather") == first
    assert mint(owner, "rome weather") != first
