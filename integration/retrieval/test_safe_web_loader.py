"""Regression for open-webui/open-webui#24560: every async page fetch by the default loader failed.

PR #24524 merged `allow_redirects` into `SafeWebBaseLoader.requests_kwargs` and `_fetch` also
passed it to `session.get()` explicitly, so each call raised `TypeError: got multiple values for
keyword argument 'allow_redirects'` before a request left. `continue_on_failure` logged it as a
bare "Error fetching" and web search went on with empty pages, so the model answered without
grounding. Web search is the caller of the async path (`aload`); attaching a link uses the
synchronous `load`, which never had the bug. PRs #24600-#24602 dropped the explicit argument.

Twin of unit/retrieval/test_safe_web_loader.py.

Discriminates: passes on dev bbfa876af; passing `allow_redirects` to `session.get()` next to the
merged options fails both tests (the page is never requested and its document is empty).
"""

from __future__ import annotations

import pytest

from harness.listener import text_answer
from harness.web_retrieval import (
    LOCAL_WEB_FETCH,
    save_web_settings,
    serve_search_results,
    web_settings_restored,
)

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

PAGE_TEXT = "Kestrels migrate south in the autumn"


@pytest.fixture(scope="module")
def fetching_instance(instance_with):
    return instance_with(LOCAL_WEB_FETCH)


@pytest.fixture
def search(fetching_instance, listener):
    """(admin client, `run(bypass_embedding)`): a web search whose one result is a local page."""
    page = f"{listener.base_url}/kestrels"
    listener.route("GET", "/kestrels", text_answer(f"<html><body><p>{PAGE_TEXT}</p></body></html>"))

    with fetching_instance.client() as client, web_settings_restored(client):

        def run(bypass_embedding: bool) -> dict:
            save_web_settings(
                client,
                **serve_search_results(listener, [page]),
                WEB_LOADER_ENGINE="safe_web",
                BYPASS_WEB_SEARCH_WEB_LOADER=False,
                BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL=bypass_embedding,
            )
            searched = client.post(
                "/api/v1/retrieval/process/web/search", json={"queries": ["kestrel"]}
            )
            assert searched.status_code == 200, searched.text
            assert listener.requests_to("/kestrels"), "the search result page was never requested"
            return searched.json()

        yield client, run


def test_a_searched_page_is_returned_with_its_text(search):
    _, run = search

    documents = run(bypass_embedding=True)["docs"]

    assert [document["content"].strip() for document in documents] == [PAGE_TEXT], (
        "the async loader returned an empty page (#24560)"
    )


def test_a_searched_page_is_stored_with_its_text(search):
    client, run = search

    collection_name = run(bypass_embedding=False)["collection_names"][0]

    stored = client.post(
        "/api/v1/retrieval/query/doc", json={"collection_name": collection_name, "query": "kestrel"}
    )
    assert stored.status_code == 200, stored.text
    assert [text.strip() for text in stored.json()["documents"][0]] == [PAGE_TEXT]
