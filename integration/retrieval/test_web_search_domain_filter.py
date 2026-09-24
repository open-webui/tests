"""Regression: the web search domain filter must match a result's host, not its host and port.

open-webui 0.10.2 fix `688bda09f`: `get_filtered_results` took the domain from
`urlparse(url).netloc`, which keeps the port and any userinfo. The filter matches host labels,
so `127.0.0.1:8443` never matched a `!127.0.0.1` block entry and the result was loaded anyway,
and `127.0.0.1:8443` never matched an allow entry for `127.0.0.1` either. The fix reads
`urlparse(url).hostname`.

A local search engine lists pages on a local listener; an IP host needs no DNS and passes the
URL validator the filter applies first, which a bare `localhost` does not.

Twin of unit/retrieval/test_web_search_domain_filter.py.

Discriminates: passes on dev bbfa876af; matching on `netloc` fails the port and userinfo cases
of the block list (the page is loaded) and the allow list (no result survives); the bare host
passes on both.
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

PAGE_TEXT = "Kestrel counts by county"


@pytest.fixture(scope="module")
def fetching_instance(instance_with):
    return instance_with(LOCAL_WEB_FETCH)


@pytest.fixture
def search_with_filter(fetching_instance, listener):
    """`search_with_filter(result_link, filter_list)` runs one search; returns the response."""
    listener.route("GET", "/kestrels", text_answer(f"<p>{PAGE_TEXT}</p>"))
    with fetching_instance.client() as client, web_settings_restored(client):

        def search(link: str, filter_list: list[str]):
            save_web_settings(
                client,
                **serve_search_results(listener, [link.format(port=listener.port)]),
                WEB_SEARCH_DOMAIN_FILTER_LIST=filter_list,
                WEB_LOADER_ENGINE="safe_web",
                BYPASS_WEB_SEARCH_WEB_LOADER=False,
                BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL=True,
            )
            return client.post(
                "/api/v1/retrieval/process/web/search", json={"queries": ["kestrels"]}
            )

        yield search


@pytest.mark.parametrize(
    "link",
    [
        "http://127.0.0.1:{port}/kestrels",
        "http://someone@127.0.0.1:{port}/kestrels",
        "http://127.0.0.1/kestrels",
    ],
    ids=["with-port", "with-userinfo", "bare"],
)
def test_a_blocked_host_is_filtered_out_of_the_results(search_with_filter, listener, link):
    searched = search_with_filter(link, ["!127.0.0.1"])

    assert searched.status_code == 404, f"the blocked result was kept: {searched.text}"
    assert listener.requests_to("/kestrels") == [], "the blocked page was loaded"


def test_an_allowed_host_with_a_port_is_kept(search_with_filter):
    searched = search_with_filter("http://127.0.0.1:{port}/kestrels", ["127.0.0.1"])

    assert searched.status_code == 200, f"the allowed result was dropped: {searched.text}"
    assert PAGE_TEXT in searched.json()["docs"][0]["content"]
