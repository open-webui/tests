"""Regression and smoke tests for the Firecrawl engine, driven through the retrieval API.

Firecrawl searches (`/v2/search`) and loads pages (`/v2/scrape`). A local service plays
Firecrawl here, so every request the instance builds and every answer shape it has to read is
seen from outside.

- #23966 Bug 2 (list parsing): v0.9.1 read `data.web` only and raised on the flat
  `{"data": [...]}` shape that v1 and some deployments answer, which the broad `except` turned
  into "no results". Both shapes list their hits now.
- An unset key sends no `Authorization` header: a self-hosted Firecrawl without auth rejects an
  empty `Bearer `.
- 429 and 5xx answers are retried, after `Retry-After` when one is given.
- A page Firecrawl returns blank, or cannot scrape in three attempts, is refused with a 400
  that names the link instead of attaching nothing (#31347, PR #31351). Those two tests fail
  on dev until that fix merges.

Twin of unit/retrieval/test_firecrawl.py, which keeps the audit that every web module calling
`requests` imports it (#23966 Bug 1, broad) and the timeout parsing no route reaches.

Discriminates: passes on dev bbfa876af; reading only `data.web` fails the list shape, an
unconditional `Bearer` header fails the keyless case, always appending `/v2` fails the `/v2`
base URLs, no retries fail both retry cases, ignoring `Retry-After` fails the wait, sending the
timeout in seconds fails the timeout cases and dropping the `[:count]` cut fails the result
count.
"""

from __future__ import annotations

import time

import pytest

from harness.listener import json_answer
from harness.web_retrieval import LOCAL_WEB_FETCH, save_web_settings, web_settings_restored

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

API_KEY = "fc-key"


@pytest.fixture(scope="module")
def fetching_instance(instance_with):
    return instance_with(LOCAL_WEB_FETCH)


@pytest.fixture
def admin_client(fetching_instance):
    with fetching_instance.client() as client, web_settings_restored(client):
        yield client


def use_firecrawl(client, listener, **settings):
    save_web_settings(
        client,
        **{
            "ENABLE_WEB_SEARCH": True,
            "WEB_SEARCH_ENGINE": "firecrawl",
            "WEB_LOADER_ENGINE": "firecrawl",
            "FIRECRAWL_API_BASE_URL": listener.base_url,
            "FIRECRAWL_API_KEY": API_KEY,
            "FIRECRAWL_TIMEOUT": "",
            "BYPASS_WEB_SEARCH_WEB_LOADER": True,
            "BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL": True,
            **settings,
        },
    )


def scraped(markdown, **data):
    return json_answer({"success": True, "data": {"markdown": markdown, **data}})


def read_page(client, page):
    response = client.post("/api/v1/retrieval/process/web?process=false", json={"url": page})
    assert response.status_code == 200, response.text
    return response.json()["content"]


def refused_link(client, page) -> str:
    response = client.post("/api/v1/retrieval/process/web?process=false", json={"url": page})
    assert response.status_code == 400, response.text
    return response.json()["detail"]


def search(client, query="who won the race"):
    return client.post("/api/v1/retrieval/process/web/search", json={"queries": [query]})


def test_a_page_is_scraped_through_v2_with_the_key(admin_client, listener):
    page = f"{listener.base_url}/article"
    listener.route("POST", "/v2/scrape", scraped("# Heading\n\nBody text."))
    use_firecrawl(admin_client, listener)

    content = read_page(admin_client, page)

    assert content == "# Heading\n\nBody text."
    call = listener.requests_to("/v2/scrape")[0]
    assert call.headers["Authorization"] == f"Bearer {API_KEY}"
    assert call.json()["url"] == page
    assert "markdown" in call.json()["formats"]


@pytest.mark.parametrize("suffix", ["/", "/v2", "/v2/"])
def test_every_spelling_of_the_base_url_reaches_v2_scrape(admin_client, listener, suffix):
    listener.route("POST", "/v2/scrape", scraped("scraped"))
    use_firecrawl(admin_client, listener, FIRECRAWL_API_BASE_URL=listener.base_url + suffix)

    assert read_page(admin_client, f"{listener.base_url}/article") == "scraped"


def test_no_key_sends_no_authorization_header(admin_client, listener):
    listener.route("POST", "/v2/scrape", scraped("open deployment"))
    use_firecrawl(admin_client, listener, FIRECRAWL_API_KEY="")

    assert read_page(admin_client, f"{listener.base_url}/article") == "open deployment"
    assert "Authorization" not in listener.requests_to("/v2/scrape")[0].headers


@pytest.mark.parametrize(("setting", "sent"), [("30", 30000), ("1000", 300000), ("", None)])
def test_the_scrape_timeout_is_sent_in_bounded_milliseconds(admin_client, listener, setting, sent):
    listener.route("POST", "/v2/scrape", scraped("scraped"))
    use_firecrawl(admin_client, listener, FIRECRAWL_TIMEOUT=setting)

    read_page(admin_client, f"{listener.base_url}/article")

    assert listener.requests_to("/v2/scrape")[0].json().get("timeout") == sent


def test_blank_markdown_is_refused_naming_the_link(admin_client, listener):
    listener.route("POST", "/v2/scrape", scraped("   "))
    use_firecrawl(admin_client, listener)

    page = f"{listener.base_url}/article"
    assert page in refused_link(admin_client, page)


def test_a_rate_limited_scrape_waits_for_retry_after_and_succeeds(admin_client, listener):
    answers = [(429, {"Retry-After": "2"}, b"{}"), scraped("second attempt")]
    arrivals = []

    def rate_limited_once(request):
        arrivals.append(time.monotonic())
        return answers.pop(0)

    listener.route("POST", "/v2/scrape", rate_limited_once)
    use_firecrawl(admin_client, listener)

    assert read_page(admin_client, f"{listener.base_url}/article") == "second attempt"
    assert len(arrivals) == 2
    assert arrivals[1] - arrivals[0] >= 1.9, "the retry did not wait for Retry-After"


def test_persistent_server_errors_give_up_after_three_attempts_naming_the_link(
    admin_client, listener
):
    listener.route("POST", "/v2/scrape", (503, {"Retry-After": "0"}, b"{}"))
    use_firecrawl(admin_client, listener)

    page = f"{listener.base_url}/article"
    assert page in refused_link(admin_client, page)
    assert len(listener.requests_to("/v2/scrape")) == 3


def hit(url, **fields):
    return {"url": url, "title": f"title of {url}", "description": f"about {url}", **fields}


@pytest.mark.parametrize("shape", ["v2", "flat list"])
def test_search_hits_are_read_from_either_answer_shape(admin_client, listener, shape):
    hits = [hit(f"{listener.base_url}/a"), hit(f"{listener.base_url}/b")]
    data = {"web": hits} if shape == "v2" else hits
    listener.route("POST", "/v2/search", json_answer({"success": True, "data": data}))
    use_firecrawl(admin_client, listener, WEB_SEARCH_RESULT_COUNT=5)

    response = search(admin_client)

    assert response.status_code == 200, response.text
    items = response.json()["items"]
    assert [item["link"] for item in items] == [entry["url"] for entry in hits]
    assert [item["snippet"] for item in items] == [entry["description"] for entry in hits]


def test_search_asks_for_the_result_count_and_keeps_that_many(admin_client, listener):
    hits = [hit(f"{listener.base_url}/{index}") for index in range(5)]
    listener.route("POST", "/v2/search", json_answer({"data": {"web": hits}}))
    use_firecrawl(admin_client, listener, WEB_SEARCH_RESULT_COUNT=2)

    response = search(admin_client, query="who won the race")

    assert len(response.json()["items"]) == 2
    call = listener.requests_to("/v2/search")[0]
    assert call.json()["limit"] == 2
    assert call.json()["query"] == "who won the race"
    assert call.headers["Authorization"] == f"Bearer {API_KEY}"


def test_search_hit_urls_fall_back_to_link_and_metadata(admin_client, listener):
    by_link, by_metadata = f"{listener.base_url}/link", f"{listener.base_url}/meta"
    hits = [
        {"link": by_link, "description": "found by link"},
        {"metadata": {"sourceURL": by_metadata, "description": "found by metadata"}},
    ]
    listener.route("POST", "/v2/search", json_answer({"data": {"web": hits}}))
    use_firecrawl(admin_client, listener, WEB_SEARCH_RESULT_COUNT=5)

    response = search(admin_client)

    assert [item["link"] for item in response.json()["items"]] == [by_link, by_metadata]


def test_a_rejected_search_is_no_results(admin_client, listener):
    listener.route("POST", "/v2/search", json_answer({"error": "Unauthorized"}, status=401))
    use_firecrawl(admin_client, listener)

    assert search(admin_client).status_code == 404


def test_searched_pages_are_scraped_with_their_title_and_description(admin_client, listener):
    page = f"{listener.base_url}/article"
    listener.route("POST", "/v2/search", json_answer({"data": {"web": [hit(page)]}}))
    metadata = {"title": "Example Page", "description": "An example."}
    listener.route("POST", "/v2/scrape", scraped("Real content.", metadata=metadata))
    use_firecrawl(
        admin_client, listener, WEB_SEARCH_RESULT_COUNT=1, BYPASS_WEB_SEARCH_WEB_LOADER=False
    )

    response = search(admin_client)

    assert response.status_code == 200, response.text
    [doc] = response.json()["docs"]
    assert doc["content"] == "Real content."
    assert doc["metadata"] == {"source": page, **metadata}
