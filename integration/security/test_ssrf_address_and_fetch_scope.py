"""Regression: IPv6 addresses that carry an internal IPv4 address, and web loaders that pace.

`1717b49` (v0.11.0) made the web fetch guard pull out the IPv4 address an IPv6 address carries
(IPv4-compatible `::a.b.c.d`, NAT64 `64:ff9b::/96`, mapped, 6to4, teredo) and judge it as well.
v0.10.2 asked `ipaddress.is_global` alone, which CPython 3.11 answers True for `::7f00:1` and
`64:ff9b::7f00:1`, so a URL on either literal passed the check and the instance connected to it.
A URL that cannot be read is a 400 either way, so the tests read the log line naming the
refused address.

`1e0ab8471` (#27528, issue #26079) unshadowed the `time` module in the web loader. `from datetime
import datetime, time, timedelta` made pacing call `datetime.time.sleep`, which raised inside
each URL's `try`, so every paced page after the first was dropped and logged as an SSL
verification failure. Microsoft Web IQ is the engine whose search path runs that pacing.

Twin of unit/security/test_ssrf_address_and_fetch_scope.py, which keeps the Playwright
interception hooks and the public-address path (showing it means fetching a public host).

Discriminates: passes on dev bbfa876af; with `_embedded_ipv4` returning nothing the four
IPv4-compatible and NAT64 cases fail (no refusal logged), and with `time` imported from
`datetime` again the paced search returns one page of two.
"""

from __future__ import annotations

import pytest

from harness.listener import json_answer, text_answer
from harness.web_retrieval import (
    LOCAL_WEB_FETCH,
    save_web_settings,
    serve_search_results,
    web_settings_restored,
)

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

PROCESS_WEB = "/api/v1/retrieval/process/web"


def read_page(actor, url):
    with actor.client() as client:
        return client.post(PROCESS_WEB, json={"url": url})


@pytest.mark.parametrize(
    ("address", "embedded"),
    [
        ("::7f00:1", "127.0.0.1"),  # IPv4-compatible
        ("::a00:1", "10.0.0.1"),
        ("64:ff9b::7f00:1", "127.0.0.1"),  # NAT64 well-known prefix
        ("64:ff9b::a00:1", "10.0.0.1"),
    ],
)
def test_ipv6_carrying_an_internal_ipv4_is_refused(instance, user, address, embedded):
    log_offset = instance.log_size()

    response = read_page(user, f"http://[{address}]/")

    assert response.status_code == 400
    assert f"Blocked non-global address: {embedded}" in instance.log_since(log_offset), (
        f"http://[{address}]/ was not refused for the {embedded} it carries"
    )


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1",
        "[::ffff:127.0.0.1]",  # IPv4-mapped
        "[2002:7f00:1::]",  # 6to4
        "[2001:0:4136:e378:8000:63bf:3fff:fdd2]",  # teredo
        "[64:ff9b:1::7f00:1]",  # NAT64 local-use prefix
        "[::1]",
    ],
)
def test_every_internal_address_spelling_is_refused(instance, user, listener, host):
    listener.route("GET", "/secret", text_answer("internal only"))
    log_offset = instance.log_size()

    response = read_page(user, f"http://{host}:{listener.port}/secret")

    assert response.status_code == 400
    assert listener.received == [], f"{host} reached the local service"
    assert "Blocked" in instance.log_since(log_offset)  # by address or by the default filter list


@pytest.fixture(scope="module")
def fetching_instance(instance_with):
    return instance_with(LOCAL_WEB_FETCH)


def browse_answer(request):
    page_url = request.json()["url"]
    return json_answer({"content": f"text of {page_url}", "url": page_url})


@pytest.mark.parametrize("pages_per_second", [2, 0])
def test_paced_web_search_loads_every_page(fetching_instance, listener, pages_per_second):
    pages = [f"{listener.base_url}/first", f"{listener.base_url}/second"]
    listener.route("POST", "/browse", browse_answer)

    with fetching_instance.client() as client, web_settings_restored(client):
        save_web_settings(
            client,
            **serve_search_results(listener, pages),
            WEB_LOADER_ENGINE="microsoft_web_iq",
            MICROSOFT_WEB_IQ_API_BASE_URL=listener.base_url,
            MICROSOFT_WEB_IQ_API_KEY="web-iq-key",
            WEB_LOADER_CONCURRENT_REQUESTS=pages_per_second,
            BYPASS_WEB_SEARCH_WEB_LOADER=False,
            BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL=True,
        )
        log_offset = fetching_instance.log_size()
        response = client.post("/api/v1/retrieval/process/web/search", json={"queries": ["pace"]})

    assert response.status_code == 200, response.text
    loaded = [doc["metadata"]["source"] for doc in response.json()["docs"]]
    assert loaded == pages, f"pacing dropped a page: {loaded}"
    assert "SSL verification failed" not in fetching_instance.log_since(log_offset)
