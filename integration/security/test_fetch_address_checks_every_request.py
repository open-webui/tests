"""Regression: the web fetch filter list must reach every request the server makes.

open-webui 0.11.1 fix `e3e4bd87d` (#27823). A `WEB_FETCH_FILTER_LIST` entry written as an address
range matched nothing at all, because entries were compared as DNS labels: `!10.0.0.0/8` blocked
nothing. Entries naming an address or a range are now matched by containment. The list also ran
once, on the submitted URL, so a redirect hop reached a listed host; it now runs per request on
both transports, the requests adapter behind `/process/web` and the aiohttp connector behind
`/process/url`. The redirect gap needs `AIOHTTP_CLIENT_ALLOW_REDIRECTS=true`.

Twin of unit/security/test_fetch_address_checks_every_request.py. The checks on resolved
addresses (DNS answers, an IPv4 address carried in IPv6) stay there: no local name resolves to a
listed address.

Discriminates: passes on dev `bbfa876af`; with the containment match removed from
`_host_matches_pattern` the range tests fail (the listed address is fetched), and with the
per-request hooks removed from `_SSRFSafeAdapter.send` and `_SSRFSafeConnector.connect` the
redirect tests fail (the listed hop is fetched).
"""

from __future__ import annotations

import pytest

from harness.listener import listening, text_answer

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

LISTED_ADDRESS = "127.0.0.2"
LOCAL_FETCH = {
    "ENABLE_LOCAL_WEB_FETCH": "true",
    "WEB_FETCH_FILTER_LIST": f"!{LISTED_ADDRESS}/32,!localhost",
    "AIOHTTP_CLIENT_ALLOW_REDIRECTS": "true",
}
FETCH_ROUTES = ["/api/v1/retrieval/process/web", "/api/v1/retrieval/process/url"]

PAGE = "<html><body><p>public page text</p></body></html>"
SECRET_PAGE = "<html><body><p>internal secret text</p></body></html>"


@pytest.fixture(scope="module")
def local_fetch(instance_with):
    return instance_with(LOCAL_FETCH)


def _fetch(instance, route: str, url: str):
    with instance.client() as client:
        return client.post(f"{route}?process=false", json={"url": url})


def _redirect_to(location: str):
    return 302, {"Location": location}, b""


@pytest.mark.parametrize("route", FETCH_ROUTES)
def test_a_range_entry_blocks_the_addresses_inside_it(local_fetch, route):
    with listening(host=LISTED_ADDRESS) as listed_service:
        listed_service.route("GET", "/page", text_answer(SECRET_PAGE))
        response = _fetch(local_fetch, route, f"{listed_service.base_url}/page")
        reached = list(listed_service.received)

    assert reached == [], (
        f"{route} fetched {LISTED_ADDRESS} although `!{LISTED_ADDRESS}/32` lists it; a range "
        "entry matched nothing, so the operator's block did nothing (#27823)"
    )
    assert response.status_code == 400, response.text


@pytest.mark.parametrize("route", FETCH_ROUTES)
def test_a_redirect_hop_to_a_listed_host_is_never_fetched(local_fetch, listener, route):
    hop_url = f"http://localhost:{listener.port}/secret"
    listener.route("GET", "/start", _redirect_to(hop_url))
    listener.route("GET", "/secret", text_answer(SECRET_PAGE))

    response = _fetch(local_fetch, route, f"{listener.base_url}/start")

    assert listener.requests_to("/secret") == [], (
        f"{route} followed a redirect to {hop_url} although `!localhost` lists it; the filter "
        "list only ran on the submitted URL, never on a hop (#27823)"
    )
    assert "internal secret text" not in response.text


@pytest.mark.parametrize("route", FETCH_ROUTES)
def test_an_unlisted_address_is_still_fetched(local_fetch, listener, route):
    listener.route("GET", "/page", text_answer(PAGE))

    response = _fetch(local_fetch, route, f"{listener.base_url}/page")

    assert response.status_code == 200, response.text
    assert "public page text" in response.json()["content"]


@pytest.mark.parametrize("route", FETCH_ROUTES)
def test_a_redirect_to_an_unlisted_host_is_still_followed(local_fetch, listener, route):
    listener.route("GET", "/start", _redirect_to(f"{listener.base_url}/landing"))
    listener.route("GET", "/landing", text_answer(PAGE))

    response = _fetch(local_fetch, route, f"{listener.base_url}/start")

    assert response.status_code == 200, response.text
    assert "public page text" in response.json()["content"]
