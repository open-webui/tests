"""Regression: the web fetch address checks must run on every outgoing request.

open-webui 0.11.1 fix `e3e4bd87d` (#27823) closes three gaps in the server-side
fetch guards.

1. A filter entry written as an address RANGE matched nothing at all, silently.
   `_host_matches_pattern` compared DNS labels, so `10.0.0.0/8` could never equal
   or suffix-match `10.1.2.3`. An operator who blocked their internal range got no
   blocking, and an operator who allow-listed one got every fetch rejected. Entries
   that parse as an address or a CIDR range are now matched by containment, which
   also makes an address entry match any spelling of itself (`fd00:ec2::254`
   matches `fd00:ec2:0:0:0:0:0:254`).

2. The filter list was consulted once, in `validate_url`, on the originally
   submitted URL. A proxied request presents the proxy's host at the connection
   layer and a pooled connection skips resolution entirely, so redirect hops were
   fetched without the list ever being applied. It now runs as a per-request hook
   on both transports: `_SSRFSafeConnector.connect` for aiohttp and
   `_SSRFSafeAdapter.send` for requests.

3. The addresses a host resolved to were never judged at the connection layer at
   all. `_SSRFSafeConnector._resolve_host` now runs `_assert_addresses_allowed`
   on every resolution result, and `_embedded_ipv4` unwraps the IPv4 address an
   IPv6 answer can carry (mapped, v4-compatible, 6to4, teredo, NAT64), so a
   blocked IPv4 range cannot be reached by spelling the address in IPv6.

Discriminates: passes on v0.11.1, fails on v0.11.0.

No network is touched: the transport tests replace the base-class `send`/`connect`
with a recorder, so a checkout without the hook records the call instead of
dialling out.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import aiohttp
import pytest
import requests
import requests.adapters

pytestmark = pytest.mark.regression


# --- narrow: a filter entry naming a range or an address matches by containment ---


@pytest.mark.parametrize(
    "host, entry",
    [
        ("10.1.2.3", "!10.0.0.0/8"),
        ("192.168.7.9", "!192.168.0.0/16"),
        ("169.254.169.254", "!169.254.0.0/16"),
        ("fd00:ec2::254", "!fd00::/8"),
    ],
)
def test_block_entry_written_as_a_range_matches_addresses_inside_it(misc_module, host, entry):
    assert misc_module.is_host_allowed(host, [entry]) is False, (
        f"{host!r} was allowed through the block entry {entry!r}; a range entry matched "
        "nothing at all, so the operator's block silently did nothing (#27823)"
    )


@pytest.mark.parametrize(
    "host, entry",
    [
        ("10.1.2.3", "10.0.0.0/8"),
        ("203.0.113.7", "203.0.113.0/24"),
    ],
)
def test_allow_entry_written_as_a_range_admits_addresses_inside_it(misc_module, host, entry):
    """The same bug's other face: a range allowlist rejected every fetch."""
    assert misc_module.is_host_allowed(host, [entry]) is True, (
        f"{host!r} was rejected by the allow entry {entry!r}; a range entry matched nothing, "
        "so a range allowlist blocked everything (#27823)"
    )


@pytest.mark.parametrize(
    "host",
    [
        "fd00:ec2:0:0:0:0:0:254",
        "FD00:EC2::254",
        "fd00:0ec2::0254",
    ],
)
def test_address_entry_matches_any_spelling_of_that_address(misc_module, host):
    """`!fd00:ec2::254` is one address, however the request happens to spell it."""
    assert misc_module.is_host_allowed(host, ["!fd00:ec2::254"]) is False, (
        f"{host!r} was allowed; an address entry was compared as text, so a different "
        "spelling of the same address walked past it (#27823)"
    )


def test_search_results_are_filtered_against_a_range_entry(web_search_main_module, monkeypatch):
    """End to end through `get_filtered_results`: a range entry has to trigger the
    address lookup *and* then match, and before the fix it did neither."""
    monkeypatch.setattr(
        web_search_main_module, "resolve_hostname", lambda host: (["10.1.2.3"], [])
    )

    results = web_search_main_module.get_filtered_results(
        [{"link": "https://intranet.example.com/report"}], ["!10.0.0.0/8"]
    )

    assert results == [], (
        "a result resolving to 10.1.2.3 survived the block entry !10.0.0.0/8 (#27823)"
    )


# --- narrow: the filter list runs per request on both transports ---


def _prepared(url: str) -> requests.PreparedRequest:
    return requests.Request("GET", url).prepare()


def test_requests_adapter_applies_the_filter_list_to_every_request(
    retrieval_web_utils_module, monkeypatch
):
    """The requests transport must judge the request destination itself. Before the
    fix nothing was checked here, so a redirect hop reached a filter-listed host."""
    utils = retrieval_web_utils_module
    monkeypatch.setattr(utils, "WEB_FETCH_FILTER_LIST", ["!blocked.example"])

    sent = []

    def _recording_send(self, request, *args, **kwargs):
        sent.append(request.url)
        return "response-sentinel"

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", _recording_send)
    adapter = utils._SSRFSafeAdapter()

    with pytest.raises(ValueError):
        adapter.send(_prepared("http://blocked.example/page"))
    assert sent == [], (
        "the request to blocked.example was handed to the transport; the filter list "
        "only ran on the originally submitted URL (#27823)"
    )

    # Control: an unlisted destination still goes out, so the hook is not blanket-refusing.
    assert adapter.send(_prepared("http://allowed.example/page")) == "response-sentinel"
    assert sent == ["http://allowed.example/page"]


def test_aiohttp_connector_applies_the_filter_list_to_every_request(
    retrieval_web_utils_module, monkeypatch
):
    """Same check on the aiohttp side, driven through the connector the SSRF-safe
    session actually installs."""
    utils = retrieval_web_utils_module
    monkeypatch.setattr(utils, "WEB_FETCH_FILTER_LIST", ["!blocked.example"])

    connected = []

    async def _recording_connect(self, req, traces, timeout):
        connected.append(req.url.host)
        return "connection-sentinel"

    monkeypatch.setattr(aiohttp.TCPConnector, "connect", _recording_connect)

    def _request_to(host: str):
        return SimpleNamespace(url=SimpleNamespace(host=host))

    async def _drive():
        async with utils.get_ssrf_safe_session() as session:
            connector = session.connector

            with pytest.raises(ValueError):
                await connector.connect(_request_to("blocked.example"), [], None)
            assert connected == [], (
                "the connection to blocked.example was opened; the filter list never "
                "reached the connection layer, so redirects and pooled connections "
                "bypassed it (#27823)"
            )

            # Control: an unlisted destination still connects.
            result = await connector.connect(_request_to("allowed.example"), [], None)
            assert result == "connection-sentinel"
            assert connected == ["allowed.example"]

    asyncio.run(_drive())


@pytest.mark.parametrize(
    "resolved",
    ["::ffff:10.1.2.3", "64:ff9b::10.1.2.3"],
)
def test_aiohttp_connector_judges_the_addresses_a_host_resolved_to(
    retrieval_web_utils_module, monkeypatch, resolved
):
    """An IPv6 answer can carry a blocked IPv4 address inside it. Before the fix the
    connector never looked at resolution results, so the blocked range was reachable."""
    utils = retrieval_web_utils_module
    monkeypatch.setattr(utils, "WEB_FETCH_FILTER_LIST", ["!10.0.0.0/8"])
    # Isolate the filter list from the non-global check, which would fire on its own.
    monkeypatch.setattr(utils, "ENABLE_LOCAL_WEB_FETCH", True)

    answer = []

    async def _recording_resolve(self, host, port, traces=None):
        return [{"host": address, "port": port} for address in answer]

    monkeypatch.setattr(aiohttp.TCPConnector, "_resolve_host", _recording_resolve)

    async def _drive():
        async with utils.get_ssrf_safe_session() as session:
            connector = session.connector

            answer[:] = [resolved]
            with pytest.raises(ValueError):
                await connector._resolve_host("rebind.example", 443)

            # Control: an address outside the blocked range still resolves.
            answer[:] = ["2606:4700:4700::1111"]
            results = await connector._resolve_host("cdn.example", 443)
            assert [entry["host"] for entry in results] == ["2606:4700:4700::1111"]

    asyncio.run(_drive())


# --- broad: the surrounding filter-list semantics ---


@pytest.mark.parametrize(
    "host, filter_list, expected",
    [
        # Hostname entries keep matching on DNS label boundaries.
        ("api.corp.com", ["!corp.com"], False),
        ("corp.com", ["!corp.com"], False),
        ("evilcorp.com", ["!corp.com"], True),
        ("api.corp.com", ["corp.com"], True),
        ("other.example", ["corp.com"], False),
        # An address entry still matches itself spelled the same way.
        ("169.254.169.254", ["!169.254.169.254"], False),
        ("169.254.169.253", ["!169.254.169.254"], True),
        # A block entry beats an allow entry for the same host.
        ("api.corp.com", ["corp.com", "!api.corp.com"], False),
        # No filter list configured means no filtering.
        ("anything.example", [], True),
        ("anything.example", None, True),
    ],
)
def test_filter_list_semantics(misc_module, host, filter_list, expected):
    assert misc_module.is_host_allowed(host, filter_list) is expected


def test_a_hostname_is_never_inside_an_address_range(misc_module):
    """A range entry must not swallow hosts that merely fail to parse as addresses."""
    assert misc_module.is_host_allowed("corp.com", ["!10.0.0.0/8"]) is True


def test_any_of_the_supplied_hosts_can_trigger_a_block(misc_module):
    """Callers pass a hostname together with the addresses it resolved to."""
    assert misc_module.is_host_allowed(["corp.com", "10.1.2.3"], ["!corp.com"]) is False


def test_search_results_pass_through_an_unrelated_filter_list(web_search_main_module):
    kept = {"link": "https://good.example/a"}
    dropped = {"link": "https://blocked.example/b"}
    assert web_search_main_module.get_filtered_results(
        [kept, dropped], ["!blocked.example"]
    ) == [kept]


def test_search_results_are_untouched_without_a_filter_list(web_search_main_module):
    results = [{"link": "https://good.example/a"}]
    assert web_search_main_module.get_filtered_results(results, []) is results


# --- nearby: validate_url's other guards ---


@pytest.mark.parametrize("url", ["ftp://example.com/x", "file:///etc/passwd", "not a url", ""])
def test_validate_url_rejects_non_http_and_malformed_urls(retrieval_web_utils_module, url):
    with pytest.raises(ValueError):
        retrieval_web_utils_module.validate_url(url)
