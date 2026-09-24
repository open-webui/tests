"""Regression: the web fetch filter list judges every address a fetched host resolves to.

open-webui 0.11.1 fix `e3e4bd87d` (#27823). The addresses a host resolved to were never judged
against the filter list at the connection layer, so a DNS answer pointing at a listed range was
dialled; `_SSRFSafeConnector._resolve_host` now checks every answer, including the IPv4 address an
IPv6 answer carries. The same fix made entries naming an address or a range match by containment.

The range entries and the per-request hooks are pinned over HTTP in
integration/security/test_fetch_address_checks_every_request.py. What stays here cannot be reached
there: no local name resolves to a listed address, so the DNS answers come from a scripted
resolver, and the matching rules of `is_host_allowed` are a pure function.

Discriminates: passes on dev `bbfa876af`; with `_SSRFSafeConnector._resolve_host` removed the
listed answers are dialled, and with the containment match removed from `_host_matches_pattern`
the range and spelling rows fail.
"""

from __future__ import annotations

import socket

import pytest

from harness.listener import listening

aiohttp = pytest.importorskip("aiohttp")

from aiohttp.abc import AbstractResolver  # noqa: E402

pytestmark = pytest.mark.regression

LISTED_ADDRESS = "127.0.0.2"


class ScriptedResolver(AbstractResolver):
    """Answers every lookup with one address, the way a rebinding DNS server would."""

    answer = ""

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def resolve(self, host, port=0, family=socket.AF_INET):
        answer_family = socket.AF_INET6 if ":" in self.answer else socket.AF_INET
        return [
            {
                "hostname": host,
                "host": self.answer,
                "port": port,
                "family": answer_family,
                "proto": 0,
                "flags": socket.AI_NUMERICHOST,
            }
        ]

    async def close(self) -> None:
        pass


@pytest.fixture
def dns_answers(retrieval_web_utils_module, monkeypatch):
    """Point every lookup the fetch session makes at the scripted resolver."""
    monkeypatch.setattr(
        retrieval_web_utils_module, "WEB_FETCH_FILTER_LIST", [f"!{LISTED_ADDRESS}/32"]
    )
    # Loopback must pass the non-global rule, so only the filter list can refuse it.
    monkeypatch.setattr(retrieval_web_utils_module, "ENABLE_LOCAL_WEB_FETCH", True)
    monkeypatch.setattr(aiohttp.connector, "DefaultResolver", ScriptedResolver)
    monkeypatch.setattr(ScriptedResolver, "answer", "")
    return ScriptedResolver


async def _fetch_status(utils, url: str) -> int:
    async with utils.get_ssrf_safe_session() as session:
        async with session.get(url) as response:
            return response.status


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [LISTED_ADDRESS, f"::ffff:{LISTED_ADDRESS}"])
async def test_a_dns_answer_inside_a_listed_range_is_never_dialled(
    retrieval_web_utils_module, dns_answers, answer
):
    dns_answers.answer = answer
    with listening(host=LISTED_ADDRESS) as listed_service:
        with pytest.raises(ValueError):
            await _fetch_status(
                retrieval_web_utils_module, f"http://rebind.example:{listed_service.port}/"
            )
        reached = list(listed_service.received)

    assert reached == [], (
        f"rebind.example answered {answer} and the fetch dialled it although "
        f"`!{LISTED_ADDRESS}/32` lists it; resolved addresses were never judged (#27823)"
    )


@pytest.mark.asyncio
async def test_a_dns_answer_outside_the_listed_range_still_connects(
    retrieval_web_utils_module, dns_answers
):
    dns_answers.answer = "127.0.0.1"
    with listening() as service:
        status = await _fetch_status(
            retrieval_web_utils_module, f"http://cdn.example:{service.port}/"
        )
        reached = list(service.received)

    assert status == 404
    assert len(reached) == 1


@pytest.mark.parametrize(
    "host, filter_list, allowed",
    [
        # An allow entry written as a range admits the addresses inside it.
        ("10.1.2.3", ["10.0.0.0/8"], True),
        ("203.0.113.7", ["203.0.113.0/24"], True),
        # An address entry matches any spelling of that address.
        ("fd00:ec2:0:0:0:0:0:254", ["!fd00:ec2::254"], False),
        ("FD00:EC2::254", ["!fd00:ec2::254"], False),
        ("fd00:0ec2::0254", ["!fd00:ec2::254"], False),
        ("fd00:ec2::254", ["!fd00::/8"], False),
        # A hostname is never inside an address range.
        ("corp.com", ["!10.0.0.0/8"], True),
        # Hostname entries keep matching on DNS label boundaries.
        ("api.corp.com", ["!corp.com"], False),
        ("corp.com", ["!corp.com"], False),
        ("evilcorp.com", ["!corp.com"], True),
        ("api.corp.com", ["corp.com"], True),
        ("other.example", ["corp.com"], False),
        # An address entry still matches itself spelled the same way, and nothing else.
        ("169.254.169.254", ["!169.254.169.254"], False),
        ("169.254.169.253", ["!169.254.169.254"], True),
        # A block entry beats an allow entry for the same host.
        ("api.corp.com", ["corp.com", "!api.corp.com"], False),
        # Any of the supplied hosts, a name with its addresses, can trigger a block.
        (["corp.com", "10.1.2.3"], ["!corp.com"], False),
        # No filter list configured means no filtering.
        ("anything.example", [], True),
        ("anything.example", None, True),
    ],
)
def test_filter_list_matching_rules(misc_module, host, filter_list, allowed):
    assert misc_module.is_host_allowed(host, filter_list) is allowed, (
        f"is_host_allowed({host!r}, {filter_list!r}) should be {allowed} (#27823)"
    )


def test_search_results_are_filtered_against_a_range_entry(web_search_main_module, monkeypatch):
    """A range entry has to trigger the address lookup and then match the answer."""

    def resolve_to_internal(host, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.1.2.3", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", resolve_to_internal)

    results = web_search_main_module.get_filtered_results(
        [{"link": "https://intranet.example.com/report"}], ["!10.0.0.0/8"]
    )

    assert results == [], "a result resolving to 10.1.2.3 survived `!10.0.0.0/8` (#27823)"


def test_search_results_pass_through_an_unrelated_filter_list(web_search_main_module):
    kept = {"link": "https://good.example/a"}
    dropped = {"link": "https://blocked.example/b"}
    assert web_search_main_module.get_filtered_results([kept, dropped], ["!blocked.example"]) == [
        kept
    ]


def test_search_results_are_untouched_without_a_filter_list(web_search_main_module):
    results = [{"link": "https://good.example/a"}]
    assert web_search_main_module.get_filtered_results(results, []) is results


@pytest.mark.parametrize("url", ["ftp://example.com/x", "file:///etc/passwd", "not a url", ""])
def test_validate_url_rejects_non_http_and_malformed_urls(retrieval_web_utils_module, url):
    with pytest.raises(ValueError):
        retrieval_web_utils_module.validate_url(url)
