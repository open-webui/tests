"""Regression: the OpenAI and Ollama proxies must strip upstream headers case-insensitively.

open-webui 0.11.4 fix `44f9a4f7f` (PR #29843): `_clean_proxy_headers` in
`routers/ollama.py` and `routers/openai.py` dropped the upstream's
Content-Encoding, Content-Length and Transfer-Encoding (stale after aiohttp
decoded the body; forwarding them makes clients try to decompress an
already-decoded payload and hit ZlibError), but it compared header names
case-sensitively against title-case entries. Anything served by uvicorn
(vLLM, LiteLLM) sends lowercase header names, so none of those headers were
stripped at all, and the clients failed to decompress a body the server had
already decoded. The filter now compares lowercased names and also drops the
upstream's Server and Date, which uvicorn re-adds as its own (nginx in front
logs "upstream sent duplicate header line" for both when they are forwarded).

Discriminates: passes on dev 344ea5306, fails on `44f9a4f7f^` (lowercase and
uppercase names survive the filter, and Server/Date are not in the strip set in
any casing).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression

STRIPPED_NAMES = [
    "Content-Encoding",
    "Content-Length",
    "Transfer-Encoding",
    "Server",
    "Date",
]
CASINGS = ["title", "lower", "upper"]

KEPT_HEADERS = {
    "Content-Type": "application/json",
    "X-Request-Id": "abc123",
    "x-custom": "value",
}


def _cased(name: str, casing: str) -> str:
    if casing == "lower":
        return name.lower()
    if casing == "upper":
        return name.upper()
    return name


@pytest.fixture(scope="session", params=["ollama", "openai"])
def proxy_router(owui_module, request):
    """Both proxy modules carry the same filter with the same contract."""
    return owui_module(f"open_webui.routers.{request.param}")


# ---------------------------------------------------------------------------
# narrow: the fix itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", STRIPPED_NAMES)
@pytest.mark.parametrize("casing", CASINGS)
def test_stale_headers_are_stripped_in_every_casing(proxy_router, name, casing):
    cleaned = proxy_router._clean_proxy_headers({_cased(name, casing): "some-value"})

    assert cleaned == {}, f"{_cased(name, casing)!r} survived the proxy header filter"


def test_the_filter_is_driven_by_a_lowercase_strip_set(proxy_router):
    """Pins the mechanism, not just the outcome: the set entries are lowercase."""
    assert all(name == name.lower() for name in proxy_router._STRIP_PROXY_HEADERS), (
        "a title-case entry in the strip set reverts to case-sensitive matching"
    )


# ---------------------------------------------------------------------------
# broad: the invariant the bug was an instance of
# ---------------------------------------------------------------------------


def test_a_mixed_case_header_batch_is_cleaned_wholesale(proxy_router):
    raw = {
        "content-encoding": "gzip",
        "Content-Encoding": "br",
        "date": "now",
        "Server": "uvicorn",
        "content-type": "application/json",
    }

    cleaned = proxy_router._clean_proxy_headers(raw)

    assert cleaned == {"content-type": "application/json"}


@pytest.mark.parametrize("name", STRIPPED_NAMES)
def test_stripping_uses_the_whole_name_not_a_prefix(proxy_router, name):
    """A header whose name merely starts with a stripped name must survive."""
    cleaned = proxy_router._clean_proxy_headers({f"{name}-Suffix": "value"})

    assert cleaned == {f"{name}-Suffix": "value"}


# ---------------------------------------------------------------------------
# nearby: behaviour that was already correct
# ---------------------------------------------------------------------------


def test_unrelated_headers_are_kept_verbatim(proxy_router):
    cleaned = proxy_router._clean_proxy_headers(dict(KEPT_HEADERS))

    assert cleaned == KEPT_HEADERS


def test_an_empty_header_batch_stays_empty(proxy_router):
    assert proxy_router._clean_proxy_headers({}) == {}


def test_header_values_are_not_modified(proxy_router):
    cleaned = proxy_router._clean_proxy_headers({"Content-Disposition": "inline; a,b"})

    assert cleaned == {"Content-Disposition": "inline; a,b"}
