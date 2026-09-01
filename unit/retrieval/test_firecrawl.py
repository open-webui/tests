"""Tests for open-webui's Firecrawl integration
(`backend/open_webui/retrieval/web/firecrawl.py`).

Two flavors:

  Regressions — pinned to known issues / PRs. They lock in the specific
    failure mode so it can't come back silently.
        - open-webui/open-webui#23966 — Bug 1 (NameError), Bug 2 (list parsing)

  Smoke tests — exercise the integration with mocked HTTP. They prove
    that *if* a user had a Firecrawl API key, the request/response/retry
    plumbing would work correctly. No real network, no API key required.

The Firecrawl bugs in #23966 are silently swallowed by `continue_on_failure`
and broad try/except handlers inside Open WebUI, so an HTTP-level
integration test against /api/v1/retrieval/process/web can't observe
them. These tests therefore drive the backend Python module directly via
the `firecrawl_module` fixture (see conftest.py).
"""

from __future__ import annotations

import re
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Regressions — open-webui/open-webui#23966
# =============================================================================

# Match `requests.<word>(`, excluding things like `self.requests_kwargs`.
_REQUESTS_CALL_RE = re.compile(r"(?<![A-Za-z0-9_.])requests\.[A-Za-z_][A-Za-z0-9_]*\s*\(")
# Match `import requests`, `import requests as X`, `from requests import ...`.
_IMPORT_REQUESTS_RE = re.compile(
    r"^\s*(?:import\s+requests(?:\s|$|,|;|\sas\s)|from\s+requests\b)",
    re.MULTILINE,
)


@pytest.mark.regression
def test_retrieval_web_modules_import_requests(open_webui_backend: Path) -> None:
    """Regression for open-webui/open-webui#23966 (Bug 1, NameError).

    In v0.9.1, SafeFireCrawlLoader.lazy_load() in
    backend/open_webui/retrieval/web/utils.py called requests.post(...)
    without importing the `requests` module. Every web-fetch with
    WEB_LOADER_ENGINE='firecrawl' crashed with
        NameError: name 'requests' is not defined
    swallowed by continue_on_failure and surfaced as an empty result.

    Any module under retrieval/web/ that calls `requests.<x>(...)` must
    also `import requests` at module scope.
    """
    web_dir = open_webui_backend / "open_webui" / "retrieval" / "web"
    assert web_dir.is_dir(), web_dir

    offenders: list[str] = []
    for py in sorted(web_dir.rglob("*.py")):
        text = py.read_text(encoding="utf-8")
        if not _REQUESTS_CALL_RE.search(text):
            continue
        if not _IMPORT_REQUESTS_RE.search(text):
            offenders.append(py.relative_to(open_webui_backend).as_posix())

    assert not offenders, (
        "Regression of open-webui/open-webui#23966 (Bug 1): the following "
        f"file(s) call `requests.<x>(...)` without `import requests` at "
        f"module scope: {offenders}"
    )


@pytest.mark.regression
def test_search_firecrawl_does_not_attribute_error_on_list_data(
    firecrawl_module: ModuleType,
) -> None:
    """Regression for open-webui/open-webui#23966 (Bug 2, list parsing).

    The v0.9.1 parser did `response.get('data', {}).get('web', [])` and
    raised `AttributeError: 'list' object has no attribute 'get'` when
    Firecrawl handed back `{"data": [ ... ]}` (frost19k's comment on the
    issue). The function's outer try/except swallowed it, returning [].

    Both the buggy and the fixed code observationally `return []` on
    list-shape input, so we intercept `log.error` and assert the
    specific AttributeError stringification doesn't appear there.
    """
    fc = firecrawl_module

    list_shape: dict[str, Any] = {
        "success": True,
        "data": [
            {"url": "https://example.com/a", "title": "A", "description": "desc-a"},
            {"url": "https://example.com/b", "title": "B", "description": "desc-b"},
        ],
    }

    patch_target = _resolve_http_patch_target(fc, list_shape)
    if patch_target is None:
        pytest.skip("Could not find a stable patch target in firecrawl module")

    with patch.object(fc, "log") as mock_log, patch_target:
        results = fc.search_firecrawl(
            firecrawl_url="https://api.firecrawl.dev",
            firecrawl_api_key="dummy-key",
            query="anything",
            count=2,
        )

    assert isinstance(results, list), repr(results)

    logged_errors = [
        str(call.args[0]) if call.args else "" for call in mock_log.error.call_args_list
    ]
    bug2_signature = "'list' object has no attribute 'get'"
    matches = [msg for msg in logged_errors if bug2_signature in msg]
    assert not matches, (
        "Regression of open-webui/open-webui#23966 (Bug 2, list parsing): "
        f"search_firecrawl logged the AttributeError signature when given "
        f"a list-shape Firecrawl response: {matches!r}"
    )


def _resolve_http_patch_target(fc: ModuleType, response_body: dict[str, Any]):
    """Return a context manager that patches whatever HTTP entry point
    the firecrawl module currently uses. Lets the same test work across
    pre- and post-#23934 refactors of the file."""
    if hasattr(fc, "request_firecrawl_json"):
        return patch(
            "open_webui.retrieval.web.firecrawl.request_firecrawl_json",
            return_value=response_body,
        )
    if hasattr(fc, "requests"):
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = response_body
        fake_resp.raise_for_status.return_value = None
        return patch(
            "open_webui.retrieval.web.firecrawl.requests.post",
            return_value=fake_resp,
        )
    return None


# =============================================================================
# Smoke tests — would the integration work if we had a real API key?
# =============================================================================
#
# These don't need a Firecrawl account. They drive the module's pure
# helpers and high-level functions with mocked HTTP, and assert the
# request shape, response parsing, and retry behavior match what a real
# Firecrawl v2 deployment would expect.
# =============================================================================


# ----- URL & header construction ---------------------------------------------


@pytest.mark.parametrize(
    "configured_url, path, expected",
    [
        # Default v2 path is appended when base lacks /v2.
        ("https://api.firecrawl.dev", "search", "https://api.firecrawl.dev/v2/search"),
        ("https://api.firecrawl.dev", "scrape", "https://api.firecrawl.dev/v2/scrape"),
        # Trailing slash on base, leading slash on path — no doubled /.
        ("https://api.firecrawl.dev/", "/search", "https://api.firecrawl.dev/v2/search"),
        # A self-hosted base that already ends in /v2 is not doubled.
        ("https://fc.internal/v2", "search", "https://fc.internal/v2/search"),
        ("https://fc.internal/v2/", "search", "https://fc.internal/v2/search"),
        # None falls back to the public default.
        (None, "search", "https://api.firecrawl.dev/v2/search"),
        ("", "search", "https://api.firecrawl.dev/v2/search"),
    ],
)
def test_build_firecrawl_url(firecrawl_module, configured_url, path, expected) -> None:
    """Request URLs target Firecrawl's v2 endpoints, regardless of how
    the operator configured FIRECRAWL_API_BASE_URL."""
    assert firecrawl_module.build_firecrawl_url(configured_url, path) == expected


def test_build_firecrawl_headers_includes_bearer_token(firecrawl_module) -> None:
    headers = firecrawl_module.build_firecrawl_headers("test-key-abc")
    assert headers["Authorization"] == "Bearer test-key-abc"
    assert headers["Content-Type"] == "application/json"


@pytest.mark.parametrize("api_key", [None, ""])
def test_build_firecrawl_headers_omits_auth_when_api_key_unset(firecrawl_module, api_key) -> None:
    """An unset API key must omit the Authorization header entirely, not
    send an empty `Bearer `. Self-hosted Firecrawl deployments that allow
    unauthenticated access reject the empty credential, so sending it
    turns a working setup into a 401."""
    headers = firecrawl_module.build_firecrawl_headers(api_key)
    assert "Authorization" not in headers
    assert headers["Content-Type"] == "application/json"


# ----- Timeout parsing -------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (30, 30.0),
        ("30", 30.0),
        ("45.5", 45.5),
        (None, None),
        ("", None),
        ("not-a-number", None),
        (-5, None),
        (0, None),
    ],
)
def test_get_firecrawl_timeout_seconds(firecrawl_module, value, expected) -> None:
    """FIRECRAWL_TIMEOUT can be set as int, float, str, or unset.
    Non-positive / invalid values disable the timeout (None)."""
    assert firecrawl_module.get_firecrawl_timeout_seconds(value) == expected


def test_get_firecrawl_scrape_timeout_ms_converts_and_clamps(firecrawl_module) -> None:
    """Firecrawl v2 wants scrape timeouts in milliseconds, clamped to
    [1000, 300000]. Configured seconds go through the conversion."""
    fc = firecrawl_module
    assert fc.get_firecrawl_scrape_timeout_ms(30) == 30000
    # Below the minimum is bumped up.
    assert fc.get_firecrawl_scrape_timeout_ms(0.5) == 1000
    # Above the maximum is capped.
    assert fc.get_firecrawl_scrape_timeout_ms(10_000) == 300_000
    # Unset stays unset.
    assert fc.get_firecrawl_scrape_timeout_ms(None) is None


def test_get_firecrawl_client_timeout_seconds_pads_for_scrape(firecrawl_module) -> None:
    """The HTTP client timeout must outlast Firecrawl's own scrape
    timeout, otherwise we cut the request before the server can answer.
    """
    fc = firecrawl_module
    assert fc.get_firecrawl_client_timeout_seconds(30) == 40.0
    # Fallback when no timeout is configured.
    assert fc.get_firecrawl_client_timeout_seconds(None) == 70.0
    assert fc.get_firecrawl_client_timeout_seconds(None, fallback=120) == 130.0


# ----- Result-URL extraction -------------------------------------------------


def test_get_firecrawl_result_url_prefers_direct_fields(firecrawl_module) -> None:
    """`url` and `link` on the result are preferred over metadata
    fields — they're what Firecrawl actually exposes for search hits."""
    fc = firecrawl_module
    assert fc.get_firecrawl_result_url({"url": "https://a"}) == "https://a"
    assert fc.get_firecrawl_result_url({"link": "https://b"}) == "https://b"


def test_get_firecrawl_result_url_falls_back_to_metadata(firecrawl_module) -> None:
    """Scrape results don't carry `url` at the top level — they put it
    under metadata.sourceURL / source_url / url. The helper must look
    there before giving up."""
    fc = firecrawl_module
    for key in ("url", "sourceURL", "source_url"):
        assert fc.get_firecrawl_result_url({"metadata": {key: "https://m"}}) == "https://m"


def test_get_firecrawl_result_url_returns_empty_when_absent(firecrawl_module) -> None:
    assert firecrawl_module.get_firecrawl_result_url({}) == ""
    assert firecrawl_module.get_firecrawl_result_url({"metadata": {}}) == ""


# ----- search_firecrawl: happy path with v2 dict-shape -----------------------


def test_search_firecrawl_parses_v2_dict_shape(firecrawl_module) -> None:
    """The v2 /search endpoint returns `{"data": {"web": [...]}}`.
    search_firecrawl must lift the result list out and wrap each entry
    in a SearchResult with link / title / snippet populated."""
    fc = firecrawl_module
    v2_response = {
        "success": True,
        "data": {
            "web": [
                {
                    "url": "https://example.com/a",
                    "title": "Title A",
                    "description": "Snippet A",
                },
                {
                    "url": "https://example.com/b",
                    "title": "Title B",
                    "description": "Snippet B",
                },
            ]
        },
    }

    with patch.object(fc, "request_firecrawl_json", return_value=v2_response):
        results = fc.search_firecrawl(
            firecrawl_url="https://api.firecrawl.dev",
            firecrawl_api_key="dummy",
            query="anything",
            count=5,
        )

    assert len(results) == 2
    assert [r.link for r in results] == ["https://example.com/a", "https://example.com/b"]
    assert [r.title for r in results] == ["Title A", "Title B"]
    assert [r.snippet for r in results] == ["Snippet A", "Snippet B"]


def test_search_firecrawl_truncates_to_requested_count(firecrawl_module) -> None:
    """A query for `count=2` must yield at most 2 SearchResults even if
    Firecrawl over-delivers."""
    fc = firecrawl_module
    over_delivered = {
        "data": {
            "web": [
                {"url": f"https://example.com/{i}", "title": f"T{i}", "description": f"S{i}"}
                for i in range(10)
            ]
        }
    }

    with patch.object(fc, "request_firecrawl_json", return_value=over_delivered):
        results = fc.search_firecrawl(
            firecrawl_url="https://api.firecrawl.dev",
            firecrawl_api_key="dummy",
            query="anything",
            count=2,
        )

    assert len(results) == 2


def test_search_firecrawl_returns_empty_on_empty_results(firecrawl_module) -> None:
    fc = firecrawl_module
    with patch.object(fc, "request_firecrawl_json", return_value={"data": {"web": []}}):
        results = fc.search_firecrawl("https://api.firecrawl.dev", "k", "q", 5)
    assert results == []


def test_search_firecrawl_swallows_http_errors(firecrawl_module) -> None:
    """When Firecrawl returns a 401/403/404 etc., request_firecrawl_json
    raises HTTPError. search_firecrawl catches it and returns [] so the
    chat flow can fall back to non-grounded answers instead of 500-ing.
    """
    fc = firecrawl_module
    import requests as _requests

    err = _requests.HTTPError("401 Unauthorized")
    with patch.object(fc, "request_firecrawl_json", side_effect=err):
        results = fc.search_firecrawl("https://api.firecrawl.dev", "bad-key", "q", 5)

    assert results == []


# ----- scrape_firecrawl_url: happy path --------------------------------------


def test_scrape_firecrawl_url_returns_document_with_content_and_metadata(
    firecrawl_module,
) -> None:
    """A successful /v2/scrape returns markdown + metadata. The helper
    must wrap that into a langchain Document with the source URL and
    title/description preserved."""
    fc = firecrawl_module
    scrape_response = {
        "data": {
            "markdown": "# Page heading\n\nBody text here.",
            "metadata": {
                "title": "Example Page",
                "description": "An example.",
                "sourceURL": "https://example.com/article",
            },
        }
    }

    with patch.object(fc, "request_firecrawl_json", return_value=scrape_response):
        doc = fc.scrape_firecrawl_url(
            firecrawl_url="https://api.firecrawl.dev",
            firecrawl_api_key="dummy",
            url="https://example.com/article",
        )

    assert doc is not None
    assert doc.page_content == "# Page heading\n\nBody text here."
    assert doc.metadata["source"] == "https://example.com/article"
    assert doc.metadata["title"] == "Example Page"
    assert doc.metadata["description"] == "An example."


def test_scrape_firecrawl_url_returns_none_when_content_empty(firecrawl_module) -> None:
    """Whitespace-only markdown isn't worth surfacing as a document."""
    fc = firecrawl_module
    with patch.object(fc, "request_firecrawl_json", return_value={"data": {"markdown": "   "}}):
        doc = fc.scrape_firecrawl_url("https://api.firecrawl.dev", "dummy", "https://example.com")
    assert doc is None


def test_scrape_firecrawl_url_returns_none_when_no_data(firecrawl_module) -> None:
    fc = firecrawl_module
    with patch.object(fc, "request_firecrawl_json", return_value={}):
        doc = fc.scrape_firecrawl_url("https://api.firecrawl.dev", "dummy", "https://example.com")
    assert doc is None


def test_scrape_firecrawl_url_falls_back_to_request_url_for_source(firecrawl_module) -> None:
    """If Firecrawl doesn't echo back the URL in metadata, the
    requested URL is used as the source — so downstream RAG can still
    cite a link."""
    fc = firecrawl_module
    response = {"data": {"markdown": "Real content."}}
    with patch.object(fc, "request_firecrawl_json", return_value=response):
        doc = fc.scrape_firecrawl_url(
            "https://api.firecrawl.dev", "dummy", "https://requested.example.com/x"
        )
    assert doc is not None
    assert doc.metadata["source"] == "https://requested.example.com/x"


# ----- request_firecrawl_json: retry & error semantics -----------------------


def _fake_response(
    status: int, body: dict[str, Any] | None = None, headers: dict[str, str] | None = None
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.headers = headers or {}
    resp.json.return_value = body or {}
    if status >= 400:
        import requests as _requests

        resp.raise_for_status.side_effect = _requests.HTTPError(f"HTTP {status}")
    else:
        resp.raise_for_status.return_value = None
    return resp


def test_request_firecrawl_json_succeeds_on_first_try(firecrawl_module) -> None:
    """Happy path: 200 OK on the first call, no retries, body returned."""
    fc = firecrawl_module
    body = {"success": True, "data": {"web": []}}

    with (
        patch("time.sleep") as fake_sleep,
        patch.object(
            fc.requests, "request", return_value=_fake_response(200, body)
        ) as fake_request,
    ):
        out = fc.request_firecrawl_json(
            "POST",
            "https://api.firecrawl.dev/v2/search",
            headers={"Authorization": "Bearer x"},
            json={"query": "q"},
            timeout=30,
        )

    assert out == body
    assert fake_request.call_count == 1
    assert fake_sleep.call_count == 0


def test_request_firecrawl_json_retries_on_429_then_succeeds(firecrawl_module) -> None:
    """A 429 is retryable — wrapper waits, retries, returns the eventual
    200 body. Total HTTP calls = 2, sleeps = 1."""
    fc = firecrawl_module
    body = {"data": {"web": []}}
    responses = [
        _fake_response(429, headers={"Retry-After": "1"}),
        _fake_response(200, body),
    ]

    with (
        patch("time.sleep") as fake_sleep,
        patch.object(fc.requests, "request", side_effect=responses) as fake_request,
    ):
        out = fc.request_firecrawl_json(
            "POST",
            "https://api.firecrawl.dev/v2/search",
            headers={},
            json={},
            timeout=30,
        )

    assert out == body
    assert fake_request.call_count == 2
    assert fake_sleep.call_count == 1


def test_request_firecrawl_json_raises_after_persistent_5xx(firecrawl_module) -> None:
    """Three consecutive 500s exhaust the retry budget; the last
    response's raise_for_status() propagates."""
    fc = firecrawl_module
    import requests as _requests

    responses = [_fake_response(500) for _ in range(3)]
    with patch("time.sleep"), patch.object(fc.requests, "request", side_effect=responses):
        with pytest.raises(_requests.HTTPError):
            fc.request_firecrawl_json(
                "POST",
                "https://api.firecrawl.dev/v2/search",
                headers={},
                json={},
                timeout=30,
            )


def test_request_firecrawl_json_raises_on_persistent_connection_error(
    firecrawl_module,
) -> None:
    """A truly down Firecrawl (connection refused, DNS fail) eventually
    surfaces the requests.ConnectionError to the caller."""
    fc = firecrawl_module
    import requests as _requests

    with (
        patch("time.sleep"),
        patch.object(fc.requests, "request", side_effect=_requests.ConnectionError("refused")),
    ):
        with pytest.raises(_requests.ConnectionError):
            fc.request_firecrawl_json(
                "POST",
                "https://api.firecrawl.dev/v2/search",
                headers={},
                json={},
                timeout=30,
            )


def test_request_firecrawl_json_respects_retry_after_header(firecrawl_module) -> None:
    """The wrapper honors `Retry-After` (clamped to a reasonable max) so
    Firecrawl rate limits don't cause us to hammer the API."""
    fc = firecrawl_module
    body = {"data": {"web": []}}
    responses = [
        _fake_response(429, headers={"Retry-After": "3"}),
        _fake_response(200, body),
    ]

    with (
        patch("time.sleep") as fake_sleep,
        patch.object(fc.requests, "request", side_effect=responses),
    ):
        fc.request_firecrawl_json(
            "POST",
            "https://api.firecrawl.dev/v2/search",
            headers={},
            json={},
            timeout=30,
        )

    # One retry, slept with the Retry-After value (3s).
    assert fake_sleep.call_count == 1
    (delay,), _ = fake_sleep.call_args
    assert delay == 3.0


# ----- End-to-end smoke: search request lands on /v2/search ------------------


def test_search_firecrawl_calls_v2_search_with_bearer_token(firecrawl_module) -> None:
    """End-to-end check on the request side: search_firecrawl builds a
    POST to .../v2/search with the configured API key in the Bearer
    header. This is the request a real Firecrawl deployment would
    receive — if it passes here, the integration is wired correctly and
    the only thing missing is a real key."""
    fc = firecrawl_module
    body = {"data": {"web": []}}

    with (
        patch("time.sleep"),
        patch.object(
            fc.requests, "request", return_value=_fake_response(200, body)
        ) as fake_request,
    ):
        fc.search_firecrawl(
            firecrawl_url="https://api.firecrawl.dev",
            firecrawl_api_key="sk-abc",
            query="who won the F1 race",
            count=3,
        )

    fake_request.assert_called_once()
    _, kwargs = fake_request.call_args
    args = fake_request.call_args.args
    # `requests.request(method, url, ...)` — method+url are positional.
    method, url = args[0], args[1]
    assert method == "POST"
    assert url == "https://api.firecrawl.dev/v2/search"
    assert kwargs["headers"]["Authorization"] == "Bearer sk-abc"
    assert kwargs["json"]["query"] == "who won the F1 race"
    assert kwargs["json"]["limit"] == 3


def test_scrape_firecrawl_url_calls_v2_scrape_with_payload(firecrawl_module) -> None:
    """Same smoke check for the scrape path: lands on /v2/scrape with
    the URL in the payload and markdown requested."""
    fc = firecrawl_module
    body = {"data": {"markdown": "hello", "metadata": {}}}

    with (
        patch("time.sleep"),
        patch.object(
            fc.requests, "request", return_value=_fake_response(200, body)
        ) as fake_request,
    ):
        fc.scrape_firecrawl_url(
            firecrawl_url="https://api.firecrawl.dev",
            firecrawl_api_key="sk-abc",
            url="https://example.com/article",
        )

    method, url = fake_request.call_args.args[:2]
    payload = fake_request.call_args.kwargs["json"]
    assert method == "POST"
    assert url == "https://api.firecrawl.dev/v2/scrape"
    assert payload["url"] == "https://example.com/article"
    assert "markdown" in payload["formats"]
