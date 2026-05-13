"""Regression tests for open-webui/open-webui#24560 — SafeWebBaseLoader._fetch
crashes with TypeError on duplicate `allow_redirects` kwarg.

In v0.9.5 and on dev (confirmed by da-astro on 2026-05-12), every
async web fetch through the default `safe_web` loader raises:

    TypeError: aiohttp.client.ClientSession.get() got multiple values
    for keyword argument 'allow_redirects'

because `SafeWebBaseLoader.__init__()` merges `allow_redirects` into
`self.requests_kwargs` (line ~503 of retrieval/web/utils.py) and
`_fetch()` then both unpacks that dict AND passes `allow_redirects`
explicitly to `session.get()` (line ~519-522). The TypeError fires at
the call expression, before aiohttp's `get()` body runs.

The failure is then caught by langchain's `_fetch_with_rate_limit` with
`continue_on_failure=True`, surfacing in the logs only as
    "Error fetching <url>, skipping due to continue_on_failure=True"
with no specific cause — so web search silently degrades to no
results and the model answers without grounding.

Bug introduced by PR #24524 (the SSRF redirect protection). Fix
proposed in PR #24600 / #24601 / #24602 — drop the explicit
`allow_redirects=` from the `session.get()` call; the value is already
in `self.requests_kwargs`, so the SSRF protection is preserved.

Source: https://github.com/open-webui/open-webui/issues/24560
"""

from __future__ import annotations

import pytest


class _FakeAiohttpResponse:
    """Async-context-manager mock that yields a fake aiohttp response.

    The bug fires at the `session.get(...)` call expression, before
    aiohttp's `get()` body runs, so we never reach this class on
    buggy code. On fixed code, _fetch awaits .text() and returns.
    """

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return None

    def raise_for_status(self) -> None:
        return None

    async def text(self) -> str:
        return ""


class _FakeAiohttpSession:
    """Replaces aiohttp.ClientSession. Records the kwargs that were
    actually passed to .get() so the test can assert against them.
    """

    last_call_kwargs: dict | None = None

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return None

    def get(self, url: str, **kwargs):
        _FakeAiohttpSession.last_call_kwargs = kwargs
        return _FakeAiohttpResponse()


@pytest.mark.regression
@pytest.mark.asyncio
async def test_safe_web_base_loader_fetch_does_not_pass_allow_redirects_twice(
    retrieval_web_utils_module, monkeypatch
) -> None:
    """Regression for open-webui/open-webui#24560.

    Invokes `SafeWebBaseLoader._fetch` with `aiohttp.ClientSession`
    swapped for a fake. The TypeError this issue describes fires at
    the call expression `session.get(url, **kwargs, allow_redirects=X)`
    when `kwargs` already contains `allow_redirects` — Python rejects
    the duplicate before the function body runs, so our fake `.get()`
    never gets a chance to be called.

    On fixed code, _fetch awaits the response and returns "".
    """
    utils = retrieval_web_utils_module
    SafeWebBaseLoader = utils.SafeWebBaseLoader

    loader = SafeWebBaseLoader(web_path="https://example.invalid/x")

    # Make sure the loader's __init__ actually injects allow_redirects —
    # otherwise this whole test would be a no-op (the bug requires the
    # kwarg to be in self.requests_kwargs).
    assert "allow_redirects" in (loader.requests_kwargs or {}), (
        "SafeWebBaseLoader.__init__ no longer seeds allow_redirects into "
        "requests_kwargs — review whether this test still pins #24560."
    )

    # Reset state so we can detect whether .get() was reached at all.
    _FakeAiohttpSession.last_call_kwargs = None
    monkeypatch.setattr(utils.aiohttp, "ClientSession", _FakeAiohttpSession)

    try:
        result = await loader._fetch("https://example.invalid/x", retries=1)
    except TypeError as e:
        msg = str(e)
        if "allow_redirects" in msg and "multiple values" in msg:
            pytest.fail(
                f"Regression of open-webui/open-webui#24560: "
                f"SafeWebBaseLoader._fetch() passes allow_redirects twice "
                f"(once via self.requests_kwargs, once explicitly to "
                f"session.get()). aiohttp raised: {msg}"
            )
        raise

    # Fixed-path assertions: .get() was actually reached, and the merged
    # kwargs still carry allow_redirects (so the SSRF redirect protection
    # is preserved by the fix, not just removed).
    assert _FakeAiohttpSession.last_call_kwargs is not None, (
        "session.get() was never reached — something other than the "
        "duplicate-kwarg bug stopped _fetch from running."
    )
    assert "allow_redirects" in _FakeAiohttpSession.last_call_kwargs, (
        "The fix must keep allow_redirects in the session.get() kwargs "
        "to preserve the SSRF redirect protection from PR #24524."
    )
    assert result == ""
