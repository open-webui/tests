"""Web search and loader settings pointed at local stand-ins, and an instance that may fetch them.

The admin panel saves every web search and loader setting as the one `web` section of the
retrieval config, so a change is the whole section with some keys replaced. The built-in loader
refuses loopback pages unless `ENABLE_LOCAL_WEB_FETCH` is on, and loopback pages are the only
ones a test may make the instance fetch, so the tests that load pages run on an instance booted
with `LOCAL_WEB_FETCH`.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

import httpx

from harness.listener import Listener, json_answer

RETRIEVAL_CONFIG = ("/api/v1/retrieval/config", "/api/v1/retrieval/config/update")
LOCAL_WEB_FETCH = {"ENABLE_LOCAL_WEB_FETCH": "true"}


def _save_web_section(client: httpx.Client, web: dict) -> None:
    saved = client.post(RETRIEVAL_CONFIG[1], json={"web": web})
    assert saved.status_code == 200, f"saving the web settings failed: {saved.text}"


def save_web_settings(client: httpx.Client, **changes) -> None:
    """Save `changes` over the admin's web search and loader settings, the rest as they are."""
    current = client.get(RETRIEVAL_CONFIG[0])
    current.raise_for_status()
    _save_web_section(client, {**current.json()["web"], **changes})


@contextlib.contextmanager
def web_settings_restored(client: httpx.Client) -> Iterator[None]:
    """`preserve` for the web settings of an instance from `instance_with`."""
    current = client.get(RETRIEVAL_CONFIG[0])
    current.raise_for_status()
    try:
        yield
    finally:
        _save_web_section(client, current.json()["web"])


def serve_search_results(listener: Listener, links: list[str]) -> dict:
    """Answer as an external search engine listing `links`; returns the settings that use it."""
    results = [{"link": link, "title": link, "snippet": f"snippet of {link}"} for link in links]
    listener.route("POST", "/search", json_answer(results))
    return {
        "ENABLE_WEB_SEARCH": True,
        "WEB_SEARCH_ENGINE": "external",
        "EXTERNAL_WEB_SEARCH_URL": f"{listener.base_url}/search",
        "EXTERNAL_WEB_SEARCH_API_KEY": "search-key",
        "WEB_SEARCH_RESULT_COUNT": len(links),
    }
