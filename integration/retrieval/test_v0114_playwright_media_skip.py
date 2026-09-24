"""Regression: the Playwright web loader must drop media requests before fetching them.

open-webui 0.11.4 fix `ae01ef9c9` (PR #29742, issue #29741): SafePlaywrightURLLoader routes
every request a page makes through the backend's own HTTP client and buffers the whole body
before the browser sees it, and a browser cancelling a media request never reaches that
download. A page with a few dozen audio players pulled every file in full for a text
extraction that reads none of them, so loading crawled or timed out. Image, media and font
requests are now aborted in the interceptor before any fetch.

The instance runs its own headless Chromium here, the way a deployment without
`PLAYWRIGHT_WS_URL` does, and loads a local page through both interceptors: the synchronous
one behind an attached link and the async one behind web search.

Twin of unit/retrieval/test_v0114_playwright_media_skip.py.

Discriminates: passes on dev bbfa876af; with no resource type dropped in the interceptors the
server fetches the page's media and both cases fail.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.listener import text_answer
from harness.web_retrieval import (
    LOCAL_WEB_FETCH,
    save_web_settings,
    serve_search_results,
    web_settings_restored,
)

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.requires_browser,
]

PAGE_TEXT = "Osprey field notes"
PAGE = f"""<html><head><link rel="stylesheet" href="/style.css"><script src="/app.js"></script>
</head><body><p>{PAGE_TEXT}</p><img src="/photo.png">
<audio src="/clip.mp3" preload="auto" autoplay></audio>
<video src="/film.mp4" preload="auto"></video></body></html>"""
STYLE = "@font-face { font-family: Brand; src: url(/brand.woff2); } p { font-family: Brand; }"
MEDIA = {
    "/photo.png": "image/png",
    "/brand.woff2": "font/woff2",
    "/clip.mp3": "audio/mpeg",
    "/film.mp4": "video/mp4",
}
PAGE_RESOURCES = ["/style.css", "/app.js"]


def chromium_installed() -> bool:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        return Path(playwright.chromium.executable_path).exists()


@pytest.fixture(scope="module")
def browsing_instance(instance_with):
    if not chromium_installed():
        pytest.skip(
            "no Chromium for the instance's Playwright loader (playwright install chromium)"
        )
    return instance_with(LOCAL_WEB_FETCH)


@pytest.fixture
def media_page(browsing_instance, listener):
    """(admin client, page URL) with the Playwright loader selected and the page served."""
    listener.route("GET", "/media", text_answer(PAGE))
    listener.route("GET", "/style.css", text_answer(STYLE, content_type="text/css"))
    listener.route("GET", "/app.js", text_answer("document.title = 'x';", "text/javascript"))
    for path, content_type in MEDIA.items():
        listener.route("GET", path, (200, {"Content-Type": content_type}, b"\x00" * 4096))
    page = f"{listener.base_url}/media"
    with browsing_instance.client() as client, web_settings_restored(client):
        save_web_settings(
            client,
            **serve_search_results(listener, [page]),
            WEB_LOADER_ENGINE="playwright",
            PLAYWRIGHT_WS_URL="",
            BYPASS_WEB_SEARCH_WEB_LOADER=False,
            BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL=True,
        )
        yield client, page


def load_attached_link(client, page: str) -> str:
    loaded = client.post("/api/v1/retrieval/process/web?process=false", json={"url": page})
    assert loaded.status_code == 200, loaded.text
    return loaded.json()["content"]


def load_search_result(client, page: str) -> str:
    searched = client.post("/api/v1/retrieval/process/web/search", json={"queries": ["osprey"]})
    assert searched.status_code == 200, searched.text
    return searched.json()["docs"][0]["content"]


@pytest.mark.parametrize(
    "load", [load_attached_link, load_search_result], ids=["attached-link", "web-search"]
)
def test_media_is_never_fetched_while_the_page_loads(media_page, listener, load):
    client, page = media_page

    text = load(client, page)

    assert PAGE_TEXT in text
    fetched_media = [path for path in MEDIA if listener.requests_to(path)]
    assert fetched_media == [], f"the server downloaded {fetched_media} for a text extraction"
    fetched_resources = [path for path in PAGE_RESOURCES if listener.requests_to(path)]
    assert fetched_resources == PAGE_RESOURCES, "the page's own resources were not fetched"
