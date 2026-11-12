"""Regression: the Playwright loader returned only the site menu for a page with two `<main>`s.

Issue #28643, fix `4ef7e35b8` (#31644): the HTML parser behind the loader keeps the first `<main>`
of a page and drops the rest, so a page whose first `<main>` holds the site menu and a later one the
content came back as its menu alone. When a page has more than one `<main>` the loader now unwraps
them and reads the whole page. A page with a single `<main>` still loads only that element.

The instance runs its own headless Chromium, as in test_v0114_playwright_media_skip, and loads a
local page through both loaders: the synchronous one behind an attached link and the async one
behind web search.

Issue #34272: the same parser drops `<details>` and `<summary>` with everything inside them, so an
API reference that keeps its parameter descriptions in collapsible sections loaded as its headings
alone. The loader now unwraps them before extraction, open or closed.

Issue #34281: the parser also drops definition lists (`<dl>`, `<dt>`, `<dd>`), `<figure>` and
`<figcaption>` with their text, so a reference page that documents each entry in a definition list
loaded without the entries, and figures came back without their captions or tables.

Discriminates: passes on dev 015dbc861; with the unwrapping call removed from both loaders in a
backend copy the page with two `<main>` elements loads as its menu alone. The single `<main>` case
passes on both. The `<details>` case passes on fix 0c715d7b7 and fails on dev
31a8c5442 (the page loads without the text inside its collapsible sections). The definition list
and figure case passes on fix 8290b2f95 and fails on dev 31a8c5442 (the page loads as its heading
alone).
"""

from __future__ import annotations

import pytest

from harness.listener import text_answer
from harness.playwright_server import chromium_installed
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

MENU = "Products Support Community"
SPECS = "Antenna gain 24 dBi and range 30 km"
FOOTER = "Copyright Example Networks"
TWO_MAINS = f"""<html><body><header><main>{MENU}</main></header>
<main><h1>Tech specs</h1><p>{SPECS}</p></main><footer>{FOOTER}</footer></body></html>"""
ONE_MAIN = f"""<html><body><header><p>{MENU}</p></header>
<main><h1>Tech specs</h1><p>{SPECS}</p></main><footer><p>{FOOTER}</p></footer></body></html>"""
PARAMETERS = "Parameters"
STREAM_OPTIONS = "Options for streaming the response, sent only when stream is true"
TEMPERATURE = "Sampling temperature between 0 and 2"
COLLAPSIBLES = f"""<html><body><main><h1>Create chat completion</h1>
<details><summary>{PARAMETERS}</summary><p>{STREAM_OPTIONS}</p>
<details open><summary>temperature</summary><div><p>{TEMPERATURE}</p></div></details></details>
</main></body></html>"""
TERM = "max_retries"
DEFINITION = "How many times a failed request is sent again before giving up"
FIGURE_CELL = "Median latency 120 ms"
CAPTION = "Figure 2: latency per region"
DEFINITIONS_AND_FIGURES = f"""<html><body><main><h1>Client options</h1>
<dl><dt>{TERM}</dt><dd><p>{DEFINITION}</p></dd></dl>
<figure><table><tr><td>{FIGURE_CELL}</td></tr></table><figcaption>{CAPTION}</figcaption></figure>
</main></body></html>"""


@pytest.fixture(scope="module")
def browsing_instance(instance_with):
    if not chromium_installed():
        pytest.skip(
            "no Chromium for the instance's Playwright loader (playwright install chromium)"
        )
    return instance_with(LOCAL_WEB_FETCH)


@pytest.fixture
def playwright_client(browsing_instance, listener):
    """Admin client with the Playwright loader selected and web search listing the served page."""
    with browsing_instance.client() as client, web_settings_restored(client):
        save_web_settings(
            client,
            **serve_search_results(listener, [f"{listener.base_url}/page"]),
            WEB_LOADER_ENGINE="playwright",
            PLAYWRIGHT_WS_URL="",
            BYPASS_WEB_SEARCH_WEB_LOADER=False,
            BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL=True,
        )
        yield client


def load_attached_link(client, page: str) -> str:
    loaded = client.post("/api/v1/retrieval/process/web?process=false", json={"url": page})
    assert loaded.status_code == 200, loaded.text
    return loaded.json()["content"]


def load_search_result(client, page: str) -> str:
    searched = client.post("/api/v1/retrieval/process/web/search", json={"queries": ["specs"]})
    assert searched.status_code == 200, searched.text
    return searched.json()["docs"][0]["content"]


LOADERS = pytest.mark.parametrize(
    "load", [load_attached_link, load_search_result], ids=["attached-link", "web-search"]
)


@LOADERS
def test_a_page_with_two_main_elements_loads_its_content(playwright_client, listener, load):
    listener.route("GET", "/page", text_answer(TWO_MAINS))

    text = load(playwright_client, f"{listener.base_url}/page")

    assert SPECS in text, f"the page loaded without its content: {text!r} (#28643)"


@LOADERS
def test_a_page_with_one_main_element_loads_only_that_element(playwright_client, listener, load):
    listener.route("GET", "/page", text_answer(ONE_MAIN))

    text = load(playwright_client, f"{listener.base_url}/page")

    assert SPECS in text
    assert MENU not in text and FOOTER not in text


@LOADERS
def test_text_inside_collapsible_sections_is_loaded(playwright_client, listener, load):
    listener.route("GET", "/page", text_answer(COLLAPSIBLES))

    text = load(playwright_client, f"{listener.base_url}/page")

    missing = [part for part in (PARAMETERS, STREAM_OPTIONS, TEMPERATURE) if part not in text]
    assert not missing, f"the page loaded without its collapsible sections: {text!r} (#34272)"


@LOADERS
def test_text_inside_definition_lists_and_figures_is_loaded(playwright_client, listener, load):
    listener.route("GET", "/page", text_answer(DEFINITIONS_AND_FIGURES))

    text = load(playwright_client, f"{listener.base_url}/page")

    missing = [part for part in (TERM, DEFINITION, FIGURE_CELL, CAPTION) if part not in text]
    assert not missing, f"the page loaded without its definition list or figure: {text!r} (#34281)"
