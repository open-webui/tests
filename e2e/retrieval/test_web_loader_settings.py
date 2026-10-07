"""Journey: the Loader settings of Admin Settings > Web Search decide what the model reads.

The admin picks a Web Loader Engine in the Web Search tab and fills in its address and key, or
the Fetch URL Content Length Limit, and saves. When a user's model then reads a page with
`fetch_url`, the page comes through that engine: the external loader, Firecrawl, Tavily and
Microsoft Web IQ are each asked for the page and their text is what the model reads and what the
reply's source shows; Playwright reads it in a browser of its own. The length limit cuts what the
model reads and says it was cut. With Verify SSL Certificate on, a page whose certificate no
authority signed is not read; off, it is. With Bypass Embedding and Retrieval on, a webpage the user
attaches reaches the model whole and is never embedded; off, it is embedded first. The loaders
and pages are local services (Tavily, whose address is fixed, behind the proxy of
`harness/search_apis.py`), on an instance that may fetch loopback addresses.

Discriminates: passes on dev ebc6add67; in a backend copy whose `fetch_url` loads every page
with the default loader, every engine case but Playwright's fails and Playwright's sees no
browser connection; with `fetch_url` skipping the length limit the limit case fails; with
`process_web` always embedding the page the bypassed case fails; with the default loader always
verifying certificates the unverified case fails.
"""

from __future__ import annotations

import contextlib
import uuid
from typing import Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.actors import Actor, admin_of, create_user
from harness.listener import json_answer, listening, text_answer
from harness.playwright_server import serving_playwright
from harness.search_apis import search_apis_env, serving_search_apis
from harness.web_retrieval import (
    LOCAL_WEB_FETCH,
    save_web_settings,
    serve_search_results,
    web_settings_restored,
)
from utils.chat_ui import chat_input, conversation, expect_reply, last_reply, send

pytestmark = [
    pytest.mark.journey,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

PAGE_TEXT = "Lighthouse keepers log the fog every hour."
PAGE = f"<html><body><h1>Lighthouse log</h1><p>{PAGE_TEXT}</p></body></html>"
LOADER_KEY = "loader-key"

# engine: (fields by placeholder, `{base}` for the local service; the route it is asked on)
LOADERS = {
    "external": (
        {
            "Enter External Web Loader URL": "{base}/external",
            "Enter External Web Loader API Key": LOADER_KEY,
        },
        "/external",
    ),
    "firecrawl": (
        {"Enter Firecrawl API Base URL": "{base}", "Enter Firecrawl API Key": LOADER_KEY},
        "/v2/scrape",
    ),
    "tavily": ({"Enter Tavily API Key": LOADER_KEY}, "/extract"),
    "microsoft_web_iq": (
        {
            "Enter Microsoft Web IQ API Base URL": "{base}/web-iq",
            "Enter Microsoft Web IQ API Key": LOADER_KEY,
        },
        "/web-iq/browse",
    ),
}


def loader_text(engine: str) -> str:
    return f"Read by the {engine} loader: the fog lifted at dawn."


@pytest.fixture(scope="module")
def services():
    """The search APIs' stand-in, which here also serves the page and plays every loader."""
    with serving_search_apis() as apis:
        listener = apis.listener
        listener.route("GET", "/page", text_answer(PAGE))
        listener.route(
            "POST", "/external", json_answer([{"page_content": loader_text("external")}])
        )
        listener.route(
            "POST", "/v2/scrape", json_answer({"data": {"markdown": loader_text("firecrawl")}})
        )
        listener.route(
            "POST",
            "/extract",
            json_answer({"results": [{"url": "page", "raw_content": loader_text("tavily")}]}),
        )
        listener.route(
            "POST", "/web-iq/browse", json_answer({"content": loader_text("microsoft_web_iq")})
        )
        yield apis


@pytest.fixture(scope="module")
def secure_pages():
    """The page again, over TLS with a certificate no authority signed."""
    with listening(tls=True) as service:
        service.route("GET", "/page", text_answer(PAGE))
        yield service


@pytest.fixture(scope="module")
def reader(instance_with, services):
    launched = instance_with({**LOCAL_WEB_FETCH, **search_apis_env(services)})
    if not launched.serves_frontend:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    return launched


@contextlib.contextmanager
def searching_admin(launched, services) -> Iterator[Actor]:
    admin = admin_of(launched)
    with admin.client() as client, web_settings_restored(client):
        save_web_settings(client, **serve_search_results(services.listener, []))
        yield admin


@pytest.fixture
def reader_admin(reader, services):
    """Web search on, so users may turn it on; the settings come back after the test."""
    with searching_admin(reader, services) as admin:
        yield admin


def web_search_tab(page: Page) -> Locator:
    with page.expect_response(lambda response: response.url.endswith("/retrieval/config")):
        page.goto("/admin/settings/web")
    return page.get_by_role("dialog")


def save(page: Page, settings: Locator) -> None:
    settings.get_by_role("button", name="Save", exact=True).click()
    expect(page.get_by_text("Settings saved successfully!").first).to_be_visible()


def pick_loader(page: Page, engine: str, fields: dict[str, str]) -> None:
    settings = web_search_tab(page)
    settings.get_by_role("combobox").filter(
        has=page.get_by_role("option", name="playwright", exact=True)
    ).select_option(engine)
    for placeholder, value in fields.items():
        settings.get_by_placeholder(placeholder, exact=True).fill(value)
    save(page, settings)


def fetch_in_a_chat(page: Page, reader, link: str, question: str) -> str:
    """The model reads `link` and cites it; returns what `fetch_url` handed the model."""
    answering = reply.answering(question)
    reader.upstream.queue(
        reply.tool_call("fetch_url", {"url": link}, "call_fetch", match=answering),
        reply.text("The log says so [1].", match=answering),
    )
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("button", name="Integrations", exact=True).last.click()
    page.get_by_role("menu").get_by_role("button", name="Web Search").click()
    page.keyboard.press("Escape")
    send(page, question)
    expect_reply(page, "The log says so")
    follow_up = [body for body in reader.upstream.chat_requests() if answering(body)][-1]
    [fetched] = [entry["content"] for entry in follow_up["messages"] if entry["role"] == "tool"]
    return fetched


def open_source(page: Page, link: str) -> Locator:
    expect(conversation(page).get_by_role("button", name="Toggle 1 source")).to_be_visible()
    last_reply(page).get_by_role("button", name=f"View source: {link.split('/')[2]}").click()
    citation = page.get_by_role("dialog")
    expect(citation.get_by_role("link", name=link, exact=True)).to_be_visible()
    return citation


@pytest.mark.parametrize("engine", LOADERS)
def test_the_loader_engine_saved_by_the_admin_reads_the_page_the_model_fetches(
    engine, page_for, reader, reader_admin, services
):
    fields, route = LOADERS[engine]
    filled = {
        placeholder: value.format(base=services.base_url) for placeholder, value in fields.items()
    }
    pick_loader(page_for(reader_admin), engine, filled)
    link = f"{services.base_url}/page"
    asked_before = len(services.listener.requests_to(route))

    page = page_for(create_user(reader))
    fetched = fetch_in_a_chat(page, reader, link, f"What does the {engine} loader say?")

    assert loader_text(engine) in fetched, f"the model did not read the page through {engine}"
    assert PAGE_TEXT not in fetched
    [asked] = services.listener.requests_to(route)[asked_before:]
    assert link in asked.body.decode(), f"{engine} was not asked for the page"
    expect(open_source(page, link)).to_contain_text(loader_text(engine))


def test_the_playwright_loader_reads_the_page_in_its_browser(
    page_for, reader, reader_admin, services
):
    with serving_playwright() as browser_server:
        pick_loader(
            page_for(reader_admin),
            "playwright",
            {"Enter Playwright WebSocket URL": browser_server.ws_url},
        )
        link = f"{services.base_url}/page"
        page = page_for(create_user(reader))
        fetched = fetch_in_a_chat(page, reader, link, "What does the browser read?")

        assert PAGE_TEXT in fetched
        assert browser_server.connections >= 1, "the page was not read in the Playwright browser"
        expect(open_source(page, link)).to_contain_text(PAGE_TEXT)


def test_the_fetch_length_limit_cuts_what_the_model_reads(page_for, reader, reader_admin, services):
    admin_page = page_for(reader_admin)
    settings = web_search_tab(admin_page)
    settings.get_by_placeholder("No limit").fill("20")  # the external engine has no other
    save(admin_page, settings)

    page = page_for(create_user(reader))
    fetched = fetch_in_a_chat(page, reader, f"{services.base_url}/page", "How long is the log?")

    assert fetched.endswith("\n\n[Content truncated...]"), fetched
    assert len(fetched) == 20 + len("\n\n[Content truncated...]")


def attach_webpage(page: Page, link: str) -> None:
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("button", name="More", exact=True).last.click()
    page.get_by_role("button", name="Attach Webpage").click()
    page.get_by_role("textbox", name="Webpage URLs").fill(link)
    page.get_by_role("button", name="Add", exact=True).click()
    expect(page.get_by_text(link)).to_be_visible()


@pytest.mark.parametrize("bypass", [True, False], ids=["bypassed", "embedded"])
def test_bypass_embedding_decides_whether_an_attached_page_is_embedded(
    bypass, page_for, reader, reader_admin, services
):
    admin_page = page_for(reader_admin)
    settings = web_search_tab(admin_page)
    switch = settings.get_by_role("switch", name="Bypass Embedding and Retrieval")
    if (switch.get_attribute("aria-checked") == "true") != bypass:
        switch.click()
    save(admin_page, settings)
    question = f"What is in the lighthouse log ({'bypassed' if bypass else 'embedded'})?"
    reader.upstream.queue(reply.text("Fog every hour.", match=reply.answering(question)))

    page = page_for(create_user(reader))
    attach_webpage(page, f"{services.base_url}/page")
    send(page, question)
    expect_reply(page, "Fog every hour.")

    embedded = [
        entry
        for entry in reader.upstream.requests_to("/embeddings")
        if PAGE_TEXT in str(entry.body)
    ]
    sent = next(body for body in reader.upstream.chat_requests() if reply.answering(question)(body))
    assert PAGE_TEXT in str(sent["messages"]), "the attached page never reached the model"
    assert bool(embedded) is not bypass


@pytest.fixture(scope="module")
def plain_reader(instance_with):
    """An instance with no certificate bundle of its own in the environment."""
    return instance_with(LOCAL_WEB_FETCH)


def read_with_verification(page_for, launched, admin: Actor, link: str, verify: bool) -> str:
    """What the model reads of `link` once the admin saved Verify SSL Certificate as `verify`."""
    admin_page = page_for(admin)
    settings = web_search_tab(admin_page)
    switch = settings.get_by_role("switch", name="Verify SSL Certificate")
    if (switch.get_attribute("aria-checked") == "true") != verify:
        switch.click()
    save(admin_page, settings)
    question = f"Is the lighthouse log signed ({uuid.uuid4().hex[:6]})?"
    return fetch_in_a_chat(page_for(create_user(launched)), launched, link, question)


@pytest.mark.parametrize("verify", [True, False], ids=["verified", "unverified"])
def test_verify_ssl_certificate_decides_whether_a_self_signed_page_is_read(
    verify, page_for, plain_reader, services, secure_pages
):
    with searching_admin(plain_reader, services) as admin:
        fetched = read_with_verification(
            page_for, plain_reader, admin, f"{secure_pages.base_url}/page", verify
        )

    if verify:
        assert PAGE_TEXT not in fetched, "a page without a trusted certificate was read"
    else:
        assert PAGE_TEXT in fetched, "the page was not read with verification off"
