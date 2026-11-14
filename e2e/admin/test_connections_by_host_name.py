"""Journey: an admin connects a provider by host name, with the threaded and the c-ares resolver.

`AIOHTTP_CLIENT_ASYNC_DNS_RESOLVER` picks the resolver the server's aiohttp sessions use (the
threaded OS resolver by default, c-ares through aiodns when on, opt-in since c5ec01b1f, PR #28242,
after #28013 and #28215). On an instance booted with each value the admin adds an OpenAI-compatible
connection in Admin Settings > Connections by a host name (`localhost`, a hosts-file name for `::1`
alone and a hosts-file name for another local address), verifies it and saves it, then picks its
model and gets its reply in a chat; an Ollama connection is added and verified by name the same
way. A connection whose host name does not resolve shows the same error toast, as quickly, under
both resolvers. Twin of integration/resolver/test_model_providers.py, which runs every name form
for both providers; here the OpenAI test runs `localhost` and the `::1`-only name, and the Ollama
test runs `localhost` only, under each resolver (the hosts-file name stays in the twin).

Discriminates: on the dev 176d31d1d build, a backend copy whose `env.py` installs a resolver that
fails every lookup when the flag is on turns every c-ares by-name run red (the verify toast reads
"OpenAI: Network Problem") and leaves every threaded run green; the same resolver installed for the
flag off does the reverse. The unresolvable-name test's c-ares run skips, naming the reason, on a
machine whose DNS server keeps c-ares from refusing an unknown name at once (a cached reply with a
stale EDNS cookie, c-ares issues 1081 and 1271); a resolver that takes 20 seconds to refuse one
turns it red.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.actors import admin_of
from harness.host_names import FAILS_WITHIN, RESOLVERS, UNRESOLVABLE, serving_by_name
from harness.listener import json_answer
from harness.ollama_provider import OLLAMA_CONFIG, serve_ollama
from harness.second_provider import OPENAI_CONFIG, sse
from utils.chat_ui import chat_input, expect_reply, send
from utils.model_selector import select_model
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

OPENAI_FORMS = [pytest.param(label, id=label) for label in ("localhost", "ipv6-only")]
OLLAMA_FORMS = [pytest.param("localhost", id="localhost")]
NAMED_MODEL = "named-model"
REPLY = "answered by its host name"


def open_admin_connections(page: Page) -> Locator:
    page.goto("/admin/settings/connections")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("heading", name="Connections", exact=True)).to_be_visible()
    expect(settings.get_by_role("switch", name="OpenAI API")).to_be_visible()
    return settings


def add_connection_dialog(page: Page, section: Locator) -> Locator:
    tooltip_button(section, "Add Connection").click()
    heading = page.get_by_role("heading", name="Add Connection")
    dialog = page.get_by_role("dialog").filter(has=heading)
    expect(dialog).to_be_visible()
    return dialog


def ollama_section(page: Page, settings: Locator) -> Locator:
    """The Ollama part of the page, switched on."""
    switch = settings.get_by_role("switch", name="Ollama API")
    if not switch.is_checked():
        with page.expect_response(lambda response: "/ollama/config/update" in response.url):
            switch.click()
    return settings.get_by_text("Manage Ollama API Connections").locator("xpath=..")


@pytest.fixture(params=list(RESOLVERS))
def resolver(request) -> str:
    return request.param


@pytest.fixture
def resolving_admin(resolver, instance_with, preserve):
    resolving = instance_with(RESOLVERS[resolver])
    preserve(OPENAI_CONFIG, OLLAMA_CONFIG, on=resolving)
    return admin_of(resolving)


@pytest.mark.parametrize("label", OPENAI_FORMS)
def test_a_connection_added_by_host_name_verifies_and_answers(page_for, resolving_admin, label):
    prefix = f"named{uuid.uuid4().hex[:6]}"
    with serving_by_name(label) as provider:
        provider.route("GET", "/v1/models", json_answer({"data": [{"id": NAMED_MODEL}]}))
        provider.route("POST", "/v1/chat/completions", sse({"content": REPLY}))
        page = page_for(resolving_admin)
        settings = open_admin_connections(page)
        form = add_connection_dialog(page, settings)
        form.get_by_label("URL", exact=True).fill(f"{provider.base_url}/v1")
        form.get_by_role("textbox", name="API Key").fill("sk-named")
        form.get_by_role("button", name="Verify Connection").click()
        expect(page.get_by_text("Server connection verified")).to_be_visible()
        form.get_by_role("button", name="Advanced").click()
        form.get_by_role("textbox", name="Prefix ID").fill(prefix)
        with page.expect_response(lambda response: "/openai/config/update" in response.url):
            form.get_by_role("button", name="Save").click()
        expect(form).to_be_hidden()

        page.goto("/")
        expect(chat_input(page)).to_be_visible()
        select_model(page, f"{prefix}.{NAMED_MODEL}")
        send(page, f"Who answers by name, {prefix}?")
        expect_reply(page, REPLY)

    assert provider.requests_to("/v1/chat/completions"), "the provider was never called"


@pytest.mark.parametrize("label", OLLAMA_FORMS)
def test_an_ollama_connection_added_by_host_name_verifies(page_for, resolving_admin, label):
    with serving_by_name(label) as listener:
        serve_ollama(listener, "named-llama:1b")
        page = page_for(resolving_admin)
        form = add_connection_dialog(page, ollama_section(page, open_admin_connections(page)))
        form.get_by_label("URL", exact=True).fill(listener.base_url)
        form.get_by_role("button", name="Verify Connection").click()
        expect(page.get_by_text("Server connection verified")).to_be_visible()

    assert listener.requests_to("/api/version"), "the Ollama stand-in was never asked"


@pytest.mark.usefixtures("refuses_unknown_names")
def test_a_host_name_that_does_not_resolve_shows_the_same_error_at_once(page_for, resolving_admin):
    page = page_for(resolving_admin)
    settings = open_admin_connections(page)
    form = add_connection_dialog(page, settings)
    form.get_by_label("URL", exact=True).fill(f"http://{UNRESOLVABLE}:8000/v1")
    form.get_by_role("button", name="Verify Connection").click()
    expect(page.get_by_text("OpenAI: Network Problem")).to_be_visible(timeout=FAILS_WITHIN * 1000)

    settings = open_admin_connections(page)
    ollama_form = add_connection_dialog(page, ollama_section(page, settings))
    ollama_form.get_by_label("URL", exact=True).fill(f"http://{UNRESOLVABLE}:11434")
    ollama_form.get_by_role("button", name="Verify Connection").click()
    expect(page.get_by_text("Ollama: Network Problem")).to_be_visible(timeout=FAILS_WITHIN * 1000)
