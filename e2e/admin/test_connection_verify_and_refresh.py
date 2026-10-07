"""Journey: verifying a connection, an Anthropic connection end to end and a cached model list.

The Verify Connection button in the admin's connection dialog asks the provider for its models
through the server and says what came back: a provider refusing the key has its own reason shown,
for an OpenAI-compatible server, an Ollama server and Anthropic alike. A connection whose URL names
`api.anthropic.com` is Anthropic's: its models are listed from Anthropic's own paged model list
with the `x-api-key` header and shown by their display names, and a chat on one goes to Anthropic's
OpenAI-compatible chat endpoint with the key as a bearer token (docs: starting-with-anthropic). The
host name counts anywhere in the URL, so a listener path of that name stands in. With Cache Base
Model List on, a model the provider adds stays out of everyone's selector until the admin presses
Refresh next to the switch.

Discriminates: passes on the dev ebc6add67 build except the two refusal tests below; in a backend
copy, the Anthropic listing reading only the first page turns the Anthropic test red (the second
page's model is missing), and the model list ignoring `refresh` while the cache is on turns the
refresh test red. Red on dev, a real bug: the Ollama and the Anthropic refusal show "Network
Problem" (the dialog reads only an OpenAI-shaped error and drops the reason the server sends).
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.listener import Listener, ReceivedRequest, json_answer
from harness.ollama_provider import OLLAMA_CONFIG
from harness.second_provider import OPENAI_CONFIG, attach, sse
from utils.admin_connections import (
    connection_dialog,
    is_openai_save,
    ollama_section,
    open_admin_connections,
)
from utils.chat_ui import chat_input, expect_reply, send
from utils.model_selector import model_options, select_model
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

CONNECTIONS_CONFIG = ("/api/v1/configs/connections", "/api/v1/configs/connections")
# the host name counts anywhere in the URL, so a local path of that name stands in
ANTHROPIC_PATH = "/api.anthropic.com/v1"
ANTHROPIC_KEY = "sk-ant-harbor"
FIRST_PAGE = [{"id": "claude-harbor-1", "display_name": "Claude Harbor", "type": "model"}]
SECOND_PAGE = [{"id": "claude-lighthouse-1", "display_name": "Claude Lighthouse", "type": "model"}]


def add_dialog(page: Page, section: Locator, url: str, key: str = "") -> Locator:
    tooltip_button(section, "Add Connection").click()
    form = connection_dialog(page, "Add Connection")
    form.get_by_label("URL", exact=True).fill(url)
    if key:
        form.get_by_role("textbox", name="API Key").fill(key)
    return form


def open_chat(page: Page) -> None:
    page.goto("/")
    expect(chat_input(page)).to_be_visible()


# --------------------------------------------------------------------------- refused keys


def test_a_refused_key_shows_the_openai_compatible_providers_reason(
    page_for, make_user, preserve, listener
):
    preserve(OPENAI_CONFIG)
    refusal = {"error": {"message": "Incorrect API key provided: sk-wro***"}}
    listener.route("GET", "/v1/models", json_answer(refusal, status=401))
    page = page_for(make_user(role="admin"))
    form = add_dialog(page, open_admin_connections(page), f"{listener.base_url}/v1", "sk-wrong")

    form.get_by_role("button", name="Verify Connection").click()

    expect(page.get_by_text("OpenAI: Incorrect API key provided: sk-wro***")).to_be_visible()
    expect(page.get_by_text("Server connection verified")).to_have_count(0)
    assert listener.requests_to("/v1/models")[-1].headers["Authorization"] == "Bearer sk-wrong"


def test_a_refused_ollama_verification_shows_the_servers_reason(
    page_for, make_user, preserve, listener
):
    preserve(OLLAMA_CONFIG)
    listener.route("GET", "/api/version", json_answer({"error": "unauthorized"}, status=401))
    page = page_for(make_user(role="admin"))
    settings = open_admin_connections(page)
    form = add_dialog(page, ollama_section(page, settings), listener.base_url)

    form.get_by_role("button", name="Verify Connection").click()

    toast = page.locator("[data-sonner-toast]").filter(has_text="Ollama:")
    expect(toast).to_be_visible()
    shown = toast.inner_text()
    assert listener.requests_to("/api/version"), "the Ollama server was never asked"
    assert "unauthorized" in shown, (
        f"the server answered 401 unauthorized, the toast reads {shown!r}"
    )


def test_a_refused_anthropic_verification_shows_anthropics_reason(
    page_for, make_user, preserve, listener
):
    preserve(OPENAI_CONFIG)
    refusal = {
        "type": "error",
        "error": {"type": "authentication_error", "message": "invalid x-api-key"},
    }
    listener.route("GET", f"{ANTHROPIC_PATH}/models", json_answer(refusal, status=401))
    page = page_for(make_user(role="admin"))
    url = f"{listener.base_url}{ANTHROPIC_PATH}"
    form = add_dialog(page, open_admin_connections(page), url, "sk-ant-wrong")

    form.get_by_role("button", name="Verify Connection").click()

    toast = page.locator("[data-sonner-toast]").filter(has_text="OpenAI:")
    expect(toast).to_be_visible()
    shown = toast.inner_text()
    assert listener.requests_to(f"{ANTHROPIC_PATH}/models"), "Anthropic was never asked"
    assert "invalid x-api-key" in shown, f"Anthropic refused the key, the toast reads {shown!r}"


# --------------------------------------------------------------------------- Anthropic


def anthropic_models(request: ReceivedRequest):
    """Anthropic's model list, one model per page."""
    if "after_id=claude-harbor-1" in request.path:
        return json_answer(
            {"data": SECOND_PAGE, "has_more": False, "last_id": "claude-lighthouse-1"}
        )
    return json_answer({"data": FIRST_PAGE, "has_more": True, "last_id": "claude-harbor-1"})


def serve_anthropic(listener: Listener, answer: str) -> None:
    listener.route("GET", f"{ANTHROPIC_PATH}/models", anthropic_models)
    listener.route("POST", f"{ANTHROPIC_PATH}/chat/completions", sse({"content": answer}))


def test_an_anthropic_connection_lists_every_page_by_name_and_answers_a_chat(
    page_for, make_user, preserve, listener
):
    preserve(OPENAI_CONFIG)
    serve_anthropic(listener, "Claude answers from the harbor")
    page = page_for(make_user(role="admin"))
    url = f"{listener.base_url}{ANTHROPIC_PATH}"
    form = add_dialog(page, open_admin_connections(page), url, ANTHROPIC_KEY)

    form.get_by_role("button", name="Verify Connection").click()
    expect(page.get_by_text("Server connection verified")).to_be_visible()
    listing = listener.requests_to(f"{ANTHROPIC_PATH}/models")[0]
    assert listing.headers.get("x-api-key") == ANTHROPIC_KEY
    assert listing.headers.get("anthropic-version")
    with page.expect_response(is_openai_save):
        form.get_by_role("button", name="Save").click()
    expect(form).to_be_hidden()

    open_chat(page)
    expect(model_options(page, "Claude Lighthouse")).to_have_count(1)
    select_model(page, "Claude Harbor")
    send(page, f"Ahoy, {uuid.uuid4().hex[:6]}?")
    expect_reply(page, "Claude answers from the harbor")

    [chat] = listener.requests_to(f"{ANTHROPIC_PATH}/chat/completions")
    assert chat.headers.get("Authorization") == f"Bearer {ANTHROPIC_KEY}"
    assert chat.json()["model"] == "claude-harbor-1"


# --------------------------------------------------------------------------- cached model list


def test_the_refresh_button_brings_a_new_provider_model_into_a_cached_list(
    page_for, make_user, admin, preserve, listener
):
    preserve(OPENAI_CONFIG, CONNECTIONS_CONFIG)
    prefix = f"cache{uuid.uuid4().hex[:6]}"
    with admin.client() as client:
        attach(client, listener, "early", prefix_id=prefix)
    admin_page = page_for(make_user(role="admin"))
    settings = open_admin_connections(admin_page)
    cache_switch = settings.get_by_role("switch", name="Cache Base Model List")
    expect(cache_switch).not_to_be_checked()
    with admin_page.expect_response(lambda response: "/configs/connections" in response.url):
        cache_switch.click()
    expect(cache_switch).to_be_checked()

    served = [{"id": "early"}, {"id": "late"}]
    listener.route("GET", "/v1/models", json_answer({"object": "list", "data": served}))
    viewer = page_for(make_user(role="admin"))
    open_chat(viewer)
    expect(model_options(viewer, f"{prefix}.early")).to_have_count(1)
    expect(model_options(viewer, f"{prefix}.late")).to_have_count(0)

    settings.get_by_role("button", name="Refresh", exact=True).click()
    expect(admin_page.get_by_text("Model list refreshed")).to_be_visible()

    open_chat(viewer)
    expect(model_options(viewer, f"{prefix}.late")).to_have_count(1)
