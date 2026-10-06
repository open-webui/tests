"""Journey: an admin manages the instance's model connections in Admin Settings > Connections.

The admin adds an OpenAI-compatible connection through the Add Connection dialog (URL, key,
connection type, prefix, a model allowlist and a tag), verifies it and saves; its allowed models
then show in the chat's model selector under the prefix, and nothing else it serves does. A
connection switched off in its row, or deleted from its settings, takes its models out of the
selector; a new key saved in its settings is the one the provider gets on the next chat. Once a
connection is verified, the allowlist's model field suggests the models the provider served that
are not yet on the list, and Enter in that field does not save the dialog. An Ollama
connection added the same way lists its models under its prefix. The OpenAI API switch at the top
takes every OpenAI-compatible model out of a user's selector and brings them back. Every provider
is a local stand-in, and every test puts the connection settings back afterwards.

Twin of integration/models/test_admin_connection_settings.py.

Discriminates: passes on the dev 176d31d1d build; in a frontend build, the dialog saving an empty
allowlist turns the add test red (the model left off the list shows), the row switch not saving
turns the switch-off test red, the edited key not reaching the saved settings turns the key test
red (the provider gets the old key), the delete not saving turns the delete test red, and the
Ollama add dropping the dialog's settings turns the Ollama test red (its model shows without the
prefix). In a frontend build without the verified models' suggestions and the Enter guard, the two
allowlist tests go red (no suggestions; the dialog saves on Enter). In a frontend build of dev
30f3f6a8f whose OpenAI API switch saves the stored state, the switch test goes red.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.listener import Listener, json_answer
from harness.ollama_provider import OLLAMA_CONFIG, serve_ollama
from harness.second_provider import OPENAI_CONFIG, attach, sse
from utils.chat_ui import chat_input, expect_reply, send
from utils.model_selector import model_options, select_model
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

OPENAI_URL_PLACEHOLDER = "API Base URL"
OLLAMA_URL_PLACEHOLDER = "Enter URL (e.g. http://localhost:11434)"
SERVED_MODELS = ("alpha", "beta", "gamma")
# refused at once, so the switched-on Ollama API does not wait on a real server
UNREACHABLE_OLLAMA = "http://127.0.0.1:9"


def is_openai_save(response) -> bool:
    return "/openai/config/update" in response.url


def is_ollama_save(response) -> bool:
    return "/ollama/config/update" in response.url


def open_admin_connections(page: Page) -> Locator:
    page.goto("/admin/settings/connections")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("heading", name="Connections", exact=True)).to_be_visible()
    expect(settings.get_by_role("switch", name="OpenAI API")).to_be_visible()
    return settings


def connection_row(settings: Locator, placeholder: str, url: str) -> Locator:
    """The row of the connection saved with `url`: its address, settings button and switch."""
    addresses = settings.get_by_placeholder(placeholder)
    expect(addresses.first).to_be_visible()
    index = addresses.evaluate_all("(inputs, url) => inputs.findIndex((i) => i.value === url)", url)
    assert index >= 0, f"no connection row reads {url}"
    return addresses.nth(index).locator("xpath=ancestor::div[.//button][1]")


def connection_dialog(page: Page, heading: str) -> Locator:
    dialog = page.get_by_role("dialog").filter(has=page.get_by_role("heading", name=heading))
    expect(dialog).to_be_visible()
    return dialog


def open_chat(page: Page) -> None:
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    # the instance's own model shows the list has loaded
    expect(model_options(page, reply.MOCK_MODEL_ID)).to_have_count(1)


def saved_openai_connection(admin, url: str) -> tuple[str, dict] | None:
    """The saved key and settings of the OpenAI connection at `url`, or None when there is none."""
    with admin.client() as client:
        saved = client.get(OPENAI_CONFIG[0]).json()
    if url not in saved["OPENAI_API_BASE_URLS"]:
        return None
    index = saved["OPENAI_API_BASE_URLS"].index(url)
    return saved["OPENAI_API_KEYS"][index], saved["OPENAI_API_CONFIGS"][str(index)]


def serve_chat(listener: Listener, answer: str) -> None:
    listener.route("POST", "/v1/chat/completions", sse({"content": answer}))


@pytest.fixture
def prefix() -> str:
    return f"conn{uuid.uuid4().hex[:6]}"


@pytest.fixture
def attached(admin, preserve, listener, prefix) -> str:
    """A connection to the listener saved with `prefix`; returns its model's listed id."""
    preserve(OPENAI_CONFIG)
    with admin.client() as client:
        attach(client, listener, "alpha", prefix_id=prefix)
    return f"{prefix}.alpha"


# --------------------------------------------------------------------------- add


def test_an_added_connection_lists_only_its_allowed_models_under_the_prefix(
    page_for, make_user, admin, preserve, listener, prefix
):
    preserve(OPENAI_CONFIG)
    served = [{"id": name, "object": "model"} for name in SERVED_MODELS]
    listener.route("GET", "/v1/models", json_answer({"object": "list", "data": served}))
    url = f"{listener.base_url}/v1"
    tag = f"team-{prefix}"
    page = page_for(make_user(role="admin"))
    settings = open_admin_connections(page)

    tooltip_button(settings, "Add Connection").click()
    form = connection_dialog(page, "Add Connection")
    form.get_by_label("URL", exact=True).fill(url)
    form.get_by_role("textbox", name="API Key").fill("sk-admin-added")
    form.get_by_role("button", name="External", exact=True).click()
    expect(form.get_by_role("button", name="Local", exact=True)).to_be_visible()
    form.get_by_role("button", name="Verify Connection").click()
    expect(page.get_by_text("Server connection verified")).to_be_visible()
    verified = listener.requests_to("/v1/models")[-1]
    assert verified.headers.get("Authorization") == "Bearer sk-admin-added"

    form.get_by_role("button", name="Advanced").click()
    form.get_by_role("textbox", name="Prefix ID").fill(prefix)
    for allowed in ("alpha", "beta"):
        form.get_by_placeholder("Add a model ID").fill(allowed)
        form.get_by_role("button", name="Add", exact=True).click()
        expect(form.get_by_text(allowed, exact=True)).to_be_visible()
    form.get_by_placeholder("Add a tag...").fill(tag)
    form.get_by_placeholder("Add a tag...").press("Enter")
    with page.expect_response(is_openai_save):
        form.get_by_role("button", name="Save").click()
    expect(form).to_be_hidden()
    expect(connection_row(settings, OPENAI_URL_PLACEHOLDER, url)).to_be_visible()

    key, config = saved_openai_connection(admin, url)
    assert key == "sk-admin-added"
    assert config["connection_type"] == "local"
    assert config["tags"] == [{"name": tag}]

    open_chat(page)
    expect(model_options(page, f"{prefix}.alpha")).to_have_count(1)
    expect(model_options(page, f"{prefix}.beta")).to_have_count(1)
    expect(model_options(page, f"{prefix}.gamma")).to_have_count(0)
    expect(model_options(page, "gamma")).to_have_count(0)


# --------------------------------------------------------------------------- switch off


def test_switching_a_connection_off_in_its_row_removes_its_models(
    page_for, make_user, listener, attached
):
    page = page_for(make_user(role="admin"))
    open_chat(page)
    expect(model_options(page, attached)).to_have_count(1)

    settings = open_admin_connections(page)
    row = connection_row(settings, OPENAI_URL_PLACEHOLDER, f"{listener.base_url}/v1")
    switch = row.get_by_role("switch")
    expect(switch).to_be_checked()
    with page.expect_response(is_openai_save):
        switch.click()
    expect(switch).not_to_be_checked()

    open_chat(page)
    expect(model_options(page, attached)).to_have_count(0)


# --------------------------------------------------------------------------- key


def test_an_edited_key_is_the_one_the_provider_gets_on_the_next_chat(
    page_for, make_user, admin, listener, attached
):
    serve_chat(listener, "answered with the new key")
    url = f"{listener.base_url}/v1"
    page = page_for(make_user(role="admin"))
    settings = open_admin_connections(page)

    tooltip_button(connection_row(settings, OPENAI_URL_PLACEHOLDER, url), "Configure").click()
    editing = connection_dialog(page, "Edit Connection")
    expect(editing.get_by_role("textbox", name="API Key")).to_have_value("sk-second")
    editing.get_by_role("textbox", name="API Key").fill("sk-rotated")
    with page.expect_response(is_openai_save):
        editing.get_by_role("button", name="Save").click()
    expect(editing).to_be_hidden()
    assert saved_openai_connection(admin, url)[0] == "sk-rotated"

    open_chat(page)
    select_model(page, attached)
    send(page, f"Which key, {uuid.uuid4().hex[:6]}?")
    expect_reply(page, "answered with the new key")
    chat = listener.requests_to("/v1/chat/completions")[-1]
    assert chat.headers.get("Authorization") == "Bearer sk-rotated"
    assert chat.json()["model"] == "alpha", "the prefix reached the provider"


# --------------------------------------------------------------------------- delete


def test_deleting_a_connection_removes_it_and_its_models(
    page_for, make_user, admin, listener, attached
):
    url = f"{listener.base_url}/v1"
    page = page_for(make_user(role="admin"))
    settings = open_admin_connections(page)

    tooltip_button(connection_row(settings, OPENAI_URL_PLACEHOLDER, url), "Configure").click()
    editing = connection_dialog(page, "Edit Connection")
    editing.get_by_role("button", name="Delete").click()
    with page.expect_response(is_openai_save):
        page.get_by_role("dialog").get_by_role("button", name="Delete").last.click()
    expect(editing).to_be_hidden()
    assert saved_openai_connection(admin, url) is None, "the deleted connection is still saved"

    open_chat(page)
    expect(model_options(page, attached)).to_have_count(0)


# --------------------------------------------------------------------------- Ollama


def test_an_added_ollama_connection_lists_its_models_under_the_prefix(
    page_for, make_user, admin, preserve, listener, prefix
):
    preserve(OLLAMA_CONFIG)
    server = serve_ollama(listener, "llama3:latest")
    with admin.client() as client:
        current = client.get(OLLAMA_CONFIG[0]).json()
        switched_off = {
            **current,
            "ENABLE_OLLAMA_API": False,
            "OLLAMA_BASE_URLS": [UNREACHABLE_OLLAMA],
            "OLLAMA_API_CONFIGS": {},
        }
        client.post(OLLAMA_CONFIG[1], json=switched_off).raise_for_status()
    page = page_for(make_user(role="admin"))
    settings = open_admin_connections(page)

    with page.expect_response(is_ollama_save):
        settings.get_by_role("switch", name="Ollama API").click()
    ollama_section = settings.get_by_text("Manage Ollama API Connections").locator("xpath=..")
    tooltip_button(ollama_section, "Add Connection").click()
    form = connection_dialog(page, "Add Connection")
    form.get_by_label("URL", exact=True).fill(listener.base_url)
    form.get_by_role("button", name="Verify Connection").click()
    expect(page.get_by_text("Server connection verified")).to_be_visible()
    form.get_by_role("button", name="Advanced").click()
    form.get_by_role("textbox", name="Prefix ID").fill(prefix)
    with page.expect_response(is_ollama_save):
        form.get_by_role("button", name="Save").click()
    expect(form).to_be_hidden()
    expect(connection_row(settings, OLLAMA_URL_PLACEHOLDER, listener.base_url)).to_be_visible()

    open_chat(page)
    expect(model_options(page, f"{prefix}.llama3:latest")).to_have_count(1)
    assert server.listener.requests_to("/api/tags"), "the Ollama stand-in was never asked"


# --------------------------------------------------------------------------- allowlist suggestions


def verified_dialog(page: Page, listener: Listener, url: str) -> Locator:
    """The Add Connection dialog with `url` verified and its Advanced section open."""
    settings = open_admin_connections(page)
    tooltip_button(settings, "Add Connection").click()
    form = connection_dialog(page, "Add Connection")
    form.get_by_label("URL", exact=True).fill(url)
    form.get_by_role("button", name="Verify Connection").click()
    expect(page.get_by_text("Server connection verified")).to_be_visible()
    form.get_by_role("button", name="Advanced").click()
    return form


def suggested_models(form: Locator) -> list[str]:
    field = form.get_by_placeholder("Add a model ID")
    return field.evaluate("(input) => [...(input.list?.options ?? [])].map((o) => o.value)")


def test_the_verified_models_are_suggested_for_the_allowlist(
    page_for, make_user, preserve, listener
):
    preserve(OPENAI_CONFIG)
    served = [{"id": name, "object": "model"} for name in SERVED_MODELS]
    listener.route("GET", "/v1/models", json_answer({"object": "list", "data": served}))
    page = page_for(make_user(role="admin"))
    settings = open_admin_connections(page)
    tooltip_button(settings, "Add Connection").click()
    form = connection_dialog(page, "Add Connection")
    form.get_by_label("URL", exact=True).fill(f"{listener.base_url}/v1")
    form.get_by_role("button", name="Advanced").click()
    assert suggested_models(form) == [], "models were suggested before the connection was verified"
    form.get_by_role("button", name="Verify Connection").click()
    expect(page.get_by_text("Server connection verified")).to_be_visible()

    assert suggested_models(form) == list(SERVED_MODELS)

    form.get_by_placeholder("Add a model ID").fill("beta")
    form.get_by_role("button", name="Add", exact=True).click()
    expect(form.get_by_text("beta", exact=True)).to_be_visible()
    assert suggested_models(form) == ["alpha", "gamma"]


def test_enter_in_the_model_field_does_not_save_the_connection(
    page_for, make_user, admin, preserve, listener
):
    preserve(OPENAI_CONFIG)
    url = f"{listener.base_url}/v1"
    listener.route("GET", "/v1/models", json_answer({"object": "list", "data": [{"id": "alpha"}]}))
    page = page_for(make_user(role="admin"))
    form = verified_dialog(page, listener, url)

    form.get_by_placeholder("Add a model ID").fill("alpha")
    form.get_by_placeholder("Add a model ID").press("Enter")

    expect(form).to_be_visible()
    assert saved_openai_connection(admin, url) is None, "Enter in the model field saved the dialog"


def test_the_openai_api_switch_takes_every_openai_model_away_and_back(
    page_for, make_user, preserve
):
    preserve(OPENAI_CONFIG)
    admin_page = page_for(make_user(role="admin"))
    settings = open_admin_connections(admin_page)
    switch = settings.get_by_role("switch", name="OpenAI API")
    expect(switch).to_be_checked()
    with admin_page.expect_response(is_openai_save):
        switch.click()
    expect(switch).not_to_be_checked()

    page = page_for(make_user())
    expect(chat_input(page)).to_be_visible()
    expect(model_options(page, reply.MOCK_MODEL_ID)).to_have_count(0)

    settings = open_admin_connections(admin_page)
    with admin_page.expect_response(is_openai_save):
        settings.get_by_role("switch", name="OpenAI API").click()
    page.reload()
    expect(chat_input(page)).to_be_visible()
    expect(model_options(page, reply.MOCK_MODEL_ID)).to_have_count(1)
