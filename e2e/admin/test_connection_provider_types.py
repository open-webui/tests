"""Journey: connection types and API types, from the admin's dialog to a user's answer.

A connection whose URL names `api.anthropic.com` is Anthropic's: Verify and the model list read
Anthropic's own paged model list with the `x-api-key` header, the models show by their display
names, and a chat goes to Anthropic's OpenAI-compatible chat endpoint with the key as a bearer
token (docs: starting-with-anthropic). The host name counts anywhere in the URL, so a listener path
of that name stands in. An Azure OpenAI connection (Provider under Advanced) is verified against
`/openai/models` with the `api-key` header and the API version, refuses to save without a
deployment name, and sends a chat to that deployment's URL with the API version (docs:
starting-with-openai-compatible, Azure OpenAI). The API Type switch sends a user's chat to the
provider's Responses API, and switched back to Chat Completions, to its chat endpoint. An Ollama
model listed under its prefix answers a user's chat, and the Ollama server gets the bare name.

Discriminates: passes on the dev ebc6add67 build; in a backend copy, the Anthropic listing
reading only the first page turns the Anthropic test red (the second page's model is missing),
the Azure deployment URL left unrewritten turns the Azure test red, the chat ignoring `api_type`
turns the API type test red and the Ollama prefix kept on the name sent upstream turns the Ollama
test red.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import responses_provider as responses_api
from harness.listener import ReceivedRequest, json_answer
from harness.ollama_provider import OLLAMA_CONFIG, chat_stream, connect_ollama, serve_ollama
from harness.second_provider import OPENAI_CONFIG, sse
from utils.admin_connections import (
    OPENAI_URL_PLACEHOLDER,
    connection_dialog,
    connection_row,
    is_openai_save,
    open_admin_connections,
)
from utils.chat_ui import chat_input, expect_reply, send
from utils.model_selector import model_options, select_model
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

EVERYONE_READS = {"principal_type": "user", "principal_id": "*", "permission": "read"}
# the host name counts anywhere in the URL, so a local path of that name stands in
ANTHROPIC_PATH = "/api.anthropic.com/v1"
ANTHROPIC_KEY = "sk-ant-harbor"
AZURE_KEY = "azure-harbor-key"
AZURE_VERSION = "2024-10-21"


def publish(admin, *model_ids: str) -> None:
    """Make the models readable by every account, as the admin's visibility toggle does."""
    with admin.client() as client:
        listed = client.get("/api/models", params={"refresh": True})
        listed.raise_for_status()
        names = {model["id"]: model["name"] for model in listed.json()["data"]}
        for model_id in model_ids:
            shared = client.post(
                "/api/v1/models/model/access/update",
                json={"id": model_id, "name": names[model_id], "access_grants": [EVERYONE_READS]},
            )
            assert shared.status_code == 200, shared.text


def add_dialog(page: Page, url: str, key: str) -> Locator:
    tooltip_button(open_admin_connections(page), "Add Connection").click()
    form = connection_dialog(page, "Add Connection")
    form.get_by_label("URL", exact=True).fill(url)
    form.get_by_role("textbox", name="API Key").fill(key)
    return form


def ask(page: Page, model_name: str, answer: str) -> None:
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    select_model(page, model_name)
    send(page, f"Who answers, {uuid.uuid4().hex[:6]}?")
    expect_reply(page, answer)


def saved_openai_urls(admin) -> list[str]:
    with admin.client() as client:
        return client.get(OPENAI_CONFIG[0]).json()["OPENAI_API_BASE_URLS"]


# --------------------------------------------------------------------------- Anthropic


def anthropic_models(request: ReceivedRequest):
    """Anthropic's model list, one model per page."""
    if "after_id=claude-harbor-1" in request.path:
        page = [{"id": "claude-lighthouse-1", "display_name": "Claude Lighthouse"}]
        return json_answer({"data": page, "has_more": False, "last_id": "claude-lighthouse-1"})
    page = [{"id": "claude-harbor-1", "display_name": "Claude Harbor"}]
    return json_answer({"data": page, "has_more": True, "last_id": "claude-harbor-1"})


def test_an_anthropic_connection_lists_every_page_by_name_and_answers_a_users_chat(
    page_for, make_user, admin, preserve, listener
):
    preserve(OPENAI_CONFIG)
    listener.route("GET", f"{ANTHROPIC_PATH}/models", anthropic_models)
    listener.route(
        "POST", f"{ANTHROPIC_PATH}/chat/completions", sse({"content": "Claude from the harbor"})
    )
    page = page_for(make_user(role="admin"))
    form = add_dialog(page, f"{listener.base_url}{ANTHROPIC_PATH}", ANTHROPIC_KEY)

    form.get_by_role("button", name="Verify Connection").click()
    expect(page.get_by_text("Server connection verified")).to_be_visible()
    listing = listener.requests_to(f"{ANTHROPIC_PATH}/models")[0]
    assert listing.headers.get("x-api-key") == ANTHROPIC_KEY
    assert listing.headers.get("anthropic-version")
    with page.expect_response(is_openai_save):
        form.get_by_role("button", name="Save").click()
    expect(form).to_be_hidden()

    publish(admin, "claude-harbor-1", "claude-lighthouse-1")
    user_page = page_for(make_user())
    ask(user_page, "Claude Harbor", "Claude from the harbor")
    expect(model_options(user_page, "Claude Lighthouse")).to_have_count(1)

    [chat] = listener.requests_to(f"{ANTHROPIC_PATH}/chat/completions")
    assert chat.headers.get("Authorization") == f"Bearer {ANTHROPIC_KEY}"
    assert chat.json()["model"] == "claude-harbor-1"


# --------------------------------------------------------------------------- Azure OpenAI


def test_an_azure_connection_needs_a_deployment_and_answers_through_it(
    page_for, make_user, admin, preserve, listener
):
    preserve(OPENAI_CONFIG)
    deployment = f"harbor-gpt-{uuid.uuid4().hex[:6]}"
    listener.route("GET", "/openai/models", json_answer({"data": [{"id": deployment}]}))
    listener.route(
        "POST",
        f"/openai/deployments/{deployment}/chat/completions",
        sse({"content": "Azure answers from the harbor"}),
    )
    page = page_for(make_user(role="admin"))
    form = add_dialog(page, listener.base_url, AZURE_KEY)
    form.get_by_role("button", name="Advanced").click()
    form.get_by_role("combobox", name="Provider").select_option(label="Azure OpenAI")
    form.get_by_role("textbox", name="API Version").fill(AZURE_VERSION)

    form.get_by_role("button", name="Verify Connection").click()
    expect(page.get_by_text("Server connection verified")).to_be_visible()
    verified = listener.requests_to("/openai/models")[-1]
    assert verified.headers.get("api-key") == AZURE_KEY
    assert f"api-version={AZURE_VERSION}" in verified.path

    form.get_by_role("button", name="Save").click()
    toasts = page.locator("[data-sonner-toast]")
    expect(toasts.filter(has_text="Deployment names are required for Azure OpenAI")).to_be_visible()
    expect(form).to_be_visible()
    assert listener.base_url not in saved_openai_urls(admin)

    form.get_by_placeholder("Add a model ID").fill(deployment)
    form.get_by_role("button", name="Add", exact=True).click()
    with page.expect_response(is_openai_save):
        form.get_by_role("button", name="Save").click()
    expect(form).to_be_hidden()

    publish(admin, deployment)
    ask(page_for(make_user()), deployment, "Azure answers from the harbor")
    [chat] = listener.requests_to(f"/openai/deployments/{deployment}/chat/completions")
    assert chat.headers.get("api-key") == AZURE_KEY
    assert f"api-version={AZURE_VERSION}" in chat.path


# --------------------------------------------------------------------------- Ollama


def test_an_ollama_model_under_its_prefix_answers_a_users_chat(
    page_for, make_user, admin, preserve, listener
):
    preserve(OLLAMA_CONFIG)
    prefix = f"farm{uuid.uuid4().hex[:6]}"
    server = serve_ollama(listener, "llama3:latest")
    server.queue_chat(chat_stream("llama3:latest", {"content": "the llama answers"}))
    with admin.client() as client:
        connect_ollama(client, listener, prefix_id=prefix)
    publish(admin, f"{prefix}.llama3:latest")

    ask(page_for(make_user()), f"{prefix}.llama3:latest", "the llama answers")

    [chat] = server.sent("/api/chat")
    assert chat["model"] == "llama3:latest"


# --------------------------------------------------------------------------- Responses API


def test_the_api_type_decides_whether_a_users_chat_goes_to_responses_or_chat_completions(
    page_for, make_user, admin, preserve, listener
):
    preserve(OPENAI_CONFIG)
    prefix = f"resp{uuid.uuid4().hex[:6]}"
    url = f"{listener.base_url}/v1"
    listener.route("GET", "/v1/models", json_answer({"data": [{"id": "keeper"}]}))
    listener.route(
        "POST",
        "/v1/responses",
        responses_api.events_stream(
            *responses_api.message("answered through Responses"), responses_api.completed()
        ),
    )
    listener.route("POST", "/v1/chat/completions", sse({"content": "answered through Chat"}))
    admin_page = page_for(make_user(role="admin"))
    form = add_dialog(admin_page, url, "sk-responses")
    form.get_by_role("button", name="API Type").click()
    expect(form.get_by_role("button", name="API Type")).to_have_text("Responses")
    form.get_by_role("button", name="Advanced").click()
    form.get_by_role("textbox", name="Prefix ID").fill(prefix)
    with admin_page.expect_response(is_openai_save):
        form.get_by_role("button", name="Save").click()
    expect(form).to_be_hidden()
    publish(admin, f"{prefix}.keeper")
    page = page_for(make_user())

    ask(page, f"{prefix}.keeper", "answered through Responses")
    [sent] = listener.requests_to("/v1/responses")
    assert sent.json()["model"] == "keeper"
    assert listener.requests_to("/v1/chat/completions") == []

    settings = open_admin_connections(admin_page)
    tooltip_button(connection_row(settings, OPENAI_URL_PLACEHOLDER, url), "Configure").click()
    editing = connection_dialog(admin_page, "Edit Connection")
    expect(editing.get_by_role("button", name="API Type")).to_have_text("Responses")
    editing.get_by_role("button", name="API Type").click()
    expect(editing.get_by_role("button", name="API Type")).to_have_text("Chat Completions")
    with admin_page.expect_response(is_openai_save):
        editing.get_by_role("button", name="Save").click()
    expect(editing).to_be_hidden()

    ask(page, f"{prefix}.keeper", "answered through Chat")
    assert len(listener.requests_to("/v1/responses")) == 1
