"""Journey: what an admin's model connections mean for the people choosing and chatting with models.

The admin sets up connections in Admin Settings > Connections (or over its API, as a seeded
deployment has them) and makes their models visible; a user then sees the effect in the chat's
model selector and in the answers. A connection's tag and its Local or External type become
filters in the selector. Two connections serving one model name answer under their own prefix
each; without prefixes the first connection answers the shared name, and the second takes over
once the admin switches the first off. A connection switched off while a user chats answers the
next message with "Model not found" and leaves the selector. A provider that cannot be reached
leaves the rest of the list alone and its model's chat shows the connection error. A prefix the
admin changes renames the model, and the provider still gets the bare id. An Ollama connection's
allowlist set in its settings keeps only that model, deleting the connection takes it away and
switching it off in its row takes its models out of a user's selector until it is switched on.
Every provider is a local stand-in, and every test puts the connection settings back afterwards.

Twin of integration/models/test_connection_routing.py.

Discriminates: passes on the dev ebc6add67 build; in a backend copy, models no longer carrying
their connection's tags and type turns the filter test red, the merge letting the last connection
win a shared name turns the shared-name test red, the prefix kept on the model id sent upstream
turns both prefix tests red, the OpenAI listing ignoring `enable` turns the switch-off tests red,
and the Ollama listing ignoring `model_ids` or `enable` turns the Ollama allowlist or the Ollama
switch test red.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.listener import Listener, json_answer, listening
from harness.ollama_provider import OLLAMA_CONFIG, connect_ollama, serve_ollama
from harness.second_provider import OPENAI_CONFIG, attach, sse
from utils.admin_connections import (
    OLLAMA_URL_PLACEHOLDER,
    OPENAI_URL_PLACEHOLDER,
    connection_dialog,
    connection_row,
    is_ollama_save,
    is_openai_save,
    open_admin_connections,
)
from utils.chat_ui import chat_input, expect_reply, send
from utils.model_selector import model_options, select_model
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

EVERYONE_READS = {"principal_type": "user", "principal_id": "*", "permission": "read"}
# refused at once, so a connection to it fails without waiting
UNREACHABLE = "http://127.0.0.1:9"


@pytest.fixture
def prefix() -> str:
    return f"conn{uuid.uuid4().hex[:6]}"


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


def add_openai(admin, url: str, **config) -> None:
    with admin.client() as client:
        current = client.get(OPENAI_CONFIG[0]).json()
        index = str(len(current["OPENAI_API_BASE_URLS"]))
        changed = {
            **current,
            "OPENAI_API_BASE_URLS": [*current["OPENAI_API_BASE_URLS"], url],
            "OPENAI_API_KEYS": [*current["OPENAI_API_KEYS"], ""],
            "OPENAI_API_CONFIGS": {
                **current["OPENAI_API_CONFIGS"],
                index: {"enable": True, **config},
            },
        }
        client.post(OPENAI_CONFIG[1], json=changed).raise_for_status()


def serve_models(listener: Listener, *names: str) -> None:
    served = [{"id": name, "object": "model"} for name in names]
    listener.route("GET", "/v1/models", json_answer({"object": "list", "data": served}))


def serve_answer(listener: Listener, answer: str) -> None:
    listener.route("POST", "/v1/chat/completions", sse({"content": answer}))


def models_sent(listener: Listener) -> list[str]:
    return [request.json()["model"] for request in listener.requests_to("/v1/chat/completions")]


def open_chat(page: Page) -> None:
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    # the instance's own model shows the list has loaded
    expect(model_options(page, reply.MOCK_MODEL_ID)).to_have_count(1)


def ask(page: Page, model_id: str, answer: str) -> None:
    """Start a new chat on `model_id` and wait for `answer`."""
    open_chat(page)
    select_model(page, model_id)
    send(page, f"Who answers, {uuid.uuid4().hex[:6]}?")
    expect_reply(page, answer)


def filter_selector(page: Page, shown: str, label: str) -> None:
    """Pick `label` in the selector's filter menu, whose button reads `shown`."""
    menu_button = page.get_by_role("button", name=shown, exact=True)
    menu_button.and_(page.locator("[aria-expanded]")).click()
    page.get_by_role("button", name=label, exact=True).last.click()


# --------------------------------------------------------------------------- tags and type


def test_a_connections_tag_and_type_filter_a_users_model_selector(
    page_for, make_user, admin, preserve, listener, prefix
):
    preserve(OPENAI_CONFIG)
    tag = f"team{prefix}"
    serve_models(listener, "harbor", "lighthouse")
    add_openai(
        admin,
        f"{listener.base_url}/v1",
        prefix_id=prefix,
        tags=[{"name": tag}],
        connection_type="local",
    )
    publish(admin, f"{prefix}.harbor", f"{prefix}.lighthouse")
    page = page_for(make_user())
    open_chat(page)

    filter_selector(page, "All", tag)
    expect(model_options(page, f"{prefix}.harbor")).to_have_count(1)
    expect(model_options(page, f"{prefix}.lighthouse")).to_have_count(1)
    expect(model_options(page, reply.MOCK_MODEL_ID)).to_have_count(0)

    filter_selector(page, tag, "External")
    expect(model_options(page, reply.MOCK_MODEL_ID)).to_have_count(1)
    expect(model_options(page, f"{prefix}.harbor")).to_have_count(0)

    filter_selector(page, "External", "Local")
    expect(model_options(page, f"{prefix}.harbor")).to_have_count(1)
    expect(model_options(page, reply.MOCK_MODEL_ID)).to_have_count(0)


# --------------------------------------------------------------------------- one model name


def test_two_connections_serving_one_model_name_answer_under_their_own_prefixes(
    page_for, make_user, admin, preserve, listener
):
    preserve(OPENAI_CONFIG)
    north, south = f"north{uuid.uuid4().hex[:4]}", f"south{uuid.uuid4().hex[:4]}"
    with listening() as other:
        serve_answer(listener, "answered in the north")
        serve_answer(other, "answered in the south")
        with admin.client() as client:
            attach(client, listener, "twin", prefix_id=north)
            attach(client, other, "twin", prefix_id=south)
        publish(admin, f"{north}.twin", f"{south}.twin")
        page = page_for(make_user())

        ask(page, f"{north}.twin", "answered in the north")
        ask(page, f"{south}.twin", "answered in the south")

        assert models_sent(listener) == ["twin"]
        assert models_sent(other) == ["twin"]


def test_without_prefixes_the_first_connection_answers_until_the_admin_switches_it_off(
    page_for, make_user, admin, preserve, listener
):
    preserve(OPENAI_CONFIG)
    shared_name = f"twin-{uuid.uuid4().hex[:6]}"
    with listening() as second:
        serve_answer(listener, "the first connection answered")
        serve_answer(second, "the second connection answered")
        with admin.client() as client:
            attach(client, listener, shared_name)
            attach(client, second, shared_name)
        publish(admin, shared_name)
        page = page_for(make_user())
        open_chat(page)
        expect(model_options(page, shared_name)).to_have_count(1)
        ask(page, shared_name, "the first connection answered")

        admin_page = page_for(make_user(role="admin"))
        settings = open_admin_connections(admin_page)
        row = connection_row(settings, OPENAI_URL_PLACEHOLDER, f"{listener.base_url}/v1")
        with admin_page.expect_response(is_openai_save):
            row.get_by_role("switch").click()

        ask(page, shared_name, "the second connection answered")
        assert len(listener.requests_to("/v1/chat/completions")) == 1
        assert models_sent(second) == [shared_name]


# --------------------------------------------------------------------------- switched off mid-chat


def test_a_connection_switched_off_mid_chat_answers_model_not_found_and_leaves_the_selector(
    page_for, make_user, admin, preserve, listener, prefix
):
    preserve(OPENAI_CONFIG)
    serve_answer(listener, "still connected")
    with admin.client() as client:
        attach(client, listener, "alpha", prefix_id=prefix)
    publish(admin, f"{prefix}.alpha")
    page = page_for(make_user())
    ask(page, f"{prefix}.alpha", "still connected")

    admin_page = page_for(make_user(role="admin"))
    settings = open_admin_connections(admin_page)
    row = connection_row(settings, OPENAI_URL_PLACEHOLDER, f"{listener.base_url}/v1")
    with admin_page.expect_response(is_openai_save):
        row.get_by_role("switch").click()

    send(page, "Are you still there?")
    expect_reply(page, "Model not found")
    assert len(listener.requests_to("/v1/chat/completions")) == 1

    open_chat(page)
    expect(model_options(page, f"{prefix}.alpha")).to_have_count(0)


# --------------------------------------------------------------------------- unreachable provider


def test_an_unreachable_provider_leaves_the_list_and_its_chat_shows_the_connection_error(
    page_for, make_user, admin, preserve, upstream, prefix
):
    preserve(OPENAI_CONFIG)
    add_openai(admin, f"{UNREACHABLE}/listed/v1")
    add_openai(admin, f"{UNREACHABLE}/allowed/v1", prefix_id=prefix, model_ids=["ghost"])
    publish(admin, f"{prefix}.ghost")
    page = page_for(make_user())

    open_chat(page)
    select_model(page, f"{prefix}.ghost")
    send(page, "Is anyone out there?")
    expect_reply(page, "Open WebUI: Server Connection Error")

    open_chat(page)
    own_answer = reply.text("the instance's own model answers", match=reply.answering("And here?"))
    upstream.queue(own_answer)
    select_model(page, reply.MOCK_MODEL_ID)
    send(page, "And here?")
    expect_reply(page, "the instance's own model answers")


# --------------------------------------------------------------------------- prefix edit


def test_a_prefix_the_admin_changes_renames_the_model_and_the_provider_gets_the_bare_id(
    page_for, make_user, admin, preserve, listener, prefix
):
    preserve(OPENAI_CONFIG)
    serve_answer(listener, "answered under the new name")
    with admin.client() as client:
        attach(client, listener, "alpha", prefix_id=prefix)
    renamed = f"re{prefix}"
    page = page_for(make_user(role="admin"))
    settings = open_admin_connections(page)

    tooltip_button(
        connection_row(settings, OPENAI_URL_PLACEHOLDER, f"{listener.base_url}/v1"), "Configure"
    ).click()
    editing = connection_dialog(page, "Edit Connection")
    editing.get_by_role("button", name="Advanced").click()
    expect(editing.get_by_role("textbox", name="Prefix ID")).to_have_value(prefix)
    editing.get_by_role("textbox", name="Prefix ID").fill(renamed)
    with page.expect_response(is_openai_save):
        editing.get_by_role("button", name="Save").click()
    expect(editing).to_be_hidden()

    open_chat(page)
    expect(model_options(page, f"{prefix}.alpha")).to_have_count(0)
    ask(page, f"{renamed}.alpha", "answered under the new name")
    assert models_sent(listener) == ["alpha"]


# --------------------------------------------------------------------------- Ollama edit and delete


def test_an_ollama_allowlist_keeps_one_model_and_deleting_the_connection_removes_it(
    page_for, make_user, admin, preserve, listener, prefix
):
    preserve(OLLAMA_CONFIG)
    serve_ollama(listener, "llama3:latest", "qwen3:latest")
    with admin.client() as client:
        connect_ollama(client, listener, prefix_id=prefix)
    page = page_for(make_user(role="admin"))
    open_chat(page)
    expect(model_options(page, f"{prefix}.llama3:latest")).to_have_count(1)

    settings = open_admin_connections(page)
    row = connection_row(settings, OLLAMA_URL_PLACEHOLDER, listener.base_url)
    tooltip_button(row, "Configure").click()
    editing = connection_dialog(page, "Edit Connection")
    editing.get_by_role("button", name="Advanced").click()
    editing.get_by_placeholder("Add a model ID").fill("qwen3:latest")
    editing.get_by_role("button", name="Add", exact=True).click()
    with page.expect_response(is_ollama_save):
        editing.get_by_role("button", name="Save").click()
    expect(editing).to_be_hidden()

    open_chat(page)
    expect(model_options(page, f"{prefix}.qwen3:latest")).to_have_count(1)
    expect(model_options(page, f"{prefix}.llama3:latest")).to_have_count(0)

    settings = open_admin_connections(page)
    row = connection_row(settings, OLLAMA_URL_PLACEHOLDER, listener.base_url)
    tooltip_button(row, "Configure").click()
    editing = connection_dialog(page, "Edit Connection")
    editing.get_by_role("button", name="Delete").click()
    with page.expect_response(is_ollama_save):
        page.get_by_role("dialog").get_by_role("button", name="Delete").last.click()
    expect(editing).to_be_hidden()

    open_chat(page)
    expect(model_options(page, f"{prefix}.qwen3:latest")).to_have_count(0)


def test_an_ollama_connection_switched_off_in_its_row_leaves_a_users_selector_and_returns(
    page_for, make_user, admin, preserve, listener, prefix
):
    preserve(OLLAMA_CONFIG)
    serve_ollama(listener, "llama3:latest")
    with admin.client() as client:
        connect_ollama(client, listener, prefix_id=prefix)
    publish(admin, f"{prefix}.llama3:latest")
    page = page_for(make_user())
    open_chat(page)
    expect(model_options(page, f"{prefix}.llama3:latest")).to_have_count(1)

    admin_page = page_for(make_user(role="admin"))
    settings = open_admin_connections(admin_page)
    switch = connection_row(settings, OLLAMA_URL_PLACEHOLDER, listener.base_url).get_by_role(
        "switch"
    )
    with admin_page.expect_response(is_ollama_save):
        switch.click()
    expect(switch).not_to_be_checked()
    open_chat(page)
    expect(model_options(page, f"{prefix}.llama3:latest")).to_have_count(0)

    with admin_page.expect_response(is_ollama_save):
        switch.click()
    expect(switch).to_be_checked()
    open_chat(page)
    expect(model_options(page, f"{prefix}.llama3:latest")).to_have_count(1)
