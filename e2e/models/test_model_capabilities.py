"""Journey: each capability switch in the model editor, and what it changes in a chat on the model.

A fresh admin unticks or ticks one capability of a preset on the scripted model in the workspace's
model editor, saves it and opens a chat on it, after a first look at the same chat with the switch
as it was. Without Vision a picture attached in the chat input is refused with a notice and never
attached, while an SVG, which the models docs page says is read as a document, still goes up.
Without File Context the text of a file attached in the chat stays out of the message and the
model is offered the chat file tools to read it instead. Without Citations the reply shows no
sources, neither the inline source chips nor the sources row under it. With Usage ticked the
provider is asked to report token counts (e2e/chat/test_reply_usage.py shows them). Without Image
Generation the chat's Image switch is gone although the admin has image generation on. Without
Memory a stored memory no longer reaches the model, as context or through the memory tools.
Without Terminal the chat offers no terminal and one picked before is not handed to the model. An
unticked capability is no longer offered under Default Features. File Upload, Web Search, Builtin
Tools and the Time category are covered in e2e/models/test_model_editor.py, Status Updates in
e2e/chat/test_status_updates.py; the Code Interpreter is legacy and left out.

`test_unticking_file_context_keeps_the_file_text_out_and_offers_the_file_tools` is red now and then
on dev 896056690: since de73bb830 the reply in a new chat sometimes stays blank until a reload
although the server saved it whole (open-webui/open-webui#32091).

Discriminates: passes on the dev ebc6add67 build. In a frontend build whose chat input ignores the
Vision, Image Generation and Terminal capabilities those three tests fail, in one whose replies
ignore the Citations capability the citations test fails and in one whose editor lists every
default feature the default feature test fails. In a backend copy that injects file text whatever
File Context says the file context test fails, in one that never asks for usage the usage test
fails and in one that ignores the Memory capability the memory test fails.
Retargeted for 8d0ff76f2, whose capabilities are switches in sections that open on a click: passes
on dev 76ad6f97c (3 of 3), and in a build of it whose editor saves the model's settings as they were
loaded every test that saves an edit fails.
"""

from __future__ import annotations

import base64
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.channel_chat import serve_openai_images
from harness.image_engines import IMAGES_CONFIG, PNG_BASE64, save_image_settings
from harness.listener import json_answer
from harness.python_tools import EVERYONE_READS
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, serving_terminal
from harness.upstream import MOCK_MODEL_ID
from utils.cached_chat import attach
from utils.chat_ui import conversation, expect_reply, last_reply, send
from utils.model_editor import (
    offered_tool_names,
    open_chat_on,
    open_editor,
    open_menu,
    save,
    section,
    sent_request,
    set_checkbox,
)
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

PICTURE = base64.b64decode(PNG_BASE64)
HARBOUR_CHART = '<svg xmlns="http://www.w3.org/2000/svg"><text>Pier 7</text></svg>'
FERRY_NOTES = ("ferry.txt", "The ferry leaves pier 7 at 06:40 daily.")
CHAT_FILE_TOOLS = {"list_chat_files", "query_chat_files", "grep_chat_files", "view_file"}
MEMORY_TOOLS = {"search_memories", "add_memory", "list_memories"}
TERMINAL_NAME = "Harbour shell"
RUN_COMMAND_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Terminal", "version": "1"},
    "paths": {
        "/execute": {
            "post": {"operationId": "run_command", "responses": {"200": {"description": "ok"}}}
        }
    },
}


def unique(text: str) -> str:
    return f"{text} {uuid.uuid4().hex[:6]}"


@pytest.fixture
def builder(make_user):
    """A fresh admin; the models it made are deleted afterwards."""
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for model in client.get("/api/v1/models/list").json().get("items", []):
            if model["user_id"] == account.id:
                client.post("/api/v1/models/model/delete", json={"id": model["id"]})


@pytest.fixture
def preset(builder) -> dict:
    """A preset on the scripted model that the builder made over the API, every account reads."""
    model = {"id": f"capable-{uuid.uuid4().hex[:8]}", "name": unique("Ferry desk")}
    with builder.client() as client:
        created = client.post(
            "/api/v1/models/create",
            json={
                **model,
                "base_model_id": MOCK_MODEL_ID,
                "meta": {},
                "params": {},
                "access_grants": [EVERYONE_READS],
            },
        )
    assert created.status_code == 200, created.text
    return model


def switch_capability(page: Page, model: dict, label: str, checked: bool) -> None:
    editor = open_editor(page, model)
    set_checkbox(editor, "Capabilities", label, checked)
    save(editor)


def image_preview(page: Page) -> Locator:
    """The picture attached in the chat input, waiting to be sent."""
    return page.get_by_role("button", name="Show image preview")


def test_unticking_vision_refuses_a_picture_attached_in_the_chat(page_for, builder, preset):
    page = page_for(builder)
    open_chat_on(page, preset)
    attach(page, "harbour.png", PICTURE, "image/png")
    expect(image_preview(page)).to_be_visible()

    switch_capability(page, preset, "Vision", False)

    open_chat_on(page, preset)
    attach(page, "harbour.png", PICTURE, "image/png")
    expect(page.get_by_text("Selected model(s) do not support image inputs")).to_be_visible()
    expect(image_preview(page)).to_have_count(0)

    # an SVG is read as a document, so it still goes up as a file
    attach(page, "chart.svg", HARBOUR_CHART, "image/svg+xml")
    expect(page.locator("form").get_by_text("chart.svg")).to_be_visible()


def ask_about_attached_notes(page: Page, upstream, question: str) -> dict:
    attach(page, *FERRY_NOTES)
    expect(page.locator("form").get_by_text(FERRY_NOTES[0])).to_be_visible()
    return sent_request(page, upstream, question)


def test_unticking_file_context_keeps_the_file_text_out_and_offers_the_file_tools(
    page_for, builder, preset, upstream
):
    page = page_for(builder)
    open_chat_on(page, preset)
    with_context = ask_about_attached_notes(page, upstream, "when does the ferry leave?")
    assert "06:40" in str(with_context["messages"])
    assert not CHAT_FILE_TOOLS & offered_tool_names(with_context)

    switch_capability(page, preset, "File Context", False)

    open_chat_on(page, preset)
    without_context = ask_about_attached_notes(page, upstream, "when does the ferry go?")
    assert "06:40" not in str(without_context["messages"])
    offered = offered_tool_names(without_context)
    assert CHAT_FILE_TOOLS <= offered, f"the chat file tools were not offered: {sorted(offered)}"


def cited_answer(page: Page, upstream, question: str) -> Locator:
    """Ask about an attached file; the model cites it as `[1]`. Returns the reply."""
    attach(page, *FERRY_NOTES)
    expect(page.locator("form").get_by_text(FERRY_NOTES[0])).to_be_visible()
    upstream.queue(reply.text("It leaves at 06:40 [1].", match=reply.answering(question)))
    send(page, question)
    expect_reply(page, "It leaves at 06:40")
    return last_reply(page)


def test_unticking_citations_shows_no_sources_with_the_reply(page_for, builder, preset, upstream):
    page = page_for(builder)
    open_chat_on(page, preset)
    cited = cited_answer(page, upstream, "when does the ferry leave?")
    expect(cited.get_by_role("button", name=f"View source: {FERRY_NOTES[0]}")).to_be_visible()
    expect(cited.get_by_role("button", name="Toggle 1 source")).to_be_visible()

    switch_capability(page, preset, "Citations", False)

    open_chat_on(page, preset)
    uncited = cited_answer(page, upstream, "when does the ferry go?")
    expect(uncited).to_contain_text("It leaves at 06:40.")
    expect(uncited).not_to_contain_text("[1]")
    expect(uncited.get_by_role("button", name=f"View source: {FERRY_NOTES[0]}")).to_have_count(0)
    expect(conversation(page).get_by_role("button", name="Toggle 1 source")).to_have_count(0)


def test_ticking_usage_asks_the_provider_for_token_counts(page_for, builder, preset, upstream):
    page = page_for(builder)
    open_chat_on(page, preset)
    assert "stream_options" not in sent_request(page, upstream, "how full is the ferry?")

    switch_capability(page, preset, "Usage", True)

    open_chat_on(page, preset)
    request = sent_request(page, upstream, "how full is the ferry now?")
    assert request.get("stream_options") == {"include_usage": True}, request.get("stream_options")


@pytest.fixture
def image_generation_on(admin, preserve, listener) -> None:
    preserve(IMAGES_CONFIG)
    with admin.client() as client:
        save_image_settings(client, **serve_openai_images(listener))


def test_unticking_image_generation_takes_the_image_switch_out(
    page_for, builder, preset, image_generation_on
):
    page = page_for(builder)
    open_chat_on(page, preset)
    expect(open_menu(page, "Integrations").get_by_role("button", name="Image")).to_be_visible()

    switch_capability(page, preset, "Image Generation", False)

    open_chat_on(page, preset)
    integrations = page.get_by_role("button", name="Integrations", exact=True)
    if integrations.count():
        open_menu(page, "Integrations")
    expect(page.get_by_role("menu").get_by_role("button", name="Image")).to_have_count(0)


def test_unticking_memory_keeps_a_stored_memory_out_of_the_chat(
    page_for, builder, preset, upstream
):
    memory = unique("My boat is the Seagull, moored at")
    with builder.client() as client:
        added = client.post("/api/v1/memories/add", json={"content": memory})
    assert added.status_code == 200, added.text
    page = page_for(builder)
    open_chat_on(page, preset)
    remembered = sent_request(page, upstream, "where is my boat?")
    assert memory in str(remembered["messages"])
    assert MEMORY_TOOLS <= offered_tool_names(remembered)

    switch_capability(page, preset, "Memory", False)

    open_chat_on(page, preset)
    forgotten = sent_request(page, upstream, "where is my boat now?")
    assert memory not in str(forgotten["messages"])
    offered = offered_tool_names(forgotten)
    assert not MEMORY_TOOLS & offered, f"memory tools were still offered: {sorted(offered)}"


@pytest.fixture
def terminal(admin, preserve):
    """A terminal connection every account may use, to a fake terminal server."""
    preserve(TERMINAL_SERVERS_CONFIG)
    with serving_terminal() as server:
        server.route("GET", "/openapi.json", json_answer(RUN_COMMAND_SPEC))
        connection = server.connection(
            name=TERMINAL_NAME, config={"access_grants": [EVERYONE_READS]}
        )
        with admin.client() as client:
            configure_terminals(client, connection)
        yield server


def terminal_buttons(page: Page) -> int:
    return (
        page.get_by_role("main")
        .get_by_role("button")
        .evaluate_all(
            "(buttons) => buttons.filter("
            "(button) => button.parentElement?._tippy?.props.content === 'Terminal').length"
        )
    )


def test_unticking_terminal_takes_the_terminal_out_of_the_chat(
    page_for, builder, preset, upstream, terminal
):
    page = page_for(builder)
    open_chat_on(page, preset)
    tooltip_button(page.get_by_role("main"), "Terminal").click()
    page.get_by_role("menu").get_by_role("button", name=TERMINAL_NAME).click()
    expect(page.get_by_role("region", name="File browser")).to_be_visible()
    assert "run_command" in offered_tool_names(sent_request(page, upstream, "list the berths"))

    switch_capability(page, preset, "Terminal", False)

    open_chat_on(page, preset)
    request = sent_request(page, upstream, "list the berths again")
    assert "run_command" not in offered_tool_names(request)
    assert terminal_buttons(page) == 0, "the chat still offers a terminal"


def test_an_unticked_capability_is_not_offered_as_a_default_feature(page_for, builder, preset):
    page = page_for(builder)
    editor = open_editor(page, preset)
    defaults = section(editor, "Default Features")
    expect(defaults.get_by_role("switch", name="Web Search", exact=True)).to_be_visible()
    expect(defaults.get_by_role("switch", name="Image Generation", exact=True)).to_be_visible()

    set_checkbox(editor, "Capabilities", "Web Search", False)

    expect(defaults.get_by_role("switch", name="Web Search", exact=True)).to_have_count(0)
    expect(defaults.get_by_role("switch", name="Image Generation", exact=True)).to_be_visible()
