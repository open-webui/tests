"""Journey: a chat started from a link with URL parameters, as the URL Parameters docs page lists.

`/?q=` sends its text as the first message at once, and with `submit=false` only puts it in the
message box. `models=` opens the chat on several models, which all answer. `temporary-chat=true`
starts a chat that is never stored. `tools=` (or its older name `tool-ids=`) turns a workspace
tool on for the chat, so it is offered to the model, and `web-search=true` sends the message with
web search on. `image-generation=true` sends it with image generation on, so the model is offered
the image tool, which a chat opened without the parameter is not.

Discriminates: passes on dev 30f3f6a8f; in a frontend build that ignores the q, models,
temporary-chat, tools and web-search parameters every test fails, each on its own parameter; in
one that reads `tool_ids` in place of `tool-ids` and turns image generation on exactly when the
parameter is missing, the tool-ids and both image generation tests fail.
"""

from __future__ import annotations

import re
import uuid
from urllib.parse import quote

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.channel_chat import serve_openai_images
from harness.image_engines import IMAGES_CONFIG, save_image_settings
from harness.python_tools import python_tool
from harness.upstream import MOCK_MODEL_ID
from harness.web_retrieval import RETRIEVAL_CONFIG, save_web_settings, serve_search_results
from utils.chat_ui import chat_input, expect_reply, replies, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

TIDE_TOOL = """class Tools:
    def tide_times(self, harbour: str) -> str:
        \"\"\"Tell when the tide turns.

        :param harbour: the harbour
        \"\"\"
        return f"High tide in {harbour} at noon."
"""


def _question() -> str:
    return f"when is high tide? {uuid.uuid4().hex[:6]}"


def _open(page: Page, query: str) -> None:
    page.goto(f"/?{query}")


def _request_for(upstream, question: str) -> dict:
    return next(filter(reply.answering(question), upstream.chat_requests()))


def test_q_sends_its_text_as_the_first_message(page_for, make_user, upstream):
    page = page_for(make_user())
    question = _question()
    upstream.queue(reply.text("At noon.", match=reply.answering(question)))
    _open(page, f"q={quote(question)}")

    expect_reply(page, "At noon.")
    expect(page).to_have_url(re.compile(r"/c/"))


def test_q_with_submit_false_only_fills_the_message_box(page_for, make_user, upstream):
    page = page_for(make_user())
    question = _question()
    _open(page, f"q={quote(question)}&submit=false")

    expect(chat_input(page)).to_have_text(question)
    expect(replies(page)).to_have_count(0)
    assert not [body for body in upstream.chat_requests() if reply.answering(question)(body)]


@pytest.fixture
def second_model(make_user):
    """A fresh admin and a second model of theirs on the scripted provider; yields both."""
    account = make_user(role="admin")
    model_id = f"tide-tables-{uuid.uuid4().hex[:6]}"
    form = {"id": model_id, "name": "Tide Tables", "base_model_id": MOCK_MODEL_ID}
    with account.client() as client:
        created = client.post("/api/v1/models/create", json={**form, "meta": {}, "params": {}})
        assert created.status_code == 200, created.text
        yield account, model_id
        client.post("/api/v1/models/model/delete", json={"id": model_id})


def test_models_opens_the_chat_on_every_model_listed(page_for, second_model, upstream):
    account, model_id = second_model
    page = page_for(account)
    question = _question()
    upstream.queue(
        reply.text("First says noon.", match=reply.answering(question)),
        reply.text("Second says noon.", match=reply.answering(question)),
    )
    _open(page, f"models={MOCK_MODEL_ID},{model_id}")
    send(page, question)

    expect(replies(page)).to_have_count(2)
    expect(page.get_by_text("First says noon.")).to_be_visible()
    expect(page.get_by_text("Second says noon.")).to_be_visible()


def test_temporary_chat_true_keeps_the_chat_out_of_storage(page_for, make_user, upstream):
    account = make_user()
    page = page_for(account)
    question = _question()
    upstream.queue(reply.text("Noon, unrecorded.", match=reply.answering(question)))
    _open(page, "temporary-chat=true")
    send(page, question)

    expect_reply(page, "Noon, unrecorded.")
    expect(page).not_to_have_url(re.compile(r"/c/"))
    with account.client() as client:
        assert client.get("/api/v1/chats/", params={"page": 1}).json() == []


def test_tools_turns_the_tool_on_for_the_chat(page_for, admin, make_user, upstream):
    with python_tool(admin, TIDE_TOOL, name="Tide times") as tool_id:
        page = page_for(make_user())
        question = _question()
        upstream.queue(reply.text("Noon.", match=reply.answering(question)))
        _open(page, f"tools={tool_id}")
        send(page, question)
        expect_reply(page, "Noon.")

    offered = {tool["function"]["name"] for tool in _request_for(upstream, question)["tools"]}
    assert "tide_times" in offered, sorted(offered)


def test_tool_ids_turns_the_tool_on_like_tools(page_for, admin, make_user, upstream):
    with python_tool(admin, TIDE_TOOL, name="Tide times") as tool_id:
        page = page_for(make_user())
        question = _question()
        upstream.queue(reply.text("Noon.", match=reply.answering(question)))
        _open(page, f"tool-ids={tool_id}")
        send(page, question)
        expect_reply(page, "Noon.")

    offered = {tool["function"]["name"] for tool in _request_for(upstream, question)["tools"]}
    assert "tide_times" in offered, sorted(offered)


@pytest.fixture
def image_generation_on(admin, preserve, listener) -> None:
    preserve(IMAGES_CONFIG)
    with admin.client() as client:
        save_image_settings(client, **serve_openai_images(listener))


def _chat_request_features(page: Page, question: str) -> dict:
    def is_chat_request(request) -> bool:
        return request.method == "POST" and request.url.endswith("/api/chat/completions")

    with page.expect_request(is_chat_request) as sent:
        send(page, question)
    return sent.value.post_data_json.get("features") or {}


def test_image_generation_true_offers_the_model_the_image_tool(
    page_for, make_user, upstream, image_generation_on
):
    page = page_for(make_user())
    question = _question()
    upstream.queue(reply.text("Drawn.", match=reply.answering(question)))
    _open(page, "image-generation=true")

    features = _chat_request_features(page, question)
    expect_reply(page, "Drawn.")

    assert features.get("image_generation") is True
    offered = {tool["function"]["name"] for tool in _request_for(upstream, question)["tools"]}
    assert "generate_image" in offered, sorted(offered)


def test_without_the_parameter_image_generation_stays_off(
    page_for, make_user, upstream, image_generation_on
):
    page = page_for(make_user())
    question = _question()
    upstream.queue(reply.text("Not drawn.", match=reply.answering(question)))
    _open(page, "")

    features = _chat_request_features(page, question)
    expect_reply(page, "Not drawn.")

    assert features.get("image_generation") is False
    offered = {
        tool["function"]["name"] for tool in _request_for(upstream, question).get("tools") or []
    }
    assert "generate_image" not in offered, sorted(offered)


@pytest.fixture
def web_search_on(admin, preserve, listener) -> None:
    preserve(RETRIEVAL_CONFIG)
    with admin.client() as client:
        save_web_settings(client, **serve_search_results(listener, []))


def test_web_search_true_sends_the_message_with_web_search_on(
    page_for, make_user, upstream, web_search_on
):
    page = page_for(make_user())
    question = _question()
    upstream.queue(reply.text("Searched.", match=reply.answering(question)))

    def is_chat_request(request) -> bool:
        return request.method == "POST" and request.url.endswith("/api/chat/completions")

    _open(page, "web-search=true")
    with page.expect_request(is_chat_request) as sent:
        send(page, question)
    assert (sent.value.post_data_json.get("features") or {}).get("web_search") is True
