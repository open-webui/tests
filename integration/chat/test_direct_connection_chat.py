"""Journey: a chat with a model from the user's own direct connection, answered by their browser.

The server never calls a direct connection. It asks the user's tab over the tab's socket to run
the completion (the request, the model and a channel), then reads the stream the tab forwards
on that channel and stores the reply like any other: text, usage, tool calls it runs itself
before asking the tab again with the result. The tab here plays the web client's part.

Discriminates: in a backend copy, dropping the dict frames the tab forwards fails the streamed
reply test, ignoring the tab's refusal (treating any acknowledgement as a go) fails the refusal
test, and skipping the session owner check lets the stranger's request reach the owner's tab.
"""

from __future__ import annotations

import pytest

from harness.chat import send_message, wait_for_reply
from harness.direct_connection import answering, chunk_line, direct_model
from harness.second_provider import tool_call_delta

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

MODEL = "my-own-model"


def _send(client, tab_session_id: str, content: str, **extra):
    return send_message(
        client,
        content,
        model=MODEL,
        model_item=direct_model(MODEL),
        session_id=tab_session_id,
        **extra,
    )


def test_a_streamed_reply_from_the_tab_is_stored(make_user, upstream):
    person = make_user()
    with answering(person) as tab, person.client() as client:
        tab.stream(
            chunk_line({"role": "assistant", "content": ""}),
            chunk_line({"content": "hi "}),
            {"choices": [{"index": 0, "delta": {"content": "there"}}]},
            chunk_line({}, "stop", usage={"prompt_tokens": 3, "completion_tokens": 2}),
        )
        message = wait_for_reply(client, _send(client, tab.session_id, "hello"))

    assert message["content"] == "hi there"
    assert (message["usage"]["input_tokens"], message["usage"]["output_tokens"]) == (3, 2)
    [request] = tab.requests
    assert request["session_id"] == tab.session_id
    assert request["model"]["id"] == MODEL
    assert request["form_data"]["model"] == MODEL
    assert request["form_data"]["stream"] is True
    assert request["form_data"]["messages"] == [{"role": "user", "content": "hello"}]
    assert upstream.chat_requests() == [], "the server's own provider answered a direct chat"


def test_a_tool_call_runs_on_the_server_and_goes_back_to_the_tab(make_user, upstream):
    person = make_user()
    with answering(person) as tab, person.client() as client:
        tab.stream(
            chunk_line(tool_call_delta("get_current_timestamp", {})),
            chunk_line({}, "tool_calls"),
        )
        tab.stream(chunk_line({"content": "It is late."}), chunk_line({}, "stop"))
        message = wait_for_reply(client, _send(client, tab.session_id, "what time is it?"))

    assert message["content"] == "It is late."
    first, follow_up = tab.requests
    assert first["channel"] != follow_up["channel"]
    assistant, tool = follow_up["form_data"]["messages"][-2:]
    assert assistant["tool_calls"][0]["function"]["name"] == "get_current_timestamp"
    assert tool["role"] == "tool"
    assert tool["tool_call_id"] == assistant["tool_calls"][0]["id"]
    assert "current_timestamp" in tool["content"]


def test_a_request_the_tab_refuses_is_stored_as_an_error(make_user):
    person = make_user()
    with answering(person) as tab, person.client() as client:
        tab.refuse({"error": {"message": "invalid api key"}})
        message = wait_for_reply(client, _send(client, tab.session_id, "hello"))

    assert "invalid api key" in str(message.get("error")), message
    assert message["content"] == ""


def test_without_a_live_tab_the_reply_is_an_error(make_user):
    with make_user().client() as client:
        message = wait_for_reply(client, _send(client, "no-such-session", "hello"))

    assert "Client session disconnected" in str(message.get("error")), message


def test_another_users_tab_is_never_asked(make_user):
    owner, stranger = make_user(), make_user()
    with answering(owner) as tab, stranger.client() as client:
        message = wait_for_reply(client, _send(client, tab.session_id, "use their provider"))

    assert tab.requests == [], "the stranger's chat was sent to the owner's browser"
    assert "Client session disconnected" in str(message.get("error")), message


def test_a_non_streamed_reply_from_the_tab_is_stored(make_user):
    person = make_user()
    completion = {
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "whole answer"}}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
    }
    with answering(person) as tab, person.client() as client:
        tab.complete(completion)
        message = wait_for_reply(client, _send(client, tab.session_id, "hello", stream=False))

    assert message["content"] == "whole answer"
    assert message["usage"]["total_tokens"] == 6
    assert tab.requests[0]["form_data"]["stream"] is False
