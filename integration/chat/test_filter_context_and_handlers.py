"""A filter exposing a non-callable `inlet`, `request`, `stream` or `outlet` broke every chat.

Fix commit `2a4ef46ac` in `utils/filter.py`. The filter runner took any truthy attribute named
after a stage as that stage's handler, so a global filter with, say, `inlet = "not implemented"`
reached `inspect.signature` and every chat request on the instance failed. The guard is now
`callable(handler)`, so such an attribute is skipped.

Twin of unit/chat/test_filter_context_and_handlers.py (its handler cases; the once-per-request
fetch of the active filters stays there).

Discriminates: with 2a4ef46ac reverted every truthy non-callable case fails (an inlet, request or
stream attribute ends the chat in an error, an outlet one stops the outlet stage so the next
filter's outlet never runs); the falsy attributes, the empty filter and the working handlers
pass on both.
"""

from __future__ import annotations

import json
import time

import httpx
import pytest

from harness import upstream as reply
from harness.chat import ChatTurn, ask
from harness.plugins import installed_function
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


def filter_with(stage: str, value: str) -> str:
    return f"class Filter:\n    {stage} = {value}\n"


def direct_stream_text(client: httpx.Client) -> str:
    request = {
        "model": MOCK_MODEL_ID,
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    }
    response = client.post("/api/chat/completions", json=request)
    assert response.status_code == 200, response.text
    events = [
        json.loads(line.removeprefix("data:"))
        for line in response.text.splitlines()
        if line.startswith("data:") and line != "data: [DONE]"
    ]
    assert not [event for event in events if "error" in event], events
    return "".join(
        choice["delta"].get("content") or ""
        for event in events
        for choice in event.get("choices", [])
    )


# Runs after the filter under test; a stage that raises stops it, so its mark goes missing.
OUTLET_WITNESS = """
from pydantic import BaseModel


class Filter:
    class Valves(BaseModel):
        priority: int = 1

    def __init__(self):
        self.valves = self.Valves()

    async def outlet(self, body):
        body["messages"][-1]["content"] += " [outlet]"
        return body
"""


def stored_reply(client: httpx.Client, turn: ChatTurn, expected: str) -> str:
    """The stored reply once the outlet stage has rewritten it, or as it stands at the deadline."""
    deadline = time.monotonic() + 10
    while True:
        chat = client.get(f"/api/v1/chats/{turn.chat_id}").json()["chat"]
        content = chat["history"]["messages"][turn.assistant_message_id]["content"]
        if content == expected or time.monotonic() > deadline:
            return content
        time.sleep(0.2)


def assert_chats_still_answer(admin, user, upstream, filter_source: str) -> None:
    upstream.queue(reply.text("session reply"), reply.text("direct reply"))
    with (
        installed_function(admin, filter_source, is_global=True),
        installed_function(admin, OUTLET_WITNESS, is_global=True),
        user.client() as client,
    ):
        turn, message = ask(client, "hello")
        assert not message.get("error"), message
        assert stored_reply(client, turn, "session reply [outlet]") == "session reply [outlet]"
        assert direct_stream_text(client) == "direct reply"


@pytest.mark.parametrize("stage", ["inlet", "request", "stream", "outlet"])
@pytest.mark.parametrize(
    "value",
    ['"not implemented"', '{"enabled": True}', "[1, 2]", "7"],
    ids=["str", "dict", "list", "int"],
)
def test_a_non_callable_stage_attribute_is_skipped(admin, user, upstream, stage, value):
    assert_chats_still_answer(admin, user, upstream, filter_with(stage, value))


@pytest.mark.parametrize("value", ["None", '""', "0", "[]", "{}"])
def test_a_falsy_stage_attribute_is_skipped(admin, user, upstream, value):
    assert_chats_still_answer(admin, user, upstream, filter_with("inlet", value))


def test_a_filter_without_handlers_changes_nothing(admin, user, upstream):
    assert_chats_still_answer(admin, user, upstream, "class Filter:\n    pass\n")


MARKING_FILTER = """
class Filter:
    async def inlet(self, body):
        body["messages"][-1]["content"] += " [inlet]"
        return body

    async def request(self, body):
        body["messages"][-1]["content"] += " [request]"
        return body

    def stream(self, event):
        for choice in event.get("choices", []):
            if choice.get("delta", {}).get("content"):
                choice["delta"]["content"] += " [stream]"
        return event
"""


def test_callable_handlers_still_run(admin, user, upstream):
    upstream.queue(reply.text("direct reply"))
    with installed_function(admin, MARKING_FILTER, is_global=True):
        with user.client() as client:
            streamed = direct_stream_text(client)

    assert upstream.chat_requests()[-1]["messages"][-1]["content"] == "hi [inlet] [request]"
    assert streamed == "direct reply [stream]"
