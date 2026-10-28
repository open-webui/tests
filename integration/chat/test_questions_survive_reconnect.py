"""Journey: a question waiting for the user is asked again once they reopen the chat.

Since 0.12.4 (c596d40bd) a tool's input or confirmation dialog, or an MCP server's input request,
that is still open when the page reloads or loses its connection waits up to a minute for a new
connection of the same account to open that chat, asks it there, and hands its answer back.
With no reconnect in that minute the tool gets "Client session disconnected." A temporary chat
cannot be reopened, so its questions (Ask User, a tool approval) end as soon as the page goes.
Ask User and tool approvals in a saved chat are stored with the chat and wait there instead, so
no socket is involved in getting them back (see the browser twin).

The minute is fixed in the server, so the test of a missed reconnect waits it out.

Twin of e2e/chat/test_questions_survive_reconnect.py.

Discriminates: passes on dev 948432b43. With the resume turned off in a backend copy (no call
may wait for a reconnect) the reconnect tests fail (the tool gets "Client session disconnected."
at once and the new connection is never asked); with temporary chats allowed to wait the
temporary chat tests fail (no answer for the model within 20s); with the minute raised to an
hour the missed reconnect test fails (the reply never finishes).
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from harness import upstream as reply
from harness.chat import send_message, wait_for_reply
from harness.python_tools import python_tool
from harness.socket_client import SocketSession, connected

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

CHAT_CONFIG = ("/api/v1/chats/config", "/api/v1/chats/config")
DISCONNECTED = "Client session disconnected."
# the window is a minute; a call that may not wait for a reconnect ends within a second or two
NO_RESUME_BOUND = 20.0

ASK_THE_TAB = '''
import json


class Tools:
    async def ask_the_tab(self, kind: str, __event_call__=None) -> str:
        """
        Ask the user's open tab something.
        :param kind: input or confirmation
        """
        question = {"title": "Your name?", "message": "For the greeting."}
        answer = await __event_call__({"type": kind, "data": question})
        return json.dumps(answer)
'''

TAB_ANSWERS = {"input": "Ada", "confirmation": True}


@pytest.fixture
def tool_approval_on(admin, preserve) -> None:
    preserve(CHAT_CONFIG)
    with admin.client() as client:
        current = client.get(CHAT_CONFIG[0]).json()
        saved = client.post(CHAT_CONFIG[1], json={**current, "ENABLE_TOOL_PERMISSIONS": True})
    assert saved.status_code == 200, saved.text


class ClosingTab:
    """A tab that sees questions of one kind and closes before answering them."""

    def __init__(self, tab: SocketSession, kind: str):
        self.tab = tab
        self.kind = kind
        self.asked: list[dict] = []
        self.closed = threading.Event()
        tab.client.on("events", self.on_events)

    def on_events(self, message: dict):
        self.tab.events.append(message)
        if (message.get("data") or {}).get("type") == self.kind:
            self.asked.append(message)
            # the handler's return value is the answer, so hold it until the tab is gone
            self.closed.wait(60)

    def close_once_asked(self, timeout: float = 30.0) -> dict:
        deadline = time.monotonic() + timeout
        while not self.asked:
            if time.monotonic() > deadline:
                raise AssertionError(f"the tab was never asked for {self.kind} within {timeout}s")
            time.sleep(0.05)
        self.tab.client.disconnect()
        self.closed.set()
        return self.asked[0]


def answer_questions(tab: SocketSession, kind: str) -> list[dict]:
    """Answer every question of `kind` the server asks this tab; returns what it asked."""
    asked: list[dict] = []

    def on_events(message: dict):
        tab.events.append(message)
        if (message.get("data") or {}).get("type") != kind:
            return None
        asked.append(message)
        return TAB_ANSWERS[kind]

    tab.client.on("events", on_events)
    return asked


def open_chat(tab: SocketSession, chat_id: str) -> None:
    """Join the chat's room the way the page does when it opens a chat."""
    joined = tab.call("events:chat", {"chat_id": chat_id, "data": {"type": "join"}})
    assert joined is True, f"the new tab could not open chat {chat_id}: {joined}"


def tool_results(upstream, prompt: str) -> list[str]:
    return [
        str(entry["content"])
        for sent in upstream.chat_requests()
        if reply.answering(prompt)(sent)
        for entry in sent["messages"]
        if entry["role"] == "tool"
    ]


def wait_for_tool_result(upstream, prompt: str, timeout: float) -> str:
    deadline = time.monotonic() + timeout
    while not tool_results(upstream, prompt):
        if time.monotonic() > deadline:
            raise AssertionError(f"the model never got the tool's result within {timeout}s")
        time.sleep(0.1)
    return tool_results(upstream, prompt)[0]


@pytest.mark.parametrize("kind", ["input", "confirmation"])
def test_a_tools_question_is_asked_again_on_the_reopened_chat_and_its_answer_finishes_the_reply(
    admin, make_user, upstream, kind
):
    account = make_user()
    prompt = f"who am I? ({kind} across a reload)"
    upstream.queue(
        reply.tool_call("ask_the_tab", {"kind": kind}, match=reply.answering(prompt)),
        reply.text("Nice to meet you.", match=reply.answering(prompt)),
    )
    with (
        python_tool(admin, ASK_THE_TAB, name="Ask the tab") as tool_id,
        account.client() as client,
        connected(account) as first_tab,
    ):
        closing = ClosingTab(first_tab, kind)
        turn = send_message(
            client, prompt, tool_ids=[tool_id], session_id=first_tab.client.get_sid()
        )
        first_question = closing.close_once_asked()

        with connected(account) as second_tab:
            asked_again = answer_questions(second_tab, kind)
            open_chat(second_tab, turn.chat_id)
            message = wait_for_reply(client, turn)
            assert len(asked_again) == 1, f"the reopened chat was asked {len(asked_again)} times"
            # what closes the dialog in the reopened page
            interaction_id = first_question["data"]["data"]["interaction_id"]
            second_tab.wait_for(
                turn.chat_id, "request:interaction:done", interaction_id=interaction_id
            )

    assert asked_again[0]["data"] == first_question["data"], (
        "the reopened chat was asked something else than the question left open"
    )
    assert tool_results(upstream, prompt) == [json.dumps(TAB_ANSWERS[kind])], (
        "the answer from the reopened chat never reached the tool"
    )
    assert message["content"].endswith("Nice to meet you."), message


def test_a_connection_that_does_not_open_the_chat_is_not_asked(admin, make_user, upstream):
    account = make_user()
    prompt = "who am I? (another chat open after the reload)"
    elsewhere = "meanwhile, a chat about the weather"
    upstream.queue(
        reply.tool_call("ask_the_tab", {"kind": "input"}, match=reply.answering(prompt)),
        reply.text("Nice to meet you.", match=reply.answering(prompt)),
        reply.text("Sunny.", match=reply.answering(elsewhere)),
    )
    with (
        python_tool(admin, ASK_THE_TAB, name="Ask the tab") as tool_id,
        account.client() as client,
        connected(account) as first_tab,
    ):
        closing = ClosingTab(first_tab, "input")
        turn = send_message(
            client, prompt, tool_ids=[tool_id], session_id=first_tab.client.get_sid()
        )
        closing.close_once_asked()
        other_chat = send_message(client, elsewhere)
        wait_for_reply(client, other_chat)

        with connected(account) as other_tab, connected(account) as reopened_tab:
            asked_elsewhere = answer_questions(other_tab, "input")
            open_chat(other_tab, other_chat.chat_id)
            time.sleep(
                3
            )  # the server looks for the reopened chat every second; it must not pick this tab
            asked_again = answer_questions(reopened_tab, "input")
            open_chat(reopened_tab, turn.chat_id)
            wait_for_reply(client, turn)

    assert asked_elsewhere == [], "a tab showing another chat was asked the open question"
    assert len(asked_again) == 1, f"the reopened chat was asked {len(asked_again)} times"


@pytest.mark.slow
def test_with_no_reconnect_within_a_minute_the_tool_hears_the_tab_is_gone(
    admin, make_user, upstream
):
    account = make_user()
    prompt = "who am I? (nobody comes back)"
    upstream.queue(
        reply.tool_call("ask_the_tab", {"kind": "input"}, match=reply.answering(prompt)),
        reply.text("Nobody answered.", match=reply.answering(prompt)),
    )
    with (
        python_tool(admin, ASK_THE_TAB, name="Ask the tab") as tool_id,
        account.client() as client,
        connected(account) as first_tab,
    ):
        closing = ClosingTab(first_tab, "input")
        turn = send_message(
            client, prompt, tool_ids=[tool_id], session_id=first_tab.client.get_sid()
        )
        closing.close_once_asked()
        message = wait_for_reply(client, turn, timeout=90)

    assert tool_results(upstream, prompt) == [json.dumps({"error": DISCONNECTED})], (
        "the tool did not hear that the tab was gone once the minute was over"
    )
    assert message["content"].endswith("Nobody answered."), message


def test_ask_user_in_a_temporary_chat_ends_when_the_tab_goes(make_user, upstream):
    account = make_user()
    prompt = "which harbour? (temporary chat)"
    questions = {
        "questions": [
            {
                "id": "harbour",
                "question": "Which harbour?",
                "options": [
                    {"label": "Trieste", "description": "north"},
                    {"label": "Split", "description": "south"},
                ],
            }
        ]
    }
    upstream.queue(
        reply.tool_call("ask_user", questions, match=reply.answering(prompt)),
        reply.text("Nobody answered.", match=reply.answering(prompt)),
    )
    with account.client() as client, connected(account) as tab:
        closing = ClosingTab(tab, "request:user_input")
        chat_id = f"temporary:{tab.client.get_sid()}"
        send_message(client, prompt, chat_id=chat_id, session_id=tab.client.get_sid())
        closing.close_once_asked()

        result = wait_for_tool_result(upstream, prompt, NO_RESUME_BOUND)

    assert json.loads(result) == {"status": "error", "error": DISCONNECTED}, (
        f"Ask User in a temporary chat did not end when the tab went: {result}"
    )


def test_a_tool_approval_in_a_temporary_chat_ends_when_the_tab_goes(
    tool_approval_on, make_user, upstream
):
    account = make_user()
    prompt = "what time is it? (temporary chat, approval)"
    upstream.queue(
        reply.tool_call("get_current_timestamp", {}, match=reply.answering(prompt)),
        reply.text("I could not check.", match=reply.answering(prompt)),
    )
    with account.client() as client, connected(account) as tab:
        closing = ClosingTab(tab, "confirmation")
        chat_id = f"temporary:{tab.client.get_sid()}"
        send_message(
            client,
            prompt,
            chat_id=chat_id,
            session_id=tab.client.get_sid(),
            params={"tool_approval_mode": "ask"},
        )
        asked = closing.close_once_asked()

        result = wait_for_tool_result(upstream, prompt, NO_RESUME_BOUND)

    assert asked["data"]["data"]["tool_call"]["name"] == "get_current_timestamp", asked
    assert result == "Error: tool call was not approved.", (
        f"the tool approval in a temporary chat did not end when the tab went: {result}"
    )
