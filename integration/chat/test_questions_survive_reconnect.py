"""Journey: a question waiting for the user is asked again once they reopen the chat.

Since 0.12.4 (c596d40bd) a tool's input or confirmation dialog, or an MCP server's input request,
that is still open when the page reloads or loses its connection waits up to a minute for a new
connection of the same account to open that chat, asks it there, and hands its answer back.
With no reconnect in that minute the tool gets "Client session disconnected." Since 321701c1a a
temporary chat belongs to the tab's connection, and a page whose connection comes back resumes it
on the new one, so its questions (Ask User, a tool approval) are asked again there too, while
another account cannot take the chat over. Ask User and tool approvals in a saved chat are stored
with the chat and wait there instead, so no socket is involved in getting them back (see the
browser twin).

The minute is fixed in the server, so the test of a missed reconnect waits it out.

Twin of e2e/chat/test_questions_survive_reconnect.py.

Discriminates: passes on dev 948432b43. With the resume turned off in a backend copy (no call
may wait for a reconnect) the reconnect tests fail (the tool gets "Client session disconnected."
at once and the new connection is never asked); with the minute raised to an hour the missed
reconnect test fails (the reply never finishes). The temporary chat tests pass on dev d22c4449c
and fail on 321701c1a^, where a temporary chat's question ends as soon as the tab goes; with the
owner check on resuming dropped in a backend copy the other account takes the chat over.
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

HARBOUR = {"harbour": "Split"}
TAB_ANSWERS = {
    "input": "Ada",
    "confirmation": True,
    "request:user_input": {"status": "answered", "answers": HARBOUR},
}


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


def resume_temporary_chat(client, tab: SocketSession, chat_id: str):
    """What the page sends for its temporary chat once its connection is back."""
    return client.post(f"/api/tasks/chat/{chat_id}/resume", json={"session_id": tab.sid})


def tool_results(upstream, prompt: str) -> list[str]:
    return [
        str(entry["content"])
        for sent in upstream.chat_requests()
        if reply.answering(prompt)(sent)
        for entry in reply.conversation(sent)
        if entry["role"] == "tool"
    ]


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


def test_ask_user_in_a_temporary_chat_is_asked_again_on_the_connection_that_resumes_it(
    make_user, upstream
):
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
        reply.text("Split it is.", match=reply.answering(prompt)),
    )
    with account.client() as client, connected(account) as first_tab:
        closing = ClosingTab(first_tab, "request:user_input")
        chat_id = first_tab.temporary_chat_id()
        send_message(client, prompt, chat_id=chat_id, session_id=first_tab.sid)
        first_question = closing.close_once_asked()

        with connected(account) as second_tab:
            asked_again = answer_questions(second_tab, "request:user_input")
            resumed = resume_temporary_chat(client, second_tab, chat_id)
            assert resumed.status_code == 200, resumed.text
            done = second_tab.wait_for(chat_id, "chat:completion", done=True)

    assert [question["data"] for question in asked_again] == [first_question["data"]], (
        f"the resumed temporary chat was not asked the open question once: {asked_again}"
    )
    assert [json.loads(result) for result in tool_results(upstream, prompt)] == [
        {"status": "answered", "answers": HARBOUR}
    ], "the answer from the resumed temporary chat never reached Ask User"
    assert "Split it is." in str(done["data"]["output"]), done


def test_a_tool_approval_in_a_temporary_chat_is_asked_again_on_its_owners_connection_only(
    tool_approval_on, make_user, upstream
):
    account, stranger = make_user(), make_user()
    prompt = "what time is it? (temporary chat, approval)"
    upstream.queue(
        reply.tool_call("get_current_timestamp", {}, match=reply.answering(prompt)),
        reply.text("Checked.", match=reply.answering(prompt)),
    )
    with account.client() as client, connected(account) as first_tab:
        closing = ClosingTab(first_tab, "confirmation")
        chat_id = first_tab.temporary_chat_id()
        send_message(
            client,
            prompt,
            chat_id=chat_id,
            session_id=first_tab.sid,
            params={"tool_approval_mode": "ask"},
        )
        first_question = closing.close_once_asked()

        with (
            stranger.client() as stranger_client,
            connected(stranger) as stranger_tab,
            connected(account) as second_tab,
        ):
            asked_the_stranger = answer_questions(stranger_tab, "confirmation")
            taken_over = resume_temporary_chat(stranger_client, stranger_tab, chat_id)
            asked_again = answer_questions(second_tab, "confirmation")
            resumed = resume_temporary_chat(client, second_tab, chat_id)
            assert resumed.status_code == 200, resumed.text
            second_tab.wait_for(chat_id, "chat:completion", done=True)

    assert first_question["data"]["data"]["tool_call"]["name"] == "get_current_timestamp"
    assert taken_over.status_code == 404, (
        f"another account resumed someone's temporary chat: HTTP {taken_over.status_code}"
    )
    assert asked_the_stranger == [], "another account was asked to approve the tool call"
    assert [question["data"] for question in asked_again] == [first_question["data"]], (
        f"the resumed temporary chat was not asked the open approval once: {asked_again}"
    )
    [result] = tool_results(upstream, prompt)
    assert "current_timestamp" in result, (
        f"the approval from the resumed temporary chat never let the tool run: {result}"
    )
