"""Regression: a tool's question to the user timed out when their tab was on another instance.

PR open-webui/open-webui#31620 (288bf91f7): with `WEBSOCKET_MANAGER=redis` an instance started
listening on Redis only when the first socket connected to it. A tool asking the user something
(`__event_call__`, an input or a confirmation) sends the question through Redis to the instance
holding the user's tab, and the tab's answer comes back the same way; an instance no tab had
connected to yet never heard it, so the tool waited until the event call timed out. Every instance
now subscribes at startup.

Two instances share one database and one real Redis. The tab connects to one of them only; the
chat is sent to the other, which no socket ever reaches in this module. The event call timeout is
cut to a few seconds so the unfixed ref fails fast. Nearby, the same question answered by a tab
on the instance the chat runs on.

Twin of e2e/chat/test_event_call_answer_across_instances.py.

Discriminates: passes on dev a5bc78300, fails with 288bf91f7 reverted (the chat instance never
subscribes, so the tool gets the event call timeout in place of the answer).
"""

from __future__ import annotations

import dataclasses
import json
from typing import Iterator

import pytest

from harness import backends
from harness import upstream as reply
from harness.actors import Actor, admin_of, create_user
from harness.chat import ask
from harness.instance import LaunchedInstance
from harness.python_tools import python_tool
from harness.socket_client import SocketSession, connected
from harness.upstream import conversation

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

EVENT_CALL_TIMEOUT_SECONDS = "8"

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

TAB_ANSWERS = {"input": {"value": "Ada"}, "confirmation": True}


@pytest.fixture(scope="module")
def shared_redis() -> Iterator[backends.RedisProcess]:
    server = backends.RedisProcess()
    server.start()
    yield server
    server.close()


@pytest.fixture(scope="module")
def fleet(shared_redis, instance_with) -> dict[str, LaunchedInstance]:
    """The instance the tab connects to and the one the chat runs on, on one database and Redis."""
    shared = {
        "REDIS_URL": shared_redis.url,
        "WEBSOCKET_MANAGER": "redis",
        "WEBSOCKET_EVENT_CALLER_TIMEOUT": EVENT_CALL_TIMEOUT_SECONDS,
    }
    tab_instance = instance_with({**shared, "WEBUI_NAME": "event-call-tab"})
    chat_instance = instance_with(
        {**shared, "WEBUI_NAME": "event-call-chat", "DATABASE_URL": tab_instance.database_url}
    )
    return {"tab": tab_instance, "chat": chat_instance}


@pytest.fixture
def provider(fleet):
    """The provider both instances ask: the connection settings live in the shared database."""
    fleet["tab"].upstream.reset()
    return fleet["tab"].upstream


def signed_in_on(account: Actor, instance: LaunchedInstance) -> Actor:
    """The same account on another instance: they share the secret key and the database."""
    return dataclasses.replace(account, base_url=instance.base_url)


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


def tool_results(provider) -> list[str]:
    return [
        entry["content"]
        for sent in provider.chat_requests()
        for entry in conversation(sent)
        if entry["role"] == "tool"
    ]


def ask_through_tool(fleet, provider, kind: str, chat_on: str) -> tuple[list[dict], dict]:
    """Chat on `chat_on` with a tab on the tab instance; the tool asks that tab a `kind`."""
    account = create_user(fleet["tab"])
    prompt = f"who am I? ({kind}, chat on the {chat_on} instance)"
    provider.queue(
        reply.tool_call("ask_the_tab", {"kind": kind}, match=reply.answering(prompt)),
        reply.text("Nice to meet you.", match=reply.answering(prompt)),
    )
    with (
        python_tool(admin_of(fleet["tab"]), ASK_THE_TAB, name="Ask the tab") as tool_id,
        connected(account) as tab,
        signed_in_on(account, fleet[chat_on]).client() as client,
    ):
        asked = answer_questions(tab, kind)
        options = {"tool_ids": [tool_id], "session_id": tab.client.get_sid()}
        _, message = ask(client, prompt, **options)
    return asked, message


@pytest.mark.parametrize("kind", ["input", "confirmation"])
def test_the_tabs_answer_reaches_a_tool_on_an_instance_no_tab_connected_to(fleet, provider, kind):
    asked, message = ask_through_tool(fleet, provider, kind, chat_on="chat")

    assert len(asked) == 1, f"the tab was asked {len(asked)} times"
    assert tool_results(provider) == [json.dumps(TAB_ANSWERS[kind])], (
        "the tab's answer never reached the tool on the instance no socket had connected to "
        "(#31620)"
    )
    assert message["content"].endswith("Nice to meet you."), message


def test_the_tabs_answer_reaches_a_tool_on_the_tabs_own_instance(fleet, provider):
    asked, message = ask_through_tool(fleet, provider, "input", chat_on="tab")

    assert len(asked) == 1, f"the tab was asked {len(asked)} times"
    assert tool_results(provider) == [json.dumps(TAB_ANSWERS["input"])]
    assert message["content"].endswith("Nice to meet you."), message
