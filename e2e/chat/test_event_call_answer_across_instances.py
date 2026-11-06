"""Regression: a tool's question shown in the page never got its typed answer back to the tool.

PR open-webui/open-webui#31620 (288bf91f7): with `WEBSOCKET_MANAGER=redis` an instance started
listening on Redis only when the first socket connected to it. Behind a load balancer a page's
socket and its requests can reach different instances, so the instance running the chat may have
no socket of its own; a tool asking the user something there showed its dialog in the page, but
the typed answer went back through Redis to an instance that was not listening, and the tool
waited until the event call timed out. Every instance now subscribes at startup.

Two instances share one database and one real Redis, and one address splits them: the page's
socket goes to the first and every other request to the second, which no socket ever reaches.
The model only greets the user by the name the tool got back, so the greeting shows exactly when
the answer arrived. The event call timeout is cut to a few seconds so the unfixed ref fails fast.

Twin of integration/chat/test_event_call_answer_across_instances.py.

Discriminates: passes on dev a5bc78300, fails with 288bf91f7 reverted (the answer never reaches
the tool, which gets the event call timeout, so the greeting never shows).
"""

from __future__ import annotations

import dataclasses
import json
from typing import Iterator

import pytest

from harness import backends
from harness import upstream as reply
from harness.actors import Actor, admin_of, create_user
from harness.instance import LaunchedInstance
from harness.python_tools import python_tool
from harness.split_proxy import splitting
from utils.cached_chat import turn_on_tool
from utils.chat_ui import expect_reply, send

pytestmark = [
    pytest.mark.regression,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

NAME = "Ada"
GREETING = f"Nice to meet you, {NAME}."

ASK_THE_PAGE = '''
import json


class Tools:
    async def ask_the_page(self, __event_call__=None) -> str:
        """
        Ask the user for their name.
        """
        answer = await __event_call__(
            {"type": "input", "data": {"title": "Your name?", "message": "For the greeting."}}
        )
        return json.dumps(answer)
'''


@pytest.fixture(scope="module")
def shared_redis() -> Iterator[backends.RedisProcess]:
    server = backends.RedisProcess()
    server.start()
    yield server
    server.close()


@pytest.fixture(scope="module")
def fleet(shared_redis, instance_with) -> dict[str, LaunchedInstance]:
    """The instance the page's socket reaches and the one its requests reach."""
    shared = {
        "REDIS_URL": shared_redis.url,
        "WEBSOCKET_MANAGER": "redis",
        "WEBSOCKET_EVENT_CALLER_TIMEOUT": "8",
    }
    socket_instance = instance_with({**shared, "WEBUI_NAME": "event-call-socket"})
    request_instance = instance_with(
        {
            **shared,
            "WEBUI_NAME": "event-call-requests",
            "DATABASE_URL": socket_instance.database_url,
        }
    )
    return {"socket": socket_instance, "requests": request_instance}


@pytest.fixture(scope="module")
def split_address(fleet) -> Iterator[str]:
    with splitting(fleet["socket"], fleet["requests"]) as base_url:
        yield base_url


def at(account: Actor, base_url: str) -> Actor:
    """The same account reached at another address: they share the secret key and the database."""
    return dataclasses.replace(account, base_url=base_url)


def answered_with(name: str):
    """A `match` for the follow-up request whose tool result carries `name`."""

    def matches(body: dict) -> bool:
        results = [entry for entry in reply.conversation(body) if entry["role"] == "tool"]
        return any(name in str(entry["content"]) for entry in results)

    return matches


def test_the_typed_answer_reaches_a_tool_on_the_instance_without_the_pages_socket(
    fleet, split_address, page_for
):
    provider = fleet["socket"].upstream
    provider.reset()
    prompt = "who am I?"
    provider.queue(
        reply.tool_call("ask_the_page", {}, match=reply.answering(prompt)),
        reply.text(GREETING, match=answered_with(NAME)),
    )
    with python_tool(admin_of(fleet["socket"]), ASK_THE_PAGE, name="Ask the page"):
        page = page_for(at(create_user(fleet["socket"]), split_address))
        turn_on_tool(page, "Ask the page")

        send(page, prompt)
        dialog = page.get_by_role("dialog", name="Your name?")
        dialog.get_by_label("Enter your message").fill(NAME)
        dialog.get_by_role("button", name="Confirm").click()

        expect_reply(page, GREETING)
    tool_results = [
        entry["content"]
        for sent in provider.chat_requests()
        for entry in reply.conversation(sent)
        if entry["role"] == "tool"
    ]
    assert tool_results == [json.dumps(NAME)]
