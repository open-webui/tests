"""A terminal switched off while a chat has it selected hands that chat no tools.

Fix `753798923` (0.11.0) refused a disabled terminal connection on every entry point that
resolves one by id, where before only the list had hidden it. A user chats with a terminal picked
from the chat input's Terminal menu; the admin then switches it off through a config import, which
saves the connection without rebuilding the terminal list cached in Redis. The user's next message
in that chat still names the terminal, and the reply shows that it is disabled instead of the
model being offered its tools.

Since 21da7627b the page opens a new chat's working directory on the terminal before the first
message is sent, and a terminal that refuses ends the send with a toast before the model is
asked. So only a chat that already has a reply reaches the server's check: the first message is
sent while the terminal is on.

Discriminates: passes on dev ef67cc3fa with its built frontend; with the `enabled` check removed
from `get_terminal_tools` in a backend copy the model is offered the terminal's tools and the
reply shows no error. Retargeted for 21da7627b: passes on dev a072aaf31 with its build, and with
the same check removed there the message after the switch-off is answered with no refusal.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.actors import admin_of, create_user
from harness.listener import json_answer
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    configure_terminals,
    read_grant,
    serving_terminal,
)
from integration.stateful_redis import StatefulRedis
from utils.cached_chat import pick_terminal
from utils.chat_ui import conversation, send, split_screen

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

NAME = "Build box"
PROMPT = "list the build artefacts"
SECOND_PROMPT = "and the test reports?"
RUN_COMMAND_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Terminal", "version": "1"},
    "paths": {
        "/execute": {
            "post": {"operationId": "run_command", "responses": {"200": {"description": "ok"}}}
        }
    },
}


@pytest.fixture(scope="module")
def redis():
    store = StatefulRedis()
    yield store
    store.close()


@pytest.fixture
def cached_instance(instance_with, redis, preserve):
    launched = instance_with({"REDIS_URL": redis.url})
    if not launched.serves_frontend:
        pytest.skip("the checkout has no built frontend")
    preserve(TERMINAL_SERVERS_CONFIG, on=launched)
    return launched


@pytest.mark.slow
def test_a_terminal_switched_off_mid_chat_offers_no_tools(page_for, cached_instance):
    member = create_user(cached_instance)
    with serving_terminal() as terminal:
        terminal.route("GET", "/openapi.json", json_answer(RUN_COMMAND_SPEC))
        terminal.route("POST", "/files/cwd", json_answer({"cwd": "/"}))
        connection = terminal.connection(
            name=NAME, config={"access_grants": [read_grant(member.id)]}
        )
        with admin_of(cached_instance).client() as client:
            configure_terminals(client, connection)

        page = page_for(member)
        pick_terminal(page, NAME)
        split_screen(page)

        cached_instance.upstream.queue(reply.text("ready", match=reply.answering(PROMPT)))
        send(page, PROMPT)
        expect(conversation(page).get_by_text("ready", exact=True)).to_be_visible()
        expect(page).to_have_url(re.compile("/c/"))

        with cached_instance.client() as client:
            switched_off = {**connection, "enabled": False}
            imported = client.post(
                "/api/v1/configs/import",
                json={"config": {"terminal_server.connections": [switched_off]}},
            )
        assert imported.status_code == 200, imported.text
        terminal.clear()

        cached_instance.upstream.queue(reply.text("ready", match=reply.answering(SECOND_PROMPT)))
        send(page, SECOND_PROMPT)
        expect(conversation(page).get_by_text("is disabled")).to_be_visible()

    def offered(prompt: str) -> list[str]:
        return [
            tool["function"]["name"]
            for request in cached_instance.upstream.chat_requests()
            if reply.answering(prompt)(request)
            for tool in request.get("tools", [])
        ]

    assert "run_command" in offered(PROMPT), "the terminal's tools never reached the chat"
    assert "run_command" not in offered(SECOND_PROMPT), (
        "a switched-off terminal still hands its tools to a chat"
    )
    assert terminal.requests_to("/execute") == []
