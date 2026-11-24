"""A terminal switched off while a chat has it selected hands that chat no tools.

Fix `753798923` (0.11.0) refused a disabled terminal connection on every entry point that
resolves one by id, where before only the list had hidden it. A user chats with a terminal picked
from the chat input's Terminal menu; the admin then switches it off through a config import, which
saves the connection without rebuilding the terminal list cached in Redis. The user's next message
in that chat still names the terminal, and the reply shows that the terminal is unavailable
instead of the model being offered its tools. Since the terminal rework on dev (6f2fc7bc7) a
switched-off terminal answers as one that does not exist, so the reply reads "Machine
unavailable: Machine '<id>' not found" (worded for terminals before df6be8c3a) where it said "is
disabled" before.

Since 21da7627b the page opens a new chat's working directory on the terminal before the first
message is sent, and a terminal that refuses ends the send with a toast before the model is
asked. So only a chat that already has a reply reaches the server's check: the first message is
sent while the terminal is on.

Discriminates: passes on dev ef67cc3fa with its built frontend; with the `enabled` check removed
from `get_terminal_tools` in a backend copy the model is offered the terminal's tools and the
reply shows no error. Retargeted for 21da7627b: passes on dev a072aaf31 with its build, and with
the same check removed there the message after the switch-off is answered with no refusal.
Retargeted for 6f2fc7bc7: passes on dev c15f28649 with its build, and in a backend copy whose
`find_terminal_connection` and `get_terminal_tools` drop their `enabled` checks the second
message is answered "ready" with no refusal.

Issue open-webui/open-webui#34643: while a Terminals orchestrator could not list its terminals
for a moment, the page left them out of the terminal menu and dropped the person's pick for good,
on the new chat page and in a saved chat, also across a reload. A pick now stays until the person
changes it: its button reads "Unavailable terminal" during the outage, the Terminal menu offers
it under that name to clear it, and once the orchestrator is back the next message is offered the
terminal's tools with no further reload. Retargeted for the terminals to machines rename
(c5c42fe3b, df6be8c3a): the button reads "Unavailable machine", the empty menu "Machine", and the
config import that switches the terminal off names `machines.connections`. Discriminates: fails on
dev a43c15c7b (after the reload the pick is gone and the button is not shown), passes on dev
c6ef4a523 (fixed by 0f71ee15a).
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
from harness.terminals import share_template
from integration.stateful_redis import StatefulRedis
from utils.cached_chat import choose_terminal, pick_terminal
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
                json={"config": {"machines.connections": [switched_off]}},
            )
        assert imported.status_code == 200, imported.text
        terminal.clear()

        cached_instance.upstream.queue(reply.text("ready", match=reply.answering(SECOND_PROMPT)))
        send(page, SECOND_PROMPT)
        refusal = re.compile(r"Machine unavailable: Machine '.+' not found")
        expect(conversation(page).get_by_text(refusal)).to_be_visible()

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


# --- #34643: a pick on an orchestrator survives a brief outage and a reload ----------------

TEMPLATES = "/api/v1/terminals/templates"
TEMPLATE = {"id": "tpl-1", "data": {"name": "Workbench", "scope": "user"}}
OUTAGE_PROMPTS = ("what is in the workbench?", "and in the build folder?")


def _set_templates_up(orchestrator, up: bool) -> None:
    if up:
        orchestrator.route("GET", TEMPLATES, json_answer([TEMPLATE]))
    else:
        orchestrator.route("GET", TEMPLATES, json_answer({"detail": "Unavailable"}, status=503))


def test_a_picked_terminal_stays_through_an_orchestrator_outage_and_a_reload(
    page_for, admin, make_user, preserve, upstream
):
    preserve(TERMINAL_SERVERS_CONFIG)
    member = make_user()
    with serving_terminal() as orchestrator:
        _set_templates_up(orchestrator, True)
        orchestrator.route("GET", f"{TEMPLATES}/{TEMPLATE['id']}", json_answer(TEMPLATE))
        orchestrator.route("GET", "/openapi.json", json_answer(RUN_COMMAND_SPEC))
        orchestrator.route("POST", "/files/cwd", json_answer({"cwd": "/"}))
        connection = orchestrator.connection(
            name="Orchestrator",
            key="orchestrator-key",
            auth_type="bearer",
            server_type="orchestrator",
            config={"access_grants": [read_grant(member.id)]},
        )
        terminal_id = f"template:{connection['id']}:{TEMPLATE['id']}"
        with admin.client() as client:
            configure_terminals(client, connection)
            share_template(client, terminal_id, read_grant(member.id))

        page = page_for(member)
        choose_terminal(page, "Workbench")
        unavailable = page.get_by_role("main").get_by_role(
            "button", name="Unavailable machine", exact=True
        )
        # first on the new chat page, then in the saved chat its message creates
        for prompt in OUTAGE_PROMPTS:
            _set_templates_up(orchestrator, False)
            page.reload()
            expect(unavailable).to_be_visible()
            _set_templates_up(orchestrator, True)
            upstream.queue(reply.text("ready", match=reply.answering(prompt)))
            send(page, prompt)
            expect(conversation(page).get_by_text("ready", exact=True).last).to_be_visible()
            expect(page).to_have_url(re.compile("/c/"))

        # a pick that is missing from the menu is still cleared from it, and stays cleared
        _set_templates_up(orchestrator, False)
        page.reload()
        unavailable.click()
        page.get_by_role("menu").get_by_role("menuitemradio", name="Unavailable machine").click()
        expect(unavailable).to_have_count(0)
        _set_templates_up(orchestrator, True)
        page.reload()
        unpicked = page.get_by_role("main").get_by_role("button", name="Machine", exact=True)
        expect(unpicked).to_be_visible()

    for prompt in OUTAGE_PROMPTS:
        offered = [
            tool["function"]["name"]
            for request in upstream.chat_requests()
            if reply.answering(prompt)(request)
            for tool in request.get("tools", [])
        ]
        assert "run_command" in offered, (
            f"after the orchestrator came back, {prompt!r} went out without the picked terminal "
            f"(#34643); the model was offered {offered}"
        )
