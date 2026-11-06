"""Regression: the AGENTS.md of a terminal added in a user's own settings never reached the model.

Fix `7ad0ae468` (issue open-webui/open-webui#30410): a personal terminal is one only the user's
browser may be able to reach, such as Open Terminal on their own machine. The server fetched its
AGENTS.md and skills itself; now it asks the open tab (`request:terminal` on its socket) and
the page reads them from the terminal and answers. This drives that page half: the terminal is a
local fake the page calls across origins, and the test checks the browser read the file and the
model got it. Browser twin of integration/tools/test_personal_terminal_through_browser.py.

Discriminates: passes on the efe63bd34 build; on a build with the page's `request:terminal`
handler removed the test fails (the page never reads the terminal and the model is sent no
AGENTS.md), while an unrelated chat journey passes on that build.

Issue open-webui/open-webui#33811: since dev the reply is only "Working directory is unavailable",
as the server tries to move the personal terminal to the file browser's folder itself. Fails on
dev 144dec942, passes with the fix branch fix/personal-terminal-working-dir on top of it.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.listener import json_answer
from harness.terminal_server import serving_terminal
from utils.chat_ui import chat_input, expect_reply, send
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

NAME = "My laptop"
HOME = "/home/ada"
AGENTS_MD = "Always run the linter before committing."
OPERATION = {"operationId": "run_command", "responses": {"200": {"description": "ok"}}}
SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Open Terminal", "version": "1"},
    "paths": {"/execute": {"post": OPERATION}},
}


@pytest.fixture
def personal_terminal():
    """A terminal that answers the browser, on any origin; yields the fake server."""
    with serving_terminal() as server:
        server.cors = True
        server.route("GET", "/openapi.json", json_answer(SPEC))
        server.route("GET", "/files/cwd", json_answer({"cwd": HOME, "home": HOME}))
        server.route("GET", "/files/read", json_answer({"content": AGENTS_MD}))
        server.route("GET", "/skills", json_answer([]))
        yield server


def test_the_page_reads_the_personal_terminals_agents_md_for_the_model(
    page_for, make_user, upstream, personal_terminal, direct_integrations
):
    direct_integrations(True)
    account = make_user(role="admin")
    page = page_for(account)
    terminal = {
        "url": personal_terminal.base_url,
        "key": "",
        "name": NAME,
        "enabled": False,
        "auth_type": "bearer",
        "path": "/openapi.json",
        "config": {"enable": True},
    }
    with account.client() as client:
        saved = client.post(
            "/api/v1/users/user/settings/update",
            json={"ui": {"showChangelog": False, "terminalServers": [terminal]}},
        )
    assert saved.status_code == 200, saved.text
    page.reload()
    expect(chat_input(page)).to_be_visible()
    tooltip_button(page.get_by_role("main"), "Terminal").click()
    page.get_by_role("menu").get_by_role("menuitemradio", name=NAME).click()

    question = f"lint and commit {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Linted.", match=reply.answering(question)))
    send(page, question)
    expect_reply(page, "Linted.")

    reads = personal_terminal.requests_to("/files/read")
    assert reads and all(read.headers.get("origin") for read in reads), (
        "the personal terminal's AGENTS.md was not read by the page (#30410)"
    )
    sent = [
        entry["content"]
        for entry in next(filter(reply.answering(question), upstream.chat_requests()))["messages"]
    ]
    assert f"# AGENTS.md\n\n{AGENTS_MD}" in sent, (
        "the AGENTS.md of the terminal the page reached never got to the model (#30410)"
    )
