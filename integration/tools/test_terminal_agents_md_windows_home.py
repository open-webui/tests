"""Regression: the AGENTS.md in the home folder of a terminal running on Windows was skipped.

Fix `3f5881c52` (open-webui/open-webui#31342, issue open-webui/open-webui#31340): before each turn
on a terminal, Open WebUI asks it for its home folder and hands the AGENTS.md found there to the
model. The home was only accepted as a POSIX absolute path, so a drive-letter or UNC home such
as `C:\\ProgramData\\OpenTerminal\\inst` was treated as invalid and the file was never read.
Windows absolute paths are accepted now; relative and drive-relative homes are still skipped.

The file is handed over only when it has text and is at most 32 KiB: an empty or blank file, a
larger one, one the terminal does not have or refuses (404, 401, 403, 500) and one the terminal
takes more than five seconds to answer are left out, and the reply arrives without it.

Discriminates: passes on dev efe63bd34; with 3f5881c52 reverted in a backend copy every Windows
home case fails (the terminal is never asked for the file). The POSIX and skipped-home cases
pass on both. On dev c15f28649, in a backend copy: with the size check dropped the over-32-KiB
case fails, with the limit lowered by one byte the exactly-32-KiB test, with the blank check
dropped the empty and blank cases, with any answer read as the file whatever its status the 500
case, and with the five second limit removed the slow test (the reply waits for the terminal).
"""

from __future__ import annotations

import threading
import uuid

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.listener import json_answer
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, serving_terminal

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

AGENTS_MD = "Answer in British English."
OPERATION = {"operationId": "run_command", "responses": {"200": {"description": "ok"}}}
SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "terminal", "version": "1"},
    "paths": {"/execute": {"post": OPERATION}},
}


@pytest.fixture
def terminal(admin, preserve):
    """A terminal the admin connected; yields the fake server and the connection's id."""
    preserve(TERMINAL_SERVERS_CONFIG)
    with serving_terminal() as server:
        server.route("GET", "/openapi.json", json_answer(SPEC))
        server.route("GET", "/files/read", json_answer({"content": AGENTS_MD}))
        connection = server.connection()
        with admin.client() as client:
            configure_terminals(client, connection)
        yield server, connection["id"]


def _model_is_sent_agents_md(terminal, make_user, upstream, home: str) -> bool:
    server, terminal_id = terminal
    server.route("GET", "/files/cwd", json_answer({"cwd": home, "home": home}))
    upstream.queue(reply.text("ready"))
    with make_user(role="admin").client() as client:
        _, message = ask(client, "tidy up the docs", terminal_id=terminal_id)
    assert message["content"] == "ready", message
    sent = [entry["content"] for entry in upstream.chat_requests()[-1]["messages"]]
    return f"# AGENTS.md\n\n{AGENTS_MD}" in sent


@pytest.mark.parametrize(
    "home",
    [
        "C:\\ProgramData\\OpenTerminal\\inst",
        "C:/Users/ada",
        "D:\\",
        "\\\\fileserver\\homes\\ada",
    ],
    ids=["drive-letter", "forward-slashes", "drive-root", "unc"],
)
def test_a_windows_home_hands_its_agents_md_to_the_model(terminal, make_user, upstream, home):
    assert _model_is_sent_agents_md(terminal, make_user, upstream, home), (
        f"the AGENTS.md in the Windows home {home!r} never reached the model (#31340)"
    )


def test_a_posix_home_still_hands_its_agents_md_to_the_model(terminal, make_user, upstream):
    assert _model_is_sent_agents_md(terminal, make_user, upstream, "/home/ada")


@pytest.mark.parametrize(
    "home",
    ["Users\\ada", "C:Users\\ada", "home/ada"],
    ids=["relative", "drive-relative", "posix-relative"],
)
def test_a_home_that_is_not_absolute_is_skipped(terminal, make_user, upstream, home):
    server, _ = terminal

    assert not _model_is_sent_agents_md(terminal, make_user, upstream, home)
    assert server.requests_to("/files/read") == [], "a file was read from a relative home"


# The largest AGENTS.md handed to the model, in bytes
AGENTS_MD_LIMIT = 32 * 1024


def _turn_on_a_terminal_whose_file_read_answers(terminal, make_user, upstream, answer) -> list:
    """One turn with the home's AGENTS.md read answered by `answer`; the texts the model got."""
    server, terminal_id = terminal
    server.route("GET", "/files/cwd", json_answer({"cwd": "/home/ada", "home": "/home/ada"}))
    server.route("GET", "/files/read", answer)
    server.clear()
    prompt = f"tidy up the docs {uuid.uuid4().hex[:8]}"
    upstream.queue(reply.text("ready", match=reply.answering(prompt)))
    with make_user(role="admin").client() as client:
        _, message = ask(client, prompt, terminal_id=terminal_id)
    assert message["content"] == "ready", f"the reply did not arrive: {message}"
    [request] = [entry for entry in upstream.chat_requests() if reply.answering(prompt)(entry)]
    assert len(server.requests_to("/files/read")) == 1, "the terminal was not asked for the file"
    return [str(entry["content"]) for entry in request["messages"]]


@pytest.mark.parametrize(
    "answer",
    [
        json_answer({"content": ""}),
        json_answer({"content": " \n\t\n"}),
        json_answer({"content": "a" * (AGENTS_MD_LIMIT + 1)}),
        json_answer({"content": {"lines": ["Answer in British English."]}}),
        json_answer({"detail": "File not found"}, status=404),
        json_answer({"detail": "Not authenticated"}, status=401),
        json_answer({"detail": "Forbidden"}, status=403),
        json_answer({"content": AGENTS_MD}, status=500),
    ],
    ids=["empty", "blank", "over-32-kib", "not-text", "missing", "401", "403", "500"],
)
def test_an_agents_md_that_is_empty_too_large_or_not_readable_is_left_out(
    terminal, make_user, upstream, answer
):
    texts = _turn_on_a_terminal_whose_file_read_answers(terminal, make_user, upstream, answer)

    handed = [text[:80] for text in texts if text.startswith("# AGENTS.md")]
    assert not handed, f"the model was handed an AGENTS.md all the same: {handed}"


def test_an_agents_md_of_exactly_32_kib_is_handed_to_the_model(terminal, make_user, upstream):
    content = "b" * AGENTS_MD_LIMIT

    texts = _turn_on_a_terminal_whose_file_read_answers(
        terminal, make_user, upstream, json_answer({"content": content})
    )

    assert f"# AGENTS.md\n\n{content}" in texts, "an AGENTS.md at the size limit was left out"


@pytest.mark.slow
def test_a_terminal_too_slow_with_its_agents_md_does_not_hold_the_reply(
    terminal, make_user, upstream
):
    answered = threading.Event()

    def after_the_reply(_request):
        answered.wait(30)
        return json_answer({"content": AGENTS_MD})

    try:
        texts = _turn_on_a_terminal_whose_file_read_answers(
            terminal, make_user, upstream, after_the_reply
        )
    finally:
        answered.set()

    assert not [text for text in texts if text.startswith("# AGENTS.md")]
