"""Journey: the steps of a coding agent or of the terminal tools read as what they did.

New in 0.13.4 ("Readable agent and terminal steps"). A reply's steps show in the chat as what
they did: a command with its output and its exit code when it failed, a diff of each edited file
with its added and removed lines, and the files read. The group of steps is summed up as "Ran 2
commands, Edited 1 file, Read 3 files", and all of it reads the same once the chat is opened
again. It holds for a coding agent over ACP (OpenCode, `harness/fake_acp_agent.py`), for Codex
(`harness/fake_native_agent.py`), both run by a real Open Terminal, and for the terminal tools of
a provider model on a real Open Terminal, whose edits need Open Terminal 0.17.20 or later for
their diff. Each runs the same work: two commands of which the second fails, three files read and
one line of a file changed. ACP reports no exit code, so only Codex and the terminal tools are
held to show it.

Discriminates: all pass on the dev f80a4f782 build with Open Terminal 0.18.1. On a frontend build
of it whose steps never show the exit code and whose edits count no removed lines, the Codex and
terminal tool tests fail at "Exit code 2" and the ACP test at the edit reading "+1" for "+1 −1".
"""

from __future__ import annotations

import re
import shutil
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.coding_agent import FAST, connect_agent_terminal
from harness.fake_acp_agent import (
    COMMANDS,
    EDITED,
    EDITED_AFTER,
    EDITED_BEFORE,
    READS,
    WORK_DONE,
)
from harness.terminal_server import TERMINAL_SERVERS_CONFIG
from utils.cached_chat import calling, calling_together, choose_terminal
from utils.chat_ui import REPLY_TIMEOUT_MS, expect_reply, last_reply, send, steps_summary
from utils.file_views import connect_for
from utils.model_selector import select_model
from utils.reply_steps import diff_line, diff_sign, open_step, open_steps, step

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

SUMMARY = "Ran 2 commands, Edited 1 file, Read 3 files"
[(WORKED, WORKED_OUTPUT, _), (FAILED, FAILED_OUTPUT, FAILED_EXIT)] = COMMANDS


def _wait_until_saved(page: Page) -> None:
    page.wait_for_url(re.compile(r"/c/[^/?#]+"), timeout=REPLY_TIMEOUT_MS)
    expect(page.get_by_role("button", name="Stop")).to_have_count(0, timeout=REPLY_TIMEOUT_MS)


def _expect_readable_steps(answer: Locator, *, exit_code_shown: bool) -> None:
    """The reply's steps: summed up, then each command, read and edit as what it did."""
    expect(steps_summary(answer)).to_have_text(SUMMARY)
    open_steps(answer)
    for name in READS:
        expect(step(answer, f"Read {name}")).to_be_visible()

    worked = open_step(answer, f"Ran {WORKED}")
    expect(worked).to_contain_text(f"$ {WORKED}")
    expect(worked).to_contain_text(WORKED_OUTPUT.strip())
    expect(worked.get_by_text("Exit code")).to_have_count(0)

    failed = open_step(answer, f"Ran {FAILED}")
    expect(failed).to_contain_text(f"$ {FAILED}")
    expect(failed).to_contain_text(FAILED_OUTPUT.strip())
    if exit_code_shown:
        expect(failed.get_by_text(f"Exit code {FAILED_EXIT}", exact=True)).to_be_visible()

    edit_line = step(answer, f"Edited {EDITED}")
    expect(edit_line).to_have_accessible_name(f"Edited {EDITED} +1 −1")
    edit = open_step(answer, f"Edited {EDITED}")
    expect(edit).to_contain_text(EDITED)
    expect(diff_sign(diff_line(edit, "ferry at nine"))).to_have_text("-")
    expect(diff_sign(diff_line(edit, "ferry at ten"))).to_have_text("+")
    expect(diff_sign(diff_line(edit, "harbour log"))).to_have_text(re.compile(r"^\s*$"))


def _expect_the_work_and_again_after_a_reload(page: Page, *, exit_code_shown: bool) -> None:
    expect_reply(page, WORK_DONE)
    _wait_until_saved(page)
    _expect_readable_steps(last_reply(page), exit_code_shown=exit_code_shown)
    page.reload()
    expect(last_reply(page)).to_contain_text(WORK_DONE, timeout=REPLY_TIMEOUT_MS)
    _expect_readable_steps(last_reply(page), exit_code_shown=exit_code_shown)


def _agent_chat(page_for, make_user, admin, terminal, model: str) -> Page:
    person = make_user()
    name = f"Agent box {uuid.uuid4().hex[:6]}"
    connect_agent_terminal(admin, terminal, person, name=name)
    page = page_for(person)
    choose_terminal(page, name)
    select_model(page, model)
    return page


def test_an_acp_agents_steps_read_as_what_it_did(
    page_for, make_user, admin, preserve, agent_terminal
):
    preserve(TERMINAL_SERVERS_CONFIG)
    page = _agent_chat(page_for, make_user, admin, agent_terminal, f"opencode/{FAST}")

    send(page, f"[work] tidy the harbour {uuid.uuid4().hex[:6]}")

    _expect_the_work_and_again_after_a_reload(page, exit_code_shown=False)


def test_codexs_steps_read_as_what_it_did_with_a_failed_commands_exit_code(
    page_for, make_user, admin, preserve, native_agent_terminal
):
    preserve(TERMINAL_SERVERS_CONFIG)
    page = _agent_chat(page_for, make_user, admin, native_agent_terminal, "codex/codex-fake")

    send(page, f"[work] tidy the harbour {uuid.uuid4().hex[:6]}")

    _expect_the_work_and_again_after_a_reload(page, exit_code_shown=True)


@pytest.fixture
def harbour(open_terminal):
    """A folder of the test's own in the terminal's home with the files the work uses."""
    folder = open_terminal.home / f"harbour-{uuid.uuid4().hex[:6]}"
    folder.mkdir()
    for name, content in READS.items():
        (folder / name).write_text(content)
    (folder / EDITED).write_text(EDITED_BEFORE)
    yield folder
    shutil.rmtree(folder, ignore_errors=True)


def test_a_models_terminal_steps_read_as_what_they_did(
    page_for, make_user, admin, preserve, open_terminal, harbour, upstream
):
    preserve(TERMINAL_SERVERS_CONFIG)
    person = make_user()
    name = connect_for(admin, person, open_terminal)
    page = page_for(person)
    choose_terminal(page, name)
    prompt = f"tidy the harbour {uuid.uuid4().hex[:6]}"
    edit = {"target": "ferry at nine", "replacement": "ferry at ten"}
    work = calling_together(
        *[("run_command", {"command": command, "wait": 10}) for command, _, _ in COMMANDS],
        *[("read_file", {"path": str(harbour / name)}) for name in READS],
    )
    rewrite = calling(
        "replace_file_content", {"path": str(harbour / EDITED), "replacements": [edit]}
    )
    for scripted in (work, rewrite, reply.text(WORK_DONE)):
        scripted.match = reply.answering(prompt)
        upstream.queue(scripted)

    send(page, prompt)

    _expect_the_work_and_again_after_a_reload(page, exit_code_shown=True)
    assert (harbour / EDITED).read_text() == EDITED_AFTER, (
        f"the model's edit is not on disk: {(harbour / EDITED).read_text()!r}"
    )
