"""Regression: kb_exec grep reported every hit as line 1, in the tool result the chat shows.

open-webui 0.11.0, fix `e18e249d5` (PR #27249, issue #26744): the single-file and piped grep
branches of `tools/knowledge_fs.py` split their text on the two characters backslash-n instead
of a newline, so a hit on the fourth line of a note came back as line 1 carrying the whole
file. `504e724fd` (PR #26795, issue #26781) made "alpha|omega" an alternation instead of a
literal that matched nothing. A user opening the tool call in the reply sees what the model got.

Twin of unit/tools/test_knowledge_file_search.py.

Discriminates: passes on dev bbfa876af; fails with the literal backslash-n split restored in the
grep branches (both tests: the result is `1: ` and the whole file as one line) and with `|`
dropped from the regex detection (the alternation test: it shows no matches).
"""

from __future__ import annotations

import json
import re

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.actors import admin_of, create_user
from harness.knowledge_bases import KB_EXEC, add_text_file, knowledge_base, model_with_knowledge
from harness.python_tools import EVERYONE_READS
from utils.chat_ui import expect_reply, last_reply, send

pytestmark = [
    pytest.mark.regression,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

NOTES = "\n".join(
    [
        "alpha appears here",
        "filler line",
        "nothing of interest",
        "the needle is on line four",
        "omega closes the file",
    ]
)


@pytest.fixture(scope="module")
def notes_model(instance_with):
    """A preset every account may use, its shared knowledge base holding notes.md."""
    launched = instance_with(KB_EXEC)
    if not launched.serves_frontend:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    with (
        admin_of(launched).client() as client,
        knowledge_base(client, access_grants=[EVERYONE_READS]) as knowledge_id,
    ):
        add_text_file(client, knowledge_id, "notes.md", NOTES)
        with model_with_knowledge(client, knowledge_id) as model_id:
            yield model_id


@pytest.fixture
def chat_page(browser, instance_with, notes_model):
    """A new account's chat on the notes preset, signed in to the kb_exec instance."""
    launched = instance_with(KB_EXEC)
    account = create_user(launched)
    with account.client() as client:
        settings = {"ui": {"showChangelog": False}}
        client.post("/api/v1/users/user/settings/update", json=settings).raise_for_status()
    context = browser.new_context(
        viewport={"width": 1920, "height": 1080}, base_url=launched.base_url
    )
    context.set_default_timeout(30_000)
    page = context.new_page()
    page.add_init_script(
        f"try {{ localStorage.setItem('token', {json.dumps(account.token)}); }} catch (e) {{}}"
    )
    page.goto(f"/?models={notes_model}")
    yield page, launched.upstream
    context.close()


def _kb_exec_result(page, command: str):
    """Have the model run `command` and open its result in the reply."""
    page, upstream = page
    upstream.queue(reply.tool_call("kb_exec", {"command": command}), reply.text("searched"))
    send(page, "search the notes")
    expect_reply(page, "searched")
    answer = last_reply(page)
    answer.get_by_role("button", name=re.compile("kb_exec")).click()
    return answer


def test_a_grep_hit_is_shown_with_its_real_line_number(chat_page):
    result = _kb_exec_result(chat_page, 'grep "needle" notes.md')

    expect(result).to_contain_text("4: the needle is on line four")
    expect(result).not_to_contain_text("1: alpha appears here")


def test_an_alternation_shows_every_alternative(chat_page):
    result = _kb_exec_result(chat_page, 'grep "alpha|omega" notes.md')

    expect(result).to_contain_text("1: alpha appears here")
    expect(result).to_contain_text("5: omega closes the file")
