"""The {{CLIPBOARD}} prompt variable mangled dollar signs, open-webui/open-webui#31363.

Fix commit `b233d0d94` (PR open-webui/open-webui#31364). The chat input put the clipboard in
place of `{{CLIPBOARD}}` with `String.replaceAll(pattern, text)`, which reads `$$`, `$&` and the
like in `text` as replacement patterns: a copied `echo $$` arrived as `echo $`, breaking shell
commands and math formulas. The clipboard now goes in through a replacer function,
exactly as copied.

Over plain HTTP by address a page cannot read the clipboard at all, so the paste tests run on a
secure page. Over plain HTTP the prompt must still go in with the variable left as typed and a
message, open-webui/open-webui#33937: picking it did nothing at all.

Discriminates: passes on the efe63bd34 build, fails on it with `b233d0d94` reverted (the pasted
text loses a dollar sign and `$&` turns back into the variable). The plain HTTP test fails on
0.13.1 and dev b6534c58f (the prompt list stays open and nothing is inserted).
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from utils.chat_ui import chat_input, expect_reply
from utils.clipboard import write_text

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

COPIED = "kill $$ && echo $& costs $$5, $' or $1 today"


@pytest.fixture
def writer(make_user):
    """An account with a prompt `/<command>` reading `Run: {{CLIPBOARD}}`."""
    account = make_user(role="admin")
    command = f"paste{uuid.uuid4().hex[:6]}"
    form = {"command": command, "name": "Paste it", "content": "Run: {{CLIPBOARD}}"}
    with account.client() as client:
        created = client.post("/api/v1/prompts/create", json=form)
    assert created.status_code == 200, created.text
    return account, command


def with_clipboard(page: Page, text: str) -> None:
    page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    write_text(page, text)


def use_prompt(page: Page, command: str) -> None:
    chat_input(page).click()
    page.keyboard.type(f"/{command}")
    picker = page.get_by_role("tooltip").filter(has_text="Prompts")
    picker.get_by_role("button", name=command).click()
    expect(picker).to_be_hidden()


# the browser lets a page read the clipboard only in a secure context
@pytest.mark.secure_context
def test_the_chat_input_pastes_the_clipboard_exactly(writer, page_for, upstream):
    account, command = writer
    page = page_for(account)
    expect(chat_input(page)).to_be_visible()
    with_clipboard(page, COPIED)

    use_prompt(page, command)

    expect(chat_input(page)).to_have_text(f"Run: {COPIED}")
    upstream.queue(reply.text("Done.", match=reply.answering("Run: ")))
    page.keyboard.press("Enter")
    expect_reply(page, "Done.")
    sent = reply.conversation(upstream.chat_requests()[-1])[-1]["content"]
    assert sent == f"Run: {COPIED}"


# the browser lets a page read the clipboard only in a secure context
@pytest.mark.secure_context
def test_plain_clipboard_text_is_pasted_unchanged(writer, page_for):
    account, command = writer
    page = page_for(account)
    expect(chat_input(page)).to_be_visible()
    with_clipboard(page, "no dollars here")

    use_prompt(page, command)

    expect(chat_input(page)).to_have_text("Run: no dollars here")


def test_a_clipboard_prompt_is_still_inserted_over_plain_http(writer, page_for):
    account, command = writer
    page = page_for(account)
    expect(chat_input(page)).to_be_visible()

    use_prompt(page, command)

    expect(chat_input(page)).to_have_text("Run: {{CLIPBOARD}}")
    expect(page.get_by_text("Failed to read clipboard contents")).to_be_visible()
