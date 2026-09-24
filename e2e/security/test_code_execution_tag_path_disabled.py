"""Regression: a `<code_interpreter>` tag in the reply ran even under native function calling.

open-webui 0.11.1 fix `ac85b0f2a` (#29024): the streaming handler acted on `<code_interpreter>`
tags in every tool-calling mode. Native mode, the default, never teaches the tag format and
offers `execute_code` as a tool instead, so code a model merely quoted in its reply was run in
the user's browser. The fix only acts on the tag under legacy function calling.

With the default Pyodide engine a tag that runs turns into an "Analyzed" block and its code runs
in the page; one that does not stays in the reply as text.

Twin of unit/security/test_code_execution_tag_path_disabled.py.

Discriminates: passes on dev bbfa876af; with the `function_calling == 'legacy'` gate dropped
the native reply turns into an "Analyzed" block and its code runs in the browser.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from utils.chat_ui import expect_reply, last_reply, send, stop_button

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

QUOTED_TAG = '<code_interpreter type="code" lang="python">print(6 * 7)</code_interpreter>'
TAG_REPLY = f"The document contains this snippet:\n{QUOTED_TAG}"


@pytest.fixture
def chat_page(page_for, make_user):
    """`chat_page(function_calling)` is a new chat with the code interpreter switched on."""

    def open_chat(function_calling: str | None = None):
        account = make_user()
        if function_calling:
            with account.client() as client:
                saved = client.post(
                    "/api/v1/users/user/settings/update",
                    json={"ui": {"params": {"function_calling": function_calling}}},
                )
            saved.raise_for_status()
        page = page_for(account)
        page.goto("/?code-interpreter=true")
        return page

    return open_chat


def test_a_tag_in_the_reply_stays_text_under_native_function_calling(chat_page, upstream):
    page = chat_page()
    prompt = "what does the document say?"
    upstream.queue(reply.text(TAG_REPLY, match=reply.answering(prompt)))
    send(page, prompt)
    expect_reply(page, "The document contains this snippet:")
    expect(stop_button(page)).to_be_hidden()

    expect(last_reply(page)).to_contain_text(QUOTED_TAG)
    expect(last_reply(page).get_by_text("Analyzed")).to_have_count(0)
    about_this_chat = [body for body in upstream.chat_requests() if prompt in str(body["messages"])]
    assert len(about_this_chat) == 1, (
        "a <code_interpreter> block quoted in a native-mode reply was run in the browser and "
        "its output sent back to the model (#29024)"
    )


def test_legacy_function_calling_still_runs_the_tag_in_the_browser(chat_page, upstream):
    page = chat_page("legacy")
    upstream.queue(reply.text(TAG_REPLY), reply.text("It printed the answer."))
    send(page, "run the snippet")
    expect_reply(page, "It printed the answer.")

    expect(last_reply(page).get_by_text("Analyzed")).to_be_visible()
    sent_back = upstream.chat_requests()[-1]["messages"][-1]["content"]
    assert "42" in sent_back, f"the browser never ran the tag's code: {sent_back}"
