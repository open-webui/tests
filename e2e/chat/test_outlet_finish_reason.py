"""Regression: an outlet filter could not tell a reply cut off by the token limit.

PR #32079 (commit 039867feb) hands outlet filters the finish reason the provider reported. A
global filter here marks a reply the provider cut off at the token limit (`length`) and leaves
one it ended itself (`stop`) alone, editing the reply's text the way the chat page renders it;
in a chat typed in the browser the mark shows on the cut-off reply and only there.

Discriminates: passes on the dev 87a937459 build. With the dev build on a backend copy with
039867feb reverted the cut-off test goes red (the filter is handed no reason, so no mark).
"""

from __future__ import annotations

import textwrap
import uuid

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.plugins import installed_function
from utils.chat_ui import expect_reply, last_reply, send

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

CUT_OFF = "(cut off at the token limit)"

CUT_OFF_FILTER = textwrap.dedent(
    f"""
    class Filter:
        def outlet(self, body):
            answer = body["messages"][-1]
            if answer.get("finish_reason") != "length":
                return body
            answer["content"] += " {CUT_OFF}"
            for item in answer.get("output", []):
                if item["type"] == "message":
                    item["content"][-1]["text"] += " {CUT_OFF}"
            return body
    """
).lstrip()


@pytest.fixture
def cut_off_filter(admin):
    with installed_function(admin, CUT_OFF_FILTER, is_global=True):
        yield


def test_a_filter_marks_only_the_reply_the_provider_cut_off(
    cut_off_filter, page_for, make_user, upstream
):
    finished_prompt, cut_prompt = f"tides {uuid.uuid4().hex[:6]}", f"more {uuid.uuid4().hex[:6]}"
    upstream.queue(
        reply.text("High water is at six.", match=reply.answering(finished_prompt)),
        reply.text("Low water is at", finish_reason="length", match=reply.answering(cut_prompt)),
    )
    page = page_for(make_user())

    send(page, finished_prompt)
    expect_reply(page, "High water is at six.")
    send(page, cut_prompt)

    expect(last_reply(page)).to_contain_text(f"Low water is at {CUT_OFF}")
    expect(page.get_by_text(CUT_OFF)).to_have_count(1)
