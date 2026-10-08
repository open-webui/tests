"""Streamed reasoning: the thinking block shows how long the model thought while the answer streams.

Commit f55e09e50 (backend `utils/middleware.py`): a reasoning stream now announces its item
with `response.output_item.added` and closes it with `response.output_item.done`, which carries
the duration. Without them the block has no duration until the whole reply finishes, so it reads
a bare "Thought" while the answer is already streaming below it.

Discriminates: passes on dev b5a20423e; in a backend copy with f55e09e50 reverted the block reads
"Thought" without the duration mid-stream and the test fails.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from utils.chat_ui import conversation, last_reply, send

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

REASONING = re.compile(r"^(Thinking|Thought)")


def test_the_thinking_block_is_done_while_the_answer_still_streams(page_for, make_user, upstream):
    page = page_for(make_user())
    prompt = "weigh the tide against the wind"
    pieces = [f"word-{index} " for index in range(14)]
    upstream.queue(
        reply.text(
            pieces,
            reasoning="comparing the tide and the wind",
            chunk_delay=0.5,
            match=reply.answering(prompt),
        )
    )
    page.goto("/")
    send(page, prompt)
    expect(last_reply(page)).to_contain_text("word-2 ", timeout=30_000)

    block = last_reply(page).get_by_role("button", name=REASONING)
    label = block.inner_text()
    still_streaming = "word-13" not in conversation(page).inner_text()
    assert still_streaming, "the reply finished before the label could be read"
    assert re.match(r"Thought for", label), (
        f"the thinking block read {label!r} while the answer was already streaming"
    )
