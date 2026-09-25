"""Regression: a space was lost right after a closing think tag.

Issue open-webui/open-webui#31435, fix PR open-webui/open-webui#31438. When the chunk carrying
`</think>` went on with answer text ending in a space, that space was stripped, so the chunks
`</think>The answer ` and `is 4.` were stored as "The answeris 4.".

Discriminates: fails on dev ac00d40e3 (the stored answer reads "The answeris 4."), passes with
#31438 applied.
"""

from __future__ import annotations

import pytest

from harness import upstream as reply
from harness.chat import ask

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


def _ask_streamed(make_user, upstream, pieces: list[str]) -> dict:
    upstream.queue(reply.text(pieces, match=reply.answering("think it over")))
    with make_user().client() as client:
        _, message = ask(client, "think it over")
    return message


def test_the_space_after_the_think_block_is_kept(make_user, upstream):
    message = _ask_streamed(make_user, upstream, ["<think>pondering</think>The answer ", "is 4."])

    assert message["content"] == "The answer is 4."


def test_the_thinking_itself_is_unchanged(make_user, upstream):
    message = _ask_streamed(make_user, upstream, ["<think>pondering</think>The answer ", "is 4."])

    [reasoning] = [item for item in message["output"] if item["type"] == "reasoning"]
    assert reasoning["content"][0]["text"] == "pondering"


def test_the_same_answer_in_one_chunk_is_unchanged(make_user, upstream):
    message = _ask_streamed(make_user, upstream, ["<think>pondering</think>The answer is 4."])

    assert message["content"] == "The answer is 4."
