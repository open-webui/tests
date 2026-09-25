"""Regression: a reply wrapped in solution tags kept the closing tag and what followed it.

Issue open-webui/open-webui#31434, fix PR open-webui/open-webui#31436. The start tag
`<|begin_of_solution|>` was removed but the branch that looks for `<|end_of_solution|>` was never
reached, so the stored reply ended with the literal end tag and the text after it.

Discriminates: fails on dev ac00d40e3 (the end tag stays in the reply), passes with #31436
applied.
"""

from __future__ import annotations

import pytest

from harness import upstream as reply
from harness.chat import ask

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

START, END = "<|begin_of_solution|>", "<|end_of_solution|>"


def _ask_streamed(make_user, upstream, pieces: list[str]) -> dict:
    upstream.queue(reply.text(pieces, match=reply.answering("solve it")))
    with make_user().client() as client:
        _, message = ask(client, "solve it")
    return message


def test_the_end_tag_is_removed_from_the_reply(make_user, upstream):
    message = _ask_streamed(make_user, upstream, [f"{START}The answer is 4.{END}"])

    assert END not in message["content"], message["content"]
    assert START not in message["content"]
    assert "The answer is 4." in message["content"]


def test_the_end_tag_is_removed_when_it_arrives_in_its_own_chunk(make_user, upstream):
    message = _ask_streamed(make_user, upstream, [START, "The answer is 4.", END])

    assert END not in message["content"], message["content"]
    assert "The answer is 4." in message["content"]


def test_a_reply_without_solution_tags_is_left_alone(make_user, upstream):
    message = _ask_streamed(make_user, upstream, ["The answer ", "is 4."])

    assert message["content"] == "The answer is 4."
