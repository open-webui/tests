"""Journey: a timer the model set fires into the chat the person still has open.

With sub-agents on, the model sets a short timer through the timer tool and says so. The person
keeps the page open. When the timer fires, its prompt shows in the chat as a "Timer" row and the
model's follow-up reply appears under it, without a reload. A follow-up that is slow to start
shows the timer row first and the answer once the model has written it.

The server tells the open page to reload the chat when a timer fires, and the page follows the
current message the chat stores. Before PR #31576 (open-webui/open-webui#31566, fixed for timers as
well) the timer left that pointing at the earlier reply, so the page showed neither the timer nor
the follow-up until it was reloaded.

Every test here is red on dev 62f70a844: since de73bb830 a chat request whose reply message is
already stored in the chat, the way automations, sub-agents and timers prepare their reply, is
refused with 409 and the reply is never written (open-webui/open-webui#32066).

Discriminates: passes on dev a5bc78300; both tests fail on dev 176d31d1d for the reason above.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.chat_ui import conversation, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

SUBAGENTS = ("/api/v1/configs/subagents", "/api/v1/configs/subagents")
# the scheduler polls every second, so a 3 second timer fires within a few seconds
FIRE_TIMEOUT_MS = 30_000
SLOW_SECONDS = 4.0


@pytest.fixture
def timers_on(admin, preserve):
    preserve(SUBAGENTS)
    with admin.client() as client:
        current = client.get(SUBAGENTS[0]).json()
        client.post(SUBAGENTS[1], json={**current, "ENABLE_SUBAGENTS": True}).raise_for_status()


def unique(label: str) -> str:
    return f"{label} {uuid.uuid4().hex[:8]}"


def timer_row(page: Page, prompt: str) -> Locator:
    return conversation(page).get_by_role("button", name=re.compile(f"Timer .*{prompt}"))


def set_timer(page: Page, upstream, prompt: str, follow_up: reply.Reply) -> None:
    """The person asks and the model sets a 3 second timer; `follow_up` answers when it fires."""
    ask = unique("remind me shortly")
    upstream.queue(
        reply.tool_call("timer", {"prompt": prompt, "at": "3s"}, match=reply.answering(ask)),
        reply.text("Timer set.", match=reply.answering(ask)),
        follow_up,
    )
    send(page, ask)
    expect_reply(page, "Timer set.")
    expect(timer_row(page, prompt)).to_have_count(0)


def test_the_timer_and_the_follow_up_show_in_the_open_chat_without_a_reload(
    timers_on, page_for, make_user, upstream
):
    prompt = unique("The kettle has boiled")
    page = page_for(make_user())
    set_timer(page, upstream, prompt, reply.text("Pour the tea.", match=reply.answering(prompt)))

    expect(
        timer_row(page, prompt),
        "the fired timer never showed in the open chat (it shows only after a reload)",
    ).to_be_visible(timeout=FIRE_TIMEOUT_MS)
    expect_reply(page, "Pour the tea.")


def test_a_slow_follow_up_shows_the_timer_first_and_the_answer_when_it_is_written(
    timers_on, page_for, make_user, upstream
):
    prompt = unique("The dough has risen")
    page = page_for(make_user())
    set_timer(
        page,
        upstream,
        prompt,
        reply.text("Knead it now.", delay=SLOW_SECONDS, match=reply.answering(prompt)),
    )

    expect(
        timer_row(page, prompt),
        "the fired timer never showed in the open chat (it shows only after a reload)",
    ).to_be_visible(timeout=FIRE_TIMEOUT_MS)
    expect(conversation(page)).not_to_contain_text("Knead it now.")
    expect_reply(page, "Knead it now.")
