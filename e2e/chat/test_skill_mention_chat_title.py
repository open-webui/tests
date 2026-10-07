"""Journey: a chat named after its first message, when that message mentions a skill.

With title generation off, a new chat takes its first message as its title. A plain message
becomes the title as typed. A message that starts with a skill picked from the `$` menu should be
titled the way the message reads, but the sidebar shows the mention's raw markup, such as
`<$skill-id|Skill name> what now?`, because the fallback title is the stored message text and
the mention is only resolved where the message itself is drawn. That test is red on purpose.

Discriminates: the plain message test passes on the dev ebc6add67 build and turns red in a
backend copy whose fallback title is a fixed "Untitled"; the mention test fails on that build
(the sidebar entry reads `<$tides-...|Tides ...> when is high tide ...`).
"""

from __future__ import annotations

import uuid
from typing import Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.actors import Actor
from utils.chat_ui import chat_input, expect_reply

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def writer(make_user) -> Iterator[Actor]:
    """A fresh admin; the skills it made are deleted afterwards."""
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for skill in client.get("/api/v1/skills/").json():
            if skill["user_id"] == account.id:
                client.delete(f"/api/v1/skills/id/{skill['id']}/delete")


def _sidebar_titles(page: Page) -> Locator:
    # the message itself shows the same words, so the title is found by its sidebar entry
    return page.locator("#sidebar-chat-item div[dir='auto']")


def _start_chat(page: Page, upstream, question: str) -> None:
    upstream.queue(reply.text("Noted.", match=reply.answering(question)))
    page.keyboard.type(question)
    page.keyboard.press("Enter")
    expect_reply(page, "Noted.")


def test_a_plain_first_message_becomes_the_chat_title(page_for, writer, upstream):
    question = f"when is high tide {uuid.uuid4().hex[:6]}"
    page = page_for(writer)
    expect(chat_input(page)).to_be_visible()
    chat_input(page).click()

    _start_chat(page, upstream, question)

    expect(_sidebar_titles(page).filter(has_text=question)).to_have_text(question)


def test_a_first_message_with_a_skill_mention_is_titled_without_its_markup(
    page_for, writer, upstream
):
    suffix = uuid.uuid4().hex[:6]
    name = f"Tides {suffix}"
    with writer.client() as client:
        created = client.post(
            "/api/v1/skills/create",
            json={"id": f"tides-{suffix}", "name": name, "content": "Read the table.", "meta": {}},
        )
    assert created.status_code == 200, created.text
    question = f"when is high tide {suffix}"
    page = page_for(writer)
    expect(chat_input(page)).to_be_visible()
    chat_input(page).click()
    page.keyboard.type(f"${suffix}")
    page.get_by_role("button").filter(has_text=name).click()

    _start_chat(page, upstream, question)

    title = _sidebar_titles(page).filter(has_text=question)
    expect(title).to_have_count(1)
    expect(title, "the chat title shows the mention's raw markup").not_to_contain_text("<$")
