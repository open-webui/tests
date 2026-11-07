"""Regression: deleting the newest artifact left the artifacts pane showing it.

Fix PR open-webui/open-webui#30435 (merged as `efe63bd34`, issue open-webui/open-webui#30287) in
the frontend's artifacts pane. The pane rebuilds its list of versions whenever the chat changes
and moved its selection when the list grew or emptied, but not when it only got shorter. So
deleting the message holding the newest artifact, with the pane open on it, left the selection
past the end: the pane kept the deleted artifact on screen under "Version 2 of 2" and the page
threw an error, until some other action redrew it. The selection now moves to the last version.

Two replies each carry an HTML block, the pane opens on the second, and the second turn is
deleted from the chat.

Discriminates: passes on the dev bc2416c5d build; with the shrink branch of the pane's list
update removed, the header keeps reading "Version 2 of 2" after the deletion.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from utils.chat_ui import conversation, expect_reply, send, split_screen
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]


def html_reply(label: str) -> str:
    return f"Here it is.\n\n```html\n<h1>{label}</h1>\n```\n\nDone with {label}."


def artifacts_pane(page: Page):
    return page.locator("#artifacts-container")


def shown_artifact(page: Page):
    return artifacts_pane(page).frame_locator("iframe").locator("h1")


@pytest.fixture
def two_artifacts(page_for, make_user, upstream):
    """A chat whose two replies each hold an artifact, the pane open on the second."""
    page = page_for(make_user(role="admin"))  # deleting messages is an admin default
    upstream.queue(
        reply.text(html_reply("first card"), match=reply.answering("first page")),
        reply.text(html_reply("second card"), match=reply.answering("second page")),
    )
    send(page, "make me the first page")
    expect_reply(page, "Done with first card.")
    split_screen(page)
    send(page, "make me the second page")
    expect_reply(page, "Done with second card.")

    expect(artifacts_pane(page)).to_contain_text("Version 2 of 2")
    expect(shown_artifact(page)).to_have_text("second card")
    return page


def delete_turn(page: Page, index: int) -> None:
    question = conversation(page).locator(".chat-user").nth(index)
    question.hover()
    tooltip_button(question, "Delete").click()
    page.get_by_role("button", name="Confirm").click()


def test_deleting_the_newest_artifact_moves_the_pane_to_the_last_one_left(two_artifacts):
    page = two_artifacts

    delete_turn(page, 1)

    expect(conversation(page)).not_to_contain_text("Done with second card.")
    expect(artifacts_pane(page)).to_contain_text("Version 1 of 1")
    expect(shown_artifact(page)).to_have_text("first card")


def test_the_pane_keeps_a_selection_that_still_exists(two_artifacts):
    page = two_artifacts
    artifacts_pane(page).get_by_role("button", name="Previous version").click()
    expect(artifacts_pane(page)).to_contain_text("Version 1 of 2")

    delete_turn(page, 1)

    expect(artifacts_pane(page)).to_contain_text("Version 1 of 1")
    expect(shown_artifact(page)).to_have_text("first card")
