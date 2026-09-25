"""Journey: a reply rated in the chat reaches the admin's Evaluations page with its reason.

A fresh account rates the scripted model's reply. Thumbs down opens the rating form, where the
account picks a score, a reason and writes a comment; the admin's Feedback list then shows the
account's rating as a loss for the model, and its details carry the prompt, the reply, the score,
the reason and the comment. Thumbs up alone is listed as a win.

Discriminates: passes on dev ac00d40e3; in a backend copy, with the feedback create route
answering without storing the row both tests fail (the account has no row on the Feedback page).
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import conversation, expect_reply, last_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

QUESTION = "name a good hiking snack"
ANSWER = "trail mix with dried apricots"


@pytest.fixture
def rater(make_user, page_for, upstream):
    """A fresh account and its page, on a chat holding one scripted reply."""
    account = make_user()
    page = page_for(account)
    upstream.queue(reply.text(ANSWER, match=reply.answering(QUESTION)))
    send(page, QUESTION)
    expect_reply(page, ANSWER)
    return account, page


def _is_new_feedback(response) -> bool:
    return response.request.method == "POST" and response.url.endswith("/evaluations/feedback")


def _rate(page: Page, verdict: str) -> None:
    last_reply(page).hover()
    with page.expect_response(_is_new_feedback):
        conversation(page).get_by_role("button", name=verdict).last.click()


def _feedback_rows(page_for, admin, account, result: str):
    page = page_for(admin)
    page.goto("/admin/evaluations/feedback")
    name = re.compile(rf"^{re.escape(account.name)} {MOCK_MODEL_ID} {result}\b")
    return page.get_by_role("row", name=name)


def test_a_thumbs_down_with_a_reason_shows_in_the_feedback_details(
    rater, page_for, admin, admin_token
):
    account, page = rater
    _rate(page, "Bad Response")
    chat = conversation(page)
    chat.get_by_role("button", name="Rate 2 out of 10").click()
    chat.get_by_role("button", name="Not helpful").click()
    chat.get_by_role("textbox", name="Additional feedback comments").fill("too sugary for me")
    chat.get_by_role("button", name="Save").click()
    expect(page.get_by_text("Thanks for your feedback!")).to_be_visible()

    rows = _feedback_rows(page_for, admin, account, "Lost")
    expect(rows).to_have_count(1)
    rows.get_by_role("cell", name=MOCK_MODEL_ID).click()
    details = rows.page.get_by_role("dialog").filter(has_text="Feedback Details")
    for shown in (QUESTION, ANSWER, "Rating 2", "not_helpful", "too sugary for me"):
        expect(details).to_contain_text(shown)


def test_a_thumbs_up_is_listed_as_a_win(rater, page_for, admin, admin_token):
    account, page = rater
    _rate(page, "Good Response")
    expect(_feedback_rows(page_for, admin, account, "Won")).to_have_count(1)
