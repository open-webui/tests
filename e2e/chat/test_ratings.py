"""Journey: a reply rated in the chat reaches the admin's Evaluations page with its reason.

A fresh account rates the scripted model's reply. Thumbs down opens the rating form, where the
account picks a score, a reason and writes a comment; the admin's Feedback list then shows the
account's rating as a loss for the model, and its details carry the prompt, the reply, the score,
the reason and the comment. Thumbs up alone is listed as a win.

The thumbs down test also pins a fixed bug (fix 2d5651410, PR open-webui/open-webui#32334): the
rating asks for tags in the background, and when they arrived after the reason and comment were
saved, the rating was saved again from its state before them, so the stored reason and comment
were gone. The tags reply is held back a few seconds so it always lands after Save. It also pins
open-webui/open-webui#33715: while the tags were still being generated, Save in the rating form
stored nothing at all.

Discriminates: passes on dev ac00d40e3; in a backend copy, with the feedback create route
answering without storing the row both tests fail (the account has no row on the Feedback page).
The thumbs down test passes on the dev 5032e834e build and fails on that build with 2d5651410
reverted (the stored rating has its tags but no reason, comment or score). It also fails on the
dev 623257af7 build, where Save stores no reason or comment while the tags are pending (#33715).

A thumb pressed from the keyboard keeps the focus while its rating saves (f36807a55, PR
open-webui/open-webui#33673: the thumb was disabled while saving, which threw the focus to the
page). Passes on dev e8c0ad85e, fails on that build with the thumbs disabled again.
"""

from __future__ import annotations

import json
import re
import time
from typing import Callable

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import conversation, expect_reply, last_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

QUESTION = "name a good hiking snack"
ANSWER = "trail mix with dried apricots"
TAGS = ["Food", "Hiking"]


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


def _tags_arrive_late(admin, preserve, upstream) -> None:
    """Rating tags are generated again, and their reply lands a few seconds after it is asked."""
    preserve("tasks")
    with admin.client() as client:
        tasks = client.get("/api/v1/tasks/config").json()
        saved = client.post(
            "/api/v1/tasks/config/update", json={**tasks, "ENABLE_TAGS_GENERATION": True}
        )
    assert saved.status_code == 200, saved.text
    upstream.queue(
        reply.text(json.dumps({"tags": TAGS}), delay=4, match=_asks_for_tags),
    )


def _asks_for_tags(body: dict) -> bool:
    return "Generate 1-3 broad tags" in json.dumps(body.get("messages"))


def _stored_rating(account) -> dict:
    with account.client() as client:
        listed = client.get("/api/v1/evaluations/feedbacks/user")
    assert listed.status_code == 200, listed.text
    [feedback] = listed.json()["items"]
    return feedback["data"]


def _stored_rating_once(account, holds: Callable[[dict], object], timeout: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout
    data = _stored_rating(account)
    while not holds(data) and time.monotonic() < deadline:
        time.sleep(0.2)
        data = _stored_rating(account)
    assert holds(data), f"the stored rating never got there: {data}"
    return data


def _feedback_rows(page_for, admin, account, result: str):
    page = page_for(admin, path="/admin/evaluations/feedback")
    name = re.compile(rf"^{re.escape(account.name)} {MOCK_MODEL_ID} {result}\b")
    return page.get_by_role("row", name=name)


@pytest.mark.regression
def test_a_thumbs_down_with_a_reason_shows_in_the_feedback_details(
    rater, page_for, admin, admin_token, upstream, preserve
):
    account, page = rater
    _tags_arrive_late(admin, preserve, upstream)
    _rate(page, "Bad Response")
    chat = conversation(page)
    chat.get_by_role("button", name="Rate 2 out of 10").click()
    chat.get_by_role("button", name="Not helpful").click()
    chat.get_by_role("textbox", name="Additional feedback comments").fill("too sugary for me")
    chat.get_by_role("button", name="Save").click()
    expect(page.get_by_text("Thanks for your feedback!")).to_be_visible()
    saved = _stored_rating_once(account, lambda data: data.get("comment"))
    assert not saved.get("tags"), "the tags arrived before the reason was saved"

    rated = _stored_rating_once(account, lambda data: data.get("tags"))
    assert rated["tags"] == TAGS
    kept = (rated.get("details"), rated.get("reason"), rated.get("comment"))
    assert kept == ({"rating": 2}, "not_helpful", "too sugary for me"), (
        "the late tags wiped the reason and comment"
    )

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


def test_a_thumb_pressed_from_the_keyboard_keeps_focus_while_the_rating_saves(rater):
    # the thumb was disabled while it saved, which threw keyboard focus to the page (#33673)
    _, page = rater
    held: list = []
    page.route("**/api/v1/evaluations/feedback", lambda route: held.append(route))
    thumb = conversation(page).get_by_role("button", name="Good Response").last
    last_reply(page).hover()
    thumb.focus()
    expect(thumb).to_be_focused()

    with page.expect_request(lambda request: request.url.endswith("/evaluations/feedback")):
        page.keyboard.press("Enter")

    expect(thumb, "keyboard focus left the thumb while its rating was saving").to_be_focused()
    page.unroute("**/api/v1/evaluations/feedback")
    for route in held:
        route.continue_()
