"""Journey: the model's autocomplete suggestions in the chat input, once both switches are on.

With Autocomplete Generation on in the admin's interface settings and Prompt Autocompletion on in
a person's own settings, a pause in typing asks the model to continue the text, and its answer
shows as a suggestion after the cursor. In a running chat the request carries the conversation so
far. Typing on drops the suggestion and asks again for the longer text. With the person's own
switch off the model is never asked, and a text longer than the admin's input limit is refused
before it reaches the model, so it gets no suggestion. Tab taking a suggestion is covered in
e2e/admin/test_admin_interface_tasks.py.

Discriminates: passes on the dev 30f3f6a8f build and backend; in a frontend build that ignores the
person's switch, leaves the conversation out of the request and keeps a suggestion through further
typing, every test fails (the last on the kept suggestion); with the input length check removed
from a backend copy, the last test fails (the long text is continued).
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.chat_history import seed_chat
from utils.chat_ui import chat_input

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

AUTOCOMPLETE_MARKER = "You are an autocompletion system"
AUTOCOMPLETE_ROUTE = "/api/v1/tasks/auto/completions"
# twice the input's typing pause before it asks
QUIET_MS = 2_500


@pytest.fixture
def switch_autocomplete(admin, preserve):
    preserve("tasks")

    def switch(turn_on: bool, max_length: int = -1) -> None:
        with admin.client() as client:
            current = client.get("/api/v1/tasks/config").json()
            saved = client.post(
                "/api/v1/tasks/config/update",
                json={
                    **current,
                    "ENABLE_AUTOCOMPLETE_GENERATION": turn_on,
                    "AUTOCOMPLETE_GENERATION_INPUT_MAX_LENGTH": max_length,
                },
            )
        saved.raise_for_status()

    return switch


def _open_composer(page_for, account, personal_switch: bool, path: str = "/") -> Page:
    page = page_for(account)
    with account.client() as client:
        saved = client.post(
            "/api/v1/users/user/settings/update",
            json={"ui": {"showChangelog": False, "promptAutocomplete": personal_switch}},
        )
    saved.raise_for_status()
    page.goto(path)
    expect(chat_input(page)).to_be_visible()
    chat_input(page).click()
    return page


def _suggest(upstream, text: str) -> None:
    upstream.queue(reply.text(f'{{"text": "{text}"}}', match=reply.answering(AUTOCOMPLETE_MARKER)))


def _autocomplete_requests(upstream) -> list[dict]:
    return [
        body for body in upstream.chat_requests() if AUTOCOMPLETE_MARKER in str(body["messages"])
    ]


def _suggestion(page: Page):
    return page.locator("#chat-input [data-suggestion]")


def test_a_suggestion_in_a_running_chat_is_written_with_the_conversation(
    page_for, make_user, upstream, switch_autocomplete
):
    switch_autocomplete(turn_on=True)
    account = make_user()
    with account.client() as client:
        chat_id, _ = seed_chat(
            client,
            [
                {"role": "user", "content": "Which harbour has the oldest lighthouse?"},
                {"role": "assistant", "content": "The lighthouse at Whitby dates from 1858."},
            ],
        )
    page = _open_composer(page_for, account, personal_switch=True, path=f"/c/{chat_id}")
    _suggest(upstream, " lighthouse at Whitby")

    page.keyboard.type("How tall is the")

    expect(_suggestion(page)).to_have_attribute("data-suggestion", " lighthouse at Whitby")
    [asked] = _autocomplete_requests(upstream)
    assert "How tall is the" in str(asked["messages"])
    assert "dates from 1858" in str(asked["messages"])


def test_typing_on_drops_the_suggestion_and_asks_again_for_the_longer_text(
    page_for, make_user, upstream, switch_autocomplete
):
    switch_autocomplete(turn_on=True)
    page = _open_composer(page_for, make_user(), personal_switch=True)
    _suggest(upstream, " Hallstatt in the morning")
    page.keyboard.type("Ferries to")
    expect(_suggestion(page)).to_have_attribute("data-suggestion", " Hallstatt in the morning")

    _suggest(upstream, "")
    with page.expect_response(lambda response: AUTOCOMPLETE_ROUTE in response.url):
        page.keyboard.type(" St. Wolfgang")

    expect(_suggestion(page)).to_have_count(0)
    expect(chat_input(page)).to_have_text("Ferries to St. Wolfgang")
    asked_again = _autocomplete_requests(upstream)[-1]
    assert "Ferries to St. Wolfgang" in str(asked_again["messages"])


def test_with_the_personal_switch_off_the_model_is_never_asked(
    page_for, make_user, upstream, switch_autocomplete
):
    switch_autocomplete(turn_on=True)
    page = _open_composer(page_for, make_user(), personal_switch=False)
    _suggest(upstream, " early in the autumn.")

    page.keyboard.type("The best time to visit Hallstatt is")
    page.wait_for_timeout(QUIET_MS)

    expect(_suggestion(page)).to_have_count(0)
    assert _autocomplete_requests(upstream) == []


def test_a_text_longer_than_the_admins_limit_is_not_continued(
    page_for, make_user, upstream, switch_autocomplete
):
    switch_autocomplete(turn_on=True, max_length=20)
    page = _open_composer(page_for, make_user(), personal_switch=True)
    _suggest(upstream, " in the morning")
    page.keyboard.type("Ferries to Hallstatt")
    expect(_suggestion(page)).to_have_attribute("data-suggestion", " in the morning")

    _suggest(upstream, " every hour")
    with page.expect_response(lambda response: AUTOCOMPLETE_ROUTE in response.url) as answered:
        page.keyboard.type(" leave from the pier")

    assert answered.value.status == 400
    expect(_suggestion(page)).to_have_count(0)
    assert len(_autocomplete_requests(upstream)) == 1
