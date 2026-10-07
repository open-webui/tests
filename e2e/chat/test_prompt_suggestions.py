"""Journey: the prompt suggestions on the new chat screen, narrowed by typing, model and language.

The new chat screen lists the instance's default prompt suggestions under Suggested. Typing into
the chat input narrows the list to the suggestions that match the text, and text that matches
none leaves no suggestion and no Suggested label. A suggestion with an empty title shows its
prompt with the word Prompt beneath and sends that prompt when pressed. A model with suggestions
of its own shows those in place of the defaults, and the defaults return once the plain model is
picked again. A browser in a language the admin gave suggestions of their own sees those.

Discriminates: passes on dev ebc6add67. On a build whose suggestions ignore the typed text both
narrowing tests fail, one that ignores the model's own suggestions fails the model test, one that
ignores the localized suggestions fails the language test and one that drops the Prompt label
fails the untitled test.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import chat_input, expect_reply

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

TIDES = {"title": ["Tide table", "for the harbour"], "content": "When is high tide today?"}
KNOTS = {"title": ["Knots", "for sailors"], "content": "Teach me to tie a bowline."}
UNTITLED = {"title": ["", ""], "content": "Describe the lighthouse at dusk."}
GERMAN = {"title": ["Gezeiten", "im Hafen"], "content": "Wann ist heute Hochwasser?"}


@pytest.fixture
def default_suggestions(admin):
    """`default_suggestions(suggestions, i18n)` saves the instance's defaults, restored after."""
    with admin.client() as client:
        before = client.get("/api/config").json()

    def save(suggestions: list[dict], i18n: dict | None = None) -> None:
        with admin.client() as client:
            saved = client.post(
                "/api/v1/configs/suggestions", json={"suggestions": suggestions, "i18n": i18n or {}}
            )
        saved.raise_for_status()

    yield save
    save(before["default_prompt_suggestions"], before.get("default_prompt_suggestions_i18n"))


def suggestion(page: Page, text: str) -> Locator:
    return page.get_by_role("listitem").filter(has_text=text)


def suggestions(page: Page) -> Locator:
    return page.get_by_role("list").get_by_role("listitem")


def type_into_input(page: Page, text: str) -> None:
    expect(chat_input(page)).to_be_visible()
    chat_input(page).click()
    page.keyboard.type(text)


def test_typing_narrows_the_suggestions_to_the_matching_ones(
    page_for, make_user, default_suggestions
):
    default_suggestions([TIDES, KNOTS])
    page = page_for(make_user())
    expect(suggestion(page, "Tide table")).to_be_visible()
    expect(suggestion(page, "Knots")).to_be_visible()

    type_into_input(page, "bowline")

    expect(suggestion(page, "Knots")).to_be_visible()
    expect(suggestion(page, "Tide table")).to_have_count(0)


def test_text_that_matches_no_suggestion_leaves_none(page_for, make_user, default_suggestions):
    default_suggestions([TIDES, KNOTS])
    page = page_for(make_user())
    expect(page.get_by_text("Suggested", exact=True)).to_be_visible()

    type_into_input(page, "zzzzqqqqxxxx")

    expect(suggestions(page)).to_have_count(0)
    expect(page.get_by_text("Suggested", exact=True)).to_have_count(0)


def test_a_suggestion_without_a_title_shows_and_sends_its_prompt(
    page_for, make_user, default_suggestions, upstream
):
    default_suggestions([UNTITLED])
    page = page_for(make_user())
    shown = suggestion(page, UNTITLED["content"])
    expect(shown).to_contain_text("Prompt")

    upstream.queue(reply.text("Golden light.", match=reply.answering(UNTITLED["content"])))
    shown.click()

    expect_reply(page, "Golden light.")
    sent = next(filter(reply.answering(UNTITLED["content"]), upstream.chat_requests()))
    assert sent["messages"][-1] == {"role": "user", "content": UNTITLED["content"]}


@pytest.fixture
def model_with_suggestions(admin, make_user):
    """A fresh user and a preset only they may read, with one suggestion of its own."""
    account = make_user()
    model_id = f"sailing-{uuid.uuid4().hex[:8]}"
    form = {
        "id": model_id,
        "base_model_id": MOCK_MODEL_ID,
        "name": model_id,
        "meta": {"suggestion_prompts": [KNOTS]},
        "params": {},
        "access_grants": [
            {"principal_type": "user", "principal_id": account.id, "permission": "read"}
        ],
    }
    with admin.client() as client:
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
    yield account, model_id
    with admin.client() as client:
        client.post("/api/v1/models/model/delete", json={"id": model_id})


def test_a_models_own_suggestions_replace_the_defaults(
    page_for, model_with_suggestions, default_suggestions
):
    default_suggestions([TIDES])
    account, model_id = model_with_suggestions
    page = page_for(account)

    page.goto(f"/?model={model_id}")
    expect(page.get_by_role("button", name=f"Selected model: {model_id}")).to_be_visible()
    expect(suggestion(page, "Knots")).to_be_visible()
    expect(suggestion(page, "Tide table")).to_have_count(0)

    page.goto(f"/?model={MOCK_MODEL_ID}")
    expect(
        page.get_by_role("button", name=re.compile(f"Selected model: {MOCK_MODEL_ID}"))
    ).to_be_visible()
    expect(suggestion(page, "Tide table")).to_be_visible()
    expect(suggestion(page, "Knots")).to_have_count(0)


def test_a_browser_in_another_language_sees_the_suggestions_for_it(
    page_for, make_user, default_suggestions
):
    default_suggestions([TIDES], {"de-DE": {"suggestion_prompts": [GERMAN]}})

    german = page_for(make_user(), locale="de-DE")
    expect(suggestion(german, "Gezeiten")).to_be_visible()
    expect(suggestion(german, "Tide table")).to_have_count(0)

    english = page_for(make_user(), locale="en-US")
    expect(suggestion(english, "Tide table")).to_be_visible()
    expect(suggestion(english, "Gezeiten")).to_have_count(0)
