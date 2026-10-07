"""Journey: a note is formatted, titled and measured in its editor.

A fresh account opens a new note and types into it. Selecting text raises a toolbar over it:
its bold, italic, underline and strikethrough buttons mark the selection, and its heading,
bullet, ordered, task and code block buttons turn the line into that block; each shows in the
editor at once, is stored with the note and is still there after a reload. The Formatting switch
in the note's menu decides whether typed Markdown such as "## Heading" or "**bold**" becomes
formatting or stays as typed characters. Clicking the sparkles button beside the focused title
asks the model for a title, which fills the field and is stored. Under the title the note shows
when it was created ("Today at" and the time) and a word and character count that follows the
typing. The undo and redo buttons in the header take back and bring back what was typed.

The two typed-markdown tests are red with Redis (`OWUI_TEST_REDIS=1`) on dev 0f5a58f5f: the server
handles the editor's live updates concurrently, so an update that arrives early can cancel the save
of a later one and store its older text last, and the note keeps missing the last characters typed.

Discriminates: passes on dev 176d31d1d; in a frontend copy, each test fails when its behaviour
is cut: each toolbar button toggling another mark or block, the Formatting switch inverted, the
generated title dropped, the word count stuck at zero, the created label losing its "Today at"
and the undo button doing nothing. The note has no saved indicator and no edited-time label in
this build, so neither is tested.
"""

from __future__ import annotations

import re
import time
from datetime import datetime

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.actors import Actor
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

TITLE_PROMPT = "Generate a concise title summarizing the content"


def _new_note(page: Page) -> str:
    page.goto("/notes")
    page.get_by_role("main").get_by_role("button", name="Create", exact=True).click()
    expect(page).to_have_url(re.compile(r"/notes/[0-9a-f-]+$"))
    return page.url.rsplit("/", 1)[1]


def _editor(page: Page) -> Locator:
    return page.get_by_role("main").get_by_label("Write something...")


def _stored_note(author: Actor, note_id: str) -> dict:
    with author.client() as client:
        return client.get(f"/api/v1/notes/{note_id}").json()


def _wait_until_stored(author: Actor, note_id: str, field: str, expected: str) -> None:
    deadline = time.monotonic() + 15
    stored = _stored_note(author, note_id)["data"]["content"].get(field)
    while stored != expected and time.monotonic() < deadline:
        time.sleep(0.2)
        stored = _stored_note(author, note_id)["data"]["content"].get(field)
    assert stored == expected, f"the note stored {field} {stored!r} instead of {expected!r}"


def _type_and_select_line(page: Page, text: str) -> None:
    _editor(page).click()
    page.keyboard.type(text)
    page.keyboard.press("Shift+Home")


def _toolbar_button(page: Page, tooltip: str) -> Locator:
    toolbar = page.locator("#bubble-menu")
    expect(toolbar).to_be_visible()
    return tooltip_button(toolbar, tooltip)


def _open_note_menu(page: Page) -> Locator:
    # the "..." button has no name; it is the last menu button in the editor's header
    page.get_by_role("main").locator("[aria-haspopup=true]:visible").last.click()
    return page.get_by_role("menu")


# ---------------------------------------------------------------- the formatting toolbar


@pytest.mark.parametrize(
    "tooltip, tag, stored_md",
    [
        ("Bold", "strong", "**bold words**"),
        ("Italic", "em", "_bold words_"),
        ("Underline", "u", "<u>bold words</u>"),
        ("Strikethrough", "s", "~~bold words~~"),
    ],
    ids=["bold", "italic", "underline", "strikethrough"],
)
def test_the_toolbar_marks_selected_text(page_for, make_user, tooltip, tag, stored_md):
    author = make_user()
    page = page_for(author)
    note_id = _new_note(page)
    _type_and_select_line(page, "bold words")

    _toolbar_button(page, tooltip).click()

    expect(_editor(page).locator(tag)).to_have_text("bold words")
    _wait_until_stored(author, note_id, "md", stored_md)
    assert f"<{tag}>bold words</{tag}>" in _stored_note(author, note_id)["data"]["content"]["html"]
    page.reload()
    expect(_editor(page).locator(tag)).to_have_text("bold words")


@pytest.mark.parametrize(
    "tooltip, block, stored_md",
    [
        ("H1", "h1", "# a line"),
        ("H2", "h2", "## a line"),
        ("H3", "h3", "### a line"),
        ("Bullet List", "ul:not([data-type]) li p", "*   a line"),
        ("Ordered List", "ol li p", "1.  a line"),
        ("Task List", "ul[data-type=taskList] li p", "- [ ] a line"),
        ("Code Block", "pre > code", "```\na line\n```"),
    ],
    ids=["h1", "h2", "h3", "bullet-list", "ordered-list", "task-list", "code-block"],
)
def test_the_toolbar_turns_a_line_into_a_block(page_for, make_user, tooltip, block, stored_md):
    author = make_user()
    page = page_for(author)
    note_id = _new_note(page)
    _type_and_select_line(page, "a line")

    _toolbar_button(page, tooltip).click()

    expect(_editor(page).locator(block)).to_have_text("a line")
    _wait_until_stored(author, note_id, "md", stored_md)
    page.reload()
    expect(_editor(page).locator(block)).to_have_text("a line")


# ---------------------------------------------------------------- the Formatting switch


def _set_formatting(page: Page, on: bool) -> None:
    menu = _open_note_menu(page)
    switch = menu.get_by_role("switch", name="Formatting")
    if (switch.get_attribute("aria-checked") == "true") != on:
        switch.click()
    expect(switch).to_have_attribute("aria-checked", "true" if on else "false")
    page.keyboard.press("Escape")
    expect(menu).to_have_count(0)


def test_typed_markdown_becomes_formatting_while_the_switch_is_on(page_for, make_user):
    author = make_user()
    page = page_for(author)
    note_id = _new_note(page)
    _set_formatting(page, on=True)

    _editor(page).click()
    page.keyboard.type("## Heading")
    page.keyboard.press("Enter")
    page.keyboard.type("some **bold**")

    expect(_editor(page).locator("h2")).to_have_text("Heading")
    expect(_editor(page).locator("strong")).to_have_text("bold")
    _wait_until_stored(author, note_id, "md", "## Heading\n\nsome **bold**")


def test_typed_markdown_stays_as_typed_while_the_switch_is_off(page_for, make_user):
    author = make_user()
    page = page_for(author)
    note_id = _new_note(page)
    _set_formatting(page, on=False)

    _editor(page).click()
    page.keyboard.type("## Heading")
    page.keyboard.press("Enter")
    page.keyboard.type("some **bold**")

    expect(_editor(page)).to_have_text("## Headingsome **bold**")
    expect(_editor(page).locator("h2, strong")).to_have_count(0)
    _wait_until_stored(author, note_id, "html", "<p>## Heading</p><p>some **bold**</p>")


# ---------------------------------------------------------------- the title and what sits under it


def test_the_generate_button_asks_the_model_for_a_title(page_for, make_user, upstream):
    author = make_user()
    page = page_for(author)
    note_id = _new_note(page)
    _editor(page).click()
    page.keyboard.type("buy oat milk and rye bread")
    _wait_until_stored(author, note_id, "md", "buy oat milk and rye bread")
    upstream.queue(reply.text('{"title": "Weekly Shopping"}', match=reply.answering(TITLE_PROMPT)))

    title = page.get_by_role("main").get_by_role("textbox", name="Title")
    title.click()
    tooltip_button(page.get_by_role("main"), "Generate").click()

    expect(title).to_have_value("Weekly Shopping")
    deadline = time.monotonic() + 15
    while _stored_note(author, note_id)["title"] != "Weekly Shopping":
        assert time.monotonic() < deadline, "the generated title was never stored"
        time.sleep(0.2)
    [request] = [body for body in upstream.chat_requests() if reply.answering(TITLE_PROMPT)(body)]
    assert "buy oat milk and rye bread" in str(request["messages"][-1]["content"])
    page.reload()
    expect(title).to_have_value("Weekly Shopping")


def test_the_word_and_character_count_follows_the_typing(page_for, make_user):
    author = make_user()
    page = page_for(author)
    _new_note(page)
    main = page.get_by_role("main")
    expect(main.get_by_text("0 words")).to_be_visible()
    expect(main.get_by_text("0 characters")).to_be_visible()

    _editor(page).click()
    page.keyboard.type("one two three")
    expect(main.get_by_text("3 words")).to_be_visible()
    expect(main.get_by_text("13 characters")).to_be_visible()

    page.keyboard.type(" four")
    expect(main.get_by_text("4 words")).to_be_visible()
    expect(main.get_by_text("18 characters")).to_be_visible()


def test_a_note_made_today_shows_when_it_was_created(page_for, make_user):
    author = make_user()
    page = page_for(author)
    note_id = _new_note(page)

    created_at = _stored_note(author, note_id)["created_at"]
    made = datetime.fromtimestamp(created_at / 1e9)
    expected = f"Today at {made.strftime('%I:%M %p').lstrip('0')}"
    expect(page.get_by_role("main").get_by_text(expected, exact=True)).to_be_visible()


# ---------------------------------------------------------------- undo and redo


def test_undo_takes_back_the_typing_and_redo_brings_it_back(page_for, make_user):
    author = make_user()
    page = page_for(author)
    _new_note(page)
    main = page.get_by_role("main")
    # the undo and redo buttons have neither a name nor a tooltip, only their arrows
    undo = main.locator('button:has(path[d^="M9 15 3 9"])')
    redo = main.locator('button:has(path[d^="m15 15 6-6"])')
    expect(redo).to_be_disabled()

    _editor(page).click()
    page.keyboard.type("typed by mistake")
    expect(_editor(page)).to_have_text("typed by mistake")
    expect(undo).to_be_enabled()
    undo.click()

    expect(_editor(page)).not_to_contain_text("typed by mistake")
    expect(redo).to_be_enabled()
    redo.click()

    expect(_editor(page)).to_have_text("typed by mistake")
