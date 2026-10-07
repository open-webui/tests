"""Journey: writing a note on a phone, from the drawer to the stored note and back to the list.

On a 390 by 844 touch screen Notes opens from the sidebar drawer, and Create opens a new note whose
title field, editor and header buttons lie on the screen without scrolling sideways. A title and
text typed there are stored, the note is listed on the Notes page and opens again from it with its
text. The note's menu fits the screen, and so do its Download choices.

Discriminates: passes on the ebc6add67 build apart from the Download choices test (red, see below).
In a backend copy whose live note edits are never written to the note the writing test goes red; in
a frontend build whose phone layout is 480 pixels wide (wider than the screen, so controls on the
right fall off it) the menu test goes red too.

The Download choices test is red on dev ebc6add67: on a screen this narrow the choices open beside
the menu, past the left edge of the screen, with their names cut off
(open-webui/open-webui#32015).
"""

from __future__ import annotations

import re
import time
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.actors import Actor
from utils.chat_ui import chat_input
from utils.phone import PHONE, SCREEN, expect_on_screen, expect_reachable, tap_on_screen

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

NOTE_URL = re.compile(r"/notes/[0-9a-f-]+$")
# a person's typing pace; keystrokes a few milliseconds apart can overtake each other's saves
TYPING_DELAY_MS = 100


@pytest.fixture
def author(make_user) -> Actor:
    return make_user()


@pytest.fixture
def phone(page_for, author) -> Page:
    page = page_for(author, **PHONE)
    expect_on_screen(chat_input(page))
    return page


def open_notes(page: Page) -> None:
    tap_on_screen(page.get_by_role("button", name="Open Sidebar").last)
    sidebar = page.get_by_role("navigation", name="Chat history")
    tap_on_screen(sidebar.get_by_role("link", name="Notes"))
    expect(page).to_have_url(re.compile(r"/notes$"))


def new_note(page: Page) -> str:
    open_notes(page)
    tap_on_screen(page.get_by_role("main").get_by_role("button", name="Create", exact=True))
    expect(page).to_have_url(NOTE_URL)
    return page.url.rsplit("/", 1)[1]


def editor(page: Page) -> Locator:
    return page.get_by_role("main").get_by_label("Write something...")


def stored_note(author: Actor, note_id: str) -> dict:
    with author.client() as client:
        return client.get(f"/api/v1/notes/{note_id}").json()


def wait_until_stored(author: Actor, note_id: str, text: str) -> dict:
    deadline = time.monotonic() + 15
    note = stored_note(author, note_id)
    while text not in str(note["data"]["content"].get("md")) and time.monotonic() < deadline:
        time.sleep(0.2)
        note = stored_note(author, note_id)
    assert text in str(note["data"]["content"].get("md")), note["data"]
    return note


def expect_no_sideways_scroll(page: Page) -> None:
    scroll_width = page.evaluate("document.documentElement.scrollWidth")
    assert scroll_width <= SCREEN["width"], f"the page scrolls sideways ({scroll_width}px wide)"


def open_note_menu(page: Page) -> Locator:
    # the "..." has no name; its menu trigger wraps the icon's padded box
    tap_on_screen(page.get_by_role("main").locator("[aria-haspopup=true] > div.p-1"))
    return page.get_by_role("menu")


def test_a_note_written_on_a_phone_is_stored_and_listed(phone, author):
    title = f"Packing list {uuid.uuid4().hex[:6]}"
    note_id = new_note(phone)
    title_field = phone.get_by_role("main").get_by_placeholder("Title")
    expect_on_screen(title_field)
    expect_on_screen(editor(phone))
    expect_no_sideways_scroll(phone)

    title_field.fill(title)
    tap_on_screen(editor(phone))
    phone.keyboard.type("Rain jacket and a torch", delay=TYPING_DELAY_MS)
    note = wait_until_stored(author, note_id, "Rain jacket and a torch")
    assert note["title"] == title, note["title"]

    open_notes(phone)
    listed = phone.get_by_role("main").get_by_text(title).first
    expect_reachable(listed)
    listed.tap()
    expect(phone).to_have_url(re.compile(note_id))
    expect_on_screen(editor(phone).get_by_text("Rain jacket and a torch"))


def test_the_note_menu_fits_the_screen(phone):
    new_note(phone)
    tap_on_screen(editor(phone))
    phone.keyboard.type("Something to keep")

    menu = open_note_menu(phone)
    for entry in ("Download", "Share", "Delete"):
        expect_on_screen(menu.get_by_role("button", name=entry).first)


def test_the_download_choices_of_the_note_menu_fit_the_screen(phone):
    new_note(phone)
    tap_on_screen(editor(phone))
    phone.keyboard.type("Something to keep")
    download = open_note_menu(phone).get_by_role("button", name="Download")
    expect_on_screen(download)

    # the click of a tap, without the emulated pointer leaving the trigger afterwards
    download.dispatch_event("click")

    choices = phone.get_by_role("menu").last.get_by_role("button")
    expect(choices.first).to_be_visible()
    for index in range(choices.count()):
        expect_on_screen(
            choices.nth(index),
            "the Download choices open past the edge of the screen (open-webui/open-webui#32015)",
        )
