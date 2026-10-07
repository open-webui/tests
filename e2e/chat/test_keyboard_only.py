"""Journey: the main chat flow, the settings dialog and the search with the keyboard alone.

On a desktop, with Tab, Shift+Tab, Enter, Escape and the arrow keys only, and the focused element
checked at every step. The chat opens with the focus in the message box and Enter sends; the
reply's buttons follow the conversation in the Tab order, so Shift+Tab reaches Regenerate, whose
menu takes the focus, tries again from Try Again and gives the focus back on Escape, and Copy puts
the reply on the clipboard. The model selector is reached with Tab and opened with Enter into its
search field; the arrow keys and Enter pick a model and leave the focus in the message box. Skip
to main content, reached with Shift+Tab and shown once focused, leads to the main content.
Settings open from the user menu with Enter, start with the focus on Back, keep it inside the
dialog however often Tab is pressed, open a tab with Enter and close on Escape. Ctrl+K puts the
focus in the search field and Escape gives it back to the message box.

Discriminates: passes on the ebc6add67 build apart from the settings focus test (red, see below). In
frontend builds of ebc6add67: with the dialogs' focus trap dropped the Tab cycle and Ctrl+K tests go
red; with dropdown menus leaving the focus on their trigger the two Regenerate tests and both
settings tests; with Enter in the message box sending nothing every chat test; with Copy copying
nothing the Copy test; with Enter in the model search picking nothing the selector test; and with
the skip link pointing nowhere the skip link test.

The settings focus test is red on dev ebc6add67: Settings opened from the user menu hand the focus
back on closing to the menu's Settings entry, which is gone by then, so the focus falls to the page
itself and the next Tab starts over at the top of the page.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.python_tools import EVERYONE_READS
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import chat_input, conversation, expect_reply

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

# far more presses than the dialog has stops, so a leak out of it would show
TAB_PRESSES = 120

FOCUSED = """() => {
    const element = document.activeElement;
    if (!element || element === document.body) return 'the page itself';
    return element.getAttribute('aria-label') || element.innerText.trim().split('\\n')[0]
        || element.getAttribute('placeholder') || element.id || element.tagName;
}"""


def focused(page: Page) -> str:
    return page.evaluate(FOCUSED)


def press_until_focused(page: Page, key: str, target: Locator, limit: int = 40) -> None:
    """Press `key` until `target` holds the focus, as someone tabbing towards it does."""
    for _ in range(limit):
        if target.evaluate("(element) => element === document.activeElement"):
            return
        page.keyboard.press(key)
    expect(target).to_be_focused()


@pytest.fixture
def keyboard_page(page_for, make_user) -> Page:
    page = page_for(make_user(), permissions=["clipboard-read", "clipboard-write"])
    expect(chat_input(page)).to_be_focused()
    return page


def ask_by_keyboard(page: Page, upstream, question: str, answer: str) -> None:
    upstream.queue(reply.text(answer, match=reply.answering(question)))
    expect(chat_input(page)).to_be_focused()
    page.keyboard.type(question)
    page.keyboard.press("Enter")
    expect_reply(page, answer)


def reply_button(page: Page, name: str) -> Locator:
    whole_reply = (
        conversation(page)
        .locator(".chat-assistant")
        .last.locator("xpath=ancestor::*[starts-with(@id, 'message-')][1]")
    )
    return whole_reply.locator(f"button[aria-label='{name}']")


# --------------------------------------------------------------------------- the chat


def test_enter_sends_from_the_message_box_and_leaves_the_focus_there(keyboard_page, upstream):
    question = f"kb question {uuid.uuid4().hex[:6]}"
    ask_by_keyboard(keyboard_page, upstream, question, "Answered without a mouse.")

    expect(chat_input(keyboard_page)).to_be_focused()
    expect(chat_input(keyboard_page)).to_be_empty()
    expect(keyboard_page).to_have_url(re.compile(r"/c/"))


def test_shift_tab_reaches_regenerate_and_try_again_adds_a_version(keyboard_page, upstream):
    question = f"name a lake {uuid.uuid4().hex[:6]}"
    ask_by_keyboard(keyboard_page, upstream, question, "Lake Baikal.")
    upstream.queue(reply.text("Lake Titicaca.", match=reply.answering(question)))

    press_until_focused(keyboard_page, "Shift+Tab", reply_button(keyboard_page, "Regenerate"))
    keyboard_page.keyboard.press("Enter")
    menu = keyboard_page.get_by_role("menu")
    expect(menu).to_be_focused()
    try_again = menu.get_by_role("button", name="Try Again")
    press_until_focused(keyboard_page, "Tab", try_again, limit=5)
    keyboard_page.keyboard.press("Enter")

    expect_reply(keyboard_page, "Lake Titicaca.")
    expect(conversation(keyboard_page).get_by_text("2/2")).to_be_visible()


def test_escape_closes_the_regenerate_menu_and_gives_the_focus_back(keyboard_page, upstream):
    ask_by_keyboard(keyboard_page, upstream, f"say hi {uuid.uuid4().hex[:6]}", "Hi.")
    regenerate = reply_button(keyboard_page, "Regenerate")
    press_until_focused(keyboard_page, "Shift+Tab", regenerate)
    keyboard_page.keyboard.press("Enter")
    expect(keyboard_page.get_by_role("menu")).to_be_focused()

    keyboard_page.keyboard.press("Escape")

    expect(keyboard_page.get_by_role("menu")).to_have_count(0)
    expect(regenerate).to_be_focused()


def test_enter_on_copy_puts_the_reply_on_the_clipboard(keyboard_page, upstream):
    ask_by_keyboard(keyboard_page, upstream, f"a motto {uuid.uuid4().hex[:6]}", "Slow and steady.")

    press_until_focused(keyboard_page, "Shift+Tab", reply_button(keyboard_page, "Copy"))
    keyboard_page.keyboard.press("Enter")

    keyboard_page.wait_for_function("navigator.clipboard.readText().then(text => text !== '')")
    assert keyboard_page.evaluate("navigator.clipboard.readText()") == "Slow and steady."


@pytest.fixture
def preset(admin) -> str:
    name = f"Cartographer {uuid.uuid4().hex[:6]}"
    form = {
        "id": f"carto-{uuid.uuid4().hex[:8]}",
        "base_model_id": MOCK_MODEL_ID,
        "name": name,
        "meta": {},
        "params": {},
        "access_grants": [EVERYONE_READS],
    }
    with admin.client() as client:
        client.post("/api/v1/models/create", json=form).raise_for_status()
        yield name
        client.post("/api/v1/models/model/delete", json={"id": form["id"]})


def test_the_model_selector_is_reached_with_tab_and_a_model_picked_with_the_keys(
    keyboard_page, preset
):
    keyboard_page.reload()
    expect(chat_input(keyboard_page)).to_be_focused()
    selector = keyboard_page.get_by_role("button", name=f"Selected model: {MOCK_MODEL_ID}")
    press_until_focused(keyboard_page, "Tab", selector)
    keyboard_page.keyboard.press("Enter")
    search = keyboard_page.get_by_role("textbox", name="Search In Models")
    expect(search).to_be_focused()

    keyboard_page.keyboard.type(preset)
    option = keyboard_page.get_by_role("option", name=f"Select {preset} model")
    expect(option).to_be_visible()
    keyboard_page.keyboard.press("ArrowDown")
    keyboard_page.keyboard.press("ArrowUp")
    keyboard_page.keyboard.press("Enter")

    expect(keyboard_page.get_by_role("button", name=f"Selected model: {preset}")).to_be_visible()
    expect(chat_input(keyboard_page)).to_be_focused()


def test_skip_to_main_content_leads_to_the_main_content(page_for, make_user):
    page = page_for(make_user())
    expect(chat_input(page)).to_be_focused()
    skip = page.get_by_role("link", name="Skip to main content")
    press_until_focused(page, "Shift+Tab", skip, limit=80)
    expect(skip).to_be_visible()

    page.keyboard.press("Enter")

    expect(page).to_have_url(re.compile(r"#main-content$"))
    expect(page.get_by_role("main")).to_have_attribute("id", "main-content")


# --------------------------------------------------------------------------- settings


def open_settings_from_the_user_menu(page: Page) -> Locator:
    user_menu = page.get_by_role("navigation", name="Chat history").locator(
        "button[aria-label='User menu']"
    )
    press_until_focused(page, "Shift+Tab", user_menu)
    page.keyboard.press("Enter")
    menu = page.get_by_role("menu")
    expect(menu).to_be_focused()
    press_until_focused(page, "Tab", menu.get_by_role("button", name="Settings"), limit=10)
    page.keyboard.press("Enter")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("button", name="Back", exact=True)).to_be_focused()
    return settings


def test_settings_open_from_the_user_menu_and_keep_the_focus_inside(keyboard_page):
    settings = open_settings_from_the_user_menu(keyboard_page)

    interface = settings.get_by_role("tab", name="Interface", exact=True)
    press_until_focused(keyboard_page, "Tab", interface, limit=10)
    keyboard_page.keyboard.press("Enter")
    expect(interface).to_have_attribute("aria-selected", "true")
    expect(keyboard_page.locator("#tab-interface")).to_be_visible()

    for _ in range(TAB_PRESSES):
        keyboard_page.keyboard.press("Tab")
        inside = keyboard_page.evaluate("() => !!document.activeElement?.closest('[role=dialog]')")
        assert inside, f"Tab left the settings dialog for {focused(keyboard_page)!r}"

    keyboard_page.keyboard.press("Escape")
    expect(settings).to_be_hidden()


def test_closing_settings_opened_from_the_user_menu_keeps_the_keyboard_focus(keyboard_page):
    settings = open_settings_from_the_user_menu(keyboard_page)

    keyboard_page.keyboard.press("Escape")
    expect(settings).to_be_hidden()

    assert focused(keyboard_page) != "the page itself", (
        "closing Settings dropped the keyboard focus onto the page itself"
    )


# --------------------------------------------------------------------------- the search


def test_ctrl_k_focuses_the_search_field_and_escape_returns_to_the_message_box(keyboard_page):
    keyboard_page.keyboard.press("Control+K")
    search = keyboard_page.get_by_role("dialog").get_by_placeholder("Search")
    expect(search).to_be_focused()

    keyboard_page.keyboard.press("Escape")

    expect(search).to_have_count(0)
    expect(chat_input(keyboard_page)).to_be_focused()
