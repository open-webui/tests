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

Discriminates: passes on the ba34b1c47 build. In frontend builds of ebc6add67: with the dialogs'
focus trap dropped the Tab cycle and Ctrl+K tests go red; with Enter in the message box sending
nothing every chat test; with Copy copying nothing the Copy test; and with Enter in the model search
picking nothing the selector test. In a frontend build of 5f6399eb6: with dropdown menus leaving the
focus on their trigger the two Regenerate tests and both settings tests go red, and with the skip
link pointing nowhere the skip link test.

The settings focus test was red on dev ebc6add67: Settings opened from the user menu handed the
focus back on closing to the menu's Settings entry, which is gone by then, so the focus fell to the
page itself and the next Tab started over at the top of the page (open-webui/open-webui#32017, fixed
by c7e7dbf98).

The skip link test was red from b29f7781a: the skip link reset the new chat, which wiped the typed
text and put the focus back in the message box (open-webui/open-webui#33396, fixed on dev by
3814b7b2f; the page needs open-webui/open-webui#33459 to set itself up on a fresh load).

The five chat tests (Enter sends, the Regenerate menu with Try Again and Escape, Copy, the model
selector) run in the main chat and in a side chat (`utils/chat_surface.py`); the skip link,
sign-in page, settings and search tests are about the page and stay as they were. In a side chat
the Regenerate button is missing on dev c15f28649 (open-webui/open-webui#34014, fix PR
open-webui/open-webui#34064). Picking a model with the keys leaves the focus off a side chat's
message box on both builds: the selector hands the focus to the element with the id `chat-input`,
which is the main chat's box (open-webui/open-webui#34194, fix PR open-webui/open-webui#34195).

On the sign-in page, Skip to main content moves the focus off the link (421b6947b, PR
open-webui/open-webui#33684) and the show password button is at least 24 pixels wide
(0fd01ff24, PR #33685). Both pass on dev e8c0ad85e and fail on that build with their fix
reverted (the link keeps the focus; the button is 22 pixels wide).
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.python_tools import EVERYONE_READS
from harness.upstream import MOCK_MODEL_ID
from utils.chat_surface import ChatSurface
from utils.chat_ui import chat_input, conversation, expect_reply
from utils.clipboard import wait_for_text
from utils.model_selector import model_search
from utils.user_menu import settings_item

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


def press_until_focused(
    chat: Page | ChatSurface, key: str, target: Locator, limit: int = 40
) -> None:
    """Press `key` until `target` holds the focus, as someone tabbing towards it does."""
    page = chat.page if isinstance(chat, ChatSurface) else chat
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


@pytest.fixture
def keyboard_chat(chat_for, make_user) -> ChatSurface:
    chat = chat_for(make_user(), permissions=["clipboard-read", "clipboard-write"])
    expect(chat_input(chat)).to_be_focused()
    return chat


def ask_by_keyboard(chat: ChatSurface, upstream, question: str, answer: str) -> None:
    upstream.queue(reply.text(answer, match=reply.answering(question)))
    expect(chat_input(chat)).to_be_focused()
    chat.page.keyboard.type(question)
    chat.page.keyboard.press("Enter")
    expect_reply(chat, answer)


def expect_focus_inside(page: Page, menu: Locator) -> None:
    """The open menu holds the keyboard focus, on itself or on one of its items."""
    expect(menu.and_(page.locator(":focus-within"))).to_be_visible()


def reply_button(chat: ChatSurface, name: str) -> Locator:
    whole_reply = (
        conversation(chat)
        .locator(".chat-assistant")
        .last.locator("xpath=ancestor::*[starts-with(@id, 'message-')][1]")
    )
    return whole_reply.locator(f"button[aria-label='{name}']")


# --------------------------------------------------------------------------- the chat


def test_enter_sends_from_the_message_box_and_leaves_the_focus_there(keyboard_chat, upstream):
    question = f"kb question {uuid.uuid4().hex[:6]}"
    ask_by_keyboard(keyboard_chat, upstream, question, "Answered without a mouse.")

    expect(chat_input(keyboard_chat)).to_be_focused()
    expect(chat_input(keyboard_chat)).to_be_empty()
    expect(keyboard_chat.page).to_have_url(re.compile(r"/c/"))


def test_shift_tab_reaches_regenerate_and_try_again_adds_a_version(keyboard_chat, upstream):
    """In a side chat: open-webui/open-webui#34014."""
    question = f"name a lake {uuid.uuid4().hex[:6]}"
    ask_by_keyboard(keyboard_chat, upstream, question, "Lake Baikal.")
    upstream.queue(reply.text("Lake Titicaca.", match=reply.answering(question)))

    press_until_focused(keyboard_chat, "Shift+Tab", reply_button(keyboard_chat, "Regenerate"))
    keyboard_chat.page.keyboard.press("Enter")
    menu = keyboard_chat.page.get_by_role("menu")
    expect_focus_inside(keyboard_chat.page, menu)
    try_again = menu.get_by_role("menuitem", name="Try Again")
    press_until_focused(keyboard_chat, "Tab", try_again, limit=5)
    keyboard_chat.page.keyboard.press("Enter")

    expect_reply(keyboard_chat, "Lake Titicaca.")
    expect(conversation(keyboard_chat).get_by_text("2/2")).to_be_visible()


def test_escape_closes_the_regenerate_menu_and_gives_the_focus_back(keyboard_chat, upstream):
    """In a side chat: open-webui/open-webui#34014."""
    ask_by_keyboard(keyboard_chat, upstream, f"say hi {uuid.uuid4().hex[:6]}", "Hi.")
    regenerate = reply_button(keyboard_chat, "Regenerate")
    press_until_focused(keyboard_chat, "Shift+Tab", regenerate)
    keyboard_chat.page.keyboard.press("Enter")
    expect_focus_inside(keyboard_chat.page, keyboard_chat.page.get_by_role("menu"))

    keyboard_chat.page.keyboard.press("Escape")

    expect(keyboard_chat.page.get_by_role("menu")).to_have_count(0)
    expect(regenerate).to_be_focused()


def test_enter_on_copy_puts_the_reply_on_the_clipboard(keyboard_chat, upstream):
    ask_by_keyboard(keyboard_chat, upstream, f"a motto {uuid.uuid4().hex[:6]}", "Slow and steady.")

    press_until_focused(keyboard_chat, "Shift+Tab", reply_button(keyboard_chat, "Copy"))
    keyboard_chat.page.keyboard.press("Enter")

    assert wait_for_text(keyboard_chat.page) == "Slow and steady."


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
    preset, keyboard_chat
):
    """In a side chat: open-webui/open-webui#34194, fix PR open-webui/open-webui#34195."""
    expect(chat_input(keyboard_chat)).to_be_focused()
    selector = keyboard_chat.button(f"Selected model: {MOCK_MODEL_ID}")
    press_until_focused(keyboard_chat, "Tab", selector)
    keyboard_chat.page.keyboard.press("Enter")
    search = model_search(keyboard_chat.page)
    expect(search).to_be_focused()

    keyboard_chat.page.keyboard.type(preset)
    option = keyboard_chat.page.get_by_role("option", name=f"Select {preset} model")
    expect(option).to_be_visible()
    keyboard_chat.page.keyboard.press("ArrowDown")
    keyboard_chat.page.keyboard.press("ArrowUp")
    keyboard_chat.page.keyboard.press("Enter")

    expect(keyboard_chat.button(f"Selected model: {preset}")).to_be_visible()
    expect(chat_input(keyboard_chat)).to_be_focused()


def test_skip_to_main_content_leads_to_the_main_content(page_for, make_user):
    page = page_for(make_user())
    expect(chat_input(page)).to_be_focused()
    skip = page.get_by_role("link", name="Skip to main content")
    press_until_focused(page, "Shift+Tab", skip, limit=80)
    expect(skip).to_be_visible()

    page.keyboard.press("Enter")

    expect(page).to_have_url(re.compile(r"#main-content$"))
    expect(page.get_by_role("main").locator("#main-content")).to_be_focused()


def test_skip_to_main_content_on_the_sign_in_page_leads_to_the_sign_in_form(page):
    # the sign-in page had no main content to skip to, so the focus stayed on the link (#33684)
    page.goto("/auth")
    email = page.get_by_label("Email")
    expect(email).to_be_visible()
    skip = page.get_by_role("link", name="Skip to main content")
    press_until_focused(page, "Tab", skip, limit=10)
    expect(skip).to_be_visible()

    page.keyboard.press("Enter")

    expect(skip, "the skip link kept the focus").not_to_be_focused()
    page.keyboard.press("Tab")
    expect(email, "the next Tab after the skip link missed the sign-in form").to_be_focused()


def test_the_show_password_button_on_the_sign_in_page_is_wide_enough_to_hit(page):
    page.goto("/auth")
    show = page.get_by_role("button", name=re.compile(r"^(Show|Make) password", re.IGNORECASE))
    expect(show).to_be_visible()

    box = show.bounding_box()

    # WCAG 2.2 asks for a target of at least 24 by 24 pixels (#33685)
    assert box and box["width"] >= 24, f"the show password button is {box} wide"


# --------------------------------------------------------------------------- settings


def open_settings_from_the_user_menu(page: Page) -> Locator:
    user_menu = page.get_by_role("navigation", name="Chat history").locator(
        "button[aria-label='User menu']"
    )
    press_until_focused(page, "Shift+Tab", user_menu)
    page.keyboard.press("Enter")
    menu = page.get_by_role("menu")
    expect_focus_inside(page, menu)
    press_until_focused(page, "Tab", settings_item(page), limit=10)
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

    # Escape on a control showing its tooltip only hides the tooltip, so leave from Back
    back = settings.get_by_role("button", name="Back", exact=True)
    press_until_focused(keyboard_page, "Tab", back, limit=TAB_PRESSES)
    keyboard_page.keyboard.press("Escape")
    expect(settings).to_be_hidden()


def test_closing_settings_opened_from_the_user_menu_keeps_the_keyboard_focus(keyboard_page):
    settings = open_settings_from_the_user_menu(keyboard_page)

    keyboard_page.keyboard.press("Escape")
    expect(settings).to_be_hidden()

    assert focused(keyboard_page) != "the page itself", (
        "closing Settings dropped the keyboard focus onto the page itself "
        "(open-webui/open-webui#32017)"
    )


# --------------------------------------------------------------------------- the search


def test_ctrl_k_focuses_the_search_field_and_escape_returns_to_the_message_box(keyboard_page):
    keyboard_page.keyboard.press("Control+K")
    search = keyboard_page.get_by_role("dialog").get_by_placeholder("Search")
    expect(search).to_be_focused()

    keyboard_page.keyboard.press("Escape")

    expect(search).to_have_count(0)
    expect(chat_input(keyboard_page)).to_be_focused()
