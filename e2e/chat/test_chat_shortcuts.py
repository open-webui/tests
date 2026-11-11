"""Journey: the default keyboard shortcuts of the chat page do what the shortcut list says.

Ctrl stands in for Cmd on Linux. In an answered chat: Ctrl+Shift+O starts a new chat, Ctrl+Shift+S
opens and closes the sidebar, Ctrl+R in the message box regenerates the last reply, Ctrl+Shift+C
copies it, Ctrl+Shift+Backspace deletes the chat after its confirm dialog, Ctrl+Shift+M opens the
model selector, Ctrl+. the settings, Ctrl+/ the shortcut list and Ctrl+Shift+' a temporary chat.
With a second chat answered later in a tab beside it, Ctrl+Shift+C in a chat's message box copies
that chat's reply, in the main chat and in a side chat (`utils/chat_surface.py`); red on dev
c15f28649 and e791cbdde, where it copies the other tab's (issue draft pending). Shift+Escape puts
the focus back in the message box, Escape there stops a reply still streaming
and keeps what came so far, and ArrowUp in the empty message box opens the last question for
editing. Bound to keys of the account's choosing, the unassigned Navigate to previous and next
chat step through the sidebar's chats and Toggle Controls opens and closes the side panel.
With Keyboard Shortcuts switched off in the settings the sidebar, model selector and new
chat shortcuts do nothing.

`test_escape_in_the_message_box_stops_a_streaming_reply` is red on dev 1c010b438: since de73bb830 a
stopped reply never ends on the page, so the Stop button stays (open-webui/open-webui#32081, fixed
by d12bf062b).

Discriminates: passes on dev 30f3f6a8f; in a frontend build whose keydown handler ignores every
chord but Ctrl+K every test but the switched-off one fails, and in a backend copy whose settings
update drops the switch that one fails. In a frontend build of ebc6add67 whose
message box ignores Escape and ArrowUp, and whose Shift+Escape focuses nothing, the three message
box tests fail; in one whose chat navigation and Toggle Controls shortcuts do nothing, the two
bound shortcut tests fail. The Toggle Controls test was retargeted for 2d7cf5ec9, which moved the
chat's Controls into the Model controls menu and points the shortcut at the side panel: it passes
on dev d22c4449c and fails in a build whose Toggle Controls does nothing. Since 5f1c6384e the
panel stays closed when it has no tab to show, so that test opens it as an admin, whose answered
chat offers a side chat. Since 4615d30a6 the panel opens on its empty + menu instead of
starting a side chat; retargeted, the test passes on dev c165ee14f. Since 617618c9e a window
768px or wider always shows the tabs, so the test runs in a narrower one.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.chat import ask
from utils.chat_surface import MAIN_CHAT, SIDE_CHAT, main_chat, open_new_chat_tab
from utils.chat_ui import (
    chat_input,
    conversation,
    expect_reply,
    last_reply,
    replies,
    send,
    stop_button,
)
from utils.clipboard import wait_for_text

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

QUESTION = "Name a mountain in Peru"
ANSWER = "Huascaran is the highest."
LATER_QUESTION = "Name a lake in Peru"
LATER_ANSWER = "Titicaca is the largest."
NARROW_WINDOW = {"width": 700, "height": 900}


@pytest.fixture
def account(make_user):
    return make_user()


@pytest.fixture
def answered(page_for, account, upstream) -> Page:
    """An answered chat, open with the cursor outside the message box."""
    page = page_for(account, permissions=["clipboard-read", "clipboard-write"])
    upstream.queue(reply.text(ANSWER, match=reply.answering(QUESTION)))
    send(page, QUESTION)
    expect_reply(page, ANSWER)
    expect(page).to_have_url(re.compile(r"/c/"))
    page.get_by_text(ANSWER).click()
    return page


def _open_sidebar_button(page: Page):
    """Shown on the narrow rail while the sidebar is closed."""
    return page.get_by_role("button", name="Open Sidebar", exact=True)


def test_ctrl_shift_o_starts_a_new_chat(answered):
    answered.keyboard.press("Control+Shift+O")

    expect(answered).not_to_have_url(re.compile(r"/c/"))
    expect(replies(answered)).to_have_count(0)
    expect(chat_input(answered)).to_be_empty()


def test_ctrl_shift_s_opens_and_closes_the_sidebar(answered):
    expect(_open_sidebar_button(answered)).to_be_visible()
    answered.keyboard.press("Control+Shift+S")
    expect(_open_sidebar_button(answered)).to_be_hidden()
    expect(answered.get_by_role("button", name="Chats", exact=True)).to_be_visible()
    answered.keyboard.press("Control+Shift+S")
    expect(_open_sidebar_button(answered)).to_be_visible()


def test_ctrl_r_in_the_message_box_regenerates_the_last_reply(answered, upstream):
    upstream.queue(reply.text("Alpamayo is the prettiest.", match=reply.answering(QUESTION)))
    chat_input(answered).click()
    answered.keyboard.press("Control+R")

    expect(replies(answered).last).to_contain_text("Alpamayo is the prettiest.")
    expect(conversation(answered).get_by_text("2/2")).to_be_visible()


def test_ctrl_shift_c_copies_the_last_reply(answered):
    answered.keyboard.press("Control+Shift+C")

    assert wait_for_text(answered) == ANSWER


@pytest.fixture
def side_chats_allowed(admin, preserve):
    preserve("permissions")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["chat"].update(side_chat=True)
        saved = client.post("/api/v1/users/default/permissions", json=permissions)
    assert saved.status_code == 200, saved.text


# a tab answered later is the other chat: next to the main chat, or next to a side chat
@pytest.mark.chat_surfaces(MAIN_CHAT, SIDE_CHAT)
def test_ctrl_shift_c_in_a_message_box_copies_that_chats_reply_not_a_later_one_beside_it(
    chat_for, account, upstream, side_chats_allowed
):
    """Red on dev c15f28649 and e791cbdde in both places: the shortcut copies the last answer on
    the page, the other tab's (issue draft pending; PR open-webui/open-webui#34064 names it)."""
    chat = chat_for(account, permissions=["clipboard-read", "clipboard-write"])
    upstream.queue(reply.text(ANSWER, match=reply.answering(QUESTION)))
    send(chat, QUESTION)
    expect_reply(chat, ANSWER)
    main_chat(chat.page).show()
    other = open_new_chat_tab(chat.page)
    upstream.queue(reply.text(LATER_ANSWER, match=reply.answering(LATER_QUESTION)))
    send(other, LATER_QUESTION)
    expect_reply(other, LATER_ANSWER)

    chat.show()
    chat_input(chat).click()
    chat.page.keyboard.press("Control+Shift+C")

    copied = wait_for_text(chat.page)
    assert copied == ANSWER, f"the shortcut in the {chat.place}'s box copied {copied!r}"


def test_ctrl_shift_backspace_deletes_the_chat_after_confirming(answered, account):
    chat_id = answered.url.rsplit("/", 1)[-1]
    answered.keyboard.press("Control+Shift+Backspace")
    answered.get_by_role("dialog", name="Delete chat?").get_by_role(
        "button", name="Confirm"
    ).click()

    expect(answered).not_to_have_url(re.compile(chat_id))
    with account.client() as client:
        assert client.get(f"/api/v1/chats/{chat_id}").status_code != 200


def test_ctrl_shift_m_opens_the_model_selector(answered):
    answered.keyboard.press("Control+Shift+M")
    expect(answered.get_by_placeholder("Search a model")).to_be_visible()


def test_ctrl_period_opens_the_settings(answered):
    answered.keyboard.press("Control+.")
    expect(answered.get_by_role("dialog").get_by_role("tab", name="General")).to_be_visible()


def test_ctrl_slash_opens_the_shortcut_list(answered):
    answered.keyboard.press("Control+/")
    shortcuts = answered.get_by_role("dialog")
    expect(shortcuts.get_by_text("Copy Last Response")).to_be_visible()


def test_ctrl_shift_quote_starts_a_temporary_chat(answered):
    answered.keyboard.press("Control+Shift+'")

    expect(answered).not_to_have_url(re.compile(r"/c/"))
    expect(answered.get_by_text("Temporary Chat").first).to_be_visible()


def test_no_shortcut_answers_with_keyboard_shortcuts_switched_off(page_for, account, upstream):
    with account.client() as client:
        saved = client.post(
            "/api/v1/users/user/settings/update", json={"ui": {"keyboardShortcuts": False}}
        )
        assert saved.status_code == 200, saved.text
    page = page_for(account)
    upstream.queue(reply.text(ANSWER, match=reply.answering(QUESTION)))
    send(page, QUESTION)
    expect_reply(page, ANSWER)
    page.get_by_text(ANSWER).click()

    page.keyboard.press("Control+Shift+S")
    page.keyboard.press("Control+Shift+M")
    page.keyboard.press("Control+Shift+O")
    # a shortcut that answered would have opened the sidebar, the selector or a new chat by now
    expect(_open_sidebar_button(page)).to_be_visible()
    expect(page.get_by_placeholder("Search a model")).to_have_count(0)
    expect(page).to_have_url(re.compile(r"/c/"))
    expect(replies(page)).to_have_count(1)


def test_shift_escape_puts_the_focus_back_in_the_message_box(answered):
    expect(chat_input(answered)).not_to_be_focused()

    answered.keyboard.press("Shift+Escape")

    expect(chat_input(answered)).to_be_focused()


def test_escape_in_the_message_box_stops_a_streaming_reply(page_for, account, upstream):
    page = page_for(account)
    pieces = [f"step{number} " for number in range(1, 61)]
    upstream.queue(reply.text(pieces, chunk_delay=0.2, match=reply.answering("count the steps")))
    send(page, "count the steps")
    expect(last_reply(page)).to_contain_text("step3")
    expect(chat_input(page)).to_be_focused()

    page.keyboard.press("Escape")

    expect(stop_button(page)).to_have_count(0)
    shown = last_reply(page).inner_text()
    page.wait_for_timeout(1500)  # a reply still streaming would grow by several steps here
    assert last_reply(page).inner_text() == shown
    assert "step60" not in shown, shown


def test_arrow_up_in_the_empty_message_box_edits_the_last_question(answered):
    answered.keyboard.press("Shift+Escape")
    expect(chat_input(answered)).to_be_focused()

    answered.keyboard.press("ArrowUp")

    editor = conversation(answered).locator(".chat-user").last.locator("textarea")
    expect(editor).to_be_focused()
    expect(editor).to_have_value(QUESTION)


def bind(account, bindings: dict[str, str]) -> None:
    with account.client() as client:
        saved = client.post("/api/v1/users/user/settings/update", json={"keybindings": bindings})
    assert saved.status_code == 200, saved.text


def test_bound_chat_navigation_steps_through_the_sidebars_chats(page_for, account, upstream):
    bind(account, {"navigateChatUp": "Cmd+Shift+U", "navigateChatDown": "Cmd+Shift+J"})
    upstream.queue(
        reply.text("The older answer.", match=reply.answering("older question")),
        reply.text("The newer answer.", match=reply.answering("newer question")),
    )
    with account.client() as client:
        older, _ = ask(client, "older question")
        newer, _ = ask(client, "newer question")
    answers = {older.chat_id: "The older answer.", newer.chat_id: "The newer answer."}
    page = page_for(account)
    # two chats made within one second may be listed either way round
    chat_links = page.locator('a[href^="/c/"]')
    expect(chat_links).to_have_count(2)
    top_id, below_id = [link.get_attribute("href").removeprefix("/c/") for link in chat_links.all()]
    page.goto(f"/c/{top_id}")
    expect_reply(page, answers[top_id])

    page.keyboard.press("Control+Shift+J")
    expect(page).to_have_url(re.compile(below_id))
    expect_reply(page, answers[below_id])

    page.keyboard.press("Control+Shift+U")
    expect(page).to_have_url(re.compile(top_id))
    expect_reply(page, answers[top_id])


def test_bound_toggle_controls_opens_and_closes_the_side_panel(page_for, make_user, upstream):
    # the panel stays shut unless it has a tab to show or can start a side chat, as an admin can
    admin = make_user(role="admin")
    bind(admin, {"toggleControls": "Cmd+Shift+Y"})
    # a wide window always shows the tabs; below 768px they are a panel that opens and closes
    page = page_for(admin, viewport=NARROW_WINDOW)
    upstream.queue(reply.text(ANSWER, match=reply.answering(QUESTION)))
    send(page, QUESTION)
    expect_reply(page, ANSWER)
    # the header's Tabs button opens and closes the same panel
    tabs_button = page.get_by_role("button", name="Tabs", exact=True)
    hide_tabs = page.get_by_role("button", name="Hide tabs", exact=True)
    expect(tabs_button).to_have_attribute("aria-expanded", "false")
    expect(hide_tabs).to_be_hidden()

    page.keyboard.press("Control+Shift+Y")
    expect(hide_tabs).to_be_visible()
    expect(tabs_button).to_have_attribute("aria-expanded", "true")
    expect(page.get_by_role("button", name="New tab", exact=True)).to_be_visible()

    page.keyboard.press("Control+Shift+Y")
    expect(hide_tabs).to_be_hidden()
    expect(tabs_button).to_have_attribute("aria-expanded", "false")
