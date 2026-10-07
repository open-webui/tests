"""Journey: Arabic, Persian and Hebrew interfaces, and right-to-left text among left-to-right text.

Picking a right-to-left language in Settings > General relabels the app in that language and
keeps every part of it on screen: the page never scrolls sideways, and the user menu, the
Settings dialog and the model selector open inside the window. The interface itself keeps its
left-to-right frame (the chat direction setting, e2e/chat/test_chat_direction.py, is what turns
the chat around). Text decides its own side: in a reply an Arabic paragraph and an Arabic list
sit on the right and an English paragraph under them on the left, a code block reads left to
right even with the chat direction set to RTL, and a chat titled in Arabic reads right to left in
the sidebar. Mixed lines in the user's own message are e2e/chat/test_mixed_direction_lines.py.

Discriminates: passes on the dev ebc6add67 build; in a frontend copy whose markdown paragraphs
and lists lose `dir="auto"` the reply tests go red (the Arabic text sits on the left), dropping
`dir="ltr"` from the code block turns the code test red, dropping `dir="auto"` from the sidebar
entry turns the title test red, and a stylesheet that widens the chat input past the window for
a right-to-left language turns the on-screen tests red.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.actors import Actor
from utils.chat_ui import chat_input, expect_reply, last_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

# per language: the sidebar's landmark name, its New Chat label and the user menu's label
RTL_LANGUAGES = {
    "ar": ("سجل المحادثات", "دردشة جديدة", "قائمة المستخدم"),
    "fa-IR": ("Chat history", "گپ جدید", "منوی کاربر"),
    "he-IL": ("Chat history", "צ'אט חדש", "User menu"),
}
ARABIC = "مرحبا بالعالم كيف حالك اليوم هل أنت بخير وهل كل شيء على ما يرام"
ENGLISH = "The tide turns at noon and the harbour opens after it."

# where the text of an element is drawn against the element's own box
TEXT_GAPS = """
element => {
    const text = document.createRange();
    text.selectNodeContents(element);
    const drawn = text.getBoundingClientRect();
    const box = element.getBoundingClientRect();
    return {left: drawn.left - box.left, right: box.right - drawn.right};
}
"""

OUTSIDE_THE_WINDOW = """
element => {
    const box = element.getBoundingClientRect();
    return box.left < -1 || box.right > window.innerWidth + 1;
}
"""


def pick_language(page: Page, code: str) -> None:
    page.goto("/?settings=general")
    settings = page.get_by_role("dialog")
    # found by what it offers, since its own label changes with the language
    settings.locator("select").filter(has=page.locator("option[value='en-US']")).select_option(code)
    expect(page.locator("html")).to_have_attribute("lang", code)
    page.keyboard.press("Escape")
    expect(settings).to_be_hidden()


def scrolls_sideways(page: Page) -> bool:
    return page.evaluate("document.documentElement.scrollWidth > window.innerWidth")


def assert_on_screen(locator: Locator, what: str) -> None:
    expect(locator).to_be_visible()
    assert not locator.evaluate(OUTSIDE_THE_WINDOW), f"{what} reaches past the window"


@pytest.mark.parametrize("code", list(RTL_LANGUAGES))
def test_a_right_to_left_language_relabels_the_app_and_keeps_it_on_screen(
    page_for, make_user, code
):
    page = page_for(make_user())
    pick_language(page, code)
    landmark, new_chat, user_menu = RTL_LANGUAGES[code]
    sidebar = page.get_by_role("navigation", name=landmark, exact=True)

    expect(sidebar.get_by_role("link", name=new_chat)).to_be_visible()
    assert not scrolls_sideways(page), "the chat page scrolls sideways"
    assert_on_screen(chat_input(page), "the chat input")

    sidebar.get_by_label(user_menu, exact=True).click()
    assert_on_screen(page.get_by_role("menu"), "the user menu")
    page.keyboard.press("Escape")

    page.goto("/?settings=interface")
    assert_on_screen(page.get_by_role("dialog"), "the Settings dialog")
    assert not scrolls_sideways(page), "the Settings dialog scrolls the page sideways"
    page.keyboard.press("Escape")

    page.get_by_role("main").locator("[aria-haspopup='listbox']").first.click()
    assert_on_screen(page.get_by_role("listbox"), "the model selector")


def ask(page: Page, upstream, question: str, answer: str, last_words: str) -> None:
    upstream.queue(reply.text(answer, match=reply.answering(question)))
    send(page, question)
    expect_reply(page, last_words)


def text_gaps(locator: Locator) -> dict:
    return locator.evaluate(TEXT_GAPS)


def test_an_arabic_paragraph_sits_right_and_an_english_one_left_in_a_reply(
    page_for, make_user, upstream
):
    page = page_for(make_user())
    ask(page, upstream, "say it in both", f"{ARABIC}\n\n{ENGLISH}", "harbour opens")
    answer = last_reply(page)

    arabic = text_gaps(answer.locator("p", has_text=ARABIC))
    english = text_gaps(answer.locator("p", has_text=ENGLISH))

    assert arabic["right"] < arabic["left"], "the Arabic paragraph sits on the left"
    assert english["left"] < english["right"], "the English paragraph sits on the right"


def test_an_arabic_list_reads_right_to_left_and_an_english_one_left_to_right(
    page_for, make_user, upstream
):
    page = page_for(make_user())
    answer_text = "- مرحبا\n- كيف حالك\n\nthen\n\n- first point\n- second point"
    ask(page, upstream, "list it in both", answer_text, "second point")
    answer = last_reply(page)

    expect(answer.locator("ul", has_text="مرحبا")).to_have_css("direction", "rtl")
    expect(answer.locator("ul", has_text="first point")).to_have_css("direction", "ltr")


def set_chat_direction(account: Actor, direction: str) -> None:
    with account.client() as client:
        saved = client.post(
            "/api/v1/users/user/settings/update", json={"ui": {"chatDirection": direction}}
        )
    saved.raise_for_status()


def test_a_code_block_reads_left_to_right_in_a_right_to_left_chat(page_for, make_user, upstream):
    account = make_user()
    set_chat_direction(account, "RTL")
    page = page_for(account)
    answer_text = f"{ARABIC}\n\n```python\nprint('tide')\n```\n\nانتهى"
    ask(page, upstream, "show the code", answer_text, "انتهى")
    answer = last_reply(page)

    expect(answer.locator("p", has_text=ARABIC)).to_have_css("direction", "rtl")
    expect(answer.locator(".cm-content")).to_have_css("direction", "ltr")


def test_a_chat_titled_in_arabic_reads_right_to_left_in_the_sidebar(page_for, make_user):
    account = make_user()
    title = f"{ARABIC[:24]} {uuid.uuid4().hex[:4]}"
    with account.client() as client:
        created = client.post("/api/v1/chats/new", json={"chat": {"title": title, "messages": []}})
    created.raise_for_status()
    page = page_for(account)

    entry = page.get_by_text(title)

    expect(entry).to_be_visible()
    expect(entry).to_have_css("direction", "rtl")
