"""Journey: Interface switches in Settings change what the chat shows and does.

Each switch is flipped in the Settings dialog by a fresh account, which then sees its effect in a
chat, still sees it after a reload where that is checked, and a second account in a browser of
its own does not.
Chat Bubble UI off labels the user's own message "You" and Display the Username puts the
account's name there instead; Rich Text Input off leaves typed markdown as plain characters; Copy
Formatted Text puts HTML on the clipboard next to the plain text; Title Auto-Generation off names
a new chat after its first message where the model would otherwise title it. Always Collapse Code
Blocks folds a reply's code block; Always Expand Details opens a reply's reasoning; Display Chat
Title in Tab off keeps the app name as the tab title; Insert Follow-Up Prompt to Input puts a
pressed follow-up in the input instead of sending it; Keep Follow-Up Prompts in Chat leaves an
earlier reply's follow-ups under it; Auto-Copy Response puts a finished reply on the clipboard;
Temporary Chat by Default starts a new chat temporary and unstored; Insert Suggestion Prompt to
Input fills the input from a suggestion instead of sending it; Show Formatting Toolbar shows the
toolbar over selected input text; Landing Page Mode Chat shows the conversation pane on a new
chat; Render Markdown in User Messages off shows a message as typed.

Discriminates: passes on dev 176d31d1d; in a frontend copy, reading `chatBubble` as always on
turns the bubble tests red, passing `richText` as always on turns the rich text test red, copying
a reply without the `copyFormatted` setting turns the copy test red, and sending
`title_generation` as always on turns the title test red. Also in a frontend copy, each setting
read as never set turns its own test red: `collapseCodeBlocks`, `expandDetails`,
`showChatTitleInTab` (always shown), `insertFollowUpPrompt`, `keepFollowUpPrompts`,
`responseAutoCopy`, `temporaryChatByDefault`, `insertSuggestionPrompt`, `showFormattingToolbar`,
`landingPageMode` and `renderMarkdownInUserMessages` (always rendered).
"""

from __future__ import annotations

import json
import re

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.chat_ui import chat_input, conversation, expect_reply, last_reply, replies, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

ANSWER = "The **lamp** is lit."
TITLE_TASK = "Generate a concise title"


def interface_switch(page: Page, name: str) -> Locator:
    page.goto("/?settings=interface")
    switch = page.locator("#tab-interface").get_by_role("switch", name=name, exact=True)
    expect(switch).to_be_visible()
    return switch


def flip(page: Page, name: str) -> None:
    switch = interface_switch(page, name)
    checked = switch.get_attribute("aria-checked")
    with page.expect_response(lambda response: "/user/settings/update" in response.url):
        switch.click()
    expect(switch).not_to_have_attribute("aria-checked", checked)


def chat_once(page: Page, upstream, question: str) -> None:
    upstream.queue(reply.text(ANSWER, match=reply.answering(question)))
    page.goto("/")
    send(page, question)
    expect_reply(page, "is lit")


def own_message_label(page: Page, text: str) -> Locator:
    return conversation(page).locator(".user-message").get_by_text(text, exact=True)


def test_chat_bubble_off_labels_the_users_message_you(page_for, make_user, upstream):
    page = page_for(make_user())
    flip(page, "Chat Bubble UI")

    chat_once(page, upstream, "Is the lamp lit?")
    expect(own_message_label(page, "You")).to_be_visible()

    page.reload()
    expect(own_message_label(page, "You")).to_be_visible()

    other = page_for(make_user())
    chat_once(other, upstream, "Is the lamp lit yet?")
    expect(own_message_label(other, "You")).to_have_count(0)


def test_display_the_username_puts_the_accounts_name_on_its_message(page_for, make_user, upstream):
    account = make_user(name="Keeper Brannigan")
    page = page_for(account)
    flip(page, "Chat Bubble UI")
    flip(page, "Display the Username Instead of You in the Chat")

    chat_once(page, upstream, "Who is on watch?")
    expect(own_message_label(page, "Keeper Brannigan")).to_be_visible()

    page.reload()
    expect(own_message_label(page, "Keeper Brannigan")).to_be_visible()
    expect(
        interface_switch(page, "Display the Username Instead of You in the Chat")
    ).to_have_attribute("aria-checked", "true")

    other = page_for(make_user(name="Keeper Other"))
    flip(other, "Chat Bubble UI")
    chat_once(other, upstream, "Who is on watch tonight?")
    expect(own_message_label(other, "You")).to_be_visible()


def type_markdown(page: Page) -> Locator:
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    chat_input(page).click()
    page.keyboard.type("**bold** words")
    return chat_input(page)


def test_rich_text_input_off_leaves_markdown_as_typed(page_for, make_user):
    page = page_for(make_user())
    flip(page, "Rich Text Input for Chat")

    composer = type_markdown(page)
    expect(composer).to_contain_text("**bold** words")
    expect(composer.locator("strong")).to_have_count(0)

    page.reload()
    composer = type_markdown(page)
    expect(composer).to_contain_text("**bold** words")

    other = page_for(make_user())
    composer = type_markdown(other)
    expect(composer.locator("strong")).to_have_text("bold")


def copy_reply(page: Page) -> tuple[str, str | None]:
    """The clipboard's types after the reply's Copy button, and its HTML (None when it has none)."""
    page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    last_reply(page).hover()
    conversation(page).get_by_role("button", name="Copy").last.click()
    expect(page.get_by_text("Copying to clipboard was successful!")).to_be_visible()
    return page.evaluate(
        """async () => {
            const [item] = await navigator.clipboard.read();
            const html = item.types.includes('text/html')
                ? await (await item.getType('text/html')).text()
                : null;
            return [item.types.join(','), html];
        }"""
    )


def test_copy_formatted_text_puts_html_on_the_clipboard(page_for, make_user, upstream):
    page = page_for(make_user())
    flip(page, "Copy Formatted Text")

    chat_once(page, upstream, "Is the lamp lit tonight?")
    types, html = copy_reply(page)
    assert "text/html" in types
    assert "<strong>lamp</strong>" in html

    page.reload()
    expect(interface_switch(page, "Copy Formatted Text")).to_have_attribute("aria-checked", "true")

    other = page_for(make_user())
    chat_once(other, upstream, "Is the lamp lit this evening?")
    types, html = copy_reply(other)
    assert "text/html" not in types
    assert html is None


@pytest.fixture
def title_generation_on(admin, preserve):
    preserve("tasks")
    with admin.client() as client:
        current = client.get("/api/v1/tasks/config").json()
        saved = client.post(
            "/api/v1/tasks/config/update", json={**current, "ENABLE_TITLE_GENERATION": True}
        )
    saved.raise_for_status()


def open_sidebar(page: Page) -> Locator:
    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    return page.get_by_role("navigation", name="Chat history")


def title_requests(upstream) -> list[dict]:
    return [body for body in upstream.chat_requests() if reply.answering(TITLE_TASK)(body)]


def offer_a_title(upstream) -> None:
    # queued first, so a title request that also quotes the question cannot take the chat's reply
    upstream.queue(reply.text('{"title": "Ferry Timetable"}', match=reply.answering(TITLE_TASK)))


def test_title_auto_generation_off_names_the_chat_after_its_first_message(
    page_for, make_user, upstream, title_generation_on
):
    page = page_for(make_user())
    flip(page, "Title Auto-Generation")

    question = "When does the ferry leave?"
    offer_a_title(upstream)
    chat_once(page, upstream, question)
    expect(open_sidebar(page).get_by_text(question)).to_be_visible()
    assert title_requests(upstream) == []

    page.reload()
    expect(interface_switch(page, "Title Auto-Generation")).to_have_attribute(
        "aria-checked", "false"
    )

    other = page_for(make_user())
    offer_a_title(upstream)
    chat_once(other, upstream, "When does the last ferry leave?")
    expect(open_sidebar(other).get_by_text("Ferry Timetable")).to_be_visible()


CLIPBOARD = ["clipboard-read", "clipboard-write"]
FOLLOW_UP_TASK = "Suggest 3-5 relevant follow-up questions"
LONG_ANSWER = "Here is the script:\n\n```python\nfirst = 1\nsecond = 2\n```\n\nThat is all."


def code_lines(page: Page) -> Locator:
    return last_reply(page).locator(".cm-line")


def test_always_collapse_code_blocks_shows_a_replys_code_collapsed(page_for, make_user, upstream):
    page = page_for(make_user())
    flip(page, "Always Collapse Code Blocks")

    upstream.queue(reply.text(LONG_ANSWER, match=reply.answering("Show the script")))
    page.goto("/")
    send(page, "Show the script")
    expect_reply(page, "That is all.")
    expect(last_reply(page).get_by_text("2 hidden lines")).to_be_visible()
    expect(code_lines(page)).to_have_count(0)

    page.reload()
    expect(last_reply(page).get_by_text("2 hidden lines")).to_be_visible()

    other = page_for(make_user())
    upstream.queue(reply.text(LONG_ANSWER, match=reply.answering("Show the script again")))
    other.goto("/")
    send(other, "Show the script again")
    expect_reply(other, "That is all.")
    expect(code_lines(other)).to_have_count(2)
    expect(last_reply(other).get_by_text("hidden lines")).to_have_count(0)


def ask_with_reasoning(page: Page, upstream, question: str) -> None:
    upstream.queue(
        reply.text(
            "The tide is out.", reasoning="checking the tide table", match=reply.answering(question)
        )
    )
    page.goto("/")
    send(page, question)
    expect_reply(page, "The tide is out.")


def test_always_expand_details_shows_the_reasoning_expanded(page_for, make_user, upstream):
    page = page_for(make_user())
    flip(page, "Always Expand Details")

    ask_with_reasoning(page, upstream, "Is the tide out?")
    expect(last_reply(page).get_by_text("checking the tide table")).to_be_visible()

    page.reload()
    expect(last_reply(page).get_by_text("checking the tide table")).to_be_visible()

    other = page_for(make_user())
    ask_with_reasoning(other, upstream, "Is the tide out now?")
    expect(last_reply(other).get_by_text("checking the tide table")).to_have_count(0)


def test_display_chat_title_in_tab_off_keeps_the_app_name_as_tab_title(
    page_for, make_user, upstream, title_generation_on
):
    page = page_for(make_user())
    flip(page, "Display Chat Title in Tab")

    offer_a_title(upstream)
    chat_once(page, upstream, "When does the first ferry leave?")
    expect(open_sidebar(page).get_by_text("Ferry Timetable")).to_be_visible()
    expect(page).not_to_have_title(re.compile("Ferry Timetable"))
    assert page.title() == "Open WebUI"

    other = page_for(make_user())
    offer_a_title(upstream)
    chat_once(other, upstream, "When does the second ferry leave?")
    expect(other).to_have_title("Ferry Timetable / Open WebUI")


@pytest.fixture
def follow_ups_on(admin, preserve):
    preserve("tasks")
    with admin.client() as client:
        current = client.get("/api/v1/tasks/config").json()
        saved = client.post(
            "/api/v1/tasks/config/update", json={**current, "ENABLE_FOLLOW_UP_GENERATION": True}
        )
    saved.raise_for_status()


def ask_with_follow_ups(page: Page, upstream, question: str, follow_ups: list[str]) -> None:
    # queued first, so the task request that quotes the question cannot take the chat's reply
    upstream.queue(
        reply.text(json.dumps({"follow_ups": follow_ups}), match=reply.answering(FOLLOW_UP_TASK)),
        reply.text(ANSWER, match=reply.answering(question)),
    )
    send(page, question)
    expect_reply(page, "is lit")


def follow_up_button(page: Page, question: str) -> Locator:
    return conversation(page).get_by_role("button", name=f"Follow up: {question}")


def test_insert_follow_up_prompt_puts_the_question_in_the_input_instead_of_sending(
    page_for, make_user, upstream, follow_ups_on
):
    page = page_for(make_user())
    flip(page, "Insert Follow-Up Prompt to Input")
    page.goto("/")
    ask_with_follow_ups(page, upstream, "Is the lamp lit?", ["Who lit it?", "When did it go out?"])

    follow_up_button(page, "When did it go out?").click()

    expect(chat_input(page)).to_have_text("When did it go out?")
    expect(replies(page)).to_have_count(1)
    sent = [
        body for body in upstream.chat_requests() if reply.answering("When did it go out?")(body)
    ]
    assert sent == []

    other = page_for(make_user())
    other.goto("/")
    ask_with_follow_ups(other, upstream, "Is the lamp lit yet?", ["Who lit it?", "Is it bright?"])
    upstream.queue(reply.text("Quite bright.", match=reply.answering("Is it bright?")))
    follow_up_button(other, "Is it bright?").click()
    expect(replies(other)).to_have_count(2)
    expect_reply(other, "Quite bright.")


def test_keep_follow_up_prompts_leaves_an_earlier_replys_follow_ups_in_the_chat(
    page_for, make_user, upstream, follow_ups_on
):
    page = page_for(make_user())
    flip(page, "Keep Follow-Up Prompts in Chat")
    page.goto("/")
    ask_with_follow_ups(page, upstream, "Is the lamp lit?", ["Who lit it?"])
    expect(follow_up_button(page, "Who lit it?")).to_be_visible()

    ask_with_follow_ups(page, upstream, "Is the lamp still lit?", ["Who tends it?"])

    expect(follow_up_button(page, "Who tends it?")).to_be_visible()
    expect(follow_up_button(page, "Who lit it?")).to_be_visible()

    other = page_for(make_user())
    other.goto("/")
    ask_with_follow_ups(other, upstream, "Is the lamp lit yet?", ["Who lit it?"])
    ask_with_follow_ups(other, upstream, "Is the lamp lit still?", ["Who tends it?"])
    expect(follow_up_button(other, "Who tends it?")).to_be_visible()
    expect(follow_up_button(other, "Who lit it?")).to_have_count(0)


def clipboard_text(page: Page) -> str:
    return page.evaluate("navigator.clipboard.readText()")


def clipboard_becomes(page: Page, expected: str) -> bool:
    """Whether the clipboard holds `expected` within ten seconds."""
    return page.evaluate(
        """async expected => {
            for (let attempt = 0; attempt < 100; attempt++) {
                if ((await navigator.clipboard.readText()) === expected) return true;
                await new Promise(resolve => setTimeout(resolve, 100));
            }
            return false;
        }""",
        expected,
    )


def test_auto_copy_response_puts_a_finished_reply_on_the_clipboard(page_for, make_user, upstream):
    page = page_for(make_user(), permissions=CLIPBOARD)
    flip(page, "Auto-Copy Response to Clipboard")
    page.goto("/")
    page.evaluate("navigator.clipboard.writeText('nothing copied')")

    chat_once(page, upstream, "Is the lamp lit tonight?")
    assert clipboard_becomes(page, ANSWER), (
        f"the reply never reached the clipboard: {clipboard_text(page)!r}"
    )

    other = page_for(make_user(), permissions=CLIPBOARD)
    other.goto("/")
    other.evaluate("navigator.clipboard.writeText('nothing copied')")
    chat_once(other, upstream, "Is the lamp lit this night?")
    other.wait_for_timeout(1000)
    assert clipboard_text(other) == "nothing copied"


def stored_chats(account, text: str) -> list[dict]:
    with account.client() as client:
        found = client.get("/api/v1/chats/search", params={"text": text})
    found.raise_for_status()
    return found.json()


def test_temporary_chat_by_default_starts_new_chats_temporary_and_unstored(
    page_for, make_user, upstream
):
    account = make_user()
    page = page_for(account)
    flip(page, "Temporary Chat by Default")

    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    expect(page.locator("#chat-pane").get_by_text("Temporary Chat", exact=True)).to_be_visible()
    chat_once(page, upstream, "Where is the harbour?")
    expect(page.locator('a[href^="/c/"]')).to_have_count(0)
    assert stored_chats(account, "harbour") == []

    other_account = make_user()
    other = page_for(other_account)
    other.goto("/")
    expect(chat_input(other)).to_be_visible()
    expect(other.locator("#chat-pane").get_by_text("Temporary Chat", exact=True)).to_have_count(0)
    chat_once(other, upstream, "Where is the other harbour?")
    expect(other).to_have_url(re.compile(r"/c/[\w-]+$"))
    assert len(stored_chats(other_account, "harbour")) == 1


@pytest.fixture
def one_suggestion(admin):
    with admin.client() as client:
        before = client.get("/api/config").json()
        ours = {
            "suggestions": [
                {"title": ["Tide table", "for today"], "content": "Read the tide table"}
            ]
        }
        saved = client.post("/api/v1/configs/suggestions", json=ours)
        saved.raise_for_status()
        yield
        restored = client.post(
            "/api/v1/configs/suggestions",
            json={
                "suggestions": before["default_prompt_suggestions"],
                "i18n": before.get("default_prompt_suggestions_i18n"),
            },
        )
    restored.raise_for_status()


def suggestion(page: Page) -> Locator:
    return page.get_by_role("listitem").filter(has_text="Tide table")


def test_insert_suggestion_prompt_fills_the_input_instead_of_sending(
    page_for, make_user, upstream, one_suggestion
):
    page = page_for(make_user())
    flip(page, "Insert Suggestion Prompt to Input")
    page.goto("/")

    suggestion(page).click()

    expect(chat_input(page)).to_have_text("Read the tide table")
    expect(replies(page)).to_have_count(0)
    assert upstream.chat_requests() == []

    other = page_for(make_user())
    upstream.queue(reply.text(ANSWER, match=reply.answering("Read the tide table")))
    other.goto("/")
    suggestion(other).click()
    expect_reply(other, "is lit")


def select_typed_text(page: Page) -> None:
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    chat_input(page).click()
    page.keyboard.type("select these words")
    page.keyboard.press("ControlOrMeta+a")


def test_show_formatting_toolbar_appears_over_selected_input_text(page_for, make_user):
    page = page_for(make_user())
    flip(page, "Show Formatting Toolbar")

    select_typed_text(page)
    expect(page.locator("#bubble-menu")).to_be_visible()

    page.reload()
    select_typed_text(page)
    expect(page.locator("#bubble-menu")).to_be_visible()

    other = page_for(make_user())
    select_typed_text(other)
    expect(other.locator("#bubble-menu")).to_have_count(0)


def test_landing_page_mode_chat_shows_the_conversation_pane_on_a_new_chat(page_for, make_user):
    page = page_for(make_user())
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    expect(page.locator("#messages-container")).to_have_count(0)

    page.goto("/?settings=interface")
    mode = page.locator("#tab-interface").get_by_role("button", name="Landing Page Mode")
    with page.expect_response(lambda response: "/user/settings/update" in response.url):
        mode.click()
    expect(mode).to_contain_text("Chat")

    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    expect(page.locator("#messages-container")).to_be_visible()

    other = page_for(make_user())
    other.goto("/")
    expect(chat_input(other)).to_be_visible()
    expect(other.locator("#messages-container")).to_have_count(0)


def ask_with_markdown(page: Page, upstream, question: str) -> Locator:
    upstream.queue(reply.text(ANSWER, match=reply.answering(question)))
    page.goto("/")
    send(page, question)
    expect_reply(page, "is lit")
    return conversation(page).locator(".user-message")


def test_render_markdown_in_user_messages_off_shows_the_message_as_typed(
    page_for, make_user, upstream
):
    page = page_for(make_user())
    flip(page, "Render Markdown in User Messages")

    message = ask_with_markdown(page, upstream, "Is the **lamp** lit?")
    expect(message).to_contain_text("Is the **lamp** lit?")
    expect(message.locator("strong")).to_have_count(0)

    page.reload()
    expect(conversation(page).locator(".user-message")).to_contain_text("Is the **lamp** lit?")

    other = page_for(make_user())
    message = ask_with_markdown(other, upstream, "Is the **lamp** lit yet?")
    expect(message.locator("strong")).to_have_text("lamp")
