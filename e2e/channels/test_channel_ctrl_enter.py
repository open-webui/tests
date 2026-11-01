"""The channel input ignored "Ctrl+Enter to Send", open-webui/open-webui#31318.

Fix commit `06bc8cb91` (PR open-webui/open-webui#31319). The chat input honours the setting,
but the channel and thread inputs always sent on Enter, so a line break could only be typed with
Shift+Enter. With the setting on, Enter now starts a new line there and Ctrl+Enter sends.

Discriminates: passes on the efe63bd34 build, fails on it with `06bc8cb91` reverted (Enter sends
the first line on its own).
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.channel_quotes import enable_channels, group_channel, post_message
from utils.channel_messages import message_action
from utils.chat_ui import chat_input

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def channel(admin, preserve):
    preserve("admin_config")
    enable_channels(admin)


def member_with(make_user, **ui):
    account = make_user()
    with account.client() as client:
        client.post("/api/v1/users/user/settings/update", json={"ui": ui}).raise_for_status()
    return account


def open_channel(page_for, account, channel_id: str) -> Page:
    page = page_for(account, path=f"/channels/{channel_id}")
    expect(chat_input(page)).to_be_visible()
    return page


def messages(page: Page) -> Locator:
    return page.locator("[id^='message-']")


def posted(account, channel_id: str) -> list[str]:
    with account.client() as client:
        listed = client.get(f"/api/v1/channels/{channel_id}/messages")
    listed.raise_for_status()
    return [message["content"] for message in listed.json()]


def type_two_lines(page: Page, box: Locator) -> None:
    box.click()
    page.keyboard.type("first line")
    page.keyboard.press("Enter")
    page.keyboard.type("second line")


def test_enter_starts_a_new_line_and_ctrl_enter_sends(channel, make_user, page_for):
    account = member_with(make_user, ctrlEnterToSend=True)
    channel_id = group_channel(account)
    page = open_channel(page_for, account, channel_id)

    type_two_lines(page, chat_input(page))
    page.wait_for_timeout(500)  # bounded: an Enter that sends would have posted by now
    assert posted(account, channel_id) == []
    page.keyboard.press("Control+Enter")

    expect(messages(page).filter(has_text="second line").first).to_be_visible()
    [content] = posted(account, channel_id)
    assert "first line" in content and "second line" in content


def test_ctrl_enter_to_send_holds_in_a_thread_reply(channel, make_user, page_for):
    account = member_with(make_user, ctrlEnterToSend=True)
    channel_id = group_channel(account)
    post_message(account, channel_id, "where do we meet?")
    page = open_channel(page_for, account, channel_id)
    question = messages(page).filter(has_text="where do we meet?").first
    message_action(question, "Reply in Thread").click()
    reply_box = page.get_by_label("Reply to thread...")

    type_two_lines(page, reply_box)
    page.wait_for_timeout(500)  # bounded: an Enter that sends would have posted by now
    expect(page.get_by_text("second line")).to_be_visible()
    expect(question.get_by_role("button", name="1 Replies")).to_have_count(0)
    page.keyboard.press("Control+Enter")

    expect(question.get_by_role("button", name="1 Replies")).to_be_visible()


def test_without_the_setting_enter_sends(channel, make_user, page_for):
    account = member_with(make_user, ctrlEnterToSend=False)
    channel_id = group_channel(account)
    page = open_channel(page_for, account, channel_id)

    chat_input(page).click()
    page.keyboard.type("just one line")
    page.keyboard.press("Enter")

    expect(messages(page).filter(has_text="just one line").first).to_be_visible()
    assert posted(account, channel_id) == ["just one line"]
