"""Regression: a large paste in a channel with "Paste Large Text as File" on was lost.

open-webui issue #31365, fixed by PR #31366: with Settings > Interface "Paste Large Text as File"
on, pasting more than 1,000 characters into the channel input or a thread reply neither landed in
the box nor became an attachment. The paste now turns into a `Pasted_Text_<timestamp>.txt` file
that posts with the message. A short paste stays text in the box, and with the setting off a long
one does too.

The text goes through the browser clipboard and Control+V, as a person pastes it.

Discriminates: passes on the dev a5bc78300 build; with the paste handler of the channel input
back to reading only files, the long paste leaves neither an attachment nor text in the box.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.channel_quotes import enable_channels, group_channel, post_message
from utils.channel_messages import message_action
from utils.chat_ui import chat_input

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

LONG_TEXT = "".join(f"line {number} of the long log\n" for number in range(60))
SHORT_TEXT = "just a short note"
CLIPBOARD = ["clipboard-read", "clipboard-write"]


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
    page = page_for(account, path=f"/channels/{channel_id}", permissions=CLIPBOARD)
    expect(chat_input(page)).to_be_visible()
    return page


def open_thread(page: Page, text: str) -> Locator:
    root = page.locator("[id^='message-']:not(#message-input-container)").filter(has_text=text)
    message_action(root.first, "Reply in Thread").click()
    return page.get_by_label("Reply to thread...")


def paste(page: Page, box: Locator, text: str) -> None:
    page.evaluate("(text) => navigator.clipboard.writeText(text)", text)
    box.click()
    page.keyboard.press("Control+V")


def pasted_file(page: Page) -> Locator:
    return page.get_by_text("Pasted_Text_", exact=False)


def stored(account, channel_id: str) -> list[dict]:
    with account.client() as client:
        listed = client.get(f"/api/v1/channels/{channel_id}/messages")
    listed.raise_for_status()
    return listed.json()


def attached_texts(account, channel_id: str) -> list[str]:
    texts = []
    with account.client() as client:
        for message in stored(account, channel_id):
            url = f"/api/v1/channels/{channel_id}/messages/{message['id']}/data"
            for attached in (client.get(url).json() or {}).get("files") or []:
                texts.append(client.get(f"/api/v1/files/{attached['id']}/content").text)
    return texts


def test_a_long_paste_in_the_channel_becomes_a_file_that_posts_with_the_message(
    channel, make_user, page_for
):
    account = member_with(make_user, largeTextAsFile=True)
    channel_id = group_channel(account)
    page = open_channel(page_for, account, channel_id)
    box = chat_input(page)

    paste(page, box, LONG_TEXT)

    expect(pasted_file(page), "#31365: the long paste was not attached").to_be_visible()
    assert box.inner_text().strip() == "", "the long paste was also left in the box"
    expect(page.locator("#message-input-container .spinner_ajPY")).to_have_count(0)
    box.click()
    page.keyboard.type("the full log")
    page.keyboard.press("Enter")
    expect(page.get_by_text("the full log")).to_be_visible()
    [message] = stored(account, channel_id)
    assert message["content"] == "the full log"
    assert attached_texts(account, channel_id) == [LONG_TEXT]


def test_a_long_paste_in_a_thread_reply_becomes_a_file(channel, make_user, page_for):
    account = member_with(make_user, largeTextAsFile=True)
    channel_id = group_channel(account)
    post_message(account, channel_id, "where is the log?")
    page = open_channel(page_for, account, channel_id)
    reply_box = open_thread(page, "where is the log?")

    paste(page, reply_box, LONG_TEXT)

    expect(
        pasted_file(page), "#31365: the long paste in the thread was not attached"
    ).to_be_visible()
    assert reply_box.inner_text().strip() == "", "the long paste was also left in the reply box"


def test_a_short_paste_stays_text_in_the_box(channel, make_user, page_for):
    account = member_with(make_user, largeTextAsFile=True)
    page = open_channel(page_for, account, group_channel(account))
    box = chat_input(page)

    paste(page, box, SHORT_TEXT)

    expect(box).to_have_text(SHORT_TEXT)
    expect(pasted_file(page)).to_have_count(0)


def test_with_the_setting_off_a_long_paste_stays_text_in_the_box(channel, make_user, page_for):
    account = member_with(make_user, largeTextAsFile=False)
    page = open_channel(page_for, account, group_channel(account))
    box = chat_input(page)

    paste(page, box, LONG_TEXT)

    expect(box).to_contain_text("line 59 of the long log")
    expect(pasted_file(page)).to_have_count(0)
