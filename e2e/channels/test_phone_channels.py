"""Journey: a channel on a phone, opened from the drawer, written in and answered by a swipe.

On a 390 by 844 touch screen a group channel is listed under Channels in the sidebar drawer once
that section is unfolded; tapping it opens the channel and shuts the drawer, with the message box
and its send button on the screen. A message sent with the send button shows on the screen for
the other member. A swipe to the right on someone's message quotes it in the message box
("Replying to ...") and leaves the drawer shut, so the reply can be typed at once.

Discriminates: passes on the ebc6add67 build apart from the swipe test (red, see below). In a
backend copy that sends a new channel message to no one live the sending test goes red; in a
frontend build whose channel messages ignore a swipe the swipe test fails at the quote as well.

The swipe test is red on dev ebc6add67: the same swipe also reaches the sidebar's swipe-to-open
gesture, so the drawer slides open over the channel and covers the quoted message
(open-webui/open-webui#32016).
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.channel_quotes import enable_channels, group_channel, post_message
from utils.chat_ui import chat_input
from utils.phone import (
    PHONE,
    expect_off_screen,
    expect_on_screen,
    swipe,
    tap_on_screen,
)

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def members(admin, preserve, make_user):
    preserve("admin_config")
    enable_channels(admin)
    return make_user(name=f"Phone {uuid.uuid4().hex[:6]}"), make_user(
        name=f"Desk {uuid.uuid4().hex[:6]}"
    )


def sidebar(page: Page) -> Locator:
    return page.get_by_role("navigation", name="Chat history")


def message(page: Page, text: str) -> Locator:
    return page.locator("[id^='message-']").filter(has_text=text).first


def phone_in_channel(page_for, member, channel_id: str) -> Page:
    page = page_for(member, **PHONE)
    page.goto(f"/channels/{channel_id}")
    expect_on_screen(chat_input(page))
    return page


def test_a_channel_opens_from_the_drawer_and_a_tapped_send_reaches_the_other_member(
    page_for, members
):
    on_phone, at_desk = members
    channel_id = group_channel(at_desk, on_phone)
    with at_desk.client() as client:
        name = client.get(f"/api/v1/channels/{channel_id}").json()["name"]
    phone = page_for(on_phone, **PHONE)
    expect_on_screen(chat_input(phone))
    desk = page_for(at_desk)
    desk.goto(f"/channels/{channel_id}")
    expect(chat_input(desk)).to_be_visible()

    tap_on_screen(phone.get_by_role("button", name="Open Sidebar").last)
    tap_on_screen(sidebar(phone).get_by_role("button", name="Channels", exact=True))
    tap_on_screen(sidebar(phone).get_by_role("link", name=re.compile(name)))
    expect(phone).to_have_url(re.compile(channel_id))
    expect_off_screen(sidebar(phone).get_by_role("link", name="New Chat"))

    tap_on_screen(chat_input(phone))
    phone.keyboard.type("on my way, ten minutes")
    tap_on_screen(phone.get_by_role("button", name="Send message"))

    expect_on_screen(message(phone, "on my way, ten minutes"))
    expect(message(desk, "on my way, ten minutes")).to_be_visible()


def test_a_swipe_on_a_message_quotes_it_and_leaves_the_drawer_shut(page_for, members):
    on_phone, at_desk = members
    channel_id = group_channel(at_desk, on_phone)
    post_message(at_desk, channel_id, "who has the spare key?")
    phone = phone_in_channel(page_for, on_phone, channel_id)
    asked = message(phone, "who has the spare key?")
    expect_on_screen(asked)

    box = asked.bounding_box()
    middle = box["y"] + box["height"] / 2
    swipe(phone, (60, middle), (220, middle))

    replying = phone.get_by_text(f"Replying to {at_desk.name}")
    expect(replying).to_be_visible()
    expect_off_screen(
        sidebar(phone).get_by_role("link", name="New Chat"),
        "the swipe to reply also opened the sidebar (open-webui/open-webui#32016)",
    )
    expect_on_screen(replying)
