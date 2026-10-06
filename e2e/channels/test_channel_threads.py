"""Journey: a thread as the members who read it at the same time see it.

Three fresh accounts share a group channel. A reply one of them types in a thread shows up live
in the open thread panel of another member, and a third member watching the channel sees the
parent's reply count go up with each reply. Editing a reply marks it "(edited)" in the other
member's open thread, and deleting one takes it out of that thread and lowers the count on the
parent. When the author deletes the parent itself, the thread panel of a member reading it
closes. A member who is elsewhere in the app gets a toast for a thread reply that opens the
channel with that thread already open.

Discriminates: passes on dev ebc6add67; in a frontend copy, a thread panel that ignores new
messages from the socket turns the live reply test red, a thread panel that ignores updates turns
the edit step red, a channel view that ignores the parent's reply event turns the count steps
red, a thread panel that stays open when its parent is deleted turns the parent test red and a
toast that drops the thread from its link turns the toast test red.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.channel_quotes import delete_message, enable_channels, group_channel, post_message
from utils.chat_ui import chat_input
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def people(admin, preserve, make_user):
    """The author, a member and a watcher of one group channel, with its id."""
    preserve("admin_config")
    enable_channels(admin)
    author, member, watcher = make_user(), make_user(), make_user()
    return author, member, watcher, group_channel(author, member, watcher)


def _open_channel(page_for, account, channel_id: str) -> Page:
    page = page_for(account)
    page.goto(f"/channels/{channel_id}")
    expect(chat_input(page)).to_be_visible()
    return page


def _in_channel(page: Page, message_id: str) -> Locator:
    return page.locator(f"[id='message-{message_id}']").first


def _in_thread(page: Page, parent_id: str, text: str) -> Locator:
    """A message of the open thread panel, whose rows carry the parent's id as a prefix."""
    return page.locator(f"[id^='message-{parent_id}-']").filter(has_text=text).first


def _open_thread(page: Page, parent_id: str) -> None:
    parent = _in_channel(page, parent_id)
    parent.hover()
    tooltip_button(parent, "Reply in Thread").click()
    expect(_thread_input(page)).to_be_visible()


def _thread_input(page: Page) -> Locator:
    return page.get_by_label("Reply to thread...")


def _reply_in_thread(page: Page, text: str) -> None:
    _thread_input(page).click()
    page.keyboard.type(text)
    page.keyboard.press("Enter")


def _reply_count(page: Page, parent_id: str, count: int) -> Locator:
    return _in_channel(page, parent_id).get_by_role("button", name=f"{count} Replies")


def _channel_name(account, channel_id: str) -> str:
    with account.client() as client:
        fetched = client.get(f"/api/v1/channels/{channel_id}")
    fetched.raise_for_status()
    return fetched.json()["name"]


def test_a_reply_reaches_another_members_open_thread_and_the_count_on_the_parent(people, page_for):
    author, member, watcher, channel_id = people
    parent_id = post_message(author, channel_id, "who can lend a tent?")
    author_page = _open_channel(page_for, author, channel_id)
    member_page = _open_channel(page_for, member, channel_id)
    watcher_page = _open_channel(page_for, watcher, channel_id)
    _open_thread(author_page, parent_id)
    _open_thread(member_page, parent_id)

    _reply_in_thread(member_page, "mine sleeps four")

    expect(_in_thread(author_page, parent_id, "mine sleeps four")).to_be_visible()
    expect(_reply_count(watcher_page, parent_id, 1)).to_be_visible()
    _reply_in_thread(author_page, "perfect, I will pick it up")
    expect(_in_thread(member_page, parent_id, "perfect, I will pick it up")).to_be_visible()
    expect(_reply_count(watcher_page, parent_id, 2)).to_be_visible()
    expect(_in_channel(watcher_page, parent_id)).not_to_contain_text("mine sleeps four")

    _reply_count(watcher_page, parent_id, 2).click()
    expect(_in_thread(watcher_page, parent_id, "mine sleeps four")).to_be_visible()
    expect(_in_thread(watcher_page, parent_id, "perfect, I will pick it up")).to_be_visible()


def test_an_edited_and_a_deleted_thread_reply_reach_the_other_members(people, page_for):
    author, member, watcher, channel_id = people
    parent_id = post_message(member, channel_id, "dinner plans for friday")
    post_message(author, channel_id, "I bring the salad", parent_id=parent_id)
    post_message(author, channel_id, "I can bake a cake", parent_id=parent_id)
    author_page = _open_channel(page_for, author, channel_id)
    member_page = _open_channel(page_for, member, channel_id)
    watcher_page = _open_channel(page_for, watcher, channel_id)
    expect(_reply_count(watcher_page, parent_id, 2)).to_be_visible()
    _open_thread(author_page, parent_id)
    _open_thread(member_page, parent_id)

    salad = _in_thread(author_page, parent_id, "I bring the salad")
    salad.hover()
    tooltip_button(salad, "Edit").click()
    author_page.locator(f"[id^='message-{parent_id}-'] textarea").fill("I bring the bread")
    author_page.get_by_role("button", name="Save").click()

    expect(_in_thread(member_page, parent_id, "I bring the bread")).to_contain_text("(edited)")
    expect(member_page.get_by_text("I bring the salad")).to_have_count(0)

    cake = _in_thread(author_page, parent_id, "I can bake a cake")
    cake.hover()
    tooltip_button(cake, "Delete").click()
    author_page.get_by_role("dialog").get_by_role("button", name="Confirm").click()

    expect(member_page.get_by_text("I can bake a cake")).to_have_count(0)
    expect(_reply_count(watcher_page, parent_id, 1)).to_be_visible()
    expect(_in_thread(member_page, parent_id, "I bring the bread")).to_be_visible()


def test_deleting_the_parent_closes_the_thread_for_a_member_reading_it(people, page_for):
    author, member, _, channel_id = people
    parent_id = post_message(author, channel_id, "carpool to the lake")
    post_message(member, channel_id, "I have two free seats", parent_id=parent_id)
    post_message(author, channel_id, "and the weather is fine")
    member_page = _open_channel(page_for, member, channel_id)
    _open_thread(member_page, parent_id)
    expect(_in_thread(member_page, parent_id, "I have two free seats")).to_be_visible()

    delete_message(author, channel_id, parent_id)

    expect(_thread_input(member_page)).to_have_count(0)
    expect(member_page.get_by_text("carpool to the lake")).to_have_count(0)
    expect(member_page.get_by_text("and the weather is fine")).to_be_visible()


def test_a_thread_reply_toast_opens_the_channel_on_that_thread(people, page_for):
    author, member, _, channel_id = people
    parent_id = post_message(member, channel_id, "anyone up for a hike?")
    member_page = page_for(member)
    expect(chat_input(member_page)).to_be_visible()

    post_message(author, channel_id, "count me in for sunday", parent_id=parent_id)

    toast = member_page.get_by_text(f"{author.name} (#{_channel_name(author, channel_id)})")
    expect(toast).to_be_visible()
    toast.click()
    expect(member_page).to_have_url(re.compile(f"/channels/{channel_id}"))
    expect(_thread_input(member_page)).to_be_visible()
    expect(_in_thread(member_page, parent_id, "count me in for sunday")).to_be_visible()
