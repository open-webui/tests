"""Journey: two people talk in a direct message, see the unread count, close it and reopen it.

A person starts a direct message from the "Message" button on someone's profile card in a
channel. The conversation lists in the sidebar's channels under the other person's name and what
is sent there reaches the other person live. While the other person is elsewhere the entry shows
an unread count that goes away once they open it, and stays away after a reload. Closing an entry
removes it from the sidebar for good, and messaging the person again brings back the same
conversation with its history. A third person who opens the conversation is sent home.
A closed conversation comes back by itself, with its unread count, when the other person writes
again. A group conversation started in the Create Channel modal with two people picked lists
for each of them under the other two names, and what one writes reaches both.

Discriminates: passes on dev 176d31d1d; in a frontend copy, leaving the unread count out of the
sidebar entry turns the unread test red, keeping the count after the entry is opened turns its
clearing step red, dropping the close button's request turns the close test red (the entry is
back after a reload), a profile card button that opens nothing turns the start test red, and
staying on the page when the conversation cannot be loaded turns the third person test red, and
creating the direct message with only the first person picked turns the group test red; in a
backend copy, leaving a closed member inactive when a message arrives turns the comes back test
red.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.channel_quotes import enable_channels, group_channel, post_message
from utils.chat_ui import chat_input, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def people(admin, preserve, make_user):
    """Two people who share a group channel, where the profile card offers the message button."""
    preserve("admin_config")
    enable_channels(admin)
    first, second = make_user(), make_user()
    return first, second, group_channel(first, second)


def _direct_message_id(actor, other) -> str:
    with actor.client() as client:
        opened = client.get(f"/api/v1/channels/users/{other.id}")
    opened.raise_for_status()
    return opened.json()["id"]


def _start_from_profile_card(page: Page, channel_id: str, author, text: str) -> None:
    page.goto(f"/channels/{channel_id}")
    message = page.locator("[id^='message-']").filter(has_text=text).first
    # the author's picture has no alt text, so it has no role to find it by
    message.locator(f"img[src$='/users/{author.id}/profile/image']").click()
    page.get_by_role("button", name="Message", exact=True).click()


def _channels_section(page: Page) -> Locator:
    """The sidebar, opened with its Channels section expanded."""
    expect(chat_input(page)).to_be_visible()
    sidebar = page.get_by_role("navigation", name="Chat history")
    open_sidebar = page.get_by_role("button", name="Open Sidebar", exact=True)
    if open_sidebar.is_visible():
        open_sidebar.click()
    section = sidebar.get_by_role("button", name="Channels")
    if section.get_attribute("aria-expanded") == "false":
        section.click()
    return sidebar


def _entry(page: Page, other_name: str) -> Locator:
    return _channels_section(page).get_by_role("link", name=other_name)


def _group_entry(page: Page, *others) -> Locator:
    """A group conversation's entry, listed under the other people's names in any order."""
    links = _channels_section(page).get_by_role("link")
    for other in others:
        links = links.filter(has_text=other.name)
    return links


def _message(page: Page, text: str) -> Locator:
    return page.locator("[id^='message-']").filter(has_text=text).first


def test_a_direct_message_started_from_a_profile_card_reaches_the_other_person(people, page_for):
    starter, other, channel_id = people
    post_message(other, channel_id, "who has the tide table?")
    starter_page, other_page = page_for(starter), page_for(other)

    _start_from_profile_card(starter_page, channel_id, other, "who has the tide table?")

    expect(starter_page).to_have_url(re.compile(r"/channels/(?!" + channel_id + ")"))
    send(starter_page, "I have it, low tide is at six")
    expect(_message(starter_page, "I have it, low tide is at six")).to_be_visible()
    direct_message_id = _direct_message_id(starter, other)
    other_page.goto(f"/channels/{direct_message_id}")
    expect(_message(other_page, "I have it, low tide is at six")).to_be_visible()
    send(other_page, "then we sail at five")
    expect(_message(starter_page, "then we sail at five")).to_be_visible()
    expect(_entry(starter_page, other.name)).to_be_visible()


def test_an_unread_count_shows_on_the_entry_until_the_conversation_is_opened(people, page_for):
    sender, reader, _ = people
    direct_message_id = _direct_message_id(sender, reader)
    reader_page = page_for(reader)
    entry = _entry(reader_page, sender.name)
    expect(entry).to_be_visible()

    post_message(sender, direct_message_id, "first call for dinner")
    post_message(sender, direct_message_id, "second call for dinner")

    expect(entry.get_by_title("Unread")).to_have_text("2")
    entry.click()
    expect(_message(reader_page, "second call for dinner")).to_be_visible()
    expect(entry.get_by_title("Unread")).to_have_count(0)
    reader_page.reload()
    expect(_entry(reader_page, sender.name).get_by_title("Unread")).to_have_count(0)


def test_a_closed_direct_message_stays_gone_and_reopens_with_its_history(people, page_for):
    closer, other, channel_id = people
    direct_message_id = _direct_message_id(closer, other)
    post_message(other, direct_message_id, "see you at the pier")
    post_message(other, channel_id, "anchoring at the bay")
    page = page_for(closer)
    entry = _entry(page, other.name)
    expect(entry).to_be_visible()

    entry.hover()
    # the close button has no name; it is the only button beside the link
    entry.locator("xpath=following-sibling::div//button").click()

    expect(entry).to_have_count(0)
    page.reload()
    expect(_entry(page, other.name)).to_have_count(0)
    _start_from_profile_card(page, channel_id, other, "anchoring at the bay")
    expect(_message(page, "see you at the pier")).to_be_visible()
    expect(_entry(page, other.name)).to_be_visible()


def test_someone_outside_a_direct_message_is_sent_home_and_never_sees_it(
    people, make_user, page_for
):
    sender, reader, _ = people
    outsider = make_user()
    direct_message_id = _direct_message_id(sender, reader)
    post_message(sender, direct_message_id, "the code word is lantern")
    page = page_for(outsider)

    page.goto(f"/channels/{direct_message_id}")

    expect(chat_input(page)).to_be_visible()
    expect(page).not_to_have_url(re.compile("/channels/"))
    expect(page.get_by_text("the code word is lantern")).to_have_count(0)
    with outsider.client() as client:
        assert client.get(f"/api/v1/channels/{direct_message_id}/messages").status_code == 403


def test_a_closed_direct_message_comes_back_when_the_other_person_writes(people, page_for):
    closer, other, _ = people
    direct_message_id = _direct_message_id(closer, other)
    post_message(other, direct_message_id, "did you get the tickets?")
    page = page_for(closer)
    entry = _entry(page, other.name)
    expect(entry).to_be_visible()
    entry.hover()
    entry.locator("xpath=following-sibling::div//button").click()
    expect(entry).to_have_count(0)

    post_message(other, direct_message_id, "the tickets are sold out soon")

    expect(entry.get_by_title("Unread")).to_have_text("2")
    entry.click()
    expect(_message(page, "the tickets are sold out soon")).to_be_visible()


def test_a_group_direct_message_lists_under_the_other_names_and_reaches_both(
    people, make_user, page_for
):
    starter, second, _ = people
    third = make_user()
    page = page_for(starter)
    _channels_section(page).get_by_role("button", name="Create Channel").click()
    modal = page.get_by_role("dialog").filter(has_text="Create Channel")
    modal.get_by_role("combobox").filter(has_text="Direct Message").select_option(
        label="Direct Message"
    )
    for person in (second, third):
        modal.get_by_placeholder("Search").fill(person.name)
        modal.get_by_role("button", name=person.name).click()
    modal.get_by_role("button", name="Create", exact=True).click()

    expect(page).to_have_url(re.compile("/channels/"))
    send(page, "dinner at mine on saturday?")
    second_page, third_page = page_for(second), page_for(third)
    second_entry = _group_entry(second_page, starter, third)
    third_entry = _group_entry(third_page, starter, second)
    expect(second_entry.get_by_title("Unread")).to_have_text("1")
    second_entry.click()
    expect(_message(second_page, "dinner at mine on saturday?")).to_be_visible()
    send(second_page, "I will bring dessert")
    expect(_message(page, "I will bring dessert")).to_be_visible()
    third_entry.click()
    expect(_message(third_page, "I will bring dessert")).to_be_visible()
