"""Journey: members edit, delete, pin and react to channel messages and the other member sees it.

Two fresh accounts share a group channel, each with it open in a browser of their own. The author
edits a message in place and it reads "(edited)" on both screens; deleting it after the confirm
dialog removes it from both. A member sees no edit or delete button on someone else's message; an
admin in the channel does, and a message the admin deletes leaves the author's screen. Pinning marks
the message "Pinned" for both and lists it under "Pinned Messages", where unpinning takes it off
again. A reaction the second member joins counts both, names who reacted, and drops back as each
takes theirs away.

Discriminates: passes on dev 176d31d1d; in a frontend copy, saving an edit with the old content
turns the edit test red, deleting nothing on confirm turns the delete test red, offering edit and
delete on every message turns the someone else's message test red, offering them to nobody but the
author turns the admin test red (checked on dev ebc6add67), pinning with the old pinned state turns
the pin test red and removing a reaction by adding it again turns the reaction test red.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.channel_quotes import enable_channels, group_channel, post_message
from utils.chat_ui import chat_input
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def people(admin, preserve, make_user):
    """The author and another member of one group channel, with its id."""
    preserve("admin_config")
    enable_channels(admin)
    author, member = make_user(), make_user()
    return author, member, group_channel(author, member)


def _open_channel(page_for, account, channel_id: str) -> Page:
    page = page_for(account)
    page.goto(f"/channels/{channel_id}")
    expect(chat_input(page)).to_be_visible()
    return page


def _message(page: Page, text: str) -> Locator:
    return page.locator("[id^='message-']").filter(has_text=text).first


def _message_tool(page: Page, text: str, tooltip: str) -> Locator:
    message = _message(page, text)
    message.hover()
    return tooltip_button(message, tooltip)


def _tooltips(message: Locator) -> list[str]:
    return message.get_by_role("button").evaluate_all(
        "(buttons) => buttons.map((button) => button.parentElement?._tippy?.props.content)"
    )


def _stored(account, channel_id: str) -> list[dict]:
    with account.client() as client:
        listed = client.get(f"/api/v1/channels/{channel_id}/messages")
    listed.raise_for_status()
    return listed.json()


def test_the_author_edits_a_message_and_the_other_member_sees_the_edit(people, page_for):
    author, member, channel_id = people
    post_message(author, channel_id, "we sail at nine")
    author_page = _open_channel(page_for, author, channel_id)
    member_page = _open_channel(page_for, member, channel_id)

    _message_tool(author_page, "we sail at nine", "Edit").click()
    editor = author_page.locator("[id^='message-'] textarea")
    editor.fill("we sail at ten")
    author_page.get_by_role("button", name="Save").click()

    expect(_message(author_page, "we sail at ten")).to_contain_text("(edited)")
    expect(_message(member_page, "we sail at ten")).to_contain_text("(edited)")
    expect(member_page.get_by_text("we sail at nine")).to_have_count(0)
    assert [message["content"] for message in _stored(member, channel_id)] == ["we sail at ten"]


def test_the_author_deletes_a_message_after_confirming_and_it_leaves_both_screens(people, page_for):
    author, member, channel_id = people
    post_message(author, channel_id, "the spare key is under the mat")
    post_message(author, channel_id, "bring sunscreen")
    author_page = _open_channel(page_for, author, channel_id)
    member_page = _open_channel(page_for, member, channel_id)
    expect(_message(member_page, "the spare key is under the mat")).to_be_visible()

    _message_tool(author_page, "the spare key is under the mat", "Delete").click()
    dialog = author_page.get_by_role("dialog")
    expect(dialog.get_by_text("Are you sure you want to delete this message?")).to_be_visible()
    dialog.get_by_role("button", name="Confirm").click()

    expect(author_page.get_by_text("the spare key is under the mat")).to_have_count(0)
    expect(member_page.get_by_text("the spare key is under the mat")).to_have_count(0)
    expect(_message(member_page, "bring sunscreen")).to_be_visible()
    assert [message["content"] for message in _stored(member, channel_id)] == ["bring sunscreen"]


def test_a_member_is_offered_no_edit_or_delete_on_someone_elses_message(people, page_for):
    author, member, channel_id = people
    post_message(author, channel_id, "the author's plan")
    post_message(member, channel_id, "the member's idea")
    page = _open_channel(page_for, member, channel_id)

    on_theirs = _tooltips(_message(page, "the author's plan"))
    on_own = _tooltips(_message(page, "the member's idea"))

    assert "Reply in Thread" in on_theirs
    assert "Edit" not in on_theirs and "Delete" not in on_theirs
    assert "Edit" in on_own and "Delete" in on_own


def test_a_pinned_message_is_marked_for_both_and_listed_until_unpinned(people, page_for):
    author, member, channel_id = people
    post_message(author, channel_id, "meeting point is the old lighthouse")
    post_message(author, channel_id, "weather looks fine")
    author_page = _open_channel(page_for, author, channel_id)
    member_page = _open_channel(page_for, member, channel_id)

    _message_tool(member_page, "meeting point is the old lighthouse", "Pin").click()

    pinned = _message(author_page, "meeting point is the old lighthouse")
    expect(pinned.get_by_text("Pinned", exact=True)).to_be_visible()
    expect(_message(author_page, "weather looks fine").get_by_text("Pinned")).to_have_count(0)
    author_page.get_by_role("button", name="Pinned Messages").click()
    dialog = author_page.get_by_role("dialog")
    expect(dialog.get_by_text("meeting point is the old lighthouse")).to_be_visible()
    expect(dialog.get_by_text("weather looks fine")).to_have_count(0)

    listed = dialog.locator("[id^='message-']").filter(has_text="the old lighthouse").first
    listed.hover()
    tooltip_button(listed, "Unpin").click()
    expect(dialog.get_by_text("No pinned messages")).to_be_visible()
    member_message = _message(member_page, "meeting point is the old lighthouse")
    expect(member_message.get_by_text("Pinned", exact=True)).to_have_count(0)


def test_a_joined_reaction_counts_both_names_them_and_drops_as_each_removes_it(people, page_for):
    author, member, channel_id = people
    post_message(author, channel_id, "who is coming on sunday?")
    author_page = _open_channel(page_for, author, channel_id)
    member_page = _open_channel(page_for, member, channel_id)

    _message_tool(author_page, "who is coming on sunday?", "Add Reaction").click()
    author_page.get_by_placeholder("Search all emojis").fill("rocket")
    author_page.get_by_role("menu").get_by_role("button", name="1F680").click()
    on_member = _message(member_page, "who is coming on sunday?")
    expect(on_member.get_by_role("button", name="rocket 1")).to_be_visible()
    on_member.get_by_role("button", name="rocket 1").click()

    on_author = _message(author_page, "who is coming on sunday?")
    expect(on_author.get_by_role("button", name="rocket 2")).to_be_visible()
    expect(on_member.get_by_role("button", name="rocket 2")).to_be_visible()
    assert f"You and {member.name} reacted with :rocket:" in _tooltips(on_author)
    assert f"{author.name} and You reacted with :rocket:" in _tooltips(on_member)

    on_member.get_by_role("button", name="rocket 2").click()
    expect(on_author.get_by_role("button", name="rocket 1")).to_be_visible()
    assert "You reacted with :rocket:" in _tooltips(on_author)
    on_author.get_by_role("button", name="rocket 1").click()
    expect(on_member.get_by_role("button", name="rocket")).to_have_count(0)
    [stored] = _stored(member, channel_id)
    assert stored["reactions"] == []


def test_an_admin_deletes_a_members_message_and_it_leaves_the_members_screen(
    people, admin, page_for
):
    _, member, _ = people
    channel_id = group_channel(admin, member)
    post_message(member, channel_id, "buy cheap watches here")
    post_message(member, channel_id, "when does the ferry leave?")
    member_page = _open_channel(page_for, member, channel_id)
    admin_page = _open_channel(page_for, admin, channel_id)
    expect(_message(member_page, "buy cheap watches here")).to_be_visible()

    _message_tool(admin_page, "buy cheap watches here", "Delete").click()
    admin_page.get_by_role("dialog").get_by_role("button", name="Confirm").click()

    expect(member_page.get_by_text("buy cheap watches here")).to_have_count(0)
    expect(_message(member_page, "when does the ferry leave?")).to_be_visible()
    assert "Edit" in _tooltips(_message(admin_page, "when does the ferry leave?"))
    assert [message["content"] for message in _stored(member, channel_id)] == [
        "when does the ferry leave?"
    ]
