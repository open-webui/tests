"""Journey: who may read and write in a channel, as the members find it on their screens.

In a standard channel the admin opened, a member with a read grant reads what a member with a write
grant posts, live, but their input is disabled and says why, their thread input says the same, and a
message offers them no tools and no reaction to join. Someone added to a group channel while the app
is open finds it in the sidebar without a reload, with the unread count of what is posted next.
Someone removed while the channel is open stops getting its messages, while a message in another
channel they share still arrives.

Discriminates: passes on dev ebc6add67; in a frontend copy, enabling the input whatever the write
access turns the read-only test red; in a backend copy, giving new members no `channel:created`
event and no room turns the added member test red and keeping a removed member in the channel's room
turns the removed member test red.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.channel_quotes import enable_channels, group_channel, post_message
from utils.chat_ui import chat_input, send
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

READ_ONLY = "You do not have permission to send messages in this channel."
READ_ONLY_THREAD = "You do not have permission to send messages in this thread."


@pytest.fixture
def channels_on(admin, preserve):
    preserve("admin_config")
    enable_channels(admin)


def _standard_channel(admin, name: str, grants: list[tuple[str, str]]) -> str:
    """A standard channel the admin opens with `(account id, permission)` grants."""
    access_grants = [
        {"principal_type": "user", "principal_id": account_id, "permission": permission}
        for account_id, permission in grants
    ]
    with admin.client() as client:
        created = client.post(
            "/api/v1/channels/create",
            json={"name": name, "type": None, "access_grants": access_grants},
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _channel_name(account, channel_id: str) -> str:
    with account.client() as client:
        fetched = client.get(f"/api/v1/channels/{channel_id}")
    fetched.raise_for_status()
    return fetched.json()["name"]


def _message(page: Page, text: str) -> Locator:
    return (
        page.locator("[id^='message-']:not(#message-input-container)").filter(has_text=text).first
    )


def _tooltips(message: Locator) -> list[str]:
    return message.get_by_role("button").evaluate_all(
        "(buttons) => buttons.map((button) => button.parentElement?._tippy?.props.content)"
    )


def _sidebar_entry(page: Page, channel_name: str) -> Locator:
    sidebar = page.get_by_role("navigation", name="Chat history")
    open_sidebar = page.get_by_role("button", name="Open Sidebar", exact=True)
    if open_sidebar.is_visible():
        open_sidebar.click()
    section = sidebar.get_by_role("button", name="Channels")
    if section.get_attribute("aria-expanded") == "false":
        section.click()
    return sidebar.get_by_role("link", name=channel_name)


def test_a_read_only_member_reads_live_but_cannot_post_reply_or_react(
    admin, channels_on, make_user, page_for
):
    writer, reader = make_user(), make_user()
    channel_id = _standard_channel(
        admin,
        "notice-board",
        [(writer.id, "read"), (writer.id, "write"), (reader.id, "read")],
    )
    post_message(writer, channel_id, "the pool opens at eight")
    writer_page, reader_page = page_for(writer), page_for(reader)
    for page in (writer_page, reader_page):
        page.goto(f"/channels/{channel_id}")
    expect(chat_input(writer_page)).to_be_visible()
    reader_box = reader_page.get_by_label(READ_ONLY)
    expect(reader_box).to_be_visible()

    send(writer_page, "and closes at ten")
    expect(_message(reader_page, "and closes at ten")).to_be_visible()

    expect(reader_box).to_have_attribute("contenteditable", "false")
    on_writer = _message(writer_page, "the pool opens at eight")
    on_writer.hover()
    tooltip_button(on_writer, "Add Reaction").click()
    writer_page.get_by_placeholder("Search all emojis").fill("rocket")
    writer_page.get_by_role("menu").get_by_role("button", name="1F680").click()
    on_reader = _message(reader_page, "the pool opens at eight")
    expect(on_reader.get_by_role("button", name="rocket 1")).to_be_disabled()
    on_reader.hover()
    assert "Reply in Thread" in _tooltips(on_writer)
    assert not {"Add Reaction", "Reply", "Pin", "Reply in Thread"} & set(_tooltips(on_reader))

    post_message(
        writer, channel_id, "lanes are shared", parent_id=_message_id(writer, channel_id, "eight")
    )
    on_reader.get_by_role("button", name="1 Replies").click()
    thread_box = reader_page.get_by_label(READ_ONLY_THREAD)
    expect(thread_box).to_have_attribute("contenteditable", "false")
    expect(reader_page.get_by_text("lanes are shared")).to_be_visible()


def _message_id(account, channel_id: str, text: str) -> str:
    with account.client() as client:
        listed = client.get(f"/api/v1/channels/{channel_id}/messages")
    listed.raise_for_status()
    [found] = [message["id"] for message in listed.json() if text in message["content"]]
    return found


def test_a_member_added_while_the_app_is_open_gets_the_channel_live(
    channels_on, make_user, page_for
):
    owner, member, newcomer = make_user(), make_user(), make_user()
    channel_id = group_channel(owner, member)
    page = page_for(newcomer)
    expect(chat_input(page)).to_be_visible()
    entry = _sidebar_entry(page, _channel_name(owner, channel_id))
    expect(page.get_by_role("navigation", name="Chat history")).to_be_visible()
    expect(entry).to_have_count(0)

    with owner.client() as client:
        added = client.post(
            f"/api/v1/channels/{channel_id}/update/members/add", json={"user_ids": [newcomer.id]}
        )
    assert added.status_code == 200, added.text

    expect(entry).to_be_visible()
    post_message(member, channel_id, "welcome aboard")
    expect(entry.get_by_title("Unread")).to_have_text("1")
    entry.click()
    expect(_message(page, "welcome aboard")).to_be_visible()


def test_a_member_removed_while_watching_stops_getting_its_messages(
    channels_on, make_user, page_for
):
    owner, leaving = make_user(), make_user()
    channel_id = group_channel(owner, leaving)
    other_channel_id = group_channel(owner, leaving)
    page = page_for(leaving)
    page.goto(f"/channels/{channel_id}")
    expect(chat_input(page)).to_be_visible()
    post_message(owner, channel_id, "still here?")
    expect(_message(page, "still here?")).to_be_visible()
    other_entry = _sidebar_entry(page, _channel_name(owner, other_channel_id))
    expect(other_entry).to_be_visible()

    with owner.client() as client:
        removed = client.post(
            f"/api/v1/channels/{channel_id}/update/members/remove", json={"user_ids": [leaving.id]}
        )
    assert removed.status_code == 200, removed.text
    post_message(owner, channel_id, "the plans for after you left")
    post_message(owner, other_channel_id, "see you in the other room")

    # messages reach a tab in order, so the later one arriving means the earlier was not sent
    expect(other_entry.get_by_title("Unread")).to_have_text("1")
    expect(page.get_by_text("the plans for after you left")).to_have_count(0)
