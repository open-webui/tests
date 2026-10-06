"""Journey: mentioning a person or a model and attaching a file from the channel input.

Typing "@" in the channel input offers the channel's members and the models; picking a member
posts a mention that reads "@<name>", and that member, while elsewhere in the app, sees an unread
count on the channel and a toast naming the sender and the channel that opens the channel when
clicked. Picking a model has it answer in a thread under the message, which the parent shows as
a reply count that opens the thread panel on the answer; with the admin's reply mode set to the
channel the answer lands in the channel itself for every member. A file attached through "Upload
Files" posts with the message and the other member sees it on the message and can open its
content. Typing "#" offers the person's channels; a picked channel posts as a "#<name>" link
that opens that channel for the other member who clicks it.

A message sent while its file still uploads waits in the input's queue and posts by itself once
the upload is done, with a file the other member can open; it posted at once with a file nobody
could open until dev 8a4547104 (open-webui/open-webui#31587). The mention toast showed the raw
mention markup until PR #31601 (open-webui/open-webui#31586).

Discriminates: passes on dev 176d31d1d; in a frontend copy, a picker that drops the mention type
turns the person and thread model tests red, a toast that goes nowhere on click turns the person
test red, sending the message without its files turns the file test red and a channel link
that opens nothing turns the channel link test red (checked on dev ebc6add67); in a backend copy,
answering in a thread whatever the reply mode turns the channel reply mode test red.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.channel_chat import enable_channels as enable_channels_with_reply_mode
from harness.channel_quotes import enable_channels, group_channel
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import chat_input

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def people(admin, preserve, make_user):
    """The sender and another member of one group channel, with its id."""
    preserve("admin_config")
    enable_channels(admin)
    sender, member = make_user(), make_user()
    return sender, member, group_channel(sender, member)


def _open_channel(page_for, account, channel_id: str) -> Page:
    page = page_for(account)
    page.goto(f"/channels/{channel_id}")
    expect(chat_input(page)).to_be_visible()
    return page


def _message(page: Page, text: str) -> Locator:
    posted = page.locator("[id^='message-']:not(#message-input-container)")
    return posted.filter(has_text=text).first


def _mention(page: Page, query: str, label: str) -> None:
    chat_input(page).click()
    page.keyboard.type(f"@{query}")
    suggestions = page.locator("#suggestions-container")
    suggestions.get_by_role("button", name=label).click()


def _sidebar_entry(page: Page, channel_name: str) -> Locator:
    sidebar = page.get_by_role("navigation", name="Chat history")
    open_sidebar = page.get_by_role("button", name="Open Sidebar", exact=True)
    if open_sidebar.is_visible():
        open_sidebar.click()
    section = sidebar.get_by_role("button", name="Channels")
    if section.get_attribute("aria-expanded") == "false":
        section.click()
    return sidebar.get_by_role("link", name=channel_name)


def _mention_member(page: Page, member, text: str) -> None:
    _mention(page, member.name.split()[-1], member.name)
    page.keyboard.type(f" {text}")
    page.keyboard.press("Enter")


def _stored(account, channel_id: str) -> list[dict]:
    with account.client() as client:
        listed = client.get(f"/api/v1/channels/{channel_id}/messages")
    listed.raise_for_status()
    return listed.json()


def test_a_mentioned_member_elsewhere_gets_a_toast_that_opens_the_channel(people, page_for):
    sender, member, channel_id = people
    member_page = page_for(member)
    expect(chat_input(member_page)).to_be_visible()
    page = _open_channel(page_for, sender, channel_id)
    channel_name = page.title().removeprefix("#").removesuffix(" / Open WebUI")
    entry = _sidebar_entry(member_page, channel_name)
    expect(entry).to_be_visible()

    _mention_member(page, member, "can you bring the charts?")

    posted = _message(page, "can you bring the charts?")
    expect(posted.locator(".mention")).to_have_text(f"@{member.name}")
    [stored] = _stored(sender, channel_id)
    assert stored["content"].startswith(f"<@U:{member.id}|{member.name}>")
    expect(entry.get_by_title("Unread")).to_have_text("1")
    toast = member_page.get_by_text(f"{sender.name} (#{channel_name})")
    expect(toast).to_be_visible()
    toast.click()
    expect(member_page).to_have_url(re.compile(f"/channels/{channel_id}"))
    expect(_message(member_page, "can you bring the charts?")).to_be_visible()
    expect(entry.get_by_title("Unread")).to_have_count(0)


def test_the_mention_toast_reads_the_name_and_not_the_mention_markup(people, page_for):
    """#31586, fixed by PR #31601: the toast showed the stored `<@U:<id>|<name>>` markup as text.

    The message itself reads "@<name>"; the toast stripped HTML tags from the content, which left
    the mention markup (not a tag) in place, user id included. Red on dev 176d31d1d.
    """
    sender, member, channel_id = people
    member_page = page_for(member)
    expect(chat_input(member_page)).to_be_visible()
    page = _open_channel(page_for, sender, channel_id)

    _mention_member(page, member, "the dinghy needs air")

    toast_text = member_page.get_by_text("the dinghy needs air")
    expect(toast_text).to_be_visible()
    expect(toast_text).to_contain_text(f"@{member.name}")
    expect(toast_text).not_to_contain_text(member.id)


def _mention_model(page: Page, question: str) -> None:
    _mention(page, MOCK_MODEL_ID, MOCK_MODEL_ID)
    page.keyboard.type(f" {question}")
    page.keyboard.press("Enter")


def test_a_mentioned_model_answers_in_a_thread_under_the_message(people, page_for, upstream):
    sender, _, channel_id = people
    question = f"what is the tide at noon? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("High tide, 2.4 metres", match=reply.answering(question)))
    page = _open_channel(page_for, sender, channel_id)

    _mention_model(page, question)

    asked = _message(page, question)
    replies = asked.get_by_role("button", name="1 Replies")
    expect(replies).to_be_visible()
    replies.click()
    expect(page.get_by_label("Reply to thread...")).to_be_visible()
    expect(_message(page, "High tide, 2.4 metres")).to_contain_text(MOCK_MODEL_ID)
    expect(asked).not_to_contain_text("High tide")
    [sent] = [request for request in upstream.chat_requests() if question in str(request)]
    assert f"<@M:{MOCK_MODEL_ID}" not in str(sent["messages"][-1]["content"])


def test_in_channel_reply_mode_the_model_answers_in_the_channel_for_everyone(
    admin, people, page_for, upstream
):
    sender, member, channel_id = people
    with admin.client() as client:
        enable_channels_with_reply_mode(client, reply_mode="channel")
    question = f"how far to the island? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("About six nautical miles", match=reply.answering(question)))
    page = _open_channel(page_for, sender, channel_id)
    member_page = _open_channel(page_for, member, channel_id)

    _mention_model(page, question)

    answer = _message(member_page, "About six nautical miles")
    expect(answer).to_contain_text(MOCK_MODEL_ID)
    expect(_message(page, "About six nautical miles")).to_be_visible()
    expect(_message(page, question).get_by_role("button", name="1 Replies")).to_have_count(0)


def _attach(page: Page, name: str, text: str) -> None:
    page.get_by_role("button", name="More", exact=True).last.click()
    with page.expect_file_chooser() as chooser:
        page.get_by_role("button", name="Upload Files").click()
    chooser.value.set_files({"name": name, "mimeType": "text/plain", "buffer": text.encode()})
    expect(page.locator("#message-input-container").get_by_text(name)).to_be_visible()


def _send(page: Page, text: str) -> None:
    chat_input(page).click()
    page.keyboard.type(text)
    page.keyboard.press("Enter")


def _attached_content(account, channel_id: str) -> list[str]:
    """The text of every file on the channel's messages, as `account` downloads it."""
    contents = []
    with account.client() as client:
        for message in client.get(f"/api/v1/channels/{channel_id}/messages").json():
            data = client.get(f"/api/v1/channels/{channel_id}/messages/{message['id']}/data")
            for attached in (data.json() or {}).get("files") or []:
                downloaded = client.get(f"/api/v1/files/{attached['id']}/content")
                contents.append(downloaded.text if downloaded.status_code == 200 else None)
    return contents


def test_a_file_sent_with_a_message_reaches_the_other_member(people, page_for):
    sender, member, channel_id = people
    page = _open_channel(page_for, sender, channel_id)
    member_page = _open_channel(page_for, member, channel_id)

    _attach(page, "crew-list.txt", "Ana, Ben and Cleo")
    # the spinner is the only sign the upload is still running
    expect(page.locator("#message-input-container .spinner_ajPY")).to_have_count(0)
    _send(page, "here is the crew list")

    on_member = _message(member_page, "here is the crew list")
    expect(on_member.get_by_text("crew-list.txt")).to_be_visible()
    assert _attached_content(member, channel_id) == ["Ana, Ben and Cleo"]


def test_a_message_sent_while_its_file_uploads_waits_and_posts_with_the_file(people, page_for):
    """Regression for open-webui/open-webui#31587, fixed in dev 8a4547104.

    The channel input posted a message sent during its upload at once, so the stored file had no id
    and nobody could open it. Now the message waits in the input's queue, nothing is posted until
    the upload is done, and then it posts by itself and the other member downloads the file.
    """
    sender, member, channel_id = people
    page = _open_channel(page_for, sender, channel_id)
    held_uploads = []
    page.route("**/api/v1/files/?*", lambda route: held_uploads.append(route))

    _attach(page, "tide-table.txt", "low tide at six")
    _send(page, "tide table attached")

    expect(page.get_by_role("button", name="Waiting for upload")).to_be_visible()
    assert _stored(sender, channel_id) == [], "#31587: a message was posted during the upload"
    for upload in held_uploads:
        upload.continue_()

    expect(_message(page, "tide table attached")).to_be_visible()
    expect(page.get_by_role("button", name="Waiting for upload")).to_have_count(0)
    member_page = _open_channel(page_for, member, channel_id)
    on_member = _message(member_page, "tide table attached")
    expect(on_member.get_by_text("tide-table.txt")).to_be_visible()
    assert _attached_content(member, channel_id) == ["low tide at six"]


def test_a_linked_channel_opens_for_the_member_who_clicks_it(people, page_for):
    sender, member, channel_id = people
    linked_id = group_channel(sender, member)
    with sender.client() as client:
        linked_name = client.get(f"/api/v1/channels/{linked_id}").json()["name"]
    page = _open_channel(page_for, sender, channel_id)
    member_page = _open_channel(page_for, member, channel_id)

    chat_input(page).click()
    page.keyboard.type(f"#{linked_name}")
    page.locator("#suggestions-container").get_by_role("button", name=linked_name).click()
    page.keyboard.type(" has the packing list")
    page.keyboard.press("Enter")

    link = _message(member_page, "has the packing list").locator(".mention")
    expect(link).to_have_text(f"#{linked_name}")
    [stored] = _stored(sender, channel_id)
    assert stored["content"].startswith(f"<#C:{linked_id}|{linked_name}>")
    link.click()
    expect(member_page).to_have_url(re.compile(f"/channels/{linked_id}"))
