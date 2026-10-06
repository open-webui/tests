"""Journey: images and files shared in a channel, as the other members and a model get them.

A picture attached from the channel input shows as an image on the other member's screen, opens
full size and downloads with the bytes that were uploaded. A text file shows as a file the other
member opens to read. A picture shared in a thread reaches a model mentioned later in that
thread, which answers there.

Discriminates: passes on dev ebc6add67; in a frontend copy, rendering every attachment as a plain
file turns the image test red, and a preview whose download fetches nothing turns its download
step red; in a backend copy, leaving the thread's pictures out of the model's request turns the
model test red.
"""

from __future__ import annotations

import base64
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.channel_chat import enable_channels as enable_channels_with_reply_mode
from harness.channel_quotes import group_channel, post_message
from harness.image_engines import PNG_BASE64
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import chat_input
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

PNG = base64.b64decode(PNG_BASE64)


@pytest.fixture
def people(admin, preserve, make_user):
    """The sender and another member of one group channel, with its id."""
    preserve("admin_config")
    with admin.client() as client:
        enable_channels_with_reply_mode(client, reply_mode="thread")
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


def _attach(page: Page, name: str, mime_type: str, content: bytes) -> None:
    page.get_by_role("button", name="More", exact=True).last.click()
    with page.expect_file_chooser() as chooser:
        page.get_by_role("button", name="Upload Files").click()
    chooser.value.set_files({"name": name, "mimeType": mime_type, "buffer": content})


def _send(page: Page, box: Locator, text: str) -> None:
    box.click()
    page.keyboard.type(text)
    page.keyboard.press("Enter")


def test_a_shared_image_shows_to_the_other_member_who_opens_and_downloads_it(people, page_for):
    sender, member, channel_id = people
    page = _open_channel(page_for, sender, channel_id)
    member_page = _open_channel(page_for, member, channel_id)

    _attach(page, "summit.png", "image/png", PNG)
    _send(page, chat_input(page), "the view from the summit")

    on_member = _message(member_page, "the view from the summit")
    picture = on_member.get_by_role("img", name="summit.png")
    expect(picture).to_be_visible()
    member_page.wait_for_function(
        "(image) => image.complete && image.naturalWidth > 0", arg=picture.element_handle()
    )
    on_member.get_by_role("button", name="Show image preview").click()
    with member_page.expect_download() as downloading:
        member_page.get_by_role("button", name="Download").click()
    assert open(downloading.value.path(), "rb").read() == PNG


def test_a_shared_text_file_opens_for_the_other_member(people, page_for):
    sender, member, channel_id = people
    page = _open_channel(page_for, sender, channel_id)
    member_page = _open_channel(page_for, member, channel_id)

    _attach(page, "packing-list.txt", "text/plain", b"rope, lamp and two flasks")
    expect(page.locator("#message-input-container").get_by_text("packing-list.txt")).to_be_visible()
    _send(page, chat_input(page), "what to pack")

    on_member = _message(member_page, "what to pack")
    on_member.get_by_role("button", name="packing-list.txt").click()
    expect(member_page.get_by_role("dialog")).to_contain_text("rope, lamp and two flasks")


def test_a_picture_shared_in_a_thread_reaches_a_model_asked_there(people, page_for, upstream):
    sender, member, channel_id = people
    parent_id = post_message(member, channel_id, "which trail is this?")
    question = f"what do you see? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("A pine forest trail", match=reply.answering(question)))
    page = _open_channel(page_for, sender, channel_id)
    parent = page.locator(f"[id='message-{parent_id}']").first
    parent.hover()
    tooltip_button(parent, "Reply in Thread").click()
    thread_box = page.get_by_label("Reply to thread...")

    _attach(page, "trail.png", "image/png", PNG)
    _send(page, thread_box, "here is the photo")
    expect(page.locator(f"[id^='message-{parent_id}-']").get_by_role("img")).to_be_visible()
    thread_box.click()
    page.keyboard.type(f"@{MOCK_MODEL_ID}")
    page.locator("#suggestions-container").get_by_role("button", name=MOCK_MODEL_ID).click()
    page.keyboard.type(f" {question}")
    page.keyboard.press("Enter")

    answer = page.locator(f"[id^='message-{parent_id}-']").filter(has_text="A pine forest trail")
    expect(answer.first).to_contain_text(MOCK_MODEL_ID)
    [sent] = [request for request in upstream.chat_requests() if question in str(request)]
    images = [
        part["image_url"]["url"]
        for part in sent["messages"][-1]["content"]
        if isinstance(part, dict) and part.get("type") == "image_url"
    ]
    assert images == [f"data:image/png;base64,{PNG_BASE64}"]
