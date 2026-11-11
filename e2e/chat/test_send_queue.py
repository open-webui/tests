"""The chat's send queue: a stuck queue after a failed attachment, and Send now on one message.

* open-webui/open-webui#28880, fix `5f8d8f0c5` (PR open-webui/open-webui#30447): a queued
  message whose attachment failed to upload holds the whole queue back. Deleting or editing it
  took it out of the queue but never resumed the queue, so the messages behind it stayed
  unsent until the chat was reopened. Since dev 8a4547104 its Send now button reads Upload failed.
* open-webui/open-webui#30027, fix `aea7d34f4`: Send now on one queued message stopped the
  running reply and sent that message, but the stopped reply's completion resumed the queue at
  the same moment, so the rest of the queue went out too and two replies streamed into the
  chat at once.

`test_send_now_sends_only_the_chosen_message` is red on dev 62f70a844: Send now stops the running
reply first, and since de73bb830 a stopped reply never ends on the page, so the chosen message is
not sent (open-webui/open-webui#32081, fixed by d12bf062b).

Discriminates: passes on the efe63bd34 build; with `5f8d8f0c5` reverted the delete and edit
tests fail (the message behind the failed one is never sent), and with `aea7d34f4` reverted the
Send now test fails (the rest of the queue is sent while the chosen message is answered).

Every test runs in the main chat and in a side chat (`utils/chat_surface.py`), and the Send now
test in every way a side chat starts and every state it can be in. A side chat had no send queue:
a message sent while an answer was being written, or while an attachment was uploading, stayed in
the message box (open-webui/open-webui#34022, fix PR open-webui/open-webui#34064). On dev c15f28649
every test is red in every side chat and passes on the PR's build.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Locator, Route, expect

from harness import upstream as reply
from utils.chat_surface import EVERYWHERE, ChatSurface
from utils.chat_ui import chat_input, conversation, expect_reply, last_reply, send, stop_button

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

OPENING = "hello"
FIRST, SECOND, THIRD = "queued alpha", "queued bravo", "queued charlie"
QUEUE_SEND_BUTTON = re.compile("^(Send now|Waiting for upload|Upload failed)$")


def queued_item(chat: ChatSurface, text: str) -> Locator:
    """The row of the send queue showing `text`, with its Send now, Edit and Delete buttons."""
    rows = chat.panel.locator("div").filter(has=chat.page.get_by_text(text, exact=True))
    return rows.filter(has=chat.page.get_by_role("button", name=QUEUE_SEND_BUTTON)).last


def last_user_messages(upstream) -> list[str]:
    """What this module's messages reached the provider as, in the order they were sent."""
    last_of_each = [
        [entry for entry in body["messages"] if entry["role"] == "user"][-1]["content"]
        for body in upstream.chat_requests()
    ]
    # a side chat can be opened with a question of its own, which is none of the queue's
    return [text for text in last_of_each if text in (OPENING, FIRST, SECOND, THIRD)]


@pytest.fixture
def chat(chat_for, make_user) -> ChatSurface:
    chat = chat_for(make_user())
    expect(chat_input(chat)).to_be_visible()
    return chat


# main and plain side chat only: how a side chat started decides whether it queues, which the
# Send now test crosses with every condition; a failed upload is then the queue's own business
@pytest.fixture
def failed_upload_ahead(chat: ChatSurface, upstream) -> ChatSurface:
    """An open chat whose queue holds FIRST, its attachment failed, and SECOND behind it."""
    page = chat.page
    upstream.queue(reply.text("hello there", match=reply.answering(OPENING)))
    send(chat, OPENING)
    expect_reply(chat, "hello there")
    held: list[Route] = []
    page.route("**/api/v1/files/**", lambda route: held.append(route))

    chat.button("More").last.click()
    with page.expect_file_chooser() as chooser:
        page.get_by_role("menu").get_by_role("menuitem", name="Upload Files").click()
    chooser.value.set_files({"name": "notes.txt", "mimeType": "text/plain", "buffer": b"notes"})
    send(chat, FIRST)
    expect(queued_item(chat, FIRST)).to_be_visible()
    send(chat, SECOND)
    expect(queued_item(chat, SECOND)).to_be_visible()

    for _ in range(50):
        if held:
            break
        page.wait_for_timeout(100)
    assert held, "the upload never started"
    held[0].fulfill(status=500, json={"detail": "the disk is full"})
    expect(queued_item(chat, FIRST).get_by_text("Upload failed")).to_be_visible()
    return chat


def test_deleting_the_failed_message_sends_the_rest_of_the_queue(failed_upload_ahead, upstream):
    """In a side chat: open-webui/open-webui#34022."""
    chat = failed_upload_ahead
    upstream.queue(reply.text("bravo answered", match=reply.answering(SECOND)))

    queued_item(chat, FIRST).get_by_role("button", name="Delete").click()

    expect_reply(chat, "bravo answered")
    expect(queued_item(chat, SECOND)).to_be_hidden()
    assert last_user_messages(upstream)[-1] == SECOND
    assert FIRST not in last_user_messages(upstream)


def test_editing_the_failed_message_sends_the_rest_and_keeps_its_text(
    failed_upload_ahead, upstream
):
    """In a side chat: open-webui/open-webui#34022."""
    chat = failed_upload_ahead
    upstream.queue(reply.text("bravo answered", match=reply.answering(SECOND)))

    queued_item(chat, FIRST).get_by_role("button", name="Edit").click()

    expect_reply(chat, "bravo answered")
    expect(chat_input(chat)).to_contain_text(FIRST)
    assert last_user_messages(upstream)[-1] == SECOND
    assert FIRST not in last_user_messages(upstream)


def test_a_failed_attachment_still_holds_the_queue_until_it_is_dealt_with(
    failed_upload_ahead, upstream
):
    """In a side chat: open-webui/open-webui#34022."""
    chat = failed_upload_ahead
    chat.page.wait_for_timeout(1500)  # bounded: nothing may be sent while the failed message waits
    assert last_user_messages(upstream) == [OPENING]
    expect(queued_item(chat, SECOND)).to_be_visible()
    expect(queued_item(chat, FIRST).get_by_role("button", name="Upload failed")).to_be_disabled()


# one model only: the queue holds messages per chat, however many models answer each of them
@pytest.mark.chat_surfaces(*EVERYWHERE)
def test_send_now_sends_only_the_chosen_message(chat, upstream):
    """In a side chat: open-webui/open-webui#34022."""
    slow = [f"alpha-{index} " for index in range(40)]
    bravo = [f"bravo-{index} " for index in range(8)]
    upstream.queue(
        reply.text(slow, chunk_delay=0.3, match=reply.answering(FIRST)),
        reply.text(bravo, chunk_delay=0.3, match=reply.answering(SECOND)),
        reply.text("charlie answered", match=reply.answering(THIRD)),
    )
    send(chat, FIRST)
    expect(last_reply(chat)).to_contain_text("alpha-1")
    send(chat, SECOND)
    send(chat, THIRD)
    expect(queued_item(chat, THIRD)).to_be_visible()

    queued_item(chat, SECOND).get_by_role("button", name="Send now").click()

    expect(last_reply(chat)).to_contain_text("bravo-2")
    assert last_user_messages(upstream) == [FIRST, SECOND], (
        "Send now also sent the rest of the queue"
    )
    expect(queued_item(chat, THIRD)).to_be_visible()

    expect_reply(chat, "charlie answered")
    expect(stop_button(chat)).to_be_hidden()
    assert last_user_messages(upstream) == [FIRST, SECOND, THIRD]
    # a side chat can hold the question it was opened with
    user_messages = conversation(chat).locator(".chat-user").filter(has_text="queued ")
    expect(user_messages).to_contain_text([FIRST, SECOND, THIRD])
