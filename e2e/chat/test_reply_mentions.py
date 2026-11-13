"""Journey: a person a model's reply mentions finds it in their notifications and opens the reply.

Since 0.13.4 a model can mention people in a reply (the Chat History builtin tool looks them up),
and each mentioned person who can open the chat gets an entry in their notifications (over HTTP in
integration/notifications/test_notification_inbox.py). Here the chat sits in a folder its owner
shares with the person. The person sees the count on the Notifications button, the entry says the
model mentioned them, and clicking it opens the chat at that reply.

Discriminates: passes on the dev 485103cde build. In a backend copy whose inbox tells nobody about
a mention in a chat the entry never shows.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.access import grant
from harness.actors import Actor
from harness.chat import ChatTurn, ask
from utils.chat_ui import conversation

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

ADMIN_CONFIG = "/api/v1/auths/admin/config"


@pytest.fixture
def notifications_on(admin, preserve):
    preserve("admin_config")
    with admin.client() as client:
        current = client.get(ADMIN_CONFIG)
        assert current.status_code == 200, current.text
        saved = client.post(ADMIN_CONFIG, json={**current.json(), "ENABLE_NOTIFICATIONS": True})
    assert saved.status_code == 200, saved.text


def mentioning_reply(owner: Actor, member: Actor, upstream, text: str) -> ChatTurn:
    """The owner's chat, in a folder the member may read, whose second reply mentions them."""
    with owner.client() as client:
        folder = client.post("/api/v1/folders/", json={"name": f"Crew {uuid.uuid4().hex[:8]}"})
        assert folder.status_code == 200, folder.text
        folder_id = folder.json()["id"]
        shared = client.post(
            f"/api/v1/folders/{folder_id}/access/update",
            json={"access_grants": [grant("user", member.id, "read")]},
        )
        assert shared.status_code == 200, shared.text
        first = f"who is on watch? {uuid.uuid4().hex[:8]}"
        upstream.queue(reply.text("Let me check.", match=reply.answering(first)))
        turn, _ = ask(client, first)
        filed = client.post(f"/api/v1/chats/{turn.chat_id}/folder", json={"folder_id": folder_id})
        assert filed.status_code == 200, filed.text
        second = f"tell the crew {uuid.uuid4().hex[:8]}"
        upstream.queue(reply.text(f"<@U:{member.id}> {text}", match=reply.answering(second)))
        mentioning, _ = ask(
            client, second, chat_id=turn.chat_id, parent_id=turn.assistant_message_id
        )
    return mentioning


def open_notifications(page: Page):
    opener = page.get_by_role("button", name=re.compile(r"^Open Sidebar"))
    if opener.is_visible():
        opener.click()
    button = page.get_by_role("button", name=re.compile(r"^Notifications"))
    expect(button, "the unread mention is not counted on the button").to_have_accessible_name(
        "Notifications (1)"
    )
    button.click()
    return page.get_by_role("region", name="Notifications")


def test_a_mentioned_reader_opens_the_reply_from_their_notifications(
    page_for, make_user, upstream, notifications_on
):
    owner, member = make_user(), make_user()
    text = f"the tide turns at six {uuid.uuid4().hex[:6]}"
    turn = mentioning_reply(owner, member, upstream, text)
    page = page_for(member)

    panel = open_notifications(page)
    entry = panel.get_by_role("button").filter(has_text="mentioned you")
    expect(entry).to_have_count(1)
    expect(entry).to_contain_text(text)
    entry.click()

    expect(page).to_have_url(re.compile(rf"/c/{turn.chat_id}\?message={turn.assistant_message_id}"))
    expect(conversation(page).get_by_text(text)).to_be_visible()
