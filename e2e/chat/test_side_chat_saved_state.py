"""Journey: what a side chat stores and how it behaves once saved, as the side chat rework says.

A side chat asked with `/side` from a saved chat is stored as a chat of its own. It keeps only the
parent chat's documents, never its images, so a file limit counts them as the main chat does, and
it stores no copy of the account's default parameters. A person limited to temporary chats can
carry on a side chat saved before the limit, and what they send there is stored. A saved side tab
whose chat was deleted comes back as an empty chat with a notification.

These are the claims of PR open-webui/open-webui#34064 ("Differences from the old side chat").

Discriminates: all four pass on dev e791cbdde, which has the rework, and fail on dev c15f28649
before it (the parent's image stored with the side chat, the default temperature and seed copied
into it, the saved side chat not carried on, the deleted side tab not opened).
Clone Chat on a read-only shared chat in a tab is not covered: a chat shared with a person reaches
their sidebar only through a shared folder, and the harness has no way yet to drag such a chat
into a tab.
"""

from __future__ import annotations

import base64
import uuid

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.chat_history import seed_chat
from utils.chat_surface import (
    REPLY_TIMEOUT_MS,
    ChatSurface,
    ask_aside,
    main_chat,
    reload_beside_the_chat,
)

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

PARENT = [
    {"role": "user", "content": "Which pass links Lienz and Matrei?"},
    {"role": "assistant", "content": "The Felbertauern road links them through a tunnel."},
]
ANSWER = "The tunnel is 5.3 kilometres long."
# a 1x1 transparent PNG
PIXEL = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


@pytest.fixture
def side_chat_allowed(admin, preserve) -> None:
    preserve("permissions")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["chat"].update(side_chat=True)
        saved = client.post("/api/v1/users/default/permissions", json=permissions)
    assert saved.status_code == 200, saved.text


def question() -> str:
    return f"By the way, how long is that tunnel? ({uuid.uuid4().hex[:8]})"


def open_parent(page_for, account, messages: list[dict] = PARENT):
    with account.client() as client:
        chat_id, _ = seed_chat(client, messages)
    page = page_for(account, path=f"/c/{chat_id}")
    expect(main_chat(page).last_reply).to_contain_text(
        messages[-1]["content"], timeout=REPLY_TIMEOUT_MS
    )
    return page


def answered_aside(page, upstream, asked: str) -> ChatSurface:
    upstream.queue(reply.text(ANSWER, match=reply.answering(asked)))
    side = ask_aside(page, asked)
    side.expect_reply(ANSWER)
    expect(side.stop_button).to_be_hidden(timeout=REPLY_TIMEOUT_MS)
    return side


def stored_side_chat(account, title: str) -> dict:
    """The stored chat titled `title`, as the server keeps it."""
    with account.client() as client:
        listed = client.get("/api/v1/chats/", params={"page": 1}).json()
        [chat_id] = [chat["id"] for chat in listed if chat["title"] == title]
        stored = client.get(f"/api/v1/chats/{chat_id}")
    assert stored.status_code == 200, stored.text
    return stored.json()


def upload(account, name: str, content: bytes, mime_type: str) -> dict:
    with account.client() as client:
        uploaded = client.post("/api/v1/files/", files={"file": (name, content, mime_type)})
    assert uploaded.status_code == 200, uploaded.text
    return uploaded.json()


def file_entry(uploaded: dict, content_type: str) -> dict:
    return {
        "type": "file",
        "id": uploaded["id"],
        "name": uploaded["filename"],
        "url": f"/api/v1/files/{uploaded['id']}",
        "content_type": content_type,
        "file": uploaded,
        "status": "uploaded",
    }


def test_a_side_chat_keeps_the_parents_documents_but_not_its_images(
    page_for, make_user, upstream, side_chat_allowed
):
    account = make_user()
    notes = upload(
        account, "pass-notes.txt", b"The Felbertauern tunnel opened in 1967.", "text/plain"
    )
    photo = upload(account, "tunnel.png", PIXEL, "image/png")
    with_files = [
        {**PARENT[0], "files": [file_entry(notes, "text/plain"), file_entry(photo, "image/png")]},
        PARENT[1],
    ]
    page = open_parent(page_for, account, with_files)
    asked = question()

    answered_aside(page, upstream, asked)

    stored = stored_side_chat(account, asked)["chat"]
    kept = [item.get("id") for item in stored.get("files") or []]
    assert notes["id"] in kept, f"the parent's document did not travel with the side chat: {kept}"
    assert photo["id"] not in kept, f"the parent's image travelled with the side chat: {kept}"


def test_a_new_side_chat_stores_no_copy_of_the_default_parameters(
    page_for, make_user, upstream, side_chat_allowed
):
    account = make_user()
    with account.client() as client:
        saved = client.post(
            "/api/v1/users/user/settings/update",
            json={"ui": {"params": {"temperature": 0.3, "seed": 42}}},
        )
    assert saved.status_code == 200, saved.text
    page = open_parent(page_for, account)
    asked = question()

    answered_aside(page, upstream, asked)

    params = stored_side_chat(account, asked)["chat"].get("params") or {}
    copied = {key: params[key] for key in ("temperature", "seed") if key in params}
    assert copied == {}, f"the side chat stored the account's default parameters: {copied}"


def test_a_person_limited_to_temporary_chats_carries_on_a_side_chat_saved_before(
    page_for, admin, make_user, upstream, side_chat_allowed
):
    account = make_user()
    page = open_parent(page_for, account)
    asked = question()
    side = answered_aside(page, upstream, asked)
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["chat"].update(temporary=True, temporary_enforced=True)
        limited = client.post("/api/v1/users/default/permissions", json=permissions)
    assert limited.status_code == 200, limited.text

    side = reload_beside_the_chat(side, asked)
    follow_up = f"And when did it open? ({uuid.uuid4().hex[:8]})"
    upstream.queue(reply.text("It opened in 1967.", match=reply.answering(follow_up)))
    side.send(follow_up)
    side.expect_reply("It opened in 1967.")
    expect(side.stop_button).to_be_hidden(timeout=REPLY_TIMEOUT_MS)

    messages = stored_side_chat(account, asked)["chat"]["history"]["messages"].values()
    contents = [message.get("content") for message in messages]
    assert follow_up in contents, "the follow-up in the saved side chat was not stored"


def test_a_saved_side_tab_whose_chat_was_deleted_opens_empty_with_a_notification(
    page_for, make_user, upstream, side_chat_allowed
):
    account = make_user()
    page = open_parent(page_for, account)
    asked = question()
    answered_aside(page, upstream, asked)
    side_id = stored_side_chat(account, asked)["id"]
    with account.client() as client:
        deleted = client.delete(f"/api/v1/chats/{side_id}")
    assert deleted.status_code == 200, deleted.text

    main_chat(page).show()
    page.reload()
    tab = page.get_by_role("tab", name=asked, exact=True)
    expect(tab).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    tab.click()

    expect(page.locator("[data-sonner-toast]").first).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    panel = page.get_by_role("tabpanel", name=asked, exact=True)
    expect(panel.locator("[role='textbox'][id*='chat-input']")).to_be_visible()
    expect(panel.get_by_text(ANSWER)).to_have_count(0)
