"""Journey: the "is typing" line of a chat shared with "Allow replies" or filed in a shared folder.

A tab that has the chat open sends `events:chat` with a `typing` event while its account types
in the message box. The server passes it on to every other tab in the chat's room, the sender's
own other tabs included, as a shared event that names only the sender's id and name. It drops
the event, answering false, when the tab has not joined the chat's room, when the account may
read the chat but not reply in it (a read-only folder, an admin looking at someone's chat), and
for an id that is no saved chat. So nobody outside the chat learns who is typing there, and a
reader who cannot reply never shows up as typing.

Discriminates: passes on dev 226dca18c; in a backend copy, the typing handler skipping its room
check fails the room test, skipping its write check (`permission='write'` dropped) fails the
write access test, relaying the client's `data` as sent fails the payload test, and emitting
with no `skip_sid` fails the tabs test. The id check for chats that are not saved is also
covered by the room check, since no tab can join such a chat's room.
"""

from __future__ import annotations

import time
import uuid

import pytest

from harness.access import grant, make_group
from harness.chat_history import seed_chat
from harness.socket_client import SocketSession, connected

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

FOLDER_SHARING = {"sharing": {"folders": True}}
QUIET_SECONDS = 3.0


def _owned_chat(owner) -> tuple[str, str]:
    with owner.client() as client:
        return seed_chat(
            client,
            [
                {"role": "user", "content": "when does the ferry leave?"},
                {"role": "assistant", "content": "The ferry leaves at nine."},
            ],
        )


def _share_continue(owner, chat_id: str, *grants: dict) -> None:
    """Share the chat as the dialog does with "Allow replies": create the link, then the grants."""
    with owner.client() as client:
        shared = client.post(f"/api/v1/chats/{chat_id}/share", json={"share_mode": None})
        assert shared.status_code == 200, shared.text
        saved = client.post(
            f"/api/v1/chats/shared/{chat_id}/access/update",
            json={"access_grants": list(grants), "share_mode": "continue"},
        )
    assert saved.status_code == 200, saved.text


def _join(session: SocketSession, chat_id: str) -> bool:
    return session.call("events:chat", {"chat_id": chat_id, "data": {"type": "join"}})


def _type(session: SocketSession, chat_id: str, typing: bool = True) -> bool:
    """What the message box sends while its account types, or stops typing."""
    event = {"type": "typing", "data": {"typing": typing}}
    return session.call("events:chat", {"chat_id": chat_id, "data": event})


def _typing_events(session: SocketSession, chat_id: str) -> list[dict]:
    return [
        entry
        for entry in list(session.events)
        if entry.get("chat_id") == chat_id and entry["data"].get("type") == "typing"
    ]


def _wait_for_typing(session: SocketSession, chat_id: str, timeout=30.0) -> list[dict]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        arrived = _typing_events(session, chat_id)
        if arrived:
            return arrived
        time.sleep(0.05)
    return []


def _expected(sender, chat_id: str, typing: bool = True) -> dict:
    return {
        "chat_id": chat_id,
        "user_id": sender.id,
        "user": {"id": sender.id, "name": sender.name},
        "shared": True,
        "data": {"type": "typing", "data": {"typing": typing}},
    }


def _folder_chat(admin, make_user, reader, share_mode: str | None) -> tuple[object, str, str]:
    """An owner's chat filed in a folder shared read with `reader`; returns owner, folder, chat."""
    owner = make_user()
    make_group(admin, [owner], FOLDER_SHARING)
    with owner.client() as client:
        created = client.post("/api/v1/folders/", json={"name": f"Pier {uuid.uuid4().hex[:6]}"})
        assert created.status_code == 200, created.text
        folder_id = created.json()["id"]
    chat_id, _ = _owned_chat(owner)
    with owner.client() as client:
        moved = client.post(f"/api/v1/chats/{chat_id}/folder", json={"folder_id": folder_id})
    assert moved.status_code == 200, moved.text
    _share_folder(owner, folder_id, reader, share_mode)
    return owner, folder_id, chat_id


def _share_folder(owner, folder_id: str, reader, share_mode: str | None) -> None:
    with owner.client() as client:
        saved = client.post(
            f"/api/v1/folders/{folder_id}/access/update",
            json={"access_grants": [grant("user", reader.id, "read")], "share_mode": share_mode},
        )
    assert saved.status_code == 200, saved.text


def test_typing_reaches_every_other_tab_that_has_the_chat_open(make_user):
    owner, member, bystander = make_user(), make_user(), make_user()
    chat_id, _ = _owned_chat(owner)
    _share_continue(
        owner, chat_id, grant("user", member.id, "read"), grant("user", bystander.id, "read")
    )

    with (
        connected(owner) as owner_tab,
        connected(owner) as owner_second_tab,
        connected(member) as member_tab,
        connected(member) as member_second_tab,
        connected(bystander) as bystander_elsewhere,
    ):
        for tab in (owner_tab, owner_second_tab, member_tab, member_second_tab):
            assert _join(tab, chat_id) is True

        assert _type(member_tab, chat_id) is True
        for name, tab in (
            ("the owner's tab", owner_tab),
            ("the owner's second tab", owner_second_tab),
            ("the member's own second tab", member_second_tab),
        ):
            assert _wait_for_typing(tab, chat_id) == [_expected(member, chat_id)], (
                f"{name} did not get exactly the member's typing: {_typing_events(tab, chat_id)}"
            )

        assert _type(member_tab, chat_id, typing=False) is True
        stopped = _expected(member, chat_id, typing=False)
        deadline = time.monotonic() + 30
        while stopped not in _typing_events(owner_tab, chat_id) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert stopped in _typing_events(owner_tab, chat_id), (
            "the member stopping never reached the owner"
        )

        assert _type(owner_tab, chat_id) is True
        owner_typing = _expected(owner, chat_id)
        deadline = time.monotonic() + 30
        while (
            owner_typing not in _typing_events(member_tab, chat_id) and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        assert owner_typing in _typing_events(member_tab, chat_id), (
            "the owner's typing never reached the member"
        )

        time.sleep(QUIET_SECONDS)
        assert _typing_events(member_tab, chat_id) == [owner_typing], (
            "the member's own tab got its own typing back"
        )
        assert _typing_events(bystander_elsewhere, chat_id) == [], (
            "a tab that never opened the chat got its typing"
        )


def test_typing_names_only_its_sender_and_carries_nothing_else(make_user):
    owner, member = make_user(), make_user()
    chat_id, _ = _owned_chat(owner)
    _share_continue(owner, chat_id, grant("user", member.id, "read"))

    with connected(owner) as owner_tab, connected(member) as member_tab:
        assert _join(owner_tab, chat_id) is True
        assert _join(member_tab, chat_id) is True
        sent = member_tab.call(
            "events:chat",
            {
                "chat_id": chat_id,
                "user_id": owner.id,
                "user": {"id": owner.id, "name": owner.name, "email": "owner@example.com"},
                "data": {
                    "type": "typing",
                    "data": {"typing": True, "draft": "my half-written reply"},
                    "user": {"id": owner.id, "name": owner.name},
                    "content": "my half-written reply",
                },
            },
        )
        assert sent is True

        arrived = _wait_for_typing(owner_tab, chat_id)

    assert arrived == [_expected(member, chat_id)], (
        f"the owner's tab got more than the member's id and name, or a forged sender: {arrived}"
    )


def test_typing_from_a_tab_outside_the_chat_is_dropped(make_user):
    owner, member, stranger = make_user(), make_user(), make_user()
    chat_id, _ = _owned_chat(owner)
    _share_continue(owner, chat_id, grant("user", member.id, "read"))

    with (
        connected(owner) as owner_tab,
        connected(member) as member_tab,
        connected(stranger) as stranger_tab,
    ):
        assert _join(owner_tab, chat_id) is True

        assert _type(member_tab, chat_id) is False, (
            "a member's tab that never opened the chat sent typing"
        )
        assert _type(stranger_tab, chat_id) is False, "a stranger's tab sent typing into the chat"
        assert _join(stranger_tab, chat_id) is False
        assert _type(stranger_tab, chat_id) is False, (
            "a stranger whose join was refused sent typing"
        )
        for not_saved in (f"temporary:{uuid.uuid4()}", f"local:{uuid.uuid4()}"):
            assert _type(member_tab, not_saved) is False, f"typing was accepted for {not_saved}"
        assert _type(owner_tab, chat_id, typing="yes") is False, (
            "a typing flag that is no bool was relayed"
        )
        time.sleep(QUIET_SECONDS)
        assert _typing_events(owner_tab, chat_id) == [], (
            f"the owner got typing from outside the chat: {_typing_events(owner_tab, chat_id)}"
        )

        # the same member's tab is heard once it has the chat open
        assert _join(member_tab, chat_id) is True
        assert _type(member_tab, chat_id) is True
        assert _wait_for_typing(owner_tab, chat_id) == [_expected(member, chat_id)]


def test_only_an_account_that_can_reply_shows_as_typing(admin, make_user):
    reader = make_user()
    owner, folder_id, chat_id = _folder_chat(admin, make_user, reader, share_mode=None)
    onlooker = make_user(role="admin")

    with (
        connected(owner) as owner_tab,
        connected(reader) as reader_tab,
        connected(onlooker) as onlooker_tab,
    ):
        assert _join(owner_tab, chat_id) is True
        assert _join(reader_tab, chat_id) is True, "a folder reader could not open the chat live"
        assert _join(onlooker_tab, chat_id) is True, "an admin could not open the user's chat live"

        assert _type(reader_tab, chat_id) is False, "a reader of a read-only folder sent typing"
        assert _type(onlooker_tab, chat_id) is False, "an admin who only reads the chat sent typing"
        time.sleep(QUIET_SECONDS)
        assert _typing_events(owner_tab, chat_id) == [], (
            f"the owner saw typing from a non-replier: {_typing_events(owner_tab, chat_id)}"
        )

        _share_folder(owner, folder_id, reader, "continue")
        assert _type(reader_tab, chat_id) is True, (
            "the reader could not type once replies were allowed"
        )
        assert _wait_for_typing(owner_tab, chat_id) == [_expected(reader, chat_id)]

        _share_folder(owner, folder_id, reader, None)
        assert _type(reader_tab, chat_id) is False, (
            "the reader still sent typing after replies were turned off"
        )
