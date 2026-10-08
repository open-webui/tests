"""Journey: people keep talking in a chat shared with "Allow replies" or filed in a shared folder.

The owner shares a chat with the share dialog's two calls (create the link, then save who may open
it and the mode) or shares a folder with the folder dialog, and the people with a grant join the
chat in their tab. A turn from anyone reaches every joined tab live, as a shared `chat:messages`
event with the author on the user message and then the reply through `chat:completion`. A tab
without access, or with a "Clone only" grant, cannot join. Removing a member from the group,
or taking a grant off the folder, sends the joined tabs a `chat:access` event, drops the tab that
lost access from the live feed and closes the chat to that account. A folder reader can reply in
the owner's chats only while the folder is in "Allow replies" mode. Only the owner shares a folder
and sets its mode: a member with a write grant is refused, while their ordinary Edit Folder save
still works.

Discriminates: passes on dev b5a20423e; in a backend copy, `get_event_emitter` skipping the
shared room emit (`if shared_event:` made false) fails the live turn test and the group removal
test, the `events:chat` join handler skipping the access check fails the join test, the group
removal test and the folder grant removal test, `refresh_chat_access` skipping `leave_room`
fails the folder grant removal test, `refresh_chat_access` never emitting `chat:access` fails
the group removal test and the folder grant removal test, `get_accessible_chat_by_id` dropping
the folder `share_mode == 'continue'` condition fails the folder reply test,
`update_folder_access_by_id` letting a write grant change sharing fails the folder sharing test,
and the folder update route skipping its share mode check fails the same test.
"""

from __future__ import annotations

import time
import uuid

import pytest

from harness import upstream as reply
from harness.access import grant, make_group
from harness.chat import ask, send_message
from harness.chat_history import seed_chat
from harness.socket_client import SocketSession, connected

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

FOLDER_SHARING = {"sharing": {"folders": True}}
QUIET_SECONDS = 3.0


def _start_chat(owner, upstream, question: str) -> tuple[str, str]:
    """A chat with one answered question; returns its id and the reply's id."""
    answer = f"answer to {question}"
    upstream.queue(reply.text(answer, match=reply.answering(question)))
    with owner.client() as client:
        turn, stored = ask(client, question)
    assert stored["content"] == answer
    return turn.chat_id, turn.assistant_message_id


def _share_continue(
    owner, chat_id: str, *grants: dict, share_mode: str | None = "continue"
) -> None:
    """Share the chat as the dialog does: create the link, then save the grants and the mode."""
    with owner.client() as client:
        shared = client.post(f"/api/v1/chats/{chat_id}/share", json={"share_mode": None})
        assert shared.status_code == 200, shared.text
        saved = client.post(
            f"/api/v1/chats/shared/{chat_id}/access/update",
            json={"access_grants": list(grants), "share_mode": share_mode},
        )
    assert saved.status_code == 200, saved.text


def _reply_in(actor, upstream, chat_id: str, parent_id: str, question: str):
    """The actor's next turn in the chat; returns the turn and the stored answer."""
    answer = f"answer to {question}"
    upstream.queue(reply.text(answer, match=reply.answering(question)))
    with actor.client() as client:
        turn, stored = ask(client, question, chat_id=chat_id, parent_id=parent_id)
    assert stored["content"] == answer
    return turn, stored


def _join(session: SocketSession, chat_id: str) -> bool:
    return session.client.call(
        "events:chat", {"chat_id": chat_id, "data": {"type": "join"}}, timeout=30
    )


def _shared_entries(session: SocketSession, chat_id: str, event_type: str) -> list[dict]:
    return [
        entry
        for entry in list(session.events)
        if entry.get("chat_id") == chat_id
        and entry.get("shared") is True
        and entry["data"].get("type") == event_type
    ]


def _entry_for_turn(session: SocketSession, chat_id: str, user_message_id: str, timeout=30.0):
    """The shared `chat:messages` event that carries the user message with this id."""
    deadline = time.monotonic() + timeout
    while True:
        for entry in _shared_entries(session, chat_id, "chat:messages"):
            if user_message_id in entry["data"]["data"]["messages"]:
                return entry
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.05)


def _finished_reply(session: SocketSession, chat_id: str, message_id: str, timeout=30.0):
    """The shared `chat:completion` event that ends the reply with this id."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for entry in _shared_entries(session, chat_id, "chat:completion"):
            if entry["message_id"] == message_id and entry["data"]["data"].get("done"):
                return entry
        time.sleep(0.05)
    return None


def _wait_for_access_event(session: SocketSession, chat_id: str, timeout=30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _shared_entries(session, chat_id, "chat:access"):
            return True
        time.sleep(0.05)
    return False


def _folder_owner(make_user, admin):
    owner = make_user()
    make_group(admin, [owner], FOLDER_SHARING)
    return owner


def _create_folder(owner) -> str:
    with owner.client() as client:
        created = client.post("/api/v1/folders/", json={"name": f"Quay {uuid.uuid4().hex[:6]}"})
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _filed_chat(owner, folder_id: str) -> tuple[str, str]:
    """A seeded chat of the owner's moved into the folder; returns its id and last message id."""
    with owner.client() as client:
        chat_id, last_id = seed_chat(
            client,
            [
                {"role": "user", "content": "when is high tide?"},
                {"role": "assistant", "content": "High tide at noon."},
            ],
        )
        moved = client.post(f"/api/v1/chats/{chat_id}/folder", json={"folder_id": folder_id})
    assert moved.status_code == 200, moved.text
    return chat_id, last_id


def _share_folder(owner, folder_id: str, grants: list[dict], share_mode: str | None):
    """Save the folder's grants and mode, as the folder share dialog does."""
    with owner.client() as client:
        saved = client.post(
            f"/api/v1/folders/{folder_id}/access/update",
            json={"access_grants": grants, "share_mode": share_mode},
        )
    assert saved.status_code == 200, saved.text


def _stored_folder(owner, folder_id: str) -> dict:
    with owner.client() as client:
        folder = client.get(f"/api/v1/folders/{folder_id}")
    assert folder.status_code == 200, folder.text
    return folder.json()


def _grant_keys(folder: dict) -> set[tuple[str, str, str]]:
    return {
        (item["principal_type"], item["principal_id"], item["permission"])
        for item in folder.get("access_grants") or []
    }


def test_the_owner_sees_a_members_turn_live_and_the_member_sees_the_owners(make_user, upstream):
    owner, member = make_user(), make_user()
    chat_id, reply_id = _start_chat(owner, upstream, "which tide table do you keep?")
    _share_continue(owner, chat_id, grant("user", member.id, "read"))

    with connected(owner) as owner_tab, connected(member) as member_tab:
        assert _join(owner_tab, chat_id) is True
        assert _join(member_tab, chat_id) is True

        member_turn, _ = _reply_in(member, upstream, chat_id, reply_id, "who keeps the lamp?")
        arrived = _entry_for_turn(owner_tab, chat_id, member_turn.user_message_id)
        finished = _finished_reply(owner_tab, chat_id, member_turn.assistant_message_id)

        assert arrived is not None, "the owner's tab never got the member's turn"
        messages = arrived["data"]["data"]["messages"]
        user_message = messages[member_turn.user_message_id]
        assert arrived["user_id"] == member.id
        assert user_message["content"] == "who keeps the lamp?"
        assert user_message["user_id"] == member.id
        assert user_message["user"] == {"id": member.id, "name": member.name}
        assert messages[member_turn.assistant_message_id]["role"] == "assistant"
        assert finished is not None, "the owner's tab never got the end of the reply"
        streamed = _shared_entries(owner_tab, chat_id, "chat:completion")
        assert "answer to who keeps the lamp?" in str(streamed)

        owner_turn, _ = _reply_in(
            owner, upstream, chat_id, member_turn.assistant_message_id, "and who lights it?"
        )
        seen_by_member = _entry_for_turn(member_tab, chat_id, owner_turn.user_message_id)

        assert seen_by_member is not None, "the member's tab never got the owner's turn"
        assert seen_by_member["user_id"] == owner.id
        assert seen_by_member["data"]["data"]["messages"][owner_turn.user_message_id]["user"] == {
            "id": owner.id,
            "name": owner.name,
        }


def test_only_accounts_that_can_read_the_live_chat_may_join_it(make_user, upstream):
    owner = make_user(role="admin")
    live_member, clone_only_member, stranger, open_visitor = (make_user() for _ in range(4))
    live_chat, _ = _start_chat(owner, upstream, "a chat to reply in")
    clone_chat, _ = _start_chat(owner, upstream, "a chat to clone")
    open_chat, _ = _start_chat(owner, upstream, "a chat anyone can open")
    anyone = {"principal_type": "anyone", "principal_id": "*", "permission": "read"}
    _share_continue(owner, live_chat, grant("user", live_member.id, "read"))
    _share_continue(owner, clone_chat, grant("user", clone_only_member.id, "read"), share_mode=None)
    _share_continue(owner, open_chat, anyone)

    with (
        connected(owner) as owner_tab,
        connected(live_member) as live_tab,
        connected(clone_only_member) as clone_only_tab,
        connected(stranger) as stranger_tab,
        connected(open_visitor) as open_tab,
    ):
        assert _join(owner_tab, live_chat) is True
        assert _join(live_tab, live_chat) is True
        assert _join(clone_only_tab, live_chat) is False
        assert _join(stranger_tab, live_chat) is False
        assert _join(clone_only_tab, clone_chat) is False
        assert _join(open_tab, open_chat) is False


def test_removing_a_member_from_the_group_cuts_their_live_feed(admin, make_user, upstream):
    owner, leaver, stayer = make_user(), make_user(), make_user()
    group_id = make_group(admin, [leaver, stayer])
    chat_id, reply_id = _start_chat(owner, upstream, "who is on the pier?")
    _share_continue(owner, chat_id, grant("group", group_id, "read"))

    with (
        connected(owner) as owner_tab,
        connected(leaver) as leaver_tab,
        connected(stayer) as stayer_tab,
    ):
        assert _join(owner_tab, chat_id) is True
        assert _join(leaver_tab, chat_id) is True
        assert _join(stayer_tab, chat_id) is True

        with admin.client() as client:
            removed = client.post(
                f"/api/v1/groups/id/{group_id}/users/remove", json={"user_ids": [leaver.id]}
            )
        assert removed.status_code == 200, removed.text
        assert _wait_for_access_event(stayer_tab, chat_id), "no chat:access event for the stayer"

        turn, _ = _reply_in(owner, upstream, chat_id, reply_id, "is the pier clear now?")
        assert _entry_for_turn(owner_tab, chat_id, turn.user_message_id) is not None
        assert _entry_for_turn(stayer_tab, chat_id, turn.user_message_id) is not None, (
            "the member who stayed lost the feed"
        )
        time.sleep(QUIET_SECONDS)
        assert _entry_for_turn(leaver_tab, chat_id, turn.user_message_id, timeout=0) is None

        # a reconnecting tab rejoins the chat it had open
        with connected(leaver) as reconnected_tab:
            assert _join(reconnected_tab, chat_id) is False
        with leaver.client() as client:
            read = client.get(f"/api/v1/chats/{chat_id}")
            assert read.status_code in {401, 404}, read.text
            with pytest.raises(AssertionError, match="chat request refused: HTTP 404"):
                send_message(
                    client, "let me back in", chat_id=chat_id, parent_id=turn.assistant_message_id
                )


def test_a_folder_reader_replies_only_while_the_folder_allows_replies(admin, make_user, upstream):
    owner = _folder_owner(make_user, admin)
    reader = make_user()
    folder_id = _create_folder(owner)
    chat_id, last_id = _filed_chat(owner, folder_id)
    reader_read = [grant("user", reader.id, "read")]
    _share_folder(owner, folder_id, reader_read, "continue")

    turn, _ = _reply_in(reader, upstream, chat_id, last_id, "is the tide table up to date?")

    with owner.client() as client:
        stored = client.get(f"/api/v1/chats/{chat_id}").json()["chat"]["history"]["messages"]
    assert stored[turn.user_message_id]["content"] == "is the tide table up to date?"
    assert stored[turn.user_message_id]["user_id"] == reader.id
    assert stored[turn.user_message_id]["user"] == {"id": reader.id, "name": reader.name}
    assert stored[turn.assistant_message_id]["content"] == "answer to is the tide table up to date?"

    _share_folder(owner, folder_id, reader_read, None)

    with reader.client() as client:
        assert client.get(f"/api/v1/chats/{chat_id}").status_code == 200
        with pytest.raises(AssertionError, match="chat request refused: HTTP 404"):
            send_message(
                client, "one more thing", chat_id=chat_id, parent_id=turn.assistant_message_id
            )

    with owner.client() as client:
        after = client.get(f"/api/v1/chats/{chat_id}").json()["chat"]["history"]["messages"]
    assert not [m for m in after.values() if m.get("content") == "one more thing"]


def test_taking_a_grant_off_a_folder_cuts_the_readers_live_feed(admin, make_user, upstream):
    owner = _folder_owner(make_user, admin)
    reader = make_user()
    folder_id = _create_folder(owner)
    chat_id, last_id = _filed_chat(owner, folder_id)
    _share_folder(owner, folder_id, [grant("user", reader.id, "read")], "continue")

    with connected(reader) as reader_tab:
        assert _join(reader_tab, chat_id) is True

        _share_folder(owner, folder_id, [], "continue")

        assert _wait_for_access_event(reader_tab, chat_id), "no chat:access event for the reader"
        assert _join(reader_tab, chat_id) is False
        with reader.client() as client:
            read = client.get(f"/api/v1/chats/{chat_id}")
        assert read.status_code in {401, 404}, read.text

        turn, _ = _reply_in(owner, upstream, chat_id, last_id, "is anyone still listening?")
        time.sleep(QUIET_SECONDS)
        assert _entry_for_turn(reader_tab, chat_id, turn.user_message_id, timeout=0) is None


def test_only_the_owner_shares_a_folder_and_sets_its_mode(admin, make_user):
    owner = _folder_owner(make_user, admin)
    writer, stranger = make_user(), make_user()
    folder_id = _create_folder(owner)
    writer_grants = [grant("user", writer.id, "read"), grant("user", writer.id, "write")]
    _share_folder(owner, folder_id, writer_grants, "continue")
    before = _stored_folder(owner, folder_id)

    with writer.client() as client:
        widened = client.post(
            f"/api/v1/folders/{folder_id}/access/update",
            json={
                "access_grants": [*writer_grants, grant("user", stranger.id, "read")],
                "share_mode": None,
            },
        )
        # the Edit Folder dialog sends the folder's data back with the share mode in it
        edit_form = {
            "name": f"Renamed {uuid.uuid4().hex[:6]}",
            "meta": {"background_image_url": None},
            "data": {"system_prompt": "", "files": [], **before["data"], "model_ids": []},
        }
        edited = client.post(f"/api/v1/folders/{folder_id}/update", json=edit_form)

    assert widened.status_code == 403, widened.text
    assert edited.status_code == 200, edited.text
    after_edit = _stored_folder(owner, folder_id)
    assert after_edit["name"] == edit_form["name"]
    assert after_edit["data"]["share_mode"] == "continue"
    assert _grant_keys(after_edit) == _grant_keys(before)

    _share_folder(owner, folder_id, writer_grants, None)
    with writer.client() as client:
        stale_edit = client.post(f"/api/v1/folders/{folder_id}/update", json=edit_form)

    assert stale_edit.status_code == 403, stale_edit.text
    assert _stored_folder(owner, folder_id)["data"]["share_mode"] is None

    with owner.client() as client:
        set_back = client.post(
            f"/api/v1/folders/{folder_id}/update",
            json={**edit_form, "data": {**edit_form["data"], "share_mode": "continue"}},
        )
    assert set_back.status_code == 200, set_back.text
    assert _stored_folder(owner, folder_id)["data"]["share_mode"] == "continue"
    widened_by_owner = [*writer_grants, grant("user", stranger.id, "read")]
    _share_folder(owner, folder_id, widened_by_owner, None)
    final = _stored_folder(owner, folder_id)
    assert _grant_keys(final) == {
        (g["principal_type"], g["principal_id"], g["permission"]) for g in widened_by_owner
    }
    assert final["data"]["share_mode"] is None
