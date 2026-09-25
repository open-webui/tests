"""Journey: what one account can see or change of another's through the socket.

A note's live document serves and accepts edits only from an account the note is shared with:
a stranger who joins under either spelling of the document id is served nothing, cannot follow
the note's HTTP edits through `join-note` and cannot push cursor updates to the owner's tab, and
neither a stranger's nor a reader's edit reaches the owner's open tab, a later tab or the saved
note. The routes that push an event into a chat owner's `user:{id}` room (a message event, a
message edit) take the chat id from the caller, so a stranger naming someone else's chat is
refused and the owner's socket gets nothing. Each case has a granted account or the owner as
its positive control.

Discriminates: in a backend copy, dropping the read check from `ydoc_document_join` turned the
stranger cases of the served test (both spellings) and the cursor test red;
`normalize_document_id` returning its input turned both `note_` cases of the served test red;
dropping the write check from `yjs_document_update` turned the reader case of the open-tab test
red, and together with the join check the stranger-edit test too; dropping the read check from
`join_note` turned the join-note test red; dropping the room check from `yjs_awareness_update`
turned the cursor test red; dropping the owner check from the message event route and from the
message edit route each turned that route's stranger case red.
"""

from __future__ import annotations

import threading
import time
import uuid

import pytest

from harness.socket_client import connected, note_text

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

ARRIVAL_TIMEOUT = 15.0
QUIET_PERIOD = 2.0
SAVE_WAIT = 10.0
# the server saves half a second after the last edit
REFUSED_SAVE_WAIT = 2.0


def _create_note(owner) -> str:
    with owner.client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={"title": f"scope {uuid.uuid4().hex[:8]}", "data": {"content": {"md": ""}}},
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _share_note(owner, note_id: str, account, permission: str | None) -> None:
    # a writer gets read and write, as the share dialog sets
    permissions = {None: [], "read": ["read"], "write": ["read", "write"]}[permission]
    grants = [
        {"principal_type": "user", "principal_id": account.id, "permission": granted}
        for granted in permissions
    ]
    if not grants:
        return
    with owner.client() as client:
        shared = client.post(
            f"/api/v1/notes/{note_id}/access/update", json={"access_grants": grants}
        )
    assert shared.status_code == 200, shared.text


def _saved_markdown(owner, note_id: str, expected: str, wait: float = SAVE_WAIT) -> str:
    """The note's stored markdown once it reads `expected`, or as it is when the wait ends."""
    deadline = time.monotonic() + wait
    with owner.client() as client:
        while True:
            note = client.get(f"/api/v1/notes/{note_id}")
            assert note.status_code == 200, note.text
            markdown = ((note.json().get("data") or {}).get("content") or {}).get("md")
            if markdown == expected or time.monotonic() > deadline:
                return markdown
            time.sleep(0.2)


def _flag_on(session, event: str) -> threading.Event:
    arrived = threading.Event()
    session.client.on(event, lambda *_: arrived.set())
    return arrived


def _arrives(flag: threading.Event, expected: bool) -> bool:
    return flag.wait(ARRIVAL_TIMEOUT if expected else QUIET_PERIOD)


@pytest.mark.parametrize("spelling", ["note:{}", "note_{}"])
@pytest.mark.parametrize("permission", [None, "read"])
def test_only_a_shared_account_is_served_the_notes_live_document(make_user, spelling, permission):
    owner, visitor = make_user(), make_user()
    note_id = _create_note(owner)
    _share_note(owner, note_id, visitor, permission)
    document_id = spelling.format(note_id)

    with connected(owner) as owners_tab:
        owners_tab.join_note(note_id)
        owners_tab.edit_note(note_id, "owner only")
        with connected(visitor) as visitors_tab:
            visitors_tab.call("ydoc:document:join", {"document_id": document_id})
            visitors_tab.call("ydoc:document:state", {"document_id": document_id})
            if permission is None:
                time.sleep(QUIET_PERIOD)
                served = [note_text(state["state"]) for state in visitors_tab.document_states]
            else:
                served = [visitors_tab.note_state(note_id)]

    if permission is None:
        assert served == [], f"a stranger joining {document_id!r} was served the note: {served}"
    else:
        assert served == ["<paragraph>owner only</paragraph>"]


def test_a_strangers_edit_reaches_no_tab_and_is_not_saved(make_user):
    owner, stranger = make_user(), make_user()
    note_id = _create_note(owner)

    with connected(owner) as owners_tab:
        owners_tab.join_note(note_id)
        with connected(stranger) as strangers_tab:
            strangers_tab.join_note(note_id)
            strangers_tab.edit_note(note_id, "written by a stranger")
            time.sleep(QUIET_PERIOD)
        with connected(owner) as later_tab:
            later_tab.join_note(note_id)
            document = later_tab.note_state(note_id)

    assert owners_tab.document_updates == [], "the stranger's edit reached the owner's open tab"
    assert document == "", "the stranger's edit went into the live document"
    saved = _saved_markdown(owner, note_id, "written by a stranger", wait=REFUSED_SAVE_WAIT)
    assert saved == ""


@pytest.mark.parametrize("permission", ["read", "write"])
def test_only_a_writers_edit_reaches_the_owners_open_tab(make_user, permission):
    owner, collaborator = make_user(), make_user()
    note_id = _create_note(owner)
    _share_note(owner, note_id, collaborator, permission)
    may_write = permission == "write"

    with connected(owner) as owners_tab, connected(collaborator) as collaborators_tab:
        owners_tab.join_note(note_id)
        collaborators_tab.join_note(note_id)
        collaborators_tab.edit_note(note_id, "from the collaborator")
        try:
            passed_on = owners_tab.note_update(
                note_id, timeout=ARRIVAL_TIMEOUT if may_write else QUIET_PERIOD
            )
        except AssertionError:
            passed_on = None

    if may_write:
        assert passed_on is not None
        assert note_text(passed_on["update"]) == "<paragraph>from the collaborator</paragraph>"
    else:
        assert passed_on is None, "a reader's edit reached the owner's open tab"


@pytest.mark.parametrize("permission", [None, "read"])
def test_join_note_follows_the_note_only_for_a_shared_account(make_user, permission):
    owner, visitor = make_user(), make_user()
    note_id = _create_note(owner)
    _share_note(owner, note_id, visitor, permission)
    shared = permission is not None

    with connected(visitor) as visitors_tab:
        arrived = _flag_on(visitors_tab, "events:note")
        visitors_tab.call("join-note", {"auth": {"token": visitor.token}, "note_id": note_id})
        with owner.client() as client:
            edited = client.post(f"/api/v1/notes/{note_id}/update", json={"title": "renamed"})
        assert edited.status_code == 200, edited.text
        received = _arrives(arrived, shared)

    assert received == shared, (
        f"after join-note the owner's edits {'did not arrive' if shared else 'reached a stranger'}"
    )


@pytest.mark.parametrize("permission", [None, "read"])
def test_only_an_account_in_the_document_reaches_the_owners_cursors(make_user, permission):
    owner, visitor = make_user(), make_user()
    note_id = _create_note(owner)
    _share_note(owner, note_id, visitor, permission)
    shared = permission is not None

    with connected(owner) as owners_tab, connected(visitor) as visitors_tab:
        arrived = _flag_on(owners_tab, "ydoc:awareness:update")
        owners_tab.join_note(note_id)
        visitors_tab.join_note(note_id)
        visitors_tab.call(
            "ydoc:awareness:update", {"document_id": f"note:{note_id}", "update": [1, 0]}
        )
        received = _arrives(arrived, shared)

    assert received == shared, (
        f"a {'reader' if shared else 'stranger'}'s cursor update "
        f"{'did not arrive' if shared else 'reached the owner'}"
    )


def _create_chat(owner) -> tuple[str, str]:
    """A stored chat with one assistant message; its id and the message's id."""
    message_id = str(uuid.uuid4())
    message = {
        "id": message_id,
        "parentId": None,
        "childrenIds": [],
        "role": "assistant",
        "content": "the owner's reply",
        "done": True,
    }
    chat = {
        "title": "owned",
        "history": {"currentId": message_id, "messages": {message_id: message}},
    }
    with owner.client() as client:
        created = client.post("/api/v1/chats/new", json={"chat": chat})
    assert created.status_code == 200, created.text
    return created.json()["id"], message_id


# route suffix, request body, the event type the owner's socket gets
PUSHES = {
    "message event": ("/event", {"type": "status", "data": {"description": "pushed"}}, "status"),
    "message edit": ("", {"content": "pushed"}, "chat:message"),
}


@pytest.mark.parametrize("push", list(PUSHES))
@pytest.mark.parametrize("sender", ["owner", "stranger"])
def test_only_the_chat_owner_can_push_an_event_into_their_room(make_user, push, sender):
    owner, stranger = make_user(), make_user()
    chat_id, message_id = _create_chat(owner)
    suffix, body, event_type = PUSHES[push]
    pusher = owner if sender == "owner" else stranger

    with connected(owner) as owners_tab:
        with pusher.client() as client:
            pushed = client.post(
                f"/api/v1/chats/{chat_id}/messages/{message_id}{suffix}", json=body
            )
        try:
            delivered = owners_tab.wait_for(
                chat_id, event_type, timeout=ARRIVAL_TIMEOUT if sender == "owner" else QUIET_PERIOD
            )
        except AssertionError:
            delivered = None

    if sender == "owner":
        assert pushed.status_code == 200, pushed.text
        assert delivered is not None
    else:
        assert pushed.status_code == 401, f"a stranger's {push} was accepted: {pushed.text}"
        assert delivered is None, f"a stranger's {push} reached the owner's socket: {delivered}"
