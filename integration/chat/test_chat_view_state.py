"""Journey: each person's tab layout of a chat is saved on the server, for them alone.

Since d4e8493cc (dev, migration 6a23bd107e54) the workspace of a chat (its tabs, panes, splits,
active tab and tab history) is saved per person and chat with `POST /api/v1/chats/{id}/view` and
read back with `GET`, so it comes back after a reload and on another device. A view is private
to the person who saved it, also on a chat shared with "Allow replies": the owner, a reader and
an admin who can open the chat each read back only their own, and a person who saved none reads
`null`. A person who cannot open the chat (a stranger, a "Clone only" reader, an unknown id)
gets 404 for both. A view that does not add up (a tab twice, a tab in two panes or in none, an
active tab outside its pane, a split ratio of 0 or 1, an unknown tab kind or field, a pane
missing from the layout, another schema version) is refused with 422, and one over 1 MB with
413, also when it is sent chunked; a refused save leaves the saved view as it was. Deleting an
account removes its views, deleting a chat (alone, with its folder or with all of the owner's
chats) removes every person's view of it, and deleting the owner's account removes the views
others saved of the owner's chats. A save the browser drops halfway (the tab closed or reloaded
while the layout was being saved) leaves no error traceback in the server log and the next save
works (open-webui/open-webui#33924).

Discriminates: passes on dev f659c0d12; on d4e8493cc^ and v0.13.0 there is no such route. In
backend copies of f659c0d12: `ChatViewStates.get` reading any viewer's row fails the privacy
test, both routes skipping the access check fail the 404 test, `valid_references` returning at
once fails the nine bad layout cases about references (the other six fail on the field schema),
the size cap raised to 8 MB fails the 413 test, the user and chat deletions keeping the rows fail
the account and chat deletion test, and the folder and all-chats deletions keeping them fail the
folder test. The dropped save test fails on dev b6534c58f, which logs "Exception in ASGI
application" with the client's disconnect as its cause.
"""

from __future__ import annotations

import copy
import json
import socket
import uuid
from urllib.parse import urlsplit

import pytest

from harness.access import grant
from harness.backends import read_rows, wait_until
from harness.chat_history import seed_chat

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

LIMIT = 1024 * 1024


def _view(label: str = "Chat", ratio: float = 0.35) -> dict:
    """A conversation pane beside a Files pane, as the workspace saves it after a split."""
    return {
        "schemaVersion": 1,
        "selectedTerminalId": "terminal-a",
        "panels": {
            "tabs": [
                {"id": "main-chat", "kind": "chat", "label": label},
                {
                    "id": "files:terminal-a",
                    "kind": "files",
                    "label": "Files",
                    "terminalId": "terminal-a",
                    "directory": "/home/user/project",
                },
                {"id": "comments", "kind": "comments", "label": "Comments"},
            ],
            "activeId": "files:terminal-a",
            "history": ["main-chat", "comments"],
            "groups": [
                {"id": "left", "tabIds": ["main-chat", "comments"], "activeId": "main-chat"},
                {"id": "right", "tabIds": ["files:terminal-a"], "activeId": "files:terminal-a"},
            ],
            "activeGroupId": "right",
            "layout": {
                "type": "split",
                "id": "split-1",
                "direction": "horizontal",
                "ratio": ratio,
                "first": {"type": "group", "groupId": "left"},
                "second": {"type": "group", "groupId": "right"},
            },
        },
    }


def _chat(actor) -> str:
    with actor.client() as client:
        chat_id, _ = seed_chat(client, [{"role": "user", "content": "plan the trip"}])
    return chat_id


def _share(owner, chat_id: str, *readers, share_mode: str | None = "continue") -> None:
    """Share the chat with the readers as the share dialog does: the link, then grants and mode."""
    grants = [grant("user", reader.id, "read") for reader in readers]
    with owner.client() as client:
        shared = client.post(f"/api/v1/chats/{chat_id}/share", json={"share_mode": None})
        assert shared.status_code == 200, shared.text
        saved = client.post(
            f"/api/v1/chats/shared/{chat_id}/access/update",
            json={"access_grants": grants, "share_mode": share_mode},
        )
    assert saved.status_code == 200, saved.text


def _save(actor, chat_id: str, state: dict):
    with actor.client() as client:
        return client.post(f"/api/v1/chats/{chat_id}/view", json={"state": state})


def _saved(actor, chat_id: str, state: dict) -> None:
    response = _save(actor, chat_id, state)
    assert response.status_code == 200, f"saving the view failed: {response.text}"
    assert isinstance(response.json()["updated_at"], int), response.text


def _read(actor, chat_id: str):
    with actor.client() as client:
        return client.get(f"/api/v1/chats/{chat_id}/view")


def _read_state(actor, chat_id: str):
    response = _read(actor, chat_id)
    assert response.status_code == 200, f"reading the view failed: {response.text}"
    return response.json()["state"]


def _rows(instance, user_id: str) -> list[dict]:
    return read_rows(
        instance,
        "SELECT chat_id FROM chat_view_state WHERE user_id = :user_id",
        {"user_id": user_id},
    )


def test_each_person_reads_back_only_the_view_they_saved(admin, make_user):
    owner, reader, silent = make_user(), make_user(), make_user()
    chat_id = _chat(owner)
    _share(owner, chat_id, reader, silent)

    assert _read_state(owner, chat_id) is None, "a chat nobody arranged has a saved view"
    owners_view, readers_view = _view("Owner's trip"), _view("Reader's copy", ratio=0.7)
    _saved(owner, chat_id, owners_view)
    assert _read_state(reader, chat_id) is None, "the reader got the owner's layout"
    _saved(reader, chat_id, readers_view)

    assert _read_state(owner, chat_id) == owners_view, "the owner's view changed or came back"
    assert _read_state(reader, chat_id) == readers_view, "the reader's view did not come back"
    assert _read_state(silent, chat_id) is None, "a reader who saved nothing got a layout"
    assert _read_state(admin, chat_id) is None, "the admin got someone else's layout"

    # saving again replaces the person's own view and nobody else's
    moved = _view("Owner's trip", ratio=0.5)
    _saved(owner, chat_id, moved)
    assert _read_state(owner, chat_id) == moved
    assert _read_state(reader, chat_id) == readers_view


def test_a_person_who_cannot_open_the_chat_gets_404(make_user):
    owner, stranger, clone_only = make_user(), make_user(), make_user()
    chat_id = _chat(owner)
    _share(owner, chat_id, clone_only, share_mode=None)
    _saved(owner, chat_id, _view())

    for person, who in ((stranger, "a stranger"), (clone_only, "a Clone only reader")):
        read = _read(person, chat_id)
        assert read.status_code == 404, f"{who} reading the view got {read.status_code}"
        write = _save(person, chat_id, _view("taken over"))
        assert write.status_code == 404, f"{who} saving a view got {write.status_code}"
    unknown = str(uuid.uuid4())
    assert _read(owner, unknown).status_code == 404, "an unknown chat has a view"
    assert _save(owner, unknown, _view()).status_code == 404, "an unknown chat took a view"
    assert _read_state(owner, chat_id) == _view(), "a refused save changed the owner's view"


def _without_group(state: dict) -> dict:
    state["panels"]["layout"] = {"type": "group", "groupId": "left"}
    return state


def _set(path: str, value):
    def change(state: dict) -> dict:
        target = state
        *parents, last = path.split(".")
        for key in parents:
            target = target[int(key) if key.isdigit() else key]
        target[int(last) if last.isdigit() else last] = value
        return state

    return change


BAD_VIEWS = {
    "a tab listed twice": lambda s: (s["panels"]["tabs"].append(s["panels"]["tabs"][1]), s)[1],
    "a tab in two panes": _set(
        "panels.groups.0.tabIds", ["main-chat", "comments", "files:terminal-a"]
    ),
    "a tab in no pane": _set("panels.groups.0.tabIds", ["main-chat"]),
    "an active tab outside its pane": _set("panels.groups.0.activeId", "files:terminal-a"),
    "an active tab that is not the active pane's": _set("panels.activeId", "main-chat"),
    "history naming a closed tab": _set("panels.history", ["gone"]),
    "an unknown active pane": _set("panels.activeGroupId", "nowhere"),
    "a split ratio of 1": _set("panels.layout.ratio", 1.0),
    "a split ratio of 0": _set("panels.layout.ratio", 0.0),
    "an unknown tab kind": _set("panels.tabs.2.kind", "spreadsheet"),
    "an unknown field": _set("panels.tabs.0.secret", "x"),
    "a pane missing from the layout": _without_group,
    "a pane twice in the layout": _set(
        "panels.layout.second", {"type": "group", "groupId": "left"}
    ),
    "another schema version": _set("schemaVersion", 2),
    "a label that is not text": _set("panels.tabs.0.label", 7),
}


@pytest.mark.parametrize("problem", sorted(BAD_VIEWS))
def test_a_view_that_does_not_add_up_is_refused_with_422(make_user, problem):
    owner = make_user()
    chat_id = _chat(owner)
    _saved(owner, chat_id, _view())

    refused = _save(owner, chat_id, BAD_VIEWS[problem](copy.deepcopy(_view("broken"))))
    assert refused.status_code == 422, f"a view with {problem} got {refused.status_code}"
    assert _read_state(owner, chat_id) == _view(), f"a view with {problem} replaced the saved one"


def test_a_body_that_is_not_a_view_is_refused_with_422(make_user):
    owner = make_user()
    chat_id = _chat(owner)
    with owner.client() as client:
        for body in (b"not json", b"{}", json.dumps({"state": _view(), "extra": 1}).encode()):
            refused = client.post(
                f"/api/v1/chats/{chat_id}/view",
                content=body,
                headers={"Content-Type": "application/json"},
            )
            assert refused.status_code == 422, f"{body[:20]!r} got {refused.status_code}"
    assert _read_state(owner, chat_id) is None


def test_a_view_over_one_megabyte_is_refused_with_413(make_user):
    owner = make_user()
    chat_id = _chat(owner)
    near_limit = _view("x" * (LIMIT - 4096))
    assert len(json.dumps({"state": near_limit})) < LIMIT
    _saved(owner, chat_id, near_limit)
    assert _read_state(owner, chat_id) == near_limit, "a view just under 1 MB did not come back"

    too_big = json.dumps({"state": _view("y" * LIMIT)}).encode()
    with owner.client() as client:
        refused = client.post(
            f"/api/v1/chats/{chat_id}/view",
            content=too_big,
            headers={"Content-Type": "application/json"},
        )
        assert refused.status_code == 413, f"a view over 1 MB got {refused.status_code}"
        # no Content-Length: the cap has to hold while the body streams in
        chunked = client.post(
            f"/api/v1/chats/{chat_id}/view",
            content=(too_big[start : start + 65536] for start in range(0, len(too_big), 65536)),
            headers={"Content-Type": "application/json"},
        )
    assert chunked.status_code == 413, f"a chunked view over 1 MB got {chunked.status_code}"
    assert _read_state(owner, chat_id) == near_limit, "a refused view replaced the saved one"


def test_views_go_with_the_account_and_the_chat(instance, admin, make_user):
    owner, reader, other = make_user(), make_user(), make_user()
    kept, deleted, others_chat = _chat(owner), _chat(owner), _chat(other)
    for chat_id in (kept, deleted):
        _share(owner, chat_id, reader)
        _saved(owner, chat_id, _view("owner"))
        _saved(reader, chat_id, _view("reader"))
    _share(other, others_chat, reader)
    _saved(reader, others_chat, _view("reader"))
    _saved(other, others_chat, _view("other"))
    assert {row["chat_id"] for row in _rows(instance, reader.id)} == {kept, deleted, others_chat}

    with owner.client() as client:
        assert client.delete(f"/api/v1/chats/{deleted}").status_code == 200
    assert {row["chat_id"] for row in _rows(instance, owner.id)} == {kept}
    assert {row["chat_id"] for row in _rows(instance, reader.id)} == {kept, others_chat}, (
        "deleting the chat left the reader's view of it behind"
    )

    with admin.client() as client:
        assert client.delete(f"/api/v1/users/{reader.id}").status_code == 200
    assert _rows(instance, reader.id) == [], "the deleted account's views are still stored"
    assert _read_state(owner, kept) == _view("owner"), "deleting the reader touched the owner"
    assert _read_state(other, others_chat) == _view("other")

    _share(owner, kept, other)
    _saved(other, kept, _view("other"))
    with admin.client() as client:
        assert client.delete(f"/api/v1/users/{owner.id}").status_code == 200
    assert _rows(instance, owner.id) == []
    assert {row["chat_id"] for row in _rows(instance, other.id)} == {others_chat}, (
        "the views others saved of the deleted owner's chats are still stored"
    )


def test_deleting_a_folder_with_its_chats_or_all_chats_removes_their_views(instance, make_user):
    owner, reader = make_user(), make_user()
    filed, loose = _chat(owner), _chat(owner)
    with owner.client() as client:
        folder = client.post("/api/v1/folders/", json={"name": f"trips {uuid.uuid4().hex[:8]}"})
        assert folder.status_code == 200, folder.text
        folder_id = folder.json()["id"]
        moved = client.post(f"/api/v1/chats/{filed}/folder", json={"folder_id": folder_id})
        assert moved.status_code == 200, moved.text
    for chat_id in (filed, loose):
        _share(owner, chat_id, reader)
        _saved(owner, chat_id, _view("owner"))
        _saved(reader, chat_id, _view("reader"))

    with owner.client() as client:
        deleted = client.delete(f"/api/v1/folders/{folder_id}", params={"delete_contents": True})
    assert deleted.status_code == 200, deleted.text
    for person in (owner, reader):
        assert {row["chat_id"] for row in _rows(instance, person.id)} == {loose}, (
            "deleting the folder with its chats left their views behind"
        )

    with owner.client() as client:
        assert client.delete("/api/v1/chats/").status_code == 200
    assert _rows(instance, owner.id) == [] and _rows(instance, reader.id) == [], (
        "deleting all chats left their views behind"
    )


def _drop_save_halfway(actor, chat_id: str) -> None:
    """Send the headers and half the body of a save, then close, as a tab closed mid-save does."""
    body = json.dumps({"state": _view("dropped")}).encode()
    address = urlsplit(actor.base_url)
    head = (
        f"POST /api/v1/chats/{chat_id}/view HTTP/1.1\r\n"
        f"Host: {address.netloc}\r\n"
        f"Authorization: Bearer {actor.token}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n\r\n"
    ).encode()
    with socket.create_connection((address.hostname, address.port), timeout=10) as connection:
        connection.sendall(head + body[: len(body) // 2])


def test_a_save_dropped_halfway_logs_no_error_and_the_next_save_works(instance, make_user):
    owner = make_user()
    chat_id = _chat(owner)
    offset = instance.log_size()

    _drop_save_halfway(owner, chat_id)
    _saved(owner, chat_id, _view("after the drop"))

    # bounded: the dropped request is handled long before this, nothing more should appear
    wait_until(lambda: "Traceback" in instance.log_since(offset), timeout=2)
    logged = instance.log_since(offset)
    assert "Traceback" not in logged, f"a dropped view save logged an error:\n{logged[-2000:]}"
    assert _read_state(owner, chat_id) == _view("after the drop"), "the next save was lost"
