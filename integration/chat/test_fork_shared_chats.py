"""Journey: forking a chat someone shared with you, from a share link or a chat you can only read.

Fork chat on a reply of a shared link or of someone else's chat stores a copy of the conversation
up to that reply as a new chat of the forking account. A fork through a share link holds the
snapshot the link shows and never a message the owner sent after sharing, even when asked for by
id. A fork of someone else's chat leaves the owner's folder, pin, chat variables, tags, settings
and share mode behind, while the owner's own fork keeps them. Each question in a fork names its
author: the owner's questions pass to the account that forked, another member's keep theirs. The
fork stops at the chosen reply. A reply still running blocks a fork of the live chat with 409 but
not a fork of its link, an account without the import permission cannot fork at all, and an
account that cannot read a chat or its link cannot fork either.

Discriminates: passes on dev 948432b43. In a backend copy of `fork_chat_by_id`, forking the
owner's live chat behind a share id fails the snapshot, open link and running reply tests,
treating every caller as the owner fails the non-owner and author tests, skipping the author
loop for non-owners fails the author tests, forking the whole branch whatever reply was chosen
fails the snapshot and stop tests, removing the 409 fails the running reply test, removing the
import permission check fails the permission test, and looking the chat up without the access
check fails the open link and access tests.
"""

from __future__ import annotations

import threading
import uuid

import httpx
import pytest

from harness import upstream as reply
from harness.access import grant, make_group
from harness.chat import ask, send_message, wait_for_reply

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

PERMISSIONS = "/api/v1/users/default/permissions"
OWNERS_SETTINGS = ("params", "tool_servers", "tool_ids", "filter_ids", "variables", "share_mode")
FORK_META = {"forked_from", "forked_from_message_id"}


def _answered(owner, upstream, question: str, **options) -> tuple[str, str]:
    """`owner` asks `question`; returns the chat id and the reply's id."""
    upstream.queue(reply.text(f"answer to {question}", match=reply.answering(question)))
    with owner.client() as client:
        turn, stored = ask(client, question, **options)
    assert stored["content"] == f"answer to {question}"
    return turn.chat_id, turn.assistant_message_id


def _save_access(owner, chat_id: str, grants: list[dict], share_mode: str | None) -> None:
    with owner.client() as client:
        saved = client.post(
            f"/api/v1/chats/shared/{chat_id}/access/update",
            json={"access_grants": grants, "share_mode": share_mode},
        )
    assert saved.status_code == 200, saved.text


def _share(owner, chat_id: str, members, share_mode: str | None = None, extra=()) -> str:
    """Share as the dialog does, with read access for `members`; returns the share id."""
    with owner.client() as client:
        shared = client.post(f"/api/v1/chats/{chat_id}/share", json={"share_mode": None})
    assert shared.status_code == 200, shared.text
    grants = [grant("user", member.id, "read") for member in members]
    _save_access(owner, chat_id, [*grants, *extra], share_mode)
    return shared.json()["share_id"]


def _fork(account, chat_id: str, message_id: str | None = None) -> httpx.Response:
    with account.client() as client:
        return client.post(f"/api/v1/chats/{chat_id}/fork", json={"message_id": message_id})


def _forked(account, chat_id: str, message_id: str | None = None) -> dict:
    """The fork as stored, which the fork route answers with."""
    forked = _fork(account, chat_id, message_id)
    assert forked.status_code == 200, forked.text
    return forked.json()


def _stored(account, chat_id: str) -> dict:
    with account.client() as client:
        stored = client.get(f"/api/v1/chats/{chat_id}")
    assert stored.status_code == 200, stored.text
    return stored.json()


def _own_chat_ids(account) -> set[str]:
    with account.client() as client:
        listed = client.get("/api/v1/chats/")
    assert listed.status_code == 200, listed.text
    return {chat["id"] for chat in listed.json()}


def _messages(chat: dict) -> dict:
    return chat["chat"]["history"]["messages"]


def _questions(chat: dict) -> list[dict]:
    return [message for message in _messages(chat).values() if message["role"] == "user"]


def test_a_fork_of_a_share_link_holds_the_snapshot_and_none_of_the_later_messages(
    make_user, upstream
):
    owner, reader = make_user(), make_user()
    chat_id, first_reply = _answered(owner, upstream, "what is the capital of Norway?")
    share_id = _share(owner, chat_id, [reader])
    later = f"sent after sharing {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text(f"answer to {later}", match=reply.answering(later)))
    with owner.client() as client:
        later_turn, _ = ask(client, later, chat_id=chat_id, parent_id=first_reply)

    fork = _forked(reader, share_id)
    asked_for_later = _fork(reader, share_id, later_turn.assistant_message_id)

    assert fork["user_id"] == reader.id
    assert fork["id"] in _own_chat_ids(reader)
    assert "answer to what is the capital of Norway?" in str(fork["chat"])
    assert later not in str(fork["chat"]), "the fork of a share link holds a later message"
    assert asked_for_later.status_code == 404, asked_for_later.text
    assert later not in asked_for_later.text
    assert later in str(_stored(owner, chat_id)["chat"])


def test_an_open_link_forks_as_its_snapshot_and_its_chat_id_not_at_all(admin, make_user, upstream):
    owner, stranger = make_user(), make_user()
    make_group(admin, [owner], {"sharing": {"open_chats": True}})
    chat_id, first_reply = _answered(owner, upstream, "before sharing")
    share_id = _share(owner, chat_id, [], "continue", extra=[grant("anyone", "*", "read")])
    later = f"after sharing {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text(f"answer to {later}", match=reply.answering(later)))
    with owner.client() as client:
        ask(client, later, chat_id=chat_id, parent_id=first_reply)
    chats_before = _own_chat_ids(stranger)

    fork = _forked(stranger, share_id)
    by_chat_id = _fork(stranger, chat_id)

    assert "answer to before sharing" in str(fork["chat"])
    assert later not in str(fork["chat"]), "an open link forked as the owner's live chat"
    assert by_chat_id.status_code == 401, by_chat_id.text
    assert later not in by_chat_id.text
    assert _own_chat_ids(stranger) == chats_before | {fork["id"]}


def _furnished(owner, upstream) -> tuple[str, str, dict]:
    """The owner's answered chat, filed, pinned, tagged, with variables and a system prompt.

    Returns the chat id, the folder's id and the variables.
    """
    chat_id, _ = _answered(owner, upstream, "plan my week")
    variables = {"codename": f"Bluebird {uuid.uuid4().hex[:6]}"}
    with owner.client() as client:
        folder = client.post("/api/v1/folders/", json={"name": f"folder {uuid.uuid4().hex[:6]}"})
        assert folder.status_code == 200, folder.text
        folder_id = folder.json()["id"]
        steps = [
            client.post(f"/api/v1/chats/{chat_id}/folder", json={"folder_id": folder_id}),
            client.post(f"/api/v1/chats/{chat_id}/pin"),
            client.post(f"/api/v1/chats/{chat_id}/tags", json={"name": "private-plans"}),
            client.post(
                f"/api/v1/chats/{chat_id}", json={"chat": {"params": {"system": "Brief."}}}
            ),
            client.post(f"/api/v1/chats/{chat_id}", json={"chat": {}, "variables": variables}),
        ]
    for step in steps:
        assert step.status_code == 200, step.text
    return chat_id, folder_id, variables


def _tag_names(account, chat_id: str) -> list[str]:
    with account.client() as client:
        tags = client.get(f"/api/v1/chats/{chat_id}/tags")
    assert tags.status_code == 200, tags.text
    return [tag["name"] for tag in tags.json()]


@pytest.mark.parametrize(
    "reached_by", ["allow-replies link", "allow-replies chat", "folder shared for reading"]
)
def test_a_fork_of_someone_elses_chat_leaves_their_folder_pin_variables_and_tags_behind(
    make_user, upstream, reached_by
):
    owner, member = make_user(), make_user()
    chat_id, folder_id, _ = _furnished(owner, upstream)
    if reached_by == "folder shared for reading":
        with owner.client() as client:
            shared = client.post(
                f"/api/v1/folders/{folder_id}/access/update",
                json={"access_grants": [grant("user", member.id, "read")]},
            )
        assert shared.status_code == 200, shared.text
        source = chat_id
    else:
        share_id = _share(owner, chat_id, [member], "continue")
        source = share_id if reached_by == "allow-replies link" else chat_id

    fork = _forked(member, source)

    assert fork["user_id"] == member.id
    assert "answer to plan my week" in str(fork["chat"])
    assert fork["folder_id"] is None, "the fork was filed in the owner's folder"
    assert fork["pinned"] is False, "the fork came pinned with the owner's pin"
    assert fork["variables"] == {}, f"the fork holds the owner's variables: {fork['variables']}"
    for key in OWNERS_SETTINGS:
        assert key not in fork["chat"], f"the fork holds the owner's {key}: {fork['chat'][key]}"
    assert set(fork["meta"]) <= FORK_META, f"the fork holds the owner's metadata: {fork['meta']}"
    assert _tag_names(member, fork["id"]) == []
    assert fork["share_id"] is None


def test_the_owners_own_fork_keeps_its_folder_pin_variables_and_tags(make_user, upstream):
    owner = make_user()
    chat_id, folder_id, variables = _furnished(owner, upstream)

    fork = _forked(owner, chat_id)

    assert fork["id"] != chat_id
    assert fork["folder_id"] == folder_id
    assert fork["pinned"] is True
    assert fork["variables"] == variables
    assert fork["chat"]["params"] == {"system": "Brief."}
    assert _tag_names(owner, fork["id"]) == ["private-plans"]


@pytest.mark.parametrize("forked_from", ["live chat", "frozen link"])
def test_every_question_in_a_fork_names_its_author(make_user, upstream, forked_from):
    owner, member, forker = make_user(), make_user(), make_user()
    chat_id, owner_reply = _answered(owner, upstream, "what is the capital of Norway?")
    share_id = _share(owner, chat_id, [member, forker], "continue")
    member_question = f"and of Sweden? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Stockholm.", match=reply.answering(member_question)))
    with member.client() as client:
        ask(client, member_question, chat_id=chat_id, parent_id=owner_reply)
    if forked_from == "frozen link":
        _save_access(
            owner,
            chat_id,
            [grant("user", account.id, "read") for account in (member, forker)],
            None,
        )
    source = share_id if forked_from == "frozen link" else chat_id

    questions = {message["content"]: message for message in _questions(_forked(forker, source))}

    assert set(questions) == {"what is the capital of Norway?", member_question}
    for content, question in questions.items():
        author = question.get("user") or {}
        assert author.get("name") and author.get("id") == question["user_id"], (
            f"the question {content!r} names no author in the fork: {author}"
        )
    owners_question = questions["what is the capital of Norway?"]
    assert owners_question["user"] == {"id": forker.id, "name": forker.name}, (
        "the owner's question did not pass to the account that forked"
    )
    assert questions[member_question]["user"] == {"id": member.id, "name": member.name}, (
        "the member's question lost its author in the fork"
    )


def test_a_fork_stops_at_the_chosen_reply(make_user, upstream):
    owner, reader = make_user(), make_user()
    chat_id, first_reply = _answered(owner, upstream, "first question")
    upstream.queue(reply.text("second answer", match=reply.answering("second question")))
    with owner.client() as client:
        _, second = ask(client, "second question", chat_id=chat_id, parent_id=first_reply)
    share_id = _share(owner, chat_id, [reader])

    at_first = _forked(reader, share_id, first_reply)
    whole = _forked(reader, share_id)

    assert [message["content"] for message in at_first["chat"]["messages"]] == [
        "first question",
        "answer to first question",
    ]
    assert len(_messages(at_first)) == 2, f"the fork holds {len(_messages(at_first))} messages"
    assert at_first["chat"]["history"]["currentId"] == first_reply
    assert _messages(at_first)[first_reply]["childrenIds"] == []
    assert "second" not in str(at_first["chat"]), "the fork went past the chosen reply"
    assert whole["chat"]["history"]["currentId"] == second["id"]
    assert "second answer" in str(whole["chat"])


def test_a_running_reply_blocks_a_fork_of_the_live_chat_but_not_of_its_link(make_user, upstream):
    owner, reader = make_user(), make_user()
    chat_id, first_reply = _answered(owner, upstream, "first question")
    share_id = _share(owner, chat_id, [reader])
    running = f"still answering {uuid.uuid4().hex[:6]}"
    release = threading.Event()
    upstream.queue(reply.text("done at last", hold=release, match=reply.answering(running)))
    try:
        with owner.client() as client:
            turn = send_message(client, running, chat_id=chat_id, parent_id=first_reply)
        while_running = _fork(owner, chat_id)
        link_fork = _fork(reader, share_id)
    finally:
        release.set()
    with owner.client() as client:
        wait_for_reply(client, turn)
    after = _forked(owner, chat_id)

    assert while_running.status_code == 409, while_running.text
    assert "Wait for the current response" in while_running.json()["detail"]
    assert link_fork.status_code == 200, link_fork.text
    assert "answer to first question" in str(_stored(reader, link_fork.json()["id"])["chat"])
    assert "done at last" in str(after["chat"])


def test_without_the_import_permission_no_fork_is_made(admin, make_user, upstream, preserve):
    preserve("permissions")
    owner, reader = make_user(), make_user()
    chat_id, _ = _answered(owner, upstream, "not for copying")
    share_id = _share(owner, chat_id, [reader])
    with admin.client() as client:
        permissions = client.get(PERMISSIONS).json()
        barred = {**permissions, "chat": {**permissions["chat"], "import": False}}
        client.post(PERMISSIONS, json=barred).raise_for_status()
    owners_chats, readers_chats = _own_chat_ids(owner), _own_chat_ids(reader)

    by_reader = _fork(reader, share_id)
    by_owner = _fork(owner, chat_id)

    assert by_reader.status_code == 403, by_reader.text
    assert by_owner.status_code == 403, by_owner.text
    assert _own_chat_ids(reader) == readers_chats
    assert _own_chat_ids(owner) == owners_chats


def test_an_account_that_cannot_read_a_chat_or_its_link_cannot_fork_it(
    instance, make_user, upstream
):
    owner, reader, stranger = make_user(), make_user(), make_user()
    chat_id, _ = _answered(owner, upstream, f"private question {uuid.uuid4().hex[:6]}")
    share_id = _share(owner, chat_id, [reader])
    readers_chats = _own_chat_ids(reader)

    refused = {
        "stranger by link": _fork(stranger, share_id),
        "stranger by chat id": _fork(stranger, chat_id),
        "reader by chat id": _fork(reader, chat_id),
        "signed out by link": httpx.post(
            f"{instance.base_url}/api/v1/chats/{share_id}/fork", json={}, timeout=60.0
        ),
    }
    with owner.client() as client:
        client.delete(f"/api/v1/chats/{chat_id}/share").raise_for_status()
    refused["reader after unsharing"] = _fork(reader, share_id)

    for who, response in refused.items():
        assert response.status_code in (401, 403, 404), f"{who}: {response.status_code}"
        assert "private question" not in response.text, f"{who} got the chat: {response.text}"
    assert _own_chat_ids(stranger) == set()
    assert _own_chat_ids(reader) == readers_chats
