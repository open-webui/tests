"""Journey: the workspace search shows each account only what it may open.

`GET /api/v1/search` searches the caller's own chats, the chats in folders shared with them
(their subfolders included), chats shared with them, notes they may read and messages in
channels they belong to. A chat shared to continue shows as the live chat, a chat shared as a
copy as its snapshot under its `/s/` link, and a share open to everyone with the link or to every
signed-in account, without a grant of the reader's own, does not show at all. Taking a grant
back, unsharing or removing a member takes the result out again. A folder cannot be shared with
every account, not even by an admin. Note results need Notes on and the `features.notes`
permission, message results Channels and `features.channels`. Every result an account gets is
one it can open over the API.

Discriminates: journey, no fix. In backend copies of dev 948432b43: listing every folder turns
the four folder tests and the unreachable-location test red; dropping the subfolder recursion
turns the shared-folder test red; letting any grant open a snapshot turns the open-share test
red; dropping the continue check turns the copy, open-share and switching tests red; dropping
the permission filter on notes turns the note test red; listing every channel to everyone turns
both channel tests red; always enabling notes and messages turns the four switch tests red.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from harness.access import grant, make_group
from harness.actors import Actor
from harness.channel_quotes import enable_channels, group_channel, post_message
from harness.workspace_search import (
    create_folder,
    create_note,
    found,
    import_chat,
    item,
    post,
    search,
    unique_word,
)

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

ADMIN_CONFIG = "/api/v1/auths/admin/config"
PERMISSIONS = "/api/v1/users/default/permissions"


def _get(actor: Actor, path: str) -> httpx.Response:
    with actor.client() as client:
        return client.get(path)


def _opens(actor: Actor, result: dict) -> int:
    """The status the account gets opening the record behind a search result."""
    source, record_id = result["id"].split(":", 1)
    if source == "chat":
        return _get(actor, f"/api/v1/chats/{record_id}").status_code
    if source == "shared":
        return _get(actor, f"/api/v1/chats/share/{record_id}").status_code
    if source == "note":
        return _get(actor, f"/api/v1/notes/{record_id}").status_code
    channel_id = result["url"].removeprefix("/channels/").split("?")[0]
    return _get(actor, f"/api/v1/channels/{channel_id}/messages/{record_id}").status_code


def _assert_every_result_opens(actor: Actor, answer: dict) -> None:
    assert answer["items"], "the search found nothing to check"
    refused = {entry["id"]: _opens(actor, entry) for entry in answer["items"]}
    refused = {result_id: status for result_id, status in refused.items() if status != 200}
    assert not refused, f"search listed results the account cannot open: {refused}"


def _share_folder(owner: Actor, folder_id: str, grants: list[dict]) -> None:
    with owner.client() as client:
        post(client, f"/api/v1/folders/{folder_id}/access/update", access_grants=grants)


def _share_chat(owner: Actor, chat_id: str, grants: list[dict], share_mode=None) -> str:
    """Shares the chat the way the share dialog does; returns the share id."""
    with owner.client() as client:
        shared = post(client, f"/api/v1/chats/{chat_id}/share", share_mode=share_mode)
        post(
            client,
            f"/api/v1/chats/shared/{chat_id}/access/update",
            access_grants=grants,
            share_mode=share_mode,
        )
    return shared["share_id"]


def _reader_principal(admin: Actor, reader: Actor, via: str) -> tuple[str, str]:
    return ("group", make_group(admin, [reader])) if via == "group" else ("user", reader.id)


# --------------------------------------------------------------------------- folders


@pytest.fixture
def shared_folder(admin, make_user):
    """An owner's folder with a subfolder and a private folder, each holding one matching chat."""
    owner, reader, stranger = make_user(), make_user(), make_user()
    word = unique_word()
    with owner.client() as client:
        folder_id = create_folder(client, f"Projects {word}")
        subfolder_id = create_folder(client, f"Drafts {word}", parent_id=folder_id)
        private_id = create_folder(client, f"Private {word}")
        chats = {
            "folder": import_chat(client, f"Plan {word}", folder_id=folder_id),
            "subfolder": import_chat(client, "Draft", answer=f"see {word}", folder_id=subfolder_id),
            "private": import_chat(client, f"Diary {word}", folder_id=private_id),
        }
    return owner, reader, stranger, word, folder_id, chats


@pytest.mark.parametrize("via", ["user", "group"])
def test_a_shared_folder_brings_its_chats_and_its_subfolders_chats(admin, shared_folder, via):
    owner, reader, stranger, word, folder_id, chats = shared_folder
    _share_folder(owner, folder_id, [grant(*_reader_principal(admin, reader, via), "read")])

    answer = search(reader, word)

    assert {entry["id"] for entry in answer["items"]} == {
        f"chat:{chats['folder']}",
        f"chat:{chats['subfolder']}",
    }, "the reader should find the shared folder's and its subfolder's chats, not the private one"
    assert item(answer, f"chat:{chats['folder']}")["shared"] is True
    assert item(answer, f"chat:{chats['folder']}")["context"] == f"Projects {word}"
    _assert_every_result_opens(reader, answer)
    assert found(stranger, word) == set(), "an account without a grant found the owner's chats"
    assert found(owner, word) == {f"chat:{chat_id}" for chat_id in chats.values()}


def test_taking_the_folder_grant_back_takes_its_chats_out(shared_folder):
    owner, reader, _, word, folder_id, chats = shared_folder
    _share_folder(owner, folder_id, [grant("user", reader.id, "read")])
    assert f"chat:{chats['folder']}" in found(reader, word)

    _share_folder(owner, folder_id, [])

    assert found(reader, word) == set(), "the reader still finds chats of a folder taken back"


def test_a_folder_is_never_shared_with_every_account(admin, shared_folder):
    owner, _, stranger, word, folder_id, _ = shared_folder
    with admin.client() as client:
        shared = client.post(
            f"/api/v1/folders/{folder_id}/access/update",
            json={"access_grants": [grant("user", "*", "read")]},
        )
        stored = client.get(f"/api/v1/folders/{folder_id}")
    assert stored.status_code == 200, stored.text

    assert shared.status_code == 403, f"a folder took a grant for every account: {shared.text}"
    assert stored.json()["access_grants"] == []
    assert found(stranger, word) == set(), "an unrelated account found the folder's chats"


# --------------------------------------------------------------------------- chat shares


@pytest.fixture
def owners_chat(make_user):
    owner, reader = make_user(), make_user()
    word = unique_word()
    with owner.client() as client:
        chat_id = import_chat(client, f"Budget {word}", answer=f"the {word} costs ten")
    return owner, reader, word, chat_id


@pytest.mark.parametrize("via", ["user", "group"])
def test_a_chat_shared_to_continue_shows_as_the_live_chat(admin, owners_chat, via):
    owner, reader, word, chat_id = owners_chat
    principal = _reader_principal(admin, reader, via)
    _share_chat(owner, chat_id, [grant(*principal, "read")], share_mode="continue")

    answer = search(reader, word)

    assert {entry["id"] for entry in answer["items"]} == {f"chat:{chat_id}"}
    assert item(answer, f"chat:{chat_id}")["url"] == f"/c/{chat_id}?message=a"
    assert item(answer, f"chat:{chat_id}")["context"] == "Shared chat"
    _assert_every_result_opens(reader, answer)


def test_a_chat_shared_as_a_copy_shows_as_its_snapshot(owners_chat):
    owner, reader, word, chat_id = owners_chat
    share_id = _share_chat(owner, chat_id, [grant("user", reader.id, "read")])

    answer = search(reader, word)

    assert {entry["id"] for entry in answer["items"]} == {f"shared:{share_id}"}, (
        "a copy-only share should show as its snapshot, never as the live chat"
    )
    snapshot = item(answer, f"shared:{share_id}")
    assert snapshot["url"] == f"/s/{share_id}?message=a"
    assert snapshot["context"] == "Shared snapshot"
    _assert_every_result_opens(reader, answer)
    assert found(owner, word) == {f"chat:{chat_id}"}, "the owner should find only the live chat"


@pytest.mark.parametrize(
    "audience",
    [("anyone", "*"), ("user", "*")],
    ids=["anyone with the link", "every signed-in account"],
)
def test_an_open_share_without_a_grant_of_your_own_does_not_show(admin, owners_chat, audience):
    owner, reader, word, chat_id = owners_chat
    make_group(admin, [owner], {"sharing": {"open_chats": True, "public_chats": True}})
    share_id = _share_chat(owner, chat_id, [grant(*audience, "read")])
    assert _get(reader, f"/api/v1/chats/share/{share_id}").status_code == 200

    assert found(reader, word) == set(), "an open share showed up in another account's search"

    _share_chat(owner, chat_id, [grant(*audience, "read"), grant("user", reader.id, "read")])
    assert found(reader, word) == {f"shared:{share_id}"}


@pytest.mark.parametrize("share_mode", [None, "continue"], ids=["copy", "continue"])
def test_unsharing_a_chat_takes_it_out(owners_chat, share_mode):
    owner, reader, word, chat_id = owners_chat
    _share_chat(owner, chat_id, [grant("user", reader.id, "read")], share_mode=share_mode)
    assert len(found(reader, word)) == 1

    with owner.client() as client:
        unshared = client.delete(f"/api/v1/chats/{chat_id}/share")
    assert unshared.status_code == 200, unshared.text

    assert found(reader, word) == set(), "the reader still finds a chat that was unshared"


def test_switching_a_continue_share_to_a_copy_shows_the_snapshot_instead(owners_chat):
    owner, reader, word, chat_id = owners_chat
    grants = [grant("user", reader.id, "read")]
    share_id = _share_chat(owner, chat_id, grants, share_mode="continue")
    assert found(reader, word) == {f"chat:{chat_id}"}

    with owner.client() as client:
        post(
            client,
            f"/api/v1/chats/shared/{chat_id}/access/update",
            access_grants=grants,
            share_mode=None,
        )

    assert found(reader, word) == {f"shared:{share_id}"}
    assert _get(reader, f"/api/v1/chats/{chat_id}").status_code != 200


# --------------------------------------------------------------------------- notes


@pytest.mark.parametrize("via", ["user", "group"])
def test_a_note_shows_only_with_a_read_grant(admin, make_user, via):
    owner, reader, stranger = make_user(), make_user(), make_user()
    word = unique_word()
    with owner.client() as client:
        note_id = create_note(client, "Minutes", f"we agreed on the {word}")
        post(
            client,
            f"/api/v1/notes/{note_id}/access/update",
            access_grants=[grant(*_reader_principal(admin, reader, via), "read")],
        )

    answer = search(reader, word)

    assert {entry["id"] for entry in answer["items"]} == {f"note:{note_id}"}
    assert item(answer, f"note:{note_id}")["context"] == "Shared note"
    _assert_every_result_opens(reader, answer)
    assert found(stranger, word) == set(), "an account without a grant found the note"

    with owner.client() as client:
        post(client, f"/api/v1/notes/{note_id}/access/update", access_grants=[])
    assert found(reader, word) == set(), "the reader still finds a note taken back"


# --------------------------------------------------------------------------- channels


def test_channel_messages_show_only_to_members(admin, preserve, make_user):
    preserve("admin_config")
    enable_channels(admin)
    author, member, outsider = make_user(), make_user(), make_user()
    word = unique_word()
    channel_id = group_channel(author, member)
    message_id = post_message(author, channel_id, f"lunch at the {word}")

    answer = search(member, word)

    assert {entry["id"] for entry in answer["items"]} == {f"message:{message_id}"}
    _assert_every_result_opens(member, answer)
    assert found(outsider, word) == set(), "someone outside the channel found its message"

    with author.client() as client:
        post(client, f"/api/v1/channels/{channel_id}/update/members/remove", user_ids=[member.id])
    assert found(member, word) == set(), "a removed member still finds the channel's message"


def test_standard_channel_messages_show_only_with_a_read_grant(admin, preserve, make_user):
    preserve("admin_config")
    enable_channels(admin)
    reader, ungranted = make_user(), make_user()
    word = unique_word()
    with admin.client() as client:
        channel_id = post(
            client,
            "/api/v1/channels/create",
            name=f"news-{uuid.uuid4().hex[:8]}",
            type=None,
            access_grants=[grant("user", reader.id, "read")],
        )["id"]
    message_id = post_message(admin, channel_id, f"the {word} is open")

    answer = search(reader, word)

    assert {entry["id"] for entry in answer["items"]} == {f"message:{message_id}"}
    _assert_every_result_opens(reader, answer)
    assert found(ungranted, word) == set(), "an account without a grant found the message"

    with admin.client() as client:
        post(client, f"/api/v1/channels/{channel_id}/access/update", access_grants=[])
    assert found(reader, word) == set(), "the reader still finds a message of a channel taken back"


# --------------------------------------------------------------------------- switches


def _set_admin_config(admin: Actor, **flags: bool) -> None:
    with admin.client() as client:
        current = client.get(ADMIN_CONFIG)
        assert current.status_code == 200, current.text
        post(client, ADMIN_CONFIG, **{**current.json(), **flags})


def _set_feature_permission(admin: Actor, feature: str, allowed: bool) -> None:
    with admin.client() as client:
        permissions = client.get(PERMISSIONS)
        assert permissions.status_code == 200, permissions.text
        changed = permissions.json()
        changed["features"][feature] = allowed
        post(client, PERMISSIONS, **changed)


@pytest.fixture
def note_and_message(admin, preserve, make_user):
    """An account's own matching note and a matching message in its channel."""
    preserve("admin_config", "permissions")
    enable_channels(admin)
    account, other = make_user(), make_user()
    word = unique_word()
    with account.client() as client:
        note_id = create_note(client, "Shopping", f"buy a {word}")
    channel_id = group_channel(account, other)
    message_id = post_message(other, channel_id, f"who has the {word}?")
    assert found(account, word) == {f"note:{note_id}", f"message:{message_id}"}
    return account, word, note_id, message_id


@pytest.mark.parametrize("switch", ["feature off", "permission withdrawn"])
def test_note_results_leave_with_the_notes_feature(admin, note_and_message, switch):
    account, word, note_id, message_id = note_and_message
    if switch == "feature off":
        _set_admin_config(admin, ENABLE_NOTES=False)
    else:
        _set_feature_permission(admin, "notes", False)

    answer = search(account, word)

    assert {entry["id"] for entry in answer["items"]} == {f"message:{message_id}"}, (
        f"notes still show in search with the {switch}"
    )
    assert "note" not in answer["kinds"]
    assert search(account, word, kind="note")["items"] == []


@pytest.mark.parametrize("switch", ["feature off", "permission withdrawn"])
def test_message_results_leave_with_the_channels_feature(admin, note_and_message, switch):
    account, word, note_id, message_id = note_and_message
    if switch == "feature off":
        _set_admin_config(admin, ENABLE_CHANNELS=False)
    else:
        _set_feature_permission(admin, "channels", False)

    answer = search(account, word)

    assert {entry["id"] for entry in answer["items"]} == {f"note:{note_id}"}, (
        f"channel messages still show in search with the {switch}"
    )
    assert "message" not in answer["kinds"]
    assert not [scope for scope in answer["scopes"] if scope["kind"] == "channel"]
    assert search(account, word, kind="message")["items"] == []
