"""The 0.11.1 round of chat-store fixes, seen over the chat API.

* `0800c21c6` chat search read only the legacy `$.messages` array on SQLite, so the text of
  current conversations, stored under `$.history.messages`, was never found.
* `5caa91a49` a compacted chat reopened on its summary message instead of the newest turn, and
  editing an older message dragged `currentId` back onto it.
* `b933292d6` (#28035) deleting a message walked `childrenIds[-1]` without remembering where it
  had been, so in a chat whose replies loop the request never returned.
* `7d4747dfd` (#28767) the tags endpoint resolved the chat by ownership only, so an admin or a
  reader of a shared chat or folder was refused the chat's tags.
* `1c13fedb1` (#28742) `update_chat_by_id` wrote back the whole blob it was given, so a save that
  named only `files` dropped the conversation and a stale writer dropped newer messages.

Twin of unit/models/test_chat_store.py, which keeps the null-byte sanitizing of the separate
message record (only PostgreSQL refuses it) and `get_message_list`, whose pre-fix cycle grows a
list without bound and would exhaust the server's memory.

Discriminates: passes on upstream dev `bbfa876af`; with the five fixes reverted together ten
tests fail (both history searches, the compacted reopen, the older-message edit, the looping
delete by timeout, the three tag readers and both partial saves) and the nine nearby tests pass.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

# A separate instance: on a checkout without the delete fix the looping walk wedges its loop.
LOOPING_DELETE_ENV = {"REGRESSION_THROWAWAY_INSTANCE": "looping-chat-delete"}


def message(message_id: str, parent_id: str | None, children: list[str], role: str, **extra):
    return {
        "id": message_id,
        "parentId": parent_id,
        "childrenIds": children,
        "role": role,
        "content": f"{role} {message_id}",
        "timestamp": 1_700_000_000,
        **extra,
    }


def history(current_id: str | None, *messages: dict) -> dict:
    return {"currentId": current_id, "messages": {entry["id"]: entry for entry in messages}}


def create_chat(client: httpx.Client, chat: dict) -> str:
    created = client.post("/api/v1/chats/new", json={"chat": {"title": "Untitled", **chat}})
    assert created.status_code == 200, created.text
    return created.json()["id"]


def read_chat(client: httpx.Client, chat_id: str) -> dict:
    response = client.get(f"/api/v1/chats/{chat_id}")
    assert response.status_code == 200, response.text
    return response.json()["chat"]


def search(client: httpx.Client, text: str) -> list[str]:
    response = client.get("/api/v1/chats/search", params={"text": text})
    assert response.status_code == 200, response.text
    return [chat["id"] for chat in response.json()]


def unique_word() -> str:
    return f"pterodactyl{uuid.uuid4().hex[:8]}"


@pytest.fixture
def owner(make_user):
    return make_user()


@pytest.fixture
def client(owner):
    with owner.client() as owner_client:
        yield owner_client


# --- 0800c21c6: search reads the current storage format -----------------------------------


def test_search_finds_text_that_lives_only_in_the_history(client):
    word = unique_word()
    chat_id = create_chat(
        client,
        {"history": history("m1", message("m1", None, [], "user", content=f"the {word} plan"))},
    )

    assert search(client, word) == [chat_id], "the text of a current conversation was not searched"


def test_search_matches_every_word_of_a_query_in_any_order(client):
    word = unique_word()
    content = f"quarterly {word} budget"
    chat_id = create_chat(
        client, {"history": history("m1", message("m1", None, [], "user", content=content))}
    )

    assert search(client, f"budget {word}") == [chat_id]


def test_search_still_finds_legacy_messages_and_titles(client):
    word = unique_word()
    legacy_id = create_chat(client, {"messages": [{"role": "user", "content": f"old {word}"}]})
    titled_id = create_chat(client, {"title": f"{word} notes", "history": history(None)})

    assert set(search(client, word)) == {legacy_id, titled_id}


def test_search_leaves_out_unrelated_chats_and_other_accounts(client, make_user):
    word = unique_word()
    create_chat(client, {"history": history("m1", message("m1", None, [], "user"))})
    with make_user().client() as other_client:
        create_chat(
            other_client,
            {"history": history("m1", message("m1", None, [], "user", content=word))},
        )

    assert search(client, word) == []


# --- 5caa91a49: where a chat reopens, and edits keep the current position ------------------


def compacted_history(**summary_extra) -> dict:
    return history(
        "summary",
        message("summary", None, ["reply"], "assistant", **summary_extra),
        message("reply", "summary", ["followup"], "user"),
        message("followup", "reply", [], "assistant"),
    )


def test_a_compacted_chat_reopens_on_its_newest_turn(client):
    chat_id = create_chat(client, {"history": compacted_history(contextSummary={"tokens": 512})})

    assert read_chat(client, chat_id)["history"]["currentId"] == "followup", (
        "reopening a compacted chat parked the reader on the summary message"
    )


def test_an_ordinary_current_message_stays_where_it_is(client):
    chat_id = create_chat(client, {"history": compacted_history()})

    assert read_chat(client, chat_id)["history"]["currentId"] == "summary"


def test_editing_an_older_message_keeps_the_current_position(client):
    chat_id = create_chat(
        client,
        {
            "history": history(
                "newest",
                message("old", None, ["newest"], "user"),
                message("newest", "old", [], "assistant"),
            )
        },
    )

    edited = client.post(f"/api/v1/chats/{chat_id}/messages/old", json={"content": "edited"})
    assert edited.status_code == 200, edited.text

    stored = read_chat(client, chat_id)["history"]
    assert stored["messages"]["old"]["content"] == "edited"
    assert stored["currentId"] == "newest", "editing an older message moved the reader onto it"


def test_a_new_message_still_becomes_the_current_position(client):
    chat_id = create_chat(
        client, {"history": history("old", message("old", None, ["fresh"], "user"))}
    )

    saved = client.post(f"/api/v1/chats/{chat_id}/messages/fresh", json={"content": "second"})
    assert saved.status_code == 200, saved.text

    assert read_chat(client, chat_id)["history"]["currentId"] == "fresh"


# --- b933292d6: deleting a message in a chat whose replies loop ----------------------------


def looping_history() -> dict:
    """`a` and `b` list each other as children, so a walk down `childrenIds` loops."""
    return history(
        "victim",
        message("root", None, ["victim", "a"], "user"),
        message("victim", "root", [], "assistant"),
        message("a", "root", ["b"], "user"),
        message("b", "a", ["a"], "assistant"),
    )


@pytest.mark.slow
def test_deleting_a_message_returns_when_replies_loop(instance_with):
    throwaway = instance_with(LOOPING_DELETE_ENV)
    with throwaway.client() as admin_client:
        chat_id = create_chat(admin_client, {"history": looping_history()})
        try:
            deleted = admin_client.delete(f"/api/v1/chats/{chat_id}/messages/victim", timeout=15.0)
        except httpx.TimeoutException:
            pytest.fail("deleting a message in a chat whose replies loop never returned (#28035)")

    assert deleted.status_code == 200, deleted.text
    stored = deleted.json()["chat"]["history"]
    assert "victim" not in stored["messages"]
    assert stored["currentId"] == "b"


def test_deleting_a_message_moves_to_the_deepest_remaining_reply(client):
    chat_id = create_chat(
        client,
        {
            "history": history(
                "victim",
                message("root", None, ["victim", "a"], "user"),
                message("victim", "root", [], "assistant"),
                message("a", "root", ["b"], "user"),
                message("b", "a", [], "assistant"),
            )
        },
    )

    deleted = client.delete(f"/api/v1/chats/{chat_id}/messages/victim")
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["chat"]["history"]["currentId"] == "b"


# --- 7d4747dfd: a chat's tags for everyone who may read the chat ---------------------------


@pytest.fixture
def tagged_chat(client) -> str:
    chat_id = create_chat(client, {"history": history("m1", message("m1", None, [], "user"))})
    tagged = client.post(f"/api/v1/chats/{chat_id}/tags", json={"name": "Quarterly"})
    assert tagged.status_code == 200, tagged.text
    return chat_id


def tag_names(response: httpx.Response) -> list[str]:
    assert response.status_code == 200, f"HTTP {response.status_code} {response.text}"
    return [tag["name"] for tag in response.json()]


def test_an_admin_gets_the_tags_of_another_accounts_chat(admin, owner, tagged_chat):
    with admin.client() as admin_client:
        response = admin_client.get(f"/api/v1/chats/{tagged_chat}/tags")

    assert tag_names(response) == ["Quarterly"]
    assert {tag["user_id"] for tag in response.json()} == {owner.id}


def test_a_reader_of_a_shared_chat_gets_its_tags(admin, make_user, tagged_chat):
    reader = make_user()
    grant = {"principal_type": "user", "principal_id": reader.id, "permission": "read"}
    with admin.client() as admin_client:
        shared = admin_client.post(
            f"/api/v1/chats/shared/{tagged_chat}/access/update", json={"access_grants": [grant]}
        )
    assert shared.status_code == 200, shared.text

    with reader.client() as reader_client:
        assert tag_names(reader_client.get(f"/api/v1/chats/{tagged_chat}/tags")) == ["Quarterly"]


def test_a_reader_of_a_shared_folder_gets_its_chats_tags(admin, make_user, client, tagged_chat):
    reader = make_user()
    folder = client.post("/api/v1/folders/", json={"name": f"Shared {uuid.uuid4().hex[:6]}"})
    assert folder.status_code == 200, folder.text
    folder_id = folder.json()["id"]
    moved = client.post(f"/api/v1/chats/{tagged_chat}/folder", json={"folder_id": folder_id})
    assert moved.status_code == 200, moved.text
    grant = {"principal_type": "user", "principal_id": reader.id, "permission": "read"}
    with admin.client() as admin_client:
        shared = admin_client.post(
            f"/api/v1/folders/{folder_id}/access/update", json={"access_grants": [grant]}
        )
    assert shared.status_code == 200, shared.text

    with reader.client() as reader_client:
        assert tag_names(reader_client.get(f"/api/v1/chats/{tagged_chat}/tags")) == ["Quarterly"]


def test_the_owner_still_gets_their_tags(client, tagged_chat):
    assert tag_names(client.get(f"/api/v1/chats/{tagged_chat}/tags")) == ["Quarterly"]


def test_a_stranger_and_a_missing_chat_are_refused(make_user, tagged_chat):
    with make_user().client() as stranger_client:
        assert stranger_client.get(f"/api/v1/chats/{tagged_chat}/tags").status_code == 401
        assert stranger_client.get(f"/api/v1/chats/{uuid.uuid4()}/tags").status_code == 401


# --- 1c13fedb1: a partial save keeps what it did not name ----------------------------------


def test_a_save_naming_only_files_keeps_the_conversation(client):
    chat_id = create_chat(
        client,
        {"models": ["gpt-4"], "history": history("m1", message("m1", None, [], "user"))},
    )

    saved = client.post(f"/api/v1/chats/{chat_id}", json={"chat": {"files": [{"id": "f1"}]}})
    assert saved.status_code == 200, saved.text

    stored = read_chat(client, chat_id)
    assert stored["files"] == [{"id": "f1"}]
    assert "m1" in stored.get("history", {}).get("messages", {}), (
        "a save that named only files wiped the conversation (#28742)"
    )
    assert stored["models"] == ["gpt-4"]


def test_a_stale_writer_keeps_a_message_saved_since_its_read(client):
    first = message("m1", None, [], "user")
    chat_id = create_chat(client, {"history": history("m1", first)})
    newer = history("m2", first, message("m2", "m1", [], "assistant"))
    assert client.post(f"/api/v1/chats/{chat_id}", json={"chat": {"history": newer}}).is_success

    stale = client.post(
        f"/api/v1/chats/{chat_id}", json={"chat": {"history": history("m1", first)}}
    )
    assert stale.status_code == 200, stale.text

    assert set(read_chat(client, chat_id)["history"]["messages"]) == {"m1", "m2"}, (
        "a writer holding an older copy deleted the message saved since (#28742)"
    )


def test_a_save_still_renames_and_strips_null_bytes(client):
    chat_id = create_chat(client, {"history": history(None)})

    saved = client.post(f"/api/v1/chats/{chat_id}", json={"chat": {"title": "Re\x00named"}})

    assert saved.status_code == 200, saved.text
    assert saved.json()["title"] == "Renamed"
    assert read_chat(client, chat_id)["title"] == "Renamed"


def test_a_save_to_a_missing_chat_is_refused(client):
    assert (
        client.post(f"/api/v1/chats/{uuid.uuid4()}", json={"chat": {"title": "x"}}).status_code
        == 401
    )
