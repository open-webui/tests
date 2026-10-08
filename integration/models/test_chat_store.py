"""The 0.11.1 round of chat-store fixes, seen over the chat API.

* `0800c21c6` chat search read only the legacy `$.messages` array on SQLite, so the text of
  current conversations, stored under `$.history.messages`, was never found.
* `5caa91a49` a compacted chat reopened on its summary message instead of the newest turn, and
  editing an older message dragged `currentId` back onto it.
* `b933292d6` (#28035) deleting a message walked `childrenIds[-1]` without remembering where it
  had been, so in a chat whose replies loop the request never returned.
* `7d4747dfd` (#28767) the tags endpoint resolved the chat by ownership only, so an admin or a
  reader of a shared chat or folder was refused the chat's tags. Since de73bb830 a person a chat
  is shared with reads the live chat only when it is shared to continue, so that is the share.
* `1c13fedb1` (#28742) `update_chat_by_id` wrote back the whole blob it was given, so a save that
  named only `files` dropped the conversation and a stale writer dropped newer messages.

* `ce22e0bb1` (#28820) only `message['content']` was sanitized, so a null byte in any other field
  of a message reached the message's separate record raw, and PostgreSQL refused
  that write. SQLite stores the byte, so the record is read back through the admin's message
  analytics.
* `5c79ccc9e` (#28034) `get_message_list` guarded against cycles with each message's own `id`,
  which an imported message may omit, so a history whose parents loop was walked forever while
  the walk's list grew without bound. The chat stats export walks the stored history; it runs on
  an instance whose process is capped at a gigabyte more than it holds, so the runaway walk ends
  in a refused export instead of exhausting the machine.

Twin of unit/models/test_chat_store.py.

Discriminates: passes on upstream dev `bbfa876af`; with the five fixes reverted together ten
tests fail (both history searches, the compacted reopen, the older-message edit, the looping
delete by timeout, the three tag readers and both partial saves) and the nine nearby tests pass.
On dev ef67cc3fa, the message upsert back on cleaning `content` alone (and the whole chat
blob) fails the sources test, and `get_message_list` back on the `id` field
fails the looping walk; the content, plain-message and plain-chain tests pass on both.
"""

from __future__ import annotations

import uuid
from pathlib import Path

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


def test_an_ordinary_newest_message_stays_where_it_is(client):
    history_on_leaf = compacted_history()
    history_on_leaf["currentId"] = "followup"
    chat_id = create_chat(client, {"history": history_on_leaf})

    assert read_chat(client, chat_id)["history"]["currentId"] == "followup"


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


def test_a_reader_of_a_shared_chat_gets_its_tags(admin, make_user, client, tagged_chat):
    reader = make_user()
    link = client.post(f"/api/v1/chats/{tagged_chat}/share", json={"share_mode": "continue"})
    assert link.status_code == 200, link.text
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


# --- ce22e0bb1: null bytes are cleaned from every field of a saved message -----------------


def message_records(admin, chat_id: str) -> dict:
    """The chat's separate message records, as the admin's analytics list them."""
    with admin.client() as admin_client:
        listed = admin_client.get("/api/v1/analytics/messages", params={"chat_id": chat_id})
    assert listed.status_code == 200, listed.text
    return {record["id"].removeprefix(f"{chat_id}-"): record for record in listed.json()}


def send_event(client: httpx.Client, chat_id: str, message_path: str, event: dict) -> None:
    sent = client.post(f"/api/v1/chats/{chat_id}/messages/{message_path}/event", json=event)
    assert sent.status_code == 200 and sent.json() is True, sent.text


def with_null_bytes(stored) -> list[str]:
    """Every string in `stored`, keys included, that still holds a null byte."""
    if isinstance(stored, dict):
        return [found for key, value in stored.items() for found in with_null_bytes([key, value])]
    if isinstance(stored, list):
        return [found for value in stored for found in with_null_bytes(value)]
    return [stored] if isinstance(stored, str) and "\x00" in stored else []


def test_null_bytes_are_cleaned_from_a_messages_sources(client, admin):
    chat_id = create_chat(client, {"history": history("m1", message("m1", None, [], "assistant"))})
    source = {"source": {"name": "re\x00port"}, "document": ["te\x00xt"], "metadata": [{}]}

    send_event(client, chat_id, "m1", {"type": "source", "data": source})

    stored = read_chat(client, chat_id)["history"]["messages"]["m1"]
    assert stored["sources"][0]["source"]["name"] == "report", (
        "a null byte outside the message content was saved as is; PostgreSQL refuses the write"
    )
    assert not with_null_bytes(stored)
    assert not with_null_bytes(message_records(admin, chat_id)), (
        "the message's separate record kept a null byte outside the content"
    )


def test_null_bytes_are_still_cleaned_from_the_content(client, admin):
    chat_id = create_chat(client, {"history": history("m1", message("m1", None, [], "assistant"))})

    send_event(client, chat_id, "m1", {"type": "replace", "data": {"content": "be\x00fore"}})

    assert read_chat(client, chat_id)["history"]["messages"]["m1"]["content"] == "before"
    assert message_records(admin, chat_id)["m1"]["content"] == "before"


def test_a_message_without_null_bytes_is_saved_as_sent(client, admin):
    chat_id = create_chat(client, {"history": history("m1", message("m1", None, [], "assistant"))})
    source = {"source": {"name": "report"}, "document": ["text"], "metadata": [{}]}

    send_event(client, chat_id, "m1", {"type": "source", "data": source})

    stored = read_chat(client, chat_id)["history"]["messages"]["m1"]
    assert stored["sources"][0]["source"] == {"name": "report"}
    assert message_records(admin, chat_id)["m1"]["sources"][0]["source"] == {"name": "report"}


# --- 5c79ccc9e: walking a history whose parents loop and whose messages carry no id -------

# A separate instance under a memory cap: before the fix the walk grows a list without bound.
LOOPING_WALK_ENV = {"REGRESSION_THROWAWAY_INSTANCE": "looping-parent-walk"}
MEMORY_HEADROOM = 1024**3


def cap_memory(instance) -> None:
    """Let the instance's process grow by `MEMORY_HEADROOM` at most, so a runaway walk fails."""
    resource = pytest.importorskip("resource", reason="capping a process's memory needs Linux")
    status = Path(f"/proc/{instance.pid}/status")
    if not hasattr(resource, "prlimit") or not status.exists():
        pytest.skip("capping another process's memory needs Linux")
    size_line = next(line for line in status.read_text().splitlines() if line.startswith("VmSize:"))
    cap = int(size_line.split()[1]) * 1024 + MEMORY_HEADROOM
    resource.prlimit(instance.pid, resource.RLIMIT_AS, (cap, cap))


@pytest.fixture(scope="module")
def capped_instance(instance_with):
    capped = instance_with(LOOPING_WALK_ENV)
    cap_memory(capped)
    return capped


def id_less_history(current_id: str, parents: dict[str, str | None]) -> dict:
    """A history whose messages, as an import may leave them, carry no `id` of their own."""
    messages = {
        message_id: {"parentId": parent_id, "role": "user", "content": f"message {message_id}"}
        for message_id, parent_id in parents.items()
    }
    return {"currentId": current_id, "messages": messages}


def exported_message_count(admin_client: httpx.Client, chat_history: dict) -> int:
    chat_id = create_chat(admin_client, {"history": chat_history})
    exported = admin_client.get(f"/api/v1/chats/stats/export/{chat_id}", timeout=30.0)
    assert exported.status_code == 200, exported.text
    return exported.json()["stats"]["message_count"]


@pytest.mark.slow
def test_a_history_whose_parents_loop_is_walked_once(capped_instance):
    looping = id_less_history("a", {"a": "b", "b": "a"})
    with capped_instance.client() as admin_client:
        message_count = exported_message_count(admin_client, looping)

    assert message_count == 2, "a history whose parents loop was walked more than once (#28034)"


@pytest.mark.slow
def test_a_plain_history_without_ids_is_walked_from_its_root(capped_instance):
    chain = id_less_history("c", {"a": None, "b": "a", "c": "b"})
    with capped_instance.client() as admin_client:
        assert exported_message_count(admin_client, chain) == 3
