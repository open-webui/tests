"""Journey: what the workspace search filters keep and the links its results carry.

`GET /api/v1/search` takes a content type (`kind`), whose items to show (`sharing`: the caller's
own or others') and a location (`scope`: a folder with its subfolders, or a channel). A folder
location keeps only chats and a channel location only that channel's messages; a location the
caller cannot reach answers with nothing. Archived chats and the hidden chats behind notes stay
out. One page holds 24 results and `has_more` says another follows. A chat result links to its
`/c/` page with `?message=` on the matching message, a channel message to its channel with
`?thread=` (the thread's first message) and `&message=`, and a note to the note with `?page=` on
the page holding the match (since v0.12.5, b34885efc). Since ec83763f1 (#33181) the words
`tag:`, `folder:`, `pinned:`, `shared:` and `archived:` filter the caller's own chats, and a
query with one of them leaves out notes, messages, shared snapshots and other accounts' chats.

Discriminates: journey, no fix. In backend copies of dev 948432b43: ignoring `kind` turns the
content type tests red, ignoring `sharing` the created-by tests, ignoring `scope` the two
location tests, skipping the unknown-location check the unreachable-location test; keeping
archived and internal chats turns the hidden-chat, paging and `archived` filter tests red; a page
of 25 turns the paging test red; links without the message turn both link tests red; a note
link always naming the first page turns the note link test red; with
ec83763f1 undone (no filter words), or with filter words no longer limited to the caller's own
chats, the six filter-word tests go red.

Since 07029af63 a chat whose messages are kept in their own rows is searched there, and its
result shows the matching excerpt and links to that message; before, only the copy in the chat
record was searched. The test empties the record's copy to reach that state. It passes on dev
e8c0ad85e and fails on a backend copy with 07029af63 reverted (the chat is not found).

#34162: since 07029af63 (v0.13.1) the search reads those rows, and a chat started, imported or
continued with a null byte in a message kept it in the message's row. PostgreSQL cannot read
JSON holding one, so one such row made the search and the chat list search answer 500. Messages
are now cleaned when they are saved, and migration 5d2b7e9c4a10 removes the null bytes already
stored in message rows, chat records and shared copies on PostgreSQL, leaving a message that
quotes the six characters of the escape alone. The cleanup tests store such data on an instance
of their own, run the migration again by hand and then use every reader: the search, the chat
list search, the inbox and a message update. The chat list search also stops hiding chats
that quote the escape's text or carry "x00" in their title, which a 2025 workaround for the
same bug filtered out on PostgreSQL. Discriminates: on dev 782f4b705 the three save tests
fail on both databases (the row keeps the byte), the quoting and "x00" tests fail on PostgreSQL
and the cleanup tests fail (no such migration); with a cleanup that does nothing
the cleanup tests fail with 500s, and with one that drops the quoted escape's text as well the
migration breaks the JSON and fails. All pass with the fix, f3d995ff4.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

import pytest

from harness import upstream as reply
from harness.access import grant
from harness.actors import Actor, admin_of, create_user
from harness.backends import on_postgres, read_rows, write_rows
from harness.channel_quotes import enable_channels, group_channel, post_message
from harness.chat import ask
from harness.prepared_data import manual_alembic
from harness.workspace_search import (
    PAGE_SIZE,
    conversation,
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


@pytest.fixture
def workspace(admin, preserve, make_user):
    """One word matched by the account's own chat, note and message and by others' shared ones."""
    preserve("admin_config")
    enable_channels(admin)
    account, other = make_user(), make_user()
    word = unique_word()
    ids = {}
    with account.client() as client:
        folder_id = create_folder(client, f"Garden {word}")
        ids["own chat"] = import_chat(client, f"Seeds {word}", folder_id=folder_id)
        ids["own note"] = create_note(client, "Plot", f"the {word} bed")
    with other.client() as client:
        shared_folder_id = create_folder(client, f"Allotment {word}")
        post(
            client,
            f"/api/v1/folders/{shared_folder_id}/access/update",
            access_grants=[grant("user", account.id, "read")],
        )
        ids["shared chat"] = import_chat(client, f"Tools {word}", folder_id=shared_folder_id)
        ids["shared note"] = create_note(client, "Rota", f"water the {word}")
        post(
            client,
            f"/api/v1/notes/{ids['shared note']}/access/update",
            access_grants=[grant("user", account.id, "read")],
        )
    channel_id = group_channel(account, other)
    ids["own message"] = post_message(account, channel_id, f"my {word} grew")
    ids["shared message"] = post_message(other, channel_id, f"your {word} is huge")
    result_ids = {
        "own chat": f"chat:{ids['own chat']}",
        "own note": f"note:{ids['own note']}",
        "own message": f"message:{ids['own message']}",
        "shared chat": f"chat:{ids['shared chat']}",
        "shared note": f"note:{ids['shared note']}",
        "shared message": f"message:{ids['shared message']}",
    }
    folders = {"own": folder_id, "shared": shared_folder_id}
    return account, other, word, result_ids, folders, channel_id


def _results(result_ids: dict, *names: str) -> set[str]:
    return {result_ids[name] for name in names}


def test_everything_matching_shows_without_filters(workspace):
    account, _, word, result_ids, _, _ = workspace

    answer = search(account, word)

    assert {entry["id"] for entry in answer["items"]} == set(result_ids.values())
    assert sorted(answer["kinds"]) == ["chat", "message", "note"]


@pytest.mark.parametrize(
    "kind, expected",
    [
        ("chat", ("own chat", "shared chat")),
        ("note", ("own note", "shared note")),
        ("message", ("own message", "shared message")),
    ],
)
def test_a_content_type_keeps_only_that_type(workspace, kind, expected):
    account, _, word, result_ids, _, _ = workspace

    answer = search(account, word, kind=kind)

    assert {entry["id"] for entry in answer["items"]} == _results(result_ids, *expected)
    assert {entry["kind"] for entry in answer["items"]} == {kind}


@pytest.mark.parametrize(
    "sharing, expected",
    [
        ("mine", ("own chat", "own note", "own message")),
        ("shared", ("shared chat", "shared note", "shared message")),
    ],
)
def test_created_by_you_and_shared_with_you_split_the_results(workspace, sharing, expected):
    account, _, word, result_ids, _, _ = workspace

    assert found(account, word, sharing=sharing) == _results(result_ids, *expected)


def test_a_folder_location_keeps_only_that_folders_chats(workspace):
    account, _, word, result_ids, folders, _ = workspace
    with account.client() as client:
        subfolder_id = create_folder(client, "Sprouts", parent_id=folders["own"])
        nested = import_chat(client, f"Nested {word}", folder_id=subfolder_id)

    own = search(account, word, scope=f"folder:{folders['own']}")
    shared = search(account, word, scope=f"folder:{folders['shared']}")

    assert {entry["id"] for entry in own["items"]} == {result_ids["own chat"], f"chat:{nested}"}, (
        "a folder location should keep its and its subfolders' chats, no notes or messages"
    )
    assert {entry["id"] for entry in shared["items"]} == {result_ids["shared chat"]}
    scope_ids = {scope["id"] for scope in own["scopes"]}
    assert {f"folder:{folders['own']}", f"folder:{folders['shared']}"} <= scope_ids


def test_a_channel_location_keeps_only_that_channels_messages(workspace, make_user):
    account, other, word, result_ids, _, channel_id = workspace
    elsewhere = group_channel(account, make_user())
    post_message(account, elsewhere, f"another {word}")

    answer = search(account, word, scope=f"channel:{channel_id}")

    assert {entry["id"] for entry in answer["items"]} == _results(
        result_ids, "own message", "shared message"
    ), "a channel location should keep only that channel's messages"
    assert f"channel:{channel_id}" in {scope["id"] for scope in answer["scopes"]}


def test_a_location_you_cannot_reach_finds_nothing(workspace, make_user):
    account, _, word, _, folders, channel_id = workspace
    stranger = make_user()
    assert found(account, word, scope=f"folder:{folders['own']}")

    unreachable = {
        "another account's folder": search(stranger, word, scope=f"folder:{folders['own']}"),
        "a channel you are not in": search(stranger, word, scope=f"channel:{channel_id}"),
        "a location that does not exist": search(account, word, scope=f"folder:{uuid.uuid4()}"),
    }

    leaked = {name: answer["items"] for name, answer in unreachable.items() if answer["items"]}
    assert not leaked, f"a location the account cannot reach returned results: {leaked}"


def test_archived_and_note_chats_stay_out(make_user):
    account = make_user()
    word = unique_word()
    with account.client() as client:
        kept = import_chat(client, f"Kept {word}")
        import_chat(client, f"Archived {word}", archived=True)
        note_id = create_note(client, "Recipe", "flour and water")
        note_chat = client.get(f"/api/v1/notes/{note_id}/chat")
        assert note_chat.status_code == 200, note_chat.text

    recent = found(account)

    assert found(account, word) == {f"chat:{kept}"}, "an archived chat showed in the results"
    assert f"chat:{kept}" in recent
    assert f"chat:{note_chat.json()['id']}" not in recent, "a note's hidden chat showed in search"


def test_more_than_a_page_of_results_says_another_page_follows(make_user):
    account = make_user()
    word = unique_word()
    with account.client() as client:
        entries = [{"chat": {"title": f"Log {index:02d} {word}"}} for index in range(PAGE_SIZE + 1)]
        imported = client.post("/api/v1/chats/import", json={"chats": entries})
    assert imported.status_code == 200, imported.text

    first = search(account, word)
    second = search(account, word, page=2)

    assert (len(first["items"]), first["has_more"]) == (PAGE_SIZE, True)
    assert (len(second["items"]), second["has_more"]) == (1, False)
    pages = {entry["id"] for entry in first["items"]} | {entry["id"] for entry in second["items"]}
    assert len(pages) == PAGE_SIZE + 1, "the two pages overlap or miss a chat"

    with account.client() as client:
        archived = client.post(f"/api/v1/chats/{second['items'][0]['id'][5:]}/archive")
    assert archived.status_code == 200, archived.text
    exactly_one_page = search(account, word)
    assert (len(exactly_one_page["items"]), exactly_one_page["has_more"]) == (PAGE_SIZE, False)


# --------------------------------------------------------------------------- links


def test_a_chat_result_links_to_the_matching_message(make_user):
    account = make_user()
    word = unique_word()
    with account.client() as client:
        in_answer = import_chat(client, "Weather", answer=f"rain over the {word}")
        in_title = import_chat(client, f"Forecast {word}")

    answer = search(account, word)

    assert item(answer, f"chat:{in_answer}")["url"] == f"/c/{in_answer}?message=a"
    assert word in item(answer, f"chat:{in_answer}")["snippet"]
    assert item(answer, f"chat:{in_title}")["url"] == f"/c/{in_title}"


def keep_messages_in_rows_only(instance, chat_id: str) -> None:
    """Empty the message text in the chat record, leaving it in the chat's message rows."""
    [row] = read_rows(instance, "SELECT chat FROM chat WHERE id = :id", {"id": chat_id})
    chat = row["chat"] if isinstance(row["chat"], dict) else json.loads(row["chat"])
    for message in chat["history"]["messages"].values():
        message.update(content="", output=[])
    statement = (
        "UPDATE chat SET chat = CAST(:chat AS JSON) WHERE id = :id"
        if on_postgres(instance)
        else "UPDATE chat SET chat = :chat WHERE id = :id"
    )
    write_rows(instance, statement, [{"chat": json.dumps(chat), "id": chat_id}])


def test_a_chat_whose_messages_live_in_their_own_rows_is_found_with_an_excerpt(
    instance, make_user, upstream
):
    account = make_user()
    word = unique_word()
    question = f"what grows in the shade? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text(f"ferns and the {word} do well", match=reply.answering(question)))
    with account.client() as client:
        turn, stored = ask(client, question)
    assert word in stored["content"], stored
    keep_messages_in_rows_only(instance, turn.chat_id)

    answer = search(account, word)

    assert f"chat:{turn.chat_id}" in {entry["id"] for entry in answer["items"]}, (
        "a chat whose messages are stored apart from the chat record is not found by its text"
    )
    found_chat = item(answer, f"chat:{turn.chat_id}")
    assert word in found_chat["snippet"], f"the result shows no excerpt of the reply: {found_chat}"
    assert found_chat["url"] == f"/c/{turn.chat_id}?message={turn.assistant_message_id}"


def test_a_note_result_links_to_the_page_holding_the_match(make_user):
    account = make_user()
    word = unique_word()
    page_id = str(uuid.uuid4())
    with account.client() as client:
        note_id = create_note(client, "Trip", "general packing list")
        post(
            client,
            f"/api/v1/notes/{note_id}/pages/update",
            action="create",
            new_id=page_id,
            title="Budget",
            revision=0,
        )
        post(
            client,
            f"/api/v1/notes/{note_id}/update",
            data={"content": {"md": f"the {word} tour costs extra"}},
            page_id=page_id,
        )

    result = item(search(account, word), f"note:{note_id}")

    assert result["url"] == f"/notes/{note_id}?page={page_id}&search={word}", result
    assert word in result["snippet"] and "packing" not in result["snippet"], result


def test_a_channel_message_links_to_its_thread_and_message(admin, preserve, make_user):
    preserve("admin_config")
    enable_channels(admin)
    author, member = make_user(), make_user()
    word = unique_word()
    channel_id = group_channel(author, member)
    parent_id = post_message(author, channel_id, f"who brings the {word}?")
    reply_id = post_message(member, channel_id, f"I bring the {word}", parent_id=parent_id)

    answer = search(member, word)

    base = f"/channels/{channel_id}"
    assert (
        item(answer, f"message:{parent_id}")["url"]
        == f"{base}?thread={parent_id}&message={parent_id}"
    )
    assert (
        item(answer, f"message:{reply_id}")["url"]
        == f"{base}?thread={parent_id}&message={reply_id}"
    )
    assert item(answer, f"message:{reply_id}")["title"] == member.name


# --------------------------------------------------------------------------- filter words


def _tagged(client, chat_id: str, word: str) -> str:
    post(client, f"/api/v1/chats/{chat_id}/tags", name="Road Trip")
    return f"tag:road_trip {word}"


def _filed(client, chat_id: str, word: str) -> str:
    folder_id = create_folder(client, "Travel Plans")
    post(client, f"/api/v1/chats/{chat_id}/folder", folder_id=folder_id)
    return f"folder:travel_plans {word}"


def _pinned(client, chat_id: str, word: str) -> str:
    post(client, f"/api/v1/chats/{chat_id}/pin")
    return f"pinned:true {word}"


def _shared(client, chat_id: str, word: str) -> str:
    post(client, f"/api/v1/chats/{chat_id}/share")
    return f"shared:true {word}"


def _archived(client, chat_id: str, word: str) -> str:
    post(client, f"/api/v1/chats/{chat_id}/archive")
    return f"archived:true {word}"


@pytest.mark.regression
@pytest.mark.parametrize(
    "mark", [_tagged, _filed, _pinned, _shared, _archived], ids=lambda mark: mark.__name__[1:]
)
def test_a_filter_word_keeps_only_your_own_chats_with_that_property(workspace, make_user, mark):
    account, other, word, result_ids, _, _ = workspace
    with account.client() as client:
        marked = import_chat(client, f"Marked {word}")
        query = mark(client, marked, word)
    # the same property on chats the account can see but does not own
    with other.client() as client:
        snapshot = import_chat(client, f"Snapshot {word}")
        post(client, f"/api/v1/chats/{snapshot}/share")
        post(
            client,
            f"/api/v1/chats/shared/{snapshot}/access/update",
            access_grants=[grant("user", account.id, "read")],
        )
        mark(client, result_ids["shared chat"][5:], word)

    answer = search(account, query)

    assert {entry["id"] for entry in answer["items"]} == {f"chat:{marked}"}, (
        f"'{query}' should find only the account's own chat with that property"
    )


def test_tag_none_finds_untagged_chats_and_an_unmet_filter_finds_nothing(workspace):
    account, _, word, result_ids, _, _ = workspace

    assert found(account, f"tag:none {word}") == {result_ids["own chat"]}
    assert found(account, f"pinned:true {word}") == set()


# --- #34162: a null byte in a message -----------------------------------------------------


def _imported(client, upstream, text: str) -> str:
    return import_chat(client, "Imported", question=text)


def _created(client, upstream, text: str) -> str:
    chat = {"title": "Created", "models": [], "history": conversation(text, "hi there")}
    return post(client, "/api/v1/chats/new", chat=chat)["id"]


def _sent(client, upstream, text: str) -> str:
    upstream.queue(reply.text("noted", match=reply.answering(text.split()[0])))
    turn, _ = ask(client, text)
    return turn.chat_id


@pytest.mark.regression
@pytest.mark.parametrize(
    "store", [_imported, _created, _sent], ids=lambda store: store.__name__[1:]
)
def test_a_null_byte_in_a_message_is_dropped_and_the_chat_is_still_found(
    admin, make_user, upstream, store
):
    account = make_user()
    word = unique_word()
    with account.client() as client:
        chat_id = store(client, upstream, f"{word} a\x00b")
    with admin.client() as admin_client:
        records = admin_client.get("/api/v1/analytics/messages", params={"chat_id": chat_id})
    assert records.status_code == 200, records.text

    assert f"{word} ab" in [record["content"] for record in records.json()], (
        "the message's own record kept the null byte, which breaks the search on PostgreSQL"
    )
    assert found(account, word) == {f"chat:{chat_id}"}
    assert _chat_list_search(account, word) == [chat_id]


NULL_BYTE_MIGRATION = "5d2b7e9c4a10"
NULL_BYTE_ENV = {"REGRESSION_THROWAWAY_INSTANCE": "null-byte-cleanup"}


@dataclass
class Cleaned:
    instance: object
    account: Actor
    word: str
    chats: dict[str, str]
    share_id: str


def _store_json(instance, table: str, column: str, key: str, value, key_value: str) -> None:
    write_rows(
        instance,
        f"UPDATE {table} SET {column} = CAST(:value AS JSON) WHERE {key} = :key",
        [{"value": json.dumps(value), "key": key_value}],
    )


def _share(client, chat_id: str, reader: Actor, share_mode: str | None) -> str:
    share_id = post(client, f"/api/v1/chats/{chat_id}/share")["share_id"]
    post(
        client,
        f"/api/v1/chats/shared/{chat_id}/access/update",
        access_grants=[grant("user", reader.id, "read")],
        share_mode=share_mode,
    )
    return share_id


@pytest.fixture(scope="module")
def cleaned(instance_with, tmp_path_factory) -> Cleaned:
    """Chats an earlier version stored with null bytes on PostgreSQL, after the cleanup ran."""
    instance = instance_with(NULL_BYTE_ENV)
    if not on_postgres(instance):
        pytest.skip("only PostgreSQL fails to read a stored null byte")
    admin = admin_of(instance)
    with admin.client() as client:
        config = client.get("/api/v1/auths/admin/config").json()
        post(client, "/api/v1/auths/admin/config", **{**config, "ENABLE_NOTIFICATIONS": True})
    account, other = create_user(instance), create_user(instance)
    word = unique_word()
    stored = f"{word} a\x00b"
    with account.client() as client:
        chats = {
            name: import_chat(client, name, question="hello")
            for name in ("row", "record", "quoted")
        }
    with other.client() as client:
        chats["snapshot"] = import_chat(client, "Snapshot", question="hello")
        chats["replies"] = import_chat(client, "Replies", question=f"{word} live")
        share_id = _share(client, chats["snapshot"], account, None)
        _share(client, chats["replies"], account, "continue")

    _store_json(instance, "chat_message", "content", "id", stored, f"{chats['row']}-q")
    usage = {"input_tokens": 3, "output_tokens": 5, "model": "tiny\x00model"}
    _store_json(instance, "chat_message", "usage", "id", usage, f"{chats['row']}-a")
    _store_json(
        instance, "chat_message", "content", "id", f"{word} prints \\u0000", f"{chats['quoted']}-q"
    )
    write_rows(instance, "DELETE FROM chat_message WHERE chat_id = :id", [{"id": chats["record"]}])
    record = {"title": "record", "models": [], "history": conversation(stored, "hi there")}
    _store_json(instance, "chat", "chat", "id", record, chats["record"])
    for name, share_mode in (("snapshot", None), ("replies", "continue")):
        snapshot = {"title": name, "share_mode": share_mode, "history": conversation(stored, "hi")}
        _store_json(instance, "shared_chat", "chat", "chat_id", snapshot, chats[name])

    data_dir = tmp_path_factory.mktemp("null-byte-cleanup")
    for arguments in (("downgrade", f"{NULL_BYTE_MIGRATION}-1"), ("upgrade", "head")):
        finished = manual_alembic(data_dir, *arguments, database_url=instance.database_url)
        assert finished.returncode == 0, f"alembic {' '.join(arguments)}: {finished.stderr[-3000:]}"
    return Cleaned(instance, account, word, chats, share_id)


def _stored_text(instance, statement: str, key: str) -> str:
    [row] = read_rows(instance, statement, {"key": key})
    return row["text"]


@pytest.mark.regression
def test_the_cleanup_removes_only_the_null_bytes(cleaned):
    instance, chats, word = cleaned.instance, cleaned.chats, cleaned.word
    message = "SELECT content #>> '{}' AS text FROM chat_message WHERE id = :key"
    in_chat = "SELECT chat #>> '{history,messages,q,content}' AS text FROM chat WHERE id = :key"
    in_share = (
        "SELECT chat #>> '{history,messages,q,content}' AS text "
        "FROM shared_chat WHERE chat_id = :key"
    )

    assert _stored_text(instance, message, f"{chats['row']}-q") == f"{word} ab"
    usage = "SELECT usage ->> 'model' AS text FROM chat_message WHERE id = :key"
    assert _stored_text(instance, usage, f"{chats['row']}-a") == "tinymodel"
    assert _stored_text(instance, in_chat, chats["record"]) == f"{word} ab"
    assert _stored_text(instance, in_share, chats["snapshot"]) == f"{word} ab"
    assert _stored_text(instance, message, f"{chats['quoted']}-q") == f"{word} prints \\u0000", (
        "the cleanup changed a message that only quotes the text of the escape"
    )


@pytest.mark.regression
def test_after_the_cleanup_the_search_dialog_finds_every_chat(cleaned):
    chats = cleaned.chats
    expected = {f"chat:{chats[name]}" for name in ("row", "record", "quoted", "replies")}

    assert found(cleaned.account, cleaned.word) == expected | {f"shared:{cleaned.share_id}"}


@pytest.mark.regression
def test_after_the_cleanup_the_chat_list_search_finds_every_chat(cleaned):
    expected = {cleaned.chats[name] for name in ("row", "record", "quoted")}
    assert expected <= set(_chat_list_search(cleaned.account, cleaned.word))


@pytest.mark.regression
def test_after_the_cleanup_the_inbox_and_message_updates_answer(cleaned):
    with cleaned.account.client() as client:
        inbox = client.get("/api/v1/notifications/inbox")
        voiced = client.post(
            f"/api/v1/chats/{cleaned.chats['record']}/messages/q", json={"voice": {"name": "alloy"}}
        )

    assert inbox.status_code == 200, inbox.text
    assert voiced.status_code == 200, voiced.text


@pytest.mark.regression
def test_a_message_quoting_the_escape_text_is_still_found(instance, make_user):
    account = make_user()
    word = unique_word()
    with account.client() as client:
        chat_id = import_chat(client, "Escapes", question=f"{word} prints \\u0000 as text")

    assert f"chat:{chat_id}" in found(account, word), (
        "a message that only quotes the text of the escape was left out of the search"
    )
    assert chat_id in _chat_list_search(account, word), (
        "the chat list search hid a chat that only quotes the text of the escape"
    )


@pytest.mark.regression
def test_a_title_holding_x00_is_found_by_the_chat_list_search(make_user):
    account = make_user()
    word = unique_word()
    with account.client() as client:
        chat_id = import_chat(client, f"{word} 0x00 opcode")

    assert chat_id in _chat_list_search(account, word)


def _chat_list_search(actor: Actor, text: str) -> list[str]:
    with actor.client() as client:
        listed = client.get("/api/v1/chats/search", params={"text": text})
    assert listed.status_code == 200, listed.text
    return [chat["id"] for chat in listed.json()]
