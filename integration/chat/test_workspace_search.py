"""Journey: what the workspace search filters keep and the links its results carry.

`GET /api/v1/search` takes a content type (`kind`), whose items to show (`sharing`: the caller's
own or others') and a location (`scope`: a folder with its subfolders, or a channel). A folder
location keeps only chats and a channel location only that channel's messages; a location the
caller cannot reach answers with nothing. Archived chats and the hidden chats behind notes stay
out. One page holds 24 results and `has_more` says another follows. A chat result links to its
`/c/` page with `?message=` on the matching message, a channel message to its channel with
`?thread=` (the thread's first message) and `&message=`. Since ec83763f1 (#33181) the words
`tag:`, `folder:`, `pinned:`, `shared:` and `archived:` filter the caller's own chats, and a
query with one of them leaves out notes, messages, shared snapshots and other accounts' chats.

Discriminates: journey, no fix. In backend copies of dev 948432b43: ignoring `kind` turns the
content type tests red, ignoring `sharing` the created-by tests, ignoring `scope` the two
location tests, skipping the unknown-location check the unreachable-location test; keeping
archived and internal chats turns the hidden-chat, paging and `archived` filter tests red; a page
of 25 turns the paging test red; links without the message turn both link tests red; with
ec83763f1 undone (no filter words), or with filter words no longer limited to the caller's own
chats, the six filter-word tests go red.
"""

from __future__ import annotations

import uuid

import pytest

from harness.access import grant
from harness.channel_quotes import enable_channels, group_channel, post_message
from harness.workspace_search import (
    PAGE_SIZE,
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
