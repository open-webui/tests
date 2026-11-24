"""Journey: the chat list's bulk actions and folder views, each confined to the caller's chats.

Archive all moves every chat of the account out of the sidebar and out of its folder into the
archive, so deleting the folder afterwards leaves those chats in the archive, and unarchive
all brings them back with their tags, which archiving keeps under their own names. Export all
streams every chat of the account, archived ones included, one JSON object per line. A folder
lists its chats and those of its subfolders, and its paged list holds ten chats a page. Marking
a chat unread, the whole chat list read or one folder read changes what the sidebar counts as
unread. The admin's database export holds every account's chats. In each case another
account's chats keep the state they had.

Discriminates: in a backend copy, dropping the owner filter from archive all turns the archive
test red (the stranger's chat is archived too), dropping archived chats from the tag count
(78550396d reverted) turns the tag test red, and marking chats read without the owner filter
turns the read test red. The folder delete test passes on dev f36490bf2 and fails with 97f5c4e97
reverted (the archived chat is deleted with the folder).

Export all also streams a stored 130 MB chat (`fat_chat`) whole, every message of it, and
leaves its internal sub-agent chats out; this passes on dev f8a518648 on SQLite and Postgres.

Regression open-webui/open-webui#35001: since chats keep their messages only as rows
(open-webui/open-webui#34689), listing many chats with their messages read the messages one chat
at a time, so export all, the admin's database export, a folder's chats, the archive and
unarchive all took several times longer for 1500 short chats than for one chat holding the same
3000 messages. Discriminates: passes with the fix on branch Classic298:fix/rows-list-slowdown
(each step at most about three times the one-chat time), fails on dev b9e9857b1 (four to nine
times), on SQLite.
"""

from __future__ import annotations

import json
import statistics
import time
import uuid

import httpx
import pytest

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]


def create_chat(client: httpx.Client, title: str, folder_id: str | None = None) -> str:
    created = client.post(
        "/api/v1/chats/new", json={"chat": {"title": title}, "folder_id": folder_id}
    )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def create_folder(client: httpx.Client, parent_id: str | None = None) -> str:
    form = {"name": f"Folder {uuid.uuid4().hex[:6]}", "parent_id": parent_id}
    created = client.post("/api/v1/folders/", json=form)
    assert created.status_code == 200, created.text
    return created.json()["id"]


def read_chat(client: httpx.Client, chat_id: str) -> dict:
    answer = client.get(f"/api/v1/chats/{chat_id}")
    assert answer.status_code == 200, answer.text
    return answer.json()


def sidebar(client: httpx.Client) -> dict[str, dict]:
    """The chat list the sidebar shows, folders and pinned chats included, by chat id."""
    listed = client.get("/api/v1/chats/", params={"include_folders": True, "include_pinned": True})
    assert listed.status_code == 200, listed.text
    return {chat["id"]: chat for chat in listed.json()}


def sidebar_ids(client: httpx.Client) -> set[str]:
    return set(sidebar(client))


def last_read_at(client: httpx.Client, chat_id: str) -> int | None:
    return sidebar(client)[chat_id]["last_read_at"]


@pytest.fixture
def owner(make_user):
    return make_user()


@pytest.fixture
def stranger_chat(make_user) -> tuple[httpx.Client, str]:
    """Another account holding one ordinary chat."""
    client = make_user().client()
    with client:
        yield client, create_chat(client, "The stranger's chat")


# --------------------------------------------------------------------------- archive


def test_archive_all_moves_every_chat_of_the_account_to_the_archive(owner, stranger_chat):
    stranger, stranger_chat_id = stranger_chat
    with owner.client() as client:
        folder_id = create_folder(client)
        chat_ids = {create_chat(client, "Loose"), create_chat(client, "Filed", folder_id)}

        archived = client.post("/api/v1/chats/archive/all")

        assert archived.status_code == 200 and archived.json() is True, archived.text
        assert sidebar_ids(client) & chat_ids == set()
        assert client.get("/api/v1/chats/archived/count").json() == 2
        in_archive = client.get("/api/v1/chats/all/archived").json()
        assert {chat["id"] for chat in in_archive} == chat_ids
        assert all(chat["archived"] for chat in in_archive)
    assert read_chat(stranger, stranger_chat_id)["archived"] is False
    assert stranger_chat_id in sidebar_ids(stranger)


def test_deleting_a_folder_after_archive_all_keeps_the_archived_chats(owner):
    """#32647, fix 97f5c4e97: archived chats stayed in their folder and went with it."""
    with owner.client() as client:
        folder_id = create_folder(client)
        archived_id = create_chat(client, "Archived", folder_id)
        assert client.post("/api/v1/chats/archive/all").is_success
        filed_later_id = create_chat(client, "Filed after archiving", folder_id)

        deleted = client.delete(f"/api/v1/folders/{folder_id}", params={"delete_contents": True})

        assert deleted.status_code == 200, deleted.text
        kept = client.get(f"/api/v1/chats/{archived_id}")
        assert kept.status_code == 200, f"deleting the folder deleted an archived chat: {kept.text}"
        assert (kept.json()["archived"], kept.json()["folder_id"]) == (True, None)
        in_archive = {chat["id"] for chat in client.get("/api/v1/chats/all/archived").json()}
        assert archived_id in in_archive
        # a missing chat answers 401, the same as someone else's
        gone = client.get(f"/api/v1/chats/{filed_later_id}")
        assert gone.status_code != 200, (
            "a chat still in the folder survived deleting the folder with its contents"
        )


def test_unarchive_all_brings_the_chats_and_their_tags_back(owner, stranger_chat):
    stranger, stranger_chat_id = stranger_chat
    assert stranger.post(f"/api/v1/chats/{stranger_chat_id}/archive").status_code == 200
    with owner.client() as client:
        tagged_id = create_chat(client, "Tagged")
        other_id = create_chat(client, "Plain")
        assert client.post(f"/api/v1/chats/{tagged_id}/tags", json={"name": "Quarterly"}).is_success
        # an archived chat still uses its tag, so the tag keeps its name (#30454)
        assert client.post(f"/api/v1/chats/{tagged_id}/archive").is_success
        assert client.post(f"/api/v1/chats/{other_id}/archive").is_success
        tags_while_archived = [tag["name"] for tag in client.get("/api/v1/chats/all/tags").json()]

        restored = client.post("/api/v1/chats/unarchive/all")

        assert restored.status_code == 200 and restored.json() is True, restored.text
        assert {tagged_id, other_id} <= sidebar_ids(client)
        assert client.get("/api/v1/chats/archived/count").json() == 0
        tags = [tag["name"] for tag in client.get("/api/v1/chats/all/tags").json()]
        by_tag = client.post("/api/v1/chats/tags", json={"name": "quarterly"}).json()
    assert "Quarterly" in tags_while_archived
    assert "Quarterly" in tags, "unarchiving did not bring the chat's tag back"
    assert [chat["id"] for chat in by_tag] == [tagged_id]
    assert read_chat(stranger, stranger_chat_id)["archived"] is True


# --------------------------------------------------------------------------- export


def test_export_all_streams_every_chat_of_the_account_including_archived(
    owner, stranger_chat, fat_chat
):
    _, stranger_chat_id = stranger_chat
    fat = fat_chat(owner, "fat-130mb")
    with owner.client() as client:
        kept_id = create_chat(client, "Kept")
        archived_id = create_chat(client, "Put away")
        assert client.post(f"/api/v1/chats/{archived_id}/archive").is_success

        exported = client.get("/api/v1/chats/all")

    assert exported.status_code == 200, exported.text[:500]
    assert exported.headers["content-type"].startswith("application/x-ndjson")
    lines = [json.loads(line) for line in exported.text.splitlines() if line]
    assert {chat["id"]: chat["title"] for chat in lines} == {
        kept_id: "Kept",
        archived_id: "Put away",
        fat.chat_id: fat.manifest["title"],
    }
    [long_chat] = [chat for chat in lines if chat["id"] == fat.chat_id]
    exported_messages = len(long_chat["chat"]["history"]["messages"])
    assert exported_messages == fat.manifest["messages"], (
        f"the long chat was exported with {exported_messages} of its "
        f"{fat.manifest['messages']} messages"
    )
    assert stranger_chat_id not in exported.text


MANY_CHATS = 1500
# a step may take this many times as long for many chats as for one chat of the same messages
SLOWER_AT_MOST = 3.5
FLOOR_SECONDS = 0.1


def conversation(title: str, turns: int, folder_id: str) -> dict:
    """A chat of `turns` questions and answers, one after the other, as an import form."""
    messages, parent_id = {}, None
    for index in range(2 * turns):
        message_id = str(uuid.uuid4())
        role = ("user", "assistant")[index % 2]
        messages[message_id] = {
            "id": message_id,
            "parentId": parent_id,
            "childrenIds": [],
            "role": role,
            "content": f"{title}, {role} message {index}: a few words of text",
            "timestamp": 1_700_000_000 + index,
        }
        if parent_id:
            messages[parent_id]["childrenIds"].append(message_id)
        parent_id = message_id
    history = {"messages": messages, "currentId": parent_id}
    return {"chat": {"title": title, "history": history}, "folder_id": folder_id}


def import_conversations(client: httpx.Client, count: int, turns: int, folder_id: str) -> set[str]:
    chat_ids = set()
    for start in range(0, count, 100):
        forms = [
            conversation(f"Chat {index}", turns, folder_id)
            for index in range(start, min(count, start + 100))
        ]
        imported = client.post("/api/v1/chats/import", json={"chats": forms})
        assert imported.status_code == 200, imported.text
        chat_ids |= {chat["id"] for chat in imported.json()}
    return chat_ids


def timed(step) -> tuple[float, object]:
    """The median time of three runs of `step`, and what its last run returned."""
    took = []
    for _ in range(3):
        started = time.perf_counter()
        result = step()
        took.append(time.perf_counter() - started)
    return statistics.median(took), result


def message_counts(chats: list[dict]) -> dict[str, int]:
    return {chat["id"]: len(chat["chat"]["history"]["messages"]) for chat in chats}


def walk_the_chat_list(client: httpx.Client, folder_id: str) -> dict[str, tuple[float, dict]]:
    """Each bulk step a person takes on their chats, timed, with the chats it returned."""

    def export_all():
        exported = client.get("/api/v1/chats/all")
        assert exported.status_code == 200, exported.text[:500]
        return message_counts([json.loads(line) for line in exported.text.splitlines() if line])

    def folder_chats():
        listed = client.get(f"/api/v1/chats/folder/{folder_id}")
        assert listed.status_code == 200, listed.text[:500]
        return message_counts(listed.json())

    steps = {"export all": timed(export_all), "the folder's chats": timed(folder_chats)}
    archive_list, unarchive = [], []
    for _ in range(3):
        assert client.post("/api/v1/chats/archive/all").is_success
        started = time.perf_counter()
        archived = client.get("/api/v1/chats/all/archived")
        archive_list.append(time.perf_counter() - started)
        started = time.perf_counter()
        restored = client.post("/api/v1/chats/unarchive/all")
        unarchive.append(time.perf_counter() - started)
        assert archived.status_code == 200, archived.text[:500]
        assert restored.status_code == 200 and restored.json() is True, restored.text
    steps["the archive"] = (statistics.median(archive_list), message_counts(archived.json()))
    steps["unarchive all"] = (statistics.median(unarchive), None)
    return steps


@pytest.mark.regression
def test_many_chats_list_about_as_fast_as_one_chat_of_the_same_messages(admin, make_user):
    one, many = make_user(), make_user()

    def admin_export():
        exported = admin_client.get("/api/v1/chats/all/db")
        assert exported.status_code == 200, exported.text[:500]
        return message_counts(exported.json())

    with admin.client() as admin_client:
        before, _ = timed(admin_export)
        with one.client() as client:
            one_folder = create_folder(client)
            one_ids = import_conversations(client, 1, MANY_CHATS, one_folder)
        with_one, _ = timed(admin_export)
        with many.client() as client:
            many_folder = create_folder(client)
            many_ids = import_conversations(client, MANY_CHATS, 1, many_folder)
        with_many, everything = timed(admin_export)

    with one.client() as client:
        one_steps = walk_the_chat_list(client, one_folder)
        assert sidebar_ids(client) == one_ids
    with many.client() as client:
        many_steps = walk_the_chat_list(client, many_folder)
        assert sidebar_ids(client) == many_ids

    for _, chats in [*one_steps.values(), *many_steps.values()]:
        assert chats is None or chats in (
            {chat_id: 2 * MANY_CHATS for chat_id in one_ids},
            {chat_id: 2 for chat_id in many_ids},
        ), "a step lost chats or messages"
    assert {chat_id: everything.get(chat_id) for chat_id in one_ids | many_ids} == {
        **{chat_id: 2 * MANY_CHATS for chat_id in one_ids},
        **{chat_id: 2 for chat_id in many_ids},
    }
    one_steps["the admin's database export"] = (with_one - before, None)
    many_steps["the admin's database export"] = (with_many - with_one, None)
    slow = {
        step: f"{many_steps[step][0]:.2f}s against {seconds:.2f}s"
        for step, (seconds, _) in one_steps.items()
        if many_steps[step][0] > SLOWER_AT_MOST * max(seconds, FLOOR_SECONDS)
    }
    assert not slow, (
        f"with {MANY_CHATS} short chats these steps took far longer than with one chat holding "
        f"the same messages: {slow}"
    )


def test_the_admin_database_export_holds_every_accounts_chats(admin, owner, stranger_chat):
    _, stranger_chat_id = stranger_chat
    with owner.client() as client:
        owner_chat_id = create_chat(client, "Owner's")
        refused = client.get("/api/v1/chats/all/db")

    with admin.client() as client:
        exported = {chat["id"] for chat in client.get("/api/v1/chats/all/db").json()}

    assert refused.status_code == 401
    assert {owner_chat_id, stranger_chat_id} <= exported


# --------------------------------------------------------------------------- folders


def test_a_folder_lists_its_chats_and_those_of_its_subfolders(owner, make_user):
    with owner.client() as client:
        folder_id = create_folder(client)
        subfolder_id = create_folder(client, parent_id=folder_id)
        inside = {
            create_chat(client, "Top", folder_id),
            create_chat(client, "Nested", subfolder_id),
        }
        create_chat(client, "Elsewhere")

        listed = client.get(f"/api/v1/chats/folder/{folder_id}")
        sub_listed = client.get(f"/api/v1/chats/folder/{subfolder_id}")

    assert listed.status_code == 200, listed.text
    assert {chat["id"] for chat in listed.json()} == inside
    assert [chat["title"] for chat in sub_listed.json()] == ["Nested"]
    with make_user().client() as stranger:
        assert stranger.get(f"/api/v1/chats/folder/{folder_id}").json() == []


def test_a_folders_chat_list_pages_ten_at_a_time(owner, make_user):
    with owner.client() as client:
        folder_id = create_folder(client)
        titles = [f"Paged {index:02d}" for index in range(12)]
        for title in titles:
            create_chat(client, title, folder_id)

        pages = [
            client.get(
                f"/api/v1/chats/folder/{folder_id}/list",
                params={"page": page, "sort_by": "title", "sort_dir": "asc"},
            )
            for page in (1, 2, 3)
        ]

    assert all(page.status_code == 200 for page in pages), [page.text for page in pages]
    assert [[chat["title"] for chat in page.json()] for page in pages] == [
        titles[:10],
        titles[10:],
        [],
    ]
    with make_user().client() as stranger:
        assert stranger.get(f"/api/v1/chats/folder/{folder_id}/list").json() == []


# --------------------------------------------------------------------------- read and unread


def test_marking_a_chat_unread_counts_it_in_its_folder(owner):
    with owner.client() as client:
        folder_id = create_folder(client)
        chat_id = create_chat(client, "Remind me", folder_id)

        marked = client.post(f"/api/v1/chats/{chat_id}/unread")

        assert marked.status_code == 200, marked.text
        assert marked.json()["folder_id"] == folder_id
        assert marked.json()["folder_unread_counts"][folder_id] == 1
        assert last_read_at(client, chat_id) == 0


def test_a_stranger_cannot_mark_someone_elses_chat_unread(owner, make_user):
    with owner.client() as client:
        chat_id = create_chat(client, "Mine")
        before = last_read_at(client, chat_id)
    with make_user().client() as stranger:
        assert stranger.post(f"/api/v1/chats/{chat_id}/unread").status_code == 404
    with owner.client() as client:
        assert last_read_at(client, chat_id) == before


def test_marking_everything_read_clears_the_callers_unread_chats_only(owner, stranger_chat):
    stranger, stranger_chat_id = stranger_chat
    assert stranger.post(f"/api/v1/chats/{stranger_chat_id}/unread").is_success
    with owner.client() as client:
        folder_id = create_folder(client)
        chat_ids = [create_chat(client, "One", folder_id), create_chat(client, "Two")]
        for chat_id in chat_ids:
            assert client.post(f"/api/v1/chats/{chat_id}/unread").is_success

        read_all = client.post("/api/v1/chats/read")

        assert read_all.status_code == 200, read_all.text
        assert read_all.json()["updated_count"] >= 2
        assert read_all.json()["folder_unread_counts"] == {folder_id: 0}
        for chat_id in chat_ids:
            chat = sidebar(client)[chat_id]
            assert chat["last_read_at"] == chat["updated_at"]
    assert last_read_at(stranger, stranger_chat_id) == 0, (
        "marking everything read reached another account's chat"
    )


def test_marking_a_folder_read_leaves_the_other_folders_unread(owner):
    with owner.client() as client:
        read_folder, other_folder = create_folder(client), create_folder(client)
        read_chat_id = create_chat(client, "Read me", read_folder)
        other_chat_id = create_chat(client, "Leave me", other_folder)
        for chat_id in (read_chat_id, other_chat_id):
            assert client.post(f"/api/v1/chats/{chat_id}/unread").is_success

        marked = client.post(f"/api/v1/folders/{read_folder}/read")

        assert marked.status_code == 200, marked.text
        assert marked.json()["folder_unread_counts"] == {read_folder: 0, other_folder: 1}
        assert last_read_at(client, other_chat_id) == 0
