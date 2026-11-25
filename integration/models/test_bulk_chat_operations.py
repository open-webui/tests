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

Deleting chats leaves none of their words behind. Once a user deletes one chat, an archived chat
from Archived Chats, all chats or a folder with its chats, or the admin deletes the account, the
chat's words are gone from the workspace search, the chat list search, the owner's export and
stats export, the admin's database export and the admin's list of the account's chats, and
analytics no longer counts its messages and tokens per model and per user nor lists its session.
The same holds for the conversation of a sub-agent the chat handed a task to. Discriminates: all
pass on dev f80a4f782 on SQLite and Postgres except the sub-agent case of the folder deletion, and
in a backend copy that keeps the messages of deleted chats every case fails (analytics still counts
and returns them).

Regression open-webui/open-webui#35053: deleting a folder with its chats left the sub-agent
conversations of those chats behind, still counted and listed in the admin analytics, while every
other way of deleting the chat removed them. Red on dev f80a4f782 until the fix merges.

Regression open-webui/open-webui#35057: deleting a note, on its own or with the folder holding it,
left the chat of its chat panel behind, still counted, listed and returned in the admin
analytics. Red on dev 893584700 in both cases until the fix merges.
"""

from __future__ import annotations

import json
import statistics
import time
import uuid

import httpx
import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.model_presets import preset_with_capabilities
from harness.subagents import delegated, report_of
from harness.tool_calls import without_hidden_calls

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


# --------------------------------------------------------------------------- deletion

USAGE = {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}
SUBAGENTS = ("/api/v1/configs/subagents", "/api/v1/configs/subagents")
DELETIONS = pytest.mark.parametrize(
    "delete",
    ["delete_one_chat", "delete_an_archived_chat", "delete_all_chats", "delete_the_folder"],
    ids=["one chat", "an archived chat", "all chats", "a folder with its chats"],
)


def unique_word() -> str:
    """A word no other chat holds, so any place still showing it shows the deleted chat."""
    return f"quillwort{uuid.uuid4().hex[:10]}"


def talk(client: httpx.Client, upstream, model: str, word: str) -> str:
    """A chat whose question and answer both hold `word`, answered with token usage."""
    question = f"Tell me about {word}"
    upstream.queue(reply.text(f"{word} is a fern", usage=USAGE, match=reply.answering(question)))
    turn, stored = ask(client, question, model=model)
    assert stored["content"] == f"{word} is a fern", stored
    return turn.chat_id


def talk_with_a_subagent(client: httpx.Client, upstream, model: str, word: str) -> list[str]:
    """A chat that hands `word` to a sub-agent; its id and the sub-agent's, once reported."""
    prompt, task = f"Hand {word} over", f"Look into {word}"
    report = f"The report on {word} is in."
    upstream.queue(
        reply.tool_call("delegate_task", {"task": task}, match=reply.answering(prompt)),
        reply.text(f"{word} grows in ponds", usage=USAGE, match=delegated(task)),
        reply.text("Handed over.", match=reply.answering(prompt)),
        reply.text(report, match=report_of(task)),
    )
    turn, stored = ask(client, prompt, model=model)
    [handle] = [
        json.loads(item["output"][0]["text"])
        for item in without_hidden_calls(stored["output"])
        if item["type"] == "function_call_output"
    ]
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        messages = read_chat(client, turn.chat_id)["chat"]["history"]["messages"].values()
        if any(message.get("done") and message["content"] == report for message in messages):
            return [turn.chat_id, handle["subagent_chat_id"]]
        time.sleep(0.1)
    raise AssertionError("the sub-agent's report never came back into the chat")


def into_folder(client: httpx.Client, chat_id: str, folder_id: str) -> None:
    moved = client.post(f"/api/v1/chats/{chat_id}/folder", json={"folder_id": folder_id})
    assert moved.status_code == 200, moved.text


def owner_sees(client: httpx.Client, word: str, chat_ids: list[str]) -> dict[str, bool]:
    """Where the owner still finds `word`: both searches, the export and the stats export."""
    found = client.get("/api/v1/search", params={"q": word})
    listed = client.get("/api/v1/chats/search", params={"text": word})
    exported = client.get("/api/v1/chats/all")
    stats = client.get("/api/v1/chats/stats/export")
    for answer in (found, listed, exported, stats):
        assert answer.status_code == 200, f"{answer.request.url}: {answer.text[:300]}"
    return {
        "the workspace search": word in json.dumps(found.json()["items"]),
        "the chat list search": listed.json() != [],
        "the owner's export": word in exported.text,
        # the stats export carries no message text, only each chat's numbers
        "the owner's stats export": names_any(stats.json(), chat_ids),
    }


def admin_sees(
    client: httpx.Client, owner_id: str, model: str, word: str, chat_ids: list[str]
) -> dict[str, bool]:
    """Where the admin still finds the owner's words, messages or token counts."""
    answers = {
        "the admin's database export": client.get("/api/v1/chats/all/db"),
        "the owner's chats in the admin's user list": client.get(
            f"/api/v1/chats/list/user/{owner_id}"
        ),
        "the analytics messages of the owner": client.get(
            "/api/v1/analytics/messages", params={"user_id": owner_id}
        ),
        "the analytics messages of the model": client.get(
            "/api/v1/analytics/messages", params={"model_id": model}
        ),
        "the analytics message counts per model": client.get("/api/v1/analytics/models"),
        "the analytics token counts per model": client.get("/api/v1/analytics/tokens"),
        "the analytics counts per user": client.get(
            "/api/v1/analytics/users", params={"limit": 10_000}
        ),
        "the analytics chats of the model": client.get(f"/api/v1/analytics/models/{model}/chats"),
        "the analytics sessions of the model": client.get(
            "/api/v1/analytics/sessions", params={"model_id": model}
        ),
    }
    for answer in answers.values():
        assert answer.status_code == 200, f"{answer.request.url}: {answer.text[:300]}"
    body = {where: answer.json() for where, answer in answers.items()}
    return {
        "the admin's database export": word in json.dumps(body["the admin's database export"]),
        "the owner's chats in the admin's user list": names_any(
            body["the owner's chats in the admin's user list"], chat_ids
        ),
        "the analytics messages of the owner": word
        in json.dumps(body["the analytics messages of the owner"]),
        "the analytics messages of the model": word
        in json.dumps(body["the analytics messages of the model"]),
        "the analytics message counts per model": any(
            entry["model_id"] == model and entry["count"] > 0
            for entry in body["the analytics message counts per model"]["models"]
        ),
        "the analytics token counts per model": any(
            entry["model_id"] == model and entry["total_tokens"] > 0
            for entry in body["the analytics token counts per model"]["models"]
        ),
        "the analytics counts per user": any(
            entry["user_id"] == owner_id and entry["count"] > 0
            for entry in body["the analytics counts per user"]["users"]
        ),
        "the analytics chats of the model": names_any(
            body["the analytics chats of the model"], chat_ids
        ),
        "the analytics sessions of the model": names_any(
            body["the analytics sessions of the model"], chat_ids
        ),
    }


def names_any(body, chat_ids: list[str]) -> bool:
    text = json.dumps(body)
    return any(chat_id in text for chat_id in chat_ids)


def still_showing(seen: dict[str, bool]) -> list[str]:
    return [where for where, shows in seen.items() if shows]


@pytest.fixture
def community_sharing(admin, preserve):
    """The stats export is open to users only with community sharing on."""
    preserve("admin_config")
    with admin.client() as client:
        config = client.get("/api/v1/auths/admin/config").json()
        saved = client.post(
            "/api/v1/auths/admin/config", json={**config, "ENABLE_COMMUNITY_SHARING": True}
        )
    assert saved.status_code == 200, saved.text


@pytest.fixture
def subagents_on(admin, preserve):
    preserve(SUBAGENTS)
    with admin.client() as client:
        current = client.get(SUBAGENTS[0]).json()
        saved = client.post(SUBAGENTS[1], json={**current, "ENABLE_SUBAGENTS": True})
    assert saved.status_code == 200, saved.text


@pytest.fixture
def deletion_model(admin):
    with preset_with_capabilities(admin, {}) as model:
        yield model


def delete_one_chat(client: httpx.Client, chat_id: str) -> httpx.Response:
    return client.delete(f"/api/v1/chats/{chat_id}")


def delete_an_archived_chat(client: httpx.Client, chat_id: str) -> httpx.Response:
    archived = client.post(f"/api/v1/chats/{chat_id}/archive")
    assert archived.status_code == 200 and archived.json()["archived"], archived.text
    return client.delete(f"/api/v1/chats/{chat_id}")


def delete_all_chats(client: httpx.Client, chat_id: str) -> httpx.Response:
    return client.delete("/api/v1/chats/")


def delete_the_folder(client: httpx.Client, chat_id: str) -> httpx.Response:
    folder_id = create_folder(client)
    into_folder(client, chat_id, create_folder(client, parent_id=folder_id))
    return client.delete(f"/api/v1/folders/{folder_id}", params={"delete_contents": True})


@DELETIONS
def test_a_deleted_chats_words_are_gone_from_every_search_export_and_report(
    delete, admin, make_user, upstream, deletion_model, community_sharing
):
    owner, word = make_user(), unique_word()
    with owner.client() as client, admin.client() as admin_client:
        chat_id = talk(client, upstream, deletion_model, word)

        def everywhere() -> dict[str, bool]:
            return {
                **owner_sees(client, word, [chat_id]),
                **admin_sees(admin_client, owner.id, deletion_model, word, [chat_id]),
            }

        before = everywhere()
        deleted = globals()[delete](client, chat_id)

        assert deleted.status_code == 200, deleted.text
        assert client.get(f"/api/v1/chats/{chat_id}").status_code != 200
        after = everywhere()
    missing = [where for where, shows in before.items() if not shows]
    assert missing == [], f"before the deletion the chat was already missing from {missing}"
    assert still_showing(after) == [], f"the deleted chat still shows in {still_showing(after)}"


@DELETIONS
def test_a_deleted_chat_takes_its_subagents_words_with_it(
    delete,
    admin,
    make_user,
    upstream,
    deletion_model,
    community_sharing,
    subagents_on,
):
    """The folder case is open-webui/open-webui#35053."""
    owner, word = make_user(), unique_word()
    with owner.client() as client, admin.client() as admin_client:
        chat_ids = talk_with_a_subagent(client, upstream, deletion_model, word)

        def everywhere() -> dict[str, bool]:
            return {
                **owner_sees(client, word, chat_ids),
                **admin_sees(admin_client, owner.id, deletion_model, word, chat_ids),
            }

        before = everywhere()
        deleted = globals()[delete](client, chat_ids[0])

        assert deleted.status_code == 200, deleted.text
        after = everywhere()
    missing = [where for where, shows in before.items() if not shows]
    assert missing == [], f"before the deletion the chat was already missing from {missing}"
    assert still_showing(after) == [], (
        f"the deleted chat's sub-agent still shows in {still_showing(after)}"
    )


@pytest.fixture
def notes_on(admin, preserve):
    preserve("admin_config")
    with admin.client() as client:
        config = client.get("/api/v1/auths/admin/config").json()
        saved = client.post("/api/v1/auths/admin/config", json={**config, "ENABLE_NOTES": True})
    assert saved.status_code == 200, saved.text


def talk_in_a_note(client: httpx.Client, upstream, model: str, word: str, folder_id: str):
    """A note in the folder and its chat, where `word` was asked and answered; both ids."""
    created = client.post(
        "/api/v1/notes/create",
        json={
            "title": f"Note {uuid.uuid4().hex[:6]}",
            "folder_id": folder_id,
            "data": {"content": {"md": "ferns", "html": "<p>ferns</p>", "json": None}},
        },
    )
    assert created.status_code == 200, created.text
    note_id = created.json()["id"]
    opened = client.post(f"/api/v1/notes/{note_id}/chat")
    assert opened.status_code == 200, opened.text
    chat_id = opened.json()["id"]
    question = f"Tell me about {word}"
    upstream.queue(reply.text(f"{word} is a fern", usage=USAGE, match=reply.answering(question)))
    _, stored = ask(client, question, chat_id=chat_id, model=model)
    assert stored["content"] == f"{word} is a fern", stored
    return note_id, chat_id


@pytest.mark.parametrize("delete", ["the note", "its folder"])
def test_a_deleted_notes_chat_is_gone_from_every_search_export_and_report(
    delete, admin, make_user, upstream, deletion_model, community_sharing, notes_on
):
    """open-webui/open-webui#35057."""
    owner, word = make_user(), unique_word()
    with owner.client() as client, admin.client() as admin_client:
        folder_id = create_folder(client)
        note_id, chat_id = talk_in_a_note(client, upstream, deletion_model, word, folder_id)

        def everywhere() -> dict[str, bool]:
            return {
                **owner_sees(client, word, [chat_id]),
                **admin_sees(admin_client, owner.id, deletion_model, word, [chat_id]),
            }

        before = everywhere()
        if delete == "the note":
            deleted = client.delete(f"/api/v1/notes/{note_id}/delete")
        else:
            folder = f"/api/v1/folders/{folder_id}"
            deleted = client.delete(folder, params={"delete_contents": True})

        assert deleted.status_code == 200 and deleted.json() is True, deleted.text
        assert client.get(f"/api/v1/notes/{note_id}").status_code == 404
        after = everywhere()
    # a note's chat stays out of the chat list, the searches and the exports while it exists
    shown_before = still_showing(before)
    assert "the analytics messages of the owner" in shown_before, (
        f"before the deletion analytics did not count the note's chat: {before}"
    )
    assert still_showing(after) == [], (
        f"the deleted note's chat still shows in {still_showing(after)}"
    )


def test_an_account_deleted_by_the_admin_leaves_none_of_its_words_behind(
    admin, make_user, upstream, deletion_model
):
    owner, word = make_user(), unique_word()
    with owner.client() as client:
        chat_id = talk(client, upstream, deletion_model, word)
        archived_id = talk(client, upstream, deletion_model, word)
        assert client.post(f"/api/v1/chats/{archived_id}/archive").is_success
        folder_id = create_folder(client)
        into_folder(client, chat_id, folder_id)
    with admin.client() as admin_client:
        chat_ids = [chat_id, archived_id]
        before = admin_sees(admin_client, owner.id, deletion_model, word, chat_ids)

        deleted = admin_client.delete(f"/api/v1/users/{owner.id}")

        assert deleted.status_code == 200 and deleted.json() is True, deleted.text
        after = admin_sees(admin_client, owner.id, deletion_model, word, chat_ids)
    missing = [where for where, shows in before.items() if not shows]
    assert missing == [], f"before the deletion the chats were already missing from {missing}"
    assert still_showing(after) == [], (
        f"the deleted account's chats still show in {still_showing(after)}"
    )
