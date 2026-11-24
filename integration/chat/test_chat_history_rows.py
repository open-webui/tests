"""Journey: a chat keeps every message, branch and version through everything a user does with it.

open-webui/open-webui#34688 asks to store a chat's messages one by one instead of rewriting the
whole conversation on every message (PR open-webui/open-webui#34689, with a migration that moves
existing chats). What a user expects is the same before and after it: a reopened chat shows its
turns in order with their text and models, a regenerated reply keeps both versions and the one
picked, an edited question keeps both branches, a deleted message stays gone, a reply keeps its
citations, steps and files, a clone, fork, share, export and import carry the conversation, search
finds a chat by the words of any message, a long chat keeps answering, and a chat written by an
earlier release opens with every message, its order and its times after the upgrade.
Every check reads the chat back from the server, as a reload does.

Discriminates: passes on dev b6aac5dee (before the PR) and on dev 414340682 (the PR merged), the
upgrade test on the three SQLite data sets. In a backend copy of 414340682 whose message delete
deletes nothing and whose clone keeps only the shown versions, the delete and the clone tests fail.
The Postgres data sets run only where Postgres is available; they were not run locally.
"""

from __future__ import annotations

import json
import uuid
from typing import Iterator

import httpx
import pytest
import sqlalchemy

from harness import upstream as reply
from harness.chat import NO_BACKGROUND_TASKS, ChatTurn, ask, send_message, wait_for_reply
from harness.prepared_data import ReleaseData
from harness.python_tools import EVERYONE_READS, python_tool
from harness.upgraded_release import Upgraded, data_set_params, upgraded_release
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]


def _tag() -> str:
    return uuid.uuid4().hex[:8]


def _stored(client: httpx.Client, chat_id: str) -> dict:
    opened = client.get(f"/api/v1/chats/{chat_id}")
    assert opened.status_code == 200, f"the chat did not open: {opened.text}"
    return opened.json()


def _history(client: httpx.Client, chat_id: str) -> dict:
    return _stored(client, chat_id)["chat"]["history"]


def _path(history: dict) -> list[dict]:
    """The messages the chat shows, first to last: from the current message up to the first."""
    messages = history["messages"]
    shown, message_id = [], history["currentId"]
    while message_id is not None:
        assert message_id in messages, f"message {message_id} on the shown path is missing"
        shown.append(messages[message_id])
        message_id = messages[message_id].get("parentId")
    return shown[::-1]


def _turns(history: dict) -> list[tuple[str, str]]:
    return [(message["role"], message["content"]) for message in _path(history)]


def _assert_linked(history: dict) -> None:
    """Every parent lists its children and every child names its parent."""
    messages = history["messages"]
    for message_id, message in messages.items():
        parent_id = message.get("parentId")
        if parent_id is not None:
            assert message_id in messages[parent_id]["childrenIds"], (
                f"{message_id} is not listed under its parent {parent_id}"
            )
        for child_id in message.get("childrenIds", []):
            assert messages[child_id]["parentId"] == message_id, f"{child_id} lost its parent"


def _continue(client: httpx.Client, chat_id: str, parent_id: str, content: str, **options):
    turn = send_message(client, content, chat_id=chat_id, parent_id=parent_id, **options)
    return turn, wait_for_reply(client, turn)


def _regenerate(client: httpx.Client, chat_id: str, question: dict) -> str:
    """Regenerate the reply to the stored `question` as the web client does; returns its id."""
    reply_id = str(uuid.uuid4())
    sent = client.post(
        "/api/chat/completions",
        json={
            "model": MOCK_MODEL_ID,
            "messages": [{"role": "user", "content": question["content"]}],
            "stream": True,
            "chat_id": chat_id,
            "parent_id": question["parentId"],
            "id": reply_id,
            "user_message": {**question, "childrenIds": [*question["childrenIds"], reply_id]},
            "session_id": f"harness-{uuid.uuid4().hex[:8]}",
            "background_tasks": NO_BACKGROUND_TASKS,
        },
    )
    assert sent.status_code == 200, f"regenerating was refused: {sent.text}"
    wait_for_reply(client, ChatTurn(chat_id, question["id"], reply_id))
    return reply_id


def _show(client: httpx.Client, chat_id: str, message_id: str) -> None:
    """Switch the chat to the branch ending in `message_id` and save it, as the client does."""
    saved = client.post(
        f"/api/v1/chats/{chat_id}",
        json={"chat": {"history": {"currentId": message_id, "messages": {}}}},
    )
    assert saved.status_code == 200, f"saving the switch failed: {saved.text}"


@pytest.fixture
def preset(admin) -> Iterator[str]:
    """A second model every account may pick, a preset on the scripted one."""
    model_id = f"history-preset-{_tag()}"
    form = {
        "id": model_id,
        "base_model_id": MOCK_MODEL_ID,
        "name": "History preset",
        "meta": {},
        "params": {},
        "access_grants": [EVERYONE_READS],
    }
    with admin.client() as client:
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
        yield model_id
        client.post("/api/v1/models/model/delete", json={"id": model_id})


def _chat_of_turns(client, upstream, turns: list[tuple[str, str]], models=None) -> list[ChatTurn]:
    """A chat of answered turns, each sent after the previous reply; returns the turns."""
    models = models or [MOCK_MODEL_ID] * len(turns)
    sent: list[ChatTurn] = []
    for (question, answer), model in zip(turns, models):
        upstream.queue(reply.text(answer, match=reply.answering(question)))
        options = {"model": model}
        if sent:
            options |= {"chat_id": sent[-1].chat_id, "parent_id": sent[-1].assistant_message_id}
        turn, stored = ask(client, question, **options)
        assert stored["content"] == answer, f"the reply to {question!r} reads {stored['content']!r}"
        sent.append(turn)
    return sent


def test_several_turns_reopen_in_order_with_their_text_and_models(make_user, upstream, preset):
    account = make_user()
    tag = _tag()
    turns = [(f"question {n} {tag}", f"answer {n} {tag}") for n in range(1, 5)]
    models = [MOCK_MODEL_ID, preset, MOCK_MODEL_ID, preset]
    with account.client() as client:
        sent = _chat_of_turns(client, upstream, turns, models)
        chat_id = sent[0].chat_id
        assert {turn.chat_id for turn in sent} == {chat_id}, "a later turn started another chat"
        history = _history(client, chat_id)
        listed = client.get("/api/v1/chats/list").json()

    expected = [
        entry for question, answer in turns for entry in (("user", question), ("assistant", answer))
    ]
    assert _turns(history) == expected, f"the reopened chat reads {_turns(history)}"
    assert len(history["messages"]) == 8, f"the chat holds {len(history['messages'])} messages"
    assert history["currentId"] == sent[-1].assistant_message_id
    _assert_linked(history)
    path = _path(history)
    for turn, model, (question, _), (asked, answered) in zip(
        sent, models, turns, zip(path[::2], path[1::2])
    ):
        assert (asked["id"], answered["id"]) == (turn.user_message_id, turn.assistant_message_id)
        assert asked["models"] == [model], f"{question!r} lost its model: {asked.get('models')}"
        assert answered["model"] == model, f"the reply to {question!r} names {answered['model']}"
        assert answered["done"] is True
    times = [message["timestamp"] for message in path]
    assert all(isinstance(value, int) and value > 0 for value in times), times
    assert times == sorted(times), f"the messages' times run backwards: {times}"
    assert chat_id in {chat["id"] for chat in listed}, "the chat is missing from the chat list"


def test_a_regenerated_reply_keeps_both_versions_and_the_one_picked(make_user, upstream):
    account = make_user()
    tag = _tag()
    question = f"name a colour {tag}"
    upstream.queue(
        reply.text(f"first version {tag}", match=reply.answering(question)),
        reply.text(f"second version {tag}", match=reply.answering(question)),
    )
    with account.client() as client:
        turn, _ = ask(client, question)
        asked = _history(client, turn.chat_id)["messages"][turn.user_message_id]
        second_id = _regenerate(client, turn.chat_id, asked)

        regenerated = _history(client, turn.chat_id)
        assert regenerated["currentId"] == second_id, "the chat does not show the new version"
        assert regenerated["messages"][turn.user_message_id]["childrenIds"] == [
            turn.assistant_message_id,
            second_id,
        ], "the question does not list both versions in order"
        assert _turns(regenerated) == [("user", question), ("assistant", f"second version {tag}")]

        _show(client, turn.chat_id, turn.assistant_message_id)
        switched = _history(client, turn.chat_id)
        assert switched["currentId"] == turn.assistant_message_id, "the switch was not kept"
        assert _turns(switched) == [("user", question), ("assistant", f"first version {tag}")]
        assert switched["messages"][second_id]["content"] == f"second version {tag}"

        follow_up = f"and another {tag}"
        upstream.queue(reply.text(f"another answer {tag}", match=reply.answering(follow_up)))
        _continue(client, turn.chat_id, turn.assistant_message_id, follow_up)
        continued = _history(client, turn.chat_id)

    _assert_linked(continued)
    assert _turns(continued) == [
        ("user", question),
        ("assistant", f"first version {tag}"),
        ("user", follow_up),
        ("assistant", f"another answer {tag}"),
    ], f"the turn after the picked version reads {_turns(continued)}"
    assert continued["messages"][second_id]["childrenIds"] == []
    assert len(continued["messages"]) == 5


def test_an_edited_question_starts_a_branch_and_both_branches_stay(make_user, upstream):
    account = make_user()
    tag = _tag()
    with account.client() as client:
        first, original = _chat_of_turns(
            client,
            upstream,
            [(f"plan a trip {tag}", f"where to? {tag}"), (f"to Oslo {tag}", f"Oslo plan {tag}")],
        )
        chat_id = first.chat_id
        rewritten = f"to Bergen {tag}"
        upstream.queue(reply.text(f"Bergen plan {tag}", match=reply.answering(rewritten)))
        edited, _ = _continue(client, chat_id, first.assistant_message_id, rewritten)

        branched = _history(client, chat_id)
        assert branched["messages"][first.assistant_message_id]["childrenIds"] == [
            original.user_message_id,
            edited.user_message_id,
        ], "the reply does not list both questions as branches"
        assert _turns(branched)[2:] == [("user", rewritten), ("assistant", f"Bergen plan {tag}")]

        _show(client, chat_id, original.assistant_message_id)
        switched = _history(client, chat_id)

    _assert_linked(switched)
    assert _turns(switched) == [
        ("user", f"plan a trip {tag}"),
        ("assistant", f"where to? {tag}"),
        ("user", f"to Oslo {tag}"),
        ("assistant", f"Oslo plan {tag}"),
    ], f"the first branch reads {_turns(switched)}"
    assert switched["messages"][edited.assistant_message_id]["content"] == f"Bergen plan {tag}"


def test_a_deleted_question_and_its_reply_stay_gone_and_the_next_turn_moves_up(make_user, upstream):
    account = make_user()
    tag = _tag()
    turns = [(f"step {n} {tag}", f"done {n} {tag}") for n in range(1, 4)]
    with account.client() as client:
        first, middle, last = _chat_of_turns(client, upstream, turns)
        deleted = client.delete(f"/api/v1/chats/{first.chat_id}/messages/{middle.user_message_id}")
        assert deleted.status_code == 200, f"deleting the message failed: {deleted.text}"
        history = _history(client, first.chat_id)

    _assert_linked(history)
    assert middle.user_message_id not in history["messages"], "the deleted question came back"
    assert middle.assistant_message_id not in history["messages"], "its reply came back"
    assert _turns(history) == [
        ("user", f"step 1 {tag}"),
        ("assistant", f"done 1 {tag}"),
        ("user", f"step 3 {tag}"),
        ("assistant", f"done 3 {tag}"),
    ], f"after the delete the chat reads {_turns(history)}"
    assert history["messages"][last.user_message_id]["parentId"] == first.assistant_message_id


REPORTING_TOOL = '''
class Tools:
    async def harbour_report(self, harbour: str, __event_emitter__=None) -> str:
        """Report on a harbour, citing the harbour log.

        :param harbour: Name of the harbour
        """
        await __event_emitter__(
            {"type": "status", "data": {"description": f"Reading the {harbour} log", "done": True}}
        )
        await __event_emitter__(
            {
                "type": "citation",
                "data": {
                    "document": [f"The lights of {harbour} were lit at dusk."],
                    "metadata": [{"source": "harbour log"}],
                    "source": {"name": "Harbour log", "url": "https://example.com/log"},
                },
            }
        )
        await __event_emitter__(
            {"type": "files", "data": {"files": [{"type": "image", "url": "/static/favicon.png"}]}}
        )
        return f"report for {harbour}"
'''


def test_a_replys_citations_steps_and_files_are_there_after_a_reload(admin, make_user, upstream):
    account = make_user()
    tag = _tag()
    question = f"how is the harbour? {tag}"
    with python_tool(admin, REPORTING_TOOL, name=f"Harbour {tag}") as tool_id:
        upstream.queue(
            reply.tool_call("harbour_report", {"harbour": "Hamnvik"}),
            reply.text(f"all calm {tag}"),
        )
        with account.client() as client:
            uploaded = client.post(
                "/api/v1/files/",
                files={"file": (f"notes-{tag}.txt", b"Demo User packed a lantern.", "text/plain")},
                params={"process": "false"},
            )
            assert uploaded.status_code == 200, uploaded.text
            attached = {"type": "file", "id": uploaded.json()["id"], "name": f"notes-{tag}.txt"}
            turn, answered = ask(client, question, tool_ids=[tool_id], message_files=[attached])
            assert answered["content"].endswith(f"all calm {tag}"), answered["content"]
            follow_up = f"thanks {tag}"
            upstream.queue(reply.text(f"welcome {tag}", match=reply.answering(follow_up)))
            _continue(client, turn.chat_id, turn.assistant_message_id, follow_up)
            history = _history(client, turn.chat_id)

    asked = history["messages"][turn.user_message_id]
    answer = history["messages"][turn.assistant_message_id]
    assert [file.get("id") for file in asked.get("files", [])] == [attached["id"]], (
        f"the question lost its attached file: {asked.get('files')}"
    )
    names = [source.get("source", {}).get("name") for source in answer.get("sources") or []]
    assert "Harbour log" in names, f"the reply lost its citation: {answer.get('sources')}"
    steps = [status.get("description") for status in answer.get("statusHistory") or []]
    assert "Reading the Hamnvik log" in steps, f"the reply lost its step: {steps}"
    urls = [file.get("url") for file in answer.get("files") or []]
    assert "/static/favicon.png" in urls, f"the reply lost its file: {answer.get('files')}"
    assert "all calm" in answer["content"]
    assert len(_path(history)) == 4


def _export(client: httpx.Client) -> dict[str, dict]:
    exported = client.get("/api/v1/chats/all")
    assert exported.status_code == 200, exported.text
    lines = [json.loads(line) for line in exported.text.splitlines() if line.strip()]
    return {chat["id"]: chat for chat in lines}


def _shape(history: dict) -> list:
    """The conversation as a reader sees it, without ids: every branch with its text."""

    def walk(message_id: str) -> list:
        message = history["messages"][message_id]
        return [
            message["role"],
            message["content"],
            [walk(child) for child in message["childrenIds"]],
        ]

    roots = [key for key, message in history["messages"].items() if message.get("parentId") is None]
    return [walk(root) for root in roots]


def test_clone_fork_share_export_and_import_carry_the_whole_conversation(make_user, upstream):
    owner, reader = make_user(), make_user()
    tag = _tag()
    with owner.client() as client:
        first, second = _chat_of_turns(
            client,
            upstream,
            [(f"bake bread {tag}", f"knead it {tag}"), (f"how long? {tag}", f"an hour {tag}")],
        )
        chat_id = first.chat_id
        asked = _history(client, chat_id)["messages"][second.user_message_id]
        upstream.queue(reply.text(f"two hours {tag}", match=reply.answering(f"how long? {tag}")))
        _regenerate(client, chat_id, asked)
        original = _history(client, chat_id)
        shape = _shape(original)
        shown = _turns(original)
        assert shown[-1] == ("assistant", f"two hours {tag}")

        cloned = client.post(f"/api/v1/chats/{chat_id}/clone", json={"title": f"Copy {tag}"})
        assert cloned.status_code == 200, cloned.text
        clone_history = _history(client, cloned.json()["id"])
        assert _shape(clone_history) == shape, "the clone lost part of the conversation"
        assert _turns(clone_history) == shown, "the clone opens on another version"

        forked = client.post(
            f"/api/v1/chats/{chat_id}/fork", json={"message_id": first.assistant_message_id}
        )
        assert forked.status_code == 200, forked.text
        fork_history = _history(client, forked.json()["id"])
        assert _turns(fork_history) == shown[:2], f"the fork reads {_turns(fork_history)}"

        shared = client.post(f"/api/v1/chats/{chat_id}/share")
        assert shared.status_code == 200, shared.text
        share_id = shared.json()["share_id"]
        grant = {"principal_type": "user", "principal_id": reader.id, "permission": "read"}
        granted = client.post(
            f"/api/v1/chats/shared/{chat_id}/access/update", json={"access_grants": [grant]}
        )
        assert granted.status_code == 200, granted.text

        exported = _export(client)
        assert chat_id in exported, "the export is missing the chat"
        exported_history = exported[chat_id]["chat"]["history"]
        assert _shape(exported_history) == shape, "the export lost part of the conversation"
        assert _turns(exported_history) == shown

        entry = {"chat": {**exported[chat_id]["chat"], "title": f"Imported {tag}"}, "meta": {}}
        imported = client.post("/api/v1/chats/import", json={"chats": [entry]})
        assert imported.status_code == 200, imported.text
        [imported_chat] = imported.json()
        assert imported_chat["id"] != chat_id
        imported_history = _history(client, imported_chat["id"])
        assert _shape(imported_history) == shape, "the imported chat lost part of the conversation"
        assert _turns(imported_history) == shown, "the imported chat opens on another version"

        upstream.queue(reply.text(f"enjoy {tag}", match=reply.answering(f"done {tag}")))
        _continue(client, imported_chat["id"], imported_history["currentId"], f"done {tag}")
        assert _turns(_history(client, imported_chat["id"]))[-2:] == [
            ("user", f"done {tag}"),
            ("assistant", f"enjoy {tag}"),
        ], "the imported chat does not take a new turn"

    with reader.client() as client:
        opened = client.get(f"/api/v1/chats/share/{share_id}")
    assert opened.status_code == 200, opened.text
    shared_history = opened.json()["chat"]["history"]
    assert _turns(shared_history) == shown, f"the shared link reads {_turns(shared_history)}"


def _search(client: httpx.Client, text: str) -> set[str]:
    found = client.get("/api/v1/chats/search", params={"text": text})
    assert found.status_code == 200, found.text
    return {chat["id"] for chat in found.json()}


def test_search_finds_a_chat_by_a_word_from_any_of_its_messages(make_user, upstream):
    account = make_user()
    words = {name: f"{name}{_tag()}" for name in ("asked", "answered", "later")}
    with account.client() as client:
        first, _ = _chat_of_turns(
            client,
            upstream,
            [
                (f"tell me about {words['asked']}", "a first answer"),
                ("one more thing", f"it is {words['answered']}"),
            ],
        )
        unrelated = _chat_of_turns(client, upstream, [("something else", "nothing")])[0].chat_id
        for word in ("asked", "answered"):
            found = _search(client, words[word])
            assert found == {first.chat_id}, f"searching the {word} word found {found}"

        history = _history(client, first.chat_id)
        upstream.queue(reply.text(f"so {words['later']}", match=reply.answering("and then?")))
        _continue(client, first.chat_id, history["currentId"], "and then?")
        assert _search(client, words["later"]) == {first.chat_id}, "a new reply is not searchable"
        assert unrelated not in _search(client, words["asked"])


def test_a_long_chat_keeps_answering_and_keeps_every_message(make_user, upstream):
    account = make_user()
    tag = _tag()
    count = 40
    turns = [(f"turn {n} {tag}", f"reply {n} {tag}") for n in range(1, count + 1)]
    with account.client() as client:
        sent = _chat_of_turns(client, upstream, turns)
        chat_id = sent[0].chat_id
        history = _history(client, chat_id)
        asked = history["messages"][sent[-1].user_message_id]
        upstream.queue(
            reply.text(f"reply {count} again {tag}", match=reply.answering(asked["content"]))
        )
        regenerated = _regenerate(client, chat_id, asked)
        after = _history(client, chat_id)

    expected = [
        entry for question, answer in turns for entry in (("user", question), ("assistant", answer))
    ]
    assert _turns(history) == expected, "the long chat lost or reordered a message"
    _assert_linked(after)
    assert len(after["messages"]) == 2 * count + 1, f"the chat holds {len(after['messages'])}"
    assert after["currentId"] == regenerated
    assert _turns(after) == [*expected[:-1], ("assistant", f"reply {count} again {tag}")]
    latest = upstream.chat_requests()[-1]["messages"]
    sent_texts = [
        message["content"] for message in latest if message["role"] in ("user", "assistant")
    ]
    assert sent_texts[:2] == [f"turn 1 {tag}", f"reply 1 {tag}"], (
        f"the model was not sent the start of the long chat: {sent_texts[:2]}"
    )


# --- the upgrade: chats an earlier release stored as one record each ---------------------------


def _old_chats(release: ReleaseData) -> dict[str, dict]:
    """Every chat as the release stored it, read before the checkout first starts on it."""
    engine = sqlalchemy.create_engine(release.database_url)
    try:
        with engine.connect() as connection:
            rows = connection.execute(sqlalchemy.text("SELECT id, user_id, chat FROM chat")).all()
    finally:
        engine.dispose()
    chats = {}
    for chat_id, user_id, chat in rows:
        chat = json.loads(chat) if isinstance(chat, str) else chat
        if (chat or {}).get("history", {}).get("messages"):
            chats[chat_id] = {"user_id": user_id, "history": chat["history"]}
    return chats


@pytest.fixture(scope="module", params=data_set_params())
def upgraded(request, tmp_path_factory) -> Iterator[tuple[Upgraded, dict]]:
    before: dict[str, dict] = {}
    name = request.param
    with upgraded_release(
        name,
        tmp_path_factory.mktemp(name),
        prepare=lambda release: before.update(_old_chats(release)),
    ) as release:
        yield release, before


def _publish_the_scripted_model(release: Upgraded) -> None:
    """The admin lets every account use the scripted model, as its visibility toggle does."""
    with release.client("admin") as client:
        client.get("/api/models").raise_for_status()
        published = client.post(
            "/api/v1/models/model/access/update",
            json={"id": MOCK_MODEL_ID, "name": MOCK_MODEL_ID, "access_grants": [EVERYONE_READS]},
        )
        assert published.status_code == 200, published.text


KEPT = ("role", "content", "parentId", "childrenIds", "timestamp", "model", "models")


@pytest.mark.slow
def test_chats_stored_by_an_earlier_release_keep_every_message_order_and_time(upgraded):
    release, before = upgraded
    alice = release.manifest["accounts"]["alice"]["id"]
    owned = {chat_id: old for chat_id, old in before.items() if old["user_id"] == alice}
    assert len(owned) >= 3, f"the data set holds too few of alice's chats: {len(owned)}"
    with release.client("alice") as client:
        for chat_id, old in owned.items():
            history = _history(client, chat_id)
            assert set(history["messages"]) == set(old["history"]["messages"]), (
                f"chat {chat_id} lost or gained messages in the upgrade"
            )
            for message_id, message in old["history"]["messages"].items():
                kept = history["messages"][message_id]
                changed = {
                    key: (message.get(key), kept.get(key))
                    for key in KEPT
                    if message.get(key) != kept.get(key)
                }
                assert not changed, f"message {message_id} of chat {chat_id} changed: {changed}"
            assert history["currentId"] == old["history"]["currentId"], (
                f"chat {chat_id} no longer opens on the message the user was last on"
            )

        live = release.manifest["chats"]["live"]
        tag = _tag()
        _publish_the_scripted_model(release)
        release.provider.queue(
            reply.text(f"still here {tag}", match=reply.answering(f"again {tag}"))
        )
        turn = send_message(
            client, f"again {tag}", chat_id=live["id"], parent_id=live["current_id"]
        )
        wait_for_reply(client, turn, timeout=90)
        continued = _history(client, live["id"])
        # a question of the chat, as replies streamed by v0.10.2 stored their text elsewhere
        found = _search(client, "eat it")

    old_shown = [
        (message["role"], message["content"]) for message in _path(before[live["id"]]["history"])
    ]
    assert _turns(continued) == [
        *old_shown,
        ("user", f"again {tag}"),
        ("assistant", f"still here {tag}"),
    ], f"the upgraded chat does not carry on where it stopped: {_turns(continued)}"
    assert live["id"] in found, "search no longer finds the upgraded chat by a question's words"
    times = [message["timestamp"] for message in _path(continued)]
    assert times == sorted(times), f"the messages' times run backwards after the upgrade: {times}"
