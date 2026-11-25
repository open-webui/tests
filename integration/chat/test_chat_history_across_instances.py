"""Journey: a chat used on two instances at once keeps every message, branch and the turn picked.

PR open-webui/open-webui#34689 (414340682) stores a chat's messages one by one instead of one
copy of the whole conversation. Several instances behind a load balancer share one Postgres and
one Redis, so a user's requests and tabs land on either. What a user expects is the same as on one
instance: a chat continued on one instance and then on the other (send, regenerate, edit, switch
to another version) shows every message and branch on both, once each, with the turn just made or
picked shown; a tab that has not reloaded since the other instance answered and then saves the
chat keeps the other instance's messages; and two sends or two regenerations at the same moment
from two tabs, one per instance, both land as branches of the same message. After every step the
chat is read from both instances, as a reload on either does.

Two instances on one embedded Postgres and one real Redis; a tab is a client that sends what the
web client sends and keeps the chat as it last saw it. Several instances on SQLite are no
supported setup and have no test.

Discriminates: passes on dev f80a4f782. In a backend copy of it whose chat save stores the saved
history as it is and deletes the messages it does not hold, the stale tab steps fail; in one that
stores a question sent after a message which already has a branch without its parent, the
alternating and the same moment send tests fail; in one that takes no lock on the chat for a new
turn and no longer rebuilds branches from each message's parent, the same moment regenerate test
fails. Sends at the same moment never lost a turn under the last edit, as each new question is
stored as a message of its own.
"""

from __future__ import annotations

import dataclasses
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

import httpx
import pytest

from harness import backends
from harness import upstream as reply
from harness.actors import Actor, create_user
from harness.chat import NO_BACKGROUND_TASKS, ChatTurn, wait_for_reply
from harness.instance import LaunchedInstance
from harness.upstream import MOCK_MODEL_ID

pytestmark = [
    pytest.mark.journey,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

SAME_MOMENT_ROUNDS = 3


@pytest.fixture(scope="module")
def shared_redis() -> Iterator[backends.RedisProcess]:
    server = backends.RedisProcess()
    server.start()
    yield server
    server.close()


@pytest.fixture(scope="module")
def shared_postgres() -> Iterator[str]:
    with backends.postgres_database() as url:
        yield url


@pytest.fixture(scope="module")
def pair(shared_redis, shared_postgres, instance_with) -> tuple[LaunchedInstance, LaunchedInstance]:
    """Instances A and B on one Postgres and one Redis, as behind a load balancer."""
    shared = {
        "REDIS_URL": shared_redis.url,
        "WEBSOCKET_MANAGER": "redis",
        "DATABASE_URL": shared_postgres,
    }
    first = instance_with({**shared, "WEBUI_NAME": "chat-instance-a"})
    second = instance_with({**shared, "WEBUI_NAME": "chat-instance-b"})
    return first, second


@pytest.fixture
def provider(pair):
    """The provider both instances ask: its connection lives in the shared database."""
    pair[0].upstream.reset()
    return pair[0].upstream


def _tag() -> str:
    return uuid.uuid4().hex[:8]


def _read(account: Actor, instance: LaunchedInstance, chat_id: str) -> dict:
    with dataclasses.replace(account, base_url=instance.base_url).client() as client:
        opened = client.get(f"/api/v1/chats/{chat_id}")
    assert opened.status_code == 200, f"the chat did not open on {instance.base_url}: {opened.text}"
    return opened.json()["chat"]["history"]


def _path(history: dict) -> list[dict]:
    messages, shown, message_id = history["messages"], [], history["currentId"]
    while message_id is not None:
        assert message_id in messages, f"message {message_id} on the shown path is missing"
        shown.append(messages[message_id])
        message_id = messages[message_id].get("parentId")
    return shown[::-1]


def _turns(history: dict) -> list[str]:
    return [message["content"] for message in _path(history)]


class Tab:
    """A browser tab on one instance: sends what the web client sends, keeps the chat it saw."""

    def __init__(self, account: Actor, instance: LaunchedInstance):
        self.actor = dataclasses.replace(account, base_url=instance.base_url)
        self.instance = instance
        self.chat_id: str | None = None
        self.history: dict = {"messages": {}, "currentId": None}

    def client(self) -> httpx.Client:
        return self.actor.client()

    def open(self, chat_id: str) -> dict:
        self.chat_id = chat_id
        self.history = _read(self.actor, self.instance, chat_id)
        return self.history

    def _post(self, question: dict, reply_id: str) -> ChatTurn:
        payload = {
            "model": MOCK_MODEL_ID,
            "messages": [{"role": "user", "content": question["content"]}],
            "stream": True,
            "parent_id": question["parentId"],
            "id": reply_id,
            "message_ids": [{"model_id": MOCK_MODEL_ID, "message_id": reply_id, "modelIdx": 0}],
            "user_message": question,
            "session_id": f"harness-{uuid.uuid4().hex[:8]}",
            "background_tasks": NO_BACKGROUND_TASKS,
        }
        if self.chat_id:
            payload["chat_id"] = self.chat_id
        with self.client() as client:
            accepted = client.post("/api/chat/completions", json=payload)
        assert accepted.status_code == 200, f"the send was refused: {accepted.text}"
        self.chat_id = accepted.json()["chat_id"]
        messages = self.history["messages"]
        messages[question["id"]] = question
        if question["parentId"] in messages:
            parent = messages[question["parentId"]]
            parent["childrenIds"] = list(dict.fromkeys([*parent["childrenIds"], question["id"]]))
        messages[reply_id] = {
            "id": reply_id,
            "parentId": question["id"],
            "childrenIds": [],
            "role": "assistant",
            "content": "",
            "done": False,
            "model": MOCK_MODEL_ID,
        }
        self.history["currentId"] = reply_id
        return ChatTurn(self.chat_id, question["id"], reply_id)

    def send(self, content: str, after: str | None) -> ChatTurn:
        """Send a question after the message `after` (None starts the chat)."""
        reply_id = str(uuid.uuid4())
        question = {
            "id": str(uuid.uuid4()),
            "parentId": after,
            "childrenIds": [reply_id],
            "role": "user",
            "content": content,
            "models": [MOCK_MODEL_ID],
            "timestamp": int(time.time()),
        }
        return self._post(question, reply_id)

    def edit(self, question_id: str, content: str) -> ChatTurn:
        """Edit and send a question again: a new question next to the old one."""
        return self.send(content, self.history["messages"][question_id]["parentId"])

    def regenerate(self, question_id: str) -> ChatTurn:
        reply_id = str(uuid.uuid4())
        question = dict(self.history["messages"][question_id])
        question["childrenIds"] = [*question["childrenIds"], reply_id]
        return self._post(question, reply_id)

    def finish(self, turn: ChatTurn) -> dict:
        """Wait for the reply; the tab shows it as it streamed in, the rest stays as it saw it."""
        with self.client() as client:
            answered = wait_for_reply(client, turn)
        self.history["messages"][turn.assistant_message_id] = answered
        return answered

    def show(self, message_id: str) -> None:
        """Pick the version ending in `message_id` and save the chat, as the web client does."""
        self.history["currentId"] = message_id
        listed = []
        message_id_on_path = message_id
        while message_id_on_path is not None:
            listed.append(self.history["messages"][message_id_on_path])
            message_id_on_path = listed[-1].get("parentId")
        with self.client() as client:
            saved = client.post(
                f"/api/v1/chats/{self.chat_id}",
                json={"chat": {"history": self.history, "messages": listed[::-1]}},
            )
        assert saved.status_code == 200, f"saving the switch failed: {saved.text}"


def _assert_both_show(
    account: Actor,
    pair,
    chat_id: str,
    *,
    children: dict[str, list[str]],
    contents: dict[str, str],
    current: str,
    step: str,
) -> dict:
    """Both instances show exactly these messages, branches and current turn; returns the chat."""
    first, second = (_read(account, instance, chat_id) for instance in pair)
    assert first == second, f"after {step} the two instances show different chats"
    messages = first["messages"]
    assert set(messages) == set(contents), (
        f"after {step} the chat holds {sorted(m['content'] for m in messages.values())}, "
        f"expected {sorted(contents.values())}"
    )
    for message_id, content in contents.items():
        assert messages[message_id]["content"] == content, (
            f"after {step} a message reads {messages[message_id]['content']!r}, not {content!r}"
        )
    shown_children = {message_id: messages[message_id]["childrenIds"] for message_id in children}
    assert shown_children == children, f"after {step} the branches are " + str(
        {contents.get(k, k): [contents.get(c, c) for c in v] for k, v in shown_children.items()}
    )
    for message_id, message in messages.items():
        assert len(message["childrenIds"]) == len(set(message["childrenIds"])), (
            f"after {step} {message['content']!r} lists a branch twice: {message['childrenIds']}"
        )
        if message["role"] == "assistant":
            assert message.get("done") is True, f"after {step} {message['content']!r} is not done"
    assert first["currentId"] == current, (
        f"after {step} the chat shows {contents.get(first['currentId'])!r}, "
        f"expected {contents[current]!r}"
    )
    return first


def test_a_chat_continued_on_each_instance_in_turn_keeps_every_message_and_branch(pair, provider):
    first, second = pair
    account = create_user(first)
    tag = _tag()
    on_a, on_b = Tab(account, first), Tab(account, second)
    contents: dict[str, str] = {}
    children: dict[str, list[str]] = {}

    def answered(turn: ChatTurn, tab: Tab, question: str, answer: str) -> None:
        stored = tab.finish(turn)
        assert stored["content"] == answer, f"the reply to {question!r} reads {stored['content']!r}"
        contents[turn.user_message_id] = question
        contents[turn.assistant_message_id] = answer
        children.setdefault(turn.user_message_id, []).append(turn.assistant_message_id)
        children.setdefault(turn.assistant_message_id, [])

    q1, a1 = f"plan a trip {tag}", f"where to? {tag}"
    provider.queue(reply.text(a1, match=reply.answering(q1)))
    one = on_a.send(q1, None)
    answered(one, on_a, q1, a1)
    chat_id = one.chat_id
    _assert_both_show(
        account,
        pair,
        chat_id,
        children=children,
        contents=contents,
        current=one.assistant_message_id,
        step="the first turn on A",
    )

    on_b.open(chat_id)
    q2, a2 = f"to Oslo {tag}", f"Oslo plan {tag}"
    provider.queue(reply.text(a2, match=reply.answering(q2)))
    two = on_b.send(q2, one.assistant_message_id)
    answered(two, on_b, q2, a2)
    children[one.assistant_message_id] = [two.user_message_id]
    _assert_both_show(
        account,
        pair,
        chat_id,
        children=children,
        contents=contents,
        current=two.assistant_message_id,
        step="the second turn on B",
    )

    on_a.open(chat_id)
    a2_again = f"Oslo plan, second version {tag}"
    provider.queue(reply.text(a2_again, match=reply.answering(q2)))
    regenerated = on_a.regenerate(two.user_message_id)
    answered(regenerated, on_a, q2, a2_again)
    _assert_both_show(
        account,
        pair,
        chat_id,
        children=children,
        contents=contents,
        current=regenerated.assistant_message_id,
        step="regenerating on A",
    )

    on_b.open(chat_id)
    q2_edited, a2_edited = f"to Bergen {tag}", f"Bergen plan {tag}"
    provider.queue(reply.text(a2_edited, match=reply.answering(q2_edited)))
    edited = on_b.edit(two.user_message_id, q2_edited)
    answered(edited, on_b, q2_edited, a2_edited)
    children[one.assistant_message_id] = [two.user_message_id, edited.user_message_id]
    _assert_both_show(
        account,
        pair,
        chat_id,
        children=children,
        contents=contents,
        current=edited.assistant_message_id,
        step="editing the question on B",
    )

    # A has not reloaded since its regeneration, so it never saw B's edit
    on_a.show(two.assistant_message_id)
    history = _assert_both_show(
        account,
        pair,
        chat_id,
        children=children,
        contents=contents,
        current=two.assistant_message_id,
        step="switching back to the first version on a stale A",
    )
    assert _turns(history) == [q1, a1, q2, a2], f"the picked version reads {_turns(history)}"

    # B has not reloaded since its edit either
    q3, a3 = f"what to pack in Bergen? {tag}", f"a raincoat {tag}"
    provider.queue(reply.text(a3, match=reply.answering(q3)))
    three = on_b.send(q3, edited.assistant_message_id)
    answered(three, on_b, q3, a3)
    children[edited.assistant_message_id] = [three.user_message_id]
    history = _assert_both_show(
        account,
        pair,
        chat_id,
        children=children,
        contents=contents,
        current=three.assistant_message_id,
        step="continuing the edited branch on a stale B",
    )
    assert _turns(history) == [q1, a1, q2_edited, a2_edited, q3, a3], _turns(history)


def _at_the_same_moment(provider, first_action, second_action, replies) -> list[ChatTurn]:
    """Run both actions at once; both replies are held until both requests reached a provider."""
    release = threading.Event()
    before = len(provider.chat_requests())
    provider.queue(*[dataclasses.replace(entry, hold=release) for entry in replies])
    start = threading.Barrier(2)

    def run(action):
        start.wait(timeout=10)
        return action()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            sent = [pool.submit(run, action) for action in (first_action, second_action)]
            assert backends.wait_until(
                lambda: len(provider.chat_requests()) >= before + 2, timeout=30
            ), "the two requests did not both reach the provider"
            release.set()
            return [future.result(timeout=60) for future in sent]
    finally:
        release.set()


def test_two_tabs_sending_at_the_same_moment_keep_both_turns(pair, provider):
    first, second = pair
    account = create_user(first)
    tag = _tag()
    on_a, on_b = Tab(account, first), Tab(account, second)
    question, answer = f"start {tag}", f"started {tag}"
    provider.queue(reply.text(answer, match=reply.answering(question)))
    start = on_a.send(question, None)
    on_a.finish(start)
    chat_id = start.chat_id
    contents = {start.user_message_id: question, start.assistant_message_id: answer}
    children = {start.user_message_id: [start.assistant_message_id]}
    parent_id = start.assistant_message_id

    for round_number in range(SAME_MOMENT_ROUNDS):
        on_a.open(chat_id)
        on_b.open(chat_id)
        asked = {
            "A": (f"from A, round {round_number} {tag}", f"A answered {round_number} {tag}"),
            "B": (f"from B, round {round_number} {tag}", f"B answered {round_number} {tag}"),
        }
        replies = [reply.text(a, match=reply.answering(q)) for q, a in asked.values()]
        turn_a, turn_b = _at_the_same_moment(
            provider,
            lambda: on_a.send(asked["A"][0], parent_id),
            lambda: on_b.send(asked["B"][0], parent_id),
            replies,
        )
        on_a.finish(turn_a)
        on_b.finish(turn_b)
        for turn, (q, a) in ((turn_a, asked["A"]), (turn_b, asked["B"])):
            contents[turn.user_message_id] = q
            contents[turn.assistant_message_id] = a
            children[turn.user_message_id] = [turn.assistant_message_id]
        history = _read(account, first, chat_id)
        stored_branches = history["messages"].get(parent_id, {}).get("childrenIds", [])
        assert sorted(stored_branches) == sorted(
            [turn_a.user_message_id, turn_b.user_message_id]
        ), (
            f"round {round_number}: the reply both tabs answered lists "
            f"{[contents.get(i, i) for i in stored_branches]}, expected both questions"
        )
        children[parent_id] = stored_branches
        current = history["currentId"]
        assert current in (turn_a.assistant_message_id, turn_b.assistant_message_id), (
            f"round {round_number}: the chat shows {contents.get(current, current)!r}, "
            "neither of the two new turns"
        )
        _assert_both_show(
            account,
            pair,
            chat_id,
            children=children,
            contents=contents,
            current=current,
            step=f"two sends at the same moment, round {round_number}",
        )
        parent_id = current


def test_two_tabs_regenerating_at_the_same_moment_keep_every_version(pair, provider):
    first, second = pair
    account = create_user(first)
    tag = _tag()
    on_a, on_b = Tab(account, first), Tab(account, second)
    question, answer = f"name a colour {tag}", f"blue {tag}"
    provider.queue(reply.text(answer, match=reply.answering(question)))
    start = on_a.send(question, None)
    on_a.finish(start)
    chat_id = start.chat_id
    contents = {start.user_message_id: question, start.assistant_message_id: answer}
    versions = [start.assistant_message_id]

    for round_number in range(SAME_MOMENT_ROUNDS):
        on_a.open(chat_id)
        on_b.open(chat_id)
        new_answers = {f"green {round_number} {tag}", f"red {round_number} {tag}"}
        replies = [reply.text(a, match=reply.answering(question)) for a in sorted(new_answers)]
        turn_a, turn_b = _at_the_same_moment(
            provider,
            lambda: on_a.regenerate(start.user_message_id),
            lambda: on_b.regenerate(start.user_message_id),
            replies,
        )
        got = {on_a.finish(turn_a)["content"], on_b.finish(turn_b)["content"]}
        assert got == new_answers, f"round {round_number}: the two new versions read {got}"
        contents[turn_a.assistant_message_id] = on_a.history["messages"][
            turn_a.assistant_message_id
        ]["content"]
        contents[turn_b.assistant_message_id] = on_b.history["messages"][
            turn_b.assistant_message_id
        ]["content"]
        history = _read(account, first, chat_id)
        stored_versions = history["messages"][start.user_message_id]["childrenIds"]
        expected_versions = [*versions, turn_a.assistant_message_id, turn_b.assistant_message_id]
        assert sorted(stored_versions) == sorted(expected_versions), (
            f"round {round_number}: the question lists "
            f"{[contents.get(i, i) for i in stored_versions]}, expected "
            f"{[contents[i] for i in expected_versions]}"
        )
        assert stored_versions[: len(versions)] == versions, (
            f"round {round_number}: the earlier versions moved: {stored_versions}"
        )
        versions = stored_versions
        current = history["currentId"]
        assert current in (turn_a.assistant_message_id, turn_b.assistant_message_id), (
            f"round {round_number}: the chat shows {contents.get(current, current)!r}, "
            "neither of the two new versions"
        )
        _assert_both_show(
            account,
            pair,
            chat_id,
            children={start.user_message_id: versions},
            contents=contents,
            current=current,
            step=f"two regenerations at the same moment, round {round_number}",
        )
