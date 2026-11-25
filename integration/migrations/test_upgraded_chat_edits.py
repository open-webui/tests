"""Journey: a chat an earlier release stored keeps every change its user makes after the upgrade.

The migration that moves chat messages into rows of their own (`5e7a2c9d4b13`, PR
open-webui/open-webui#34689) keeps, next to each message's row, whatever of the old message the
row's fields would read back differently, and lays it over the message whenever the chat is
read. What a user expects on an upgraded chat is what they get on a new one: an edited question
or reply, a changed rating, a deleted version, a continued or regenerated reply and a new turn
each show after a reload, the model is sent the edited text, and nothing an earlier version of
a message held comes back. The data sets under `upgrade_data/` get older shapes into two of
their chats before the checkout first starts on them, as releases before the rows left them:
times in milliseconds, a question stored as a list of parts, a reply version with no text and
a reply rated with feedback. Each step saves the chat as the web client does and reads it back
from the server, as a reload does.

Discriminates: passes on dev f80a4f782 for all six data sets. In a backend copy whose message
save keeps the old message copy the migration stored instead of replacing it, both tests fail on
every data set: the new rating is gone after a reload, and the changed rating of the rated reply
reads its old value again.
"""

from __future__ import annotations

import copy
import json
import time
import uuid
from typing import Callable, Iterator

import httpx
import pytest
import sqlalchemy

from harness import upstream as reply
from harness.chat import NO_BACKGROUND_TASKS, ChatTurn, send_message, wait_for_reply
from harness.prepared_data import ReleaseData, run_sql
from harness.python_tools import EVERYONE_READS
from harness.upgraded_release import Upgraded, data_set_params, upgraded_release
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.slow, pytest.mark.api, pytest.mark.requires_source]


def _tag() -> str:
    return uuid.uuid4().hex[:8]


# --- older shapes, written before the checkout first starts -----------------------------------


def _json_parameter(release: ReleaseData) -> Callable[[str], str]:
    if release.manifest["engine"] == "postgres":
        return lambda name: f"CAST(:{name} AS json)"
    return lambda name: f":{name}"


def _stored_chat(release: ReleaseData, chat_id: str) -> dict:
    engine = sqlalchemy.create_engine(release.database_url)
    try:
        with engine.connect() as connection:
            chat = connection.execute(
                sqlalchemy.text("SELECT chat FROM chat WHERE id = :id"), {"id": chat_id}
            ).scalar_one()
    finally:
        engine.dispose()
    return json.loads(chat) if isinstance(chat, str) else chat


def _store(release: ReleaseData, chat_id: str, chat: dict, contents: dict[str, object]) -> None:
    """Save the chat record and the changed messages' rows, as the old release kept both."""
    as_json = _json_parameter(release)
    run_sql(
        release.database_url,
        f"UPDATE chat SET chat = {as_json('chat')} WHERE id = :id",
        {"id": chat_id, "chat": json.dumps(chat)},
    )
    for message_id, content in contents.items():
        run_sql(
            release.database_url,
            f"UPDATE chat_message SET content = {as_json('content')} WHERE id = :id",
            {"id": f"{chat_id}-{message_id}", "content": json.dumps(content)},
        )


def _age(release: ReleaseData) -> None:
    chats = release.manifest["chats"]

    branched = chats["branched"]
    chat = _stored_chat(release, branched["id"])
    messages = chat["history"]["messages"]
    version, follow_up = _ids(branched)["version"], _ids(branched)["question"]
    for message in messages.values():
        message["timestamp"] = message["timestamp"] * 1000
    messages[follow_up]["content"] = [{"type": "text", "text": messages[follow_up]["content"]}]
    messages[version]["content"] = None
    _store(
        release,
        branched["id"],
        chat,
        {follow_up: messages[follow_up]["content"], version: None},
    )

    live, feedback = chats["live"], release.manifest["feedback"]
    chat = _stored_chat(release, live["id"])
    chat["history"]["messages"][live["reply_id"]].update(
        annotation={"rating": feedback["rating"], "reason": feedback["reason"]},
        feedbackId=feedback["id"],
    )
    _store(release, live["id"], chat, {})


def _ids(branched: dict) -> dict[str, str]:
    """The branched chat: a question with two reply versions, the second followed by a turn."""
    messages = branched["messages"]
    [first] = [key for key, message in messages.items() if message["parent"] is None]
    version, picked = messages[first]["children"]
    [question] = messages[picked]["children"]
    [answer] = messages[question]["children"]
    return {
        "first": first,
        "version": version,
        "picked": picked,
        "question": question,
        "answer": answer,
    }


@pytest.fixture(scope="module", params=data_set_params())
def upgraded(request, tmp_path_factory) -> Iterator[Upgraded]:
    name = request.param
    with upgraded_release(name, tmp_path_factory.mktemp(name), prepare=_age) as release:
        with release.client("admin") as client:
            client.get("/api/models").raise_for_status()
            published = client.post(
                "/api/v1/models/model/access/update",
                json={
                    "id": MOCK_MODEL_ID,
                    "name": MOCK_MODEL_ID,
                    "access_grants": [EVERYONE_READS],
                },
            )
            assert published.status_code == 200, published.text
        yield release


# --- what the web client does -----------------------------------------------------------------


def _history(client: httpx.Client, chat_id: str) -> dict:
    opened = client.get(f"/api/v1/chats/{chat_id}")
    assert opened.status_code == 200, f"the chat did not open: {opened.text}"
    return opened.json()["chat"]["history"]


def _path(history: dict) -> list[dict]:
    messages = history["messages"]
    shown, message_id = [], history["currentId"]
    while message_id is not None:
        assert message_id in messages, f"message {message_id} on the shown path is missing"
        shown.append(messages[message_id])
        message_id = messages[message_id].get("parentId")
    return shown[::-1]


def _save(client: httpx.Client, chat_id: str, history: dict) -> None:
    """Save the whole conversation the page holds, as every edit in the client does."""
    saved = client.post(
        f"/api/v1/chats/{chat_id}",
        json={"chat": {"history": history, "messages": _path(history)}},
    )
    assert saved.status_code == 200, f"saving the chat failed: {saved.text}"


def _edit(client: httpx.Client, chat_id: str, message_id: str, change: Callable[[dict], None]):
    history = copy.deepcopy(_history(client, chat_id))
    change(history["messages"][message_id])
    _save(client, chat_id, history)


def _output_text(message: dict) -> str:
    parts = [
        part.get("text", "")
        for item in message.get("output") or []
        if item.get("type") == "message"
        for part in item.get("content") or []
    ]
    return "".join(parts)


def _with_output_text(output: list[dict], text: str) -> list[dict]:
    """The reply's output with its last message reading `text`, as the reply editor saves it."""
    last = [index for index, item in enumerate(output) if item.get("type") == "message"][-1]
    return [
        {**item, "content": [{"type": "output_text", "text": text}]}
        if index == last
        else {**item, "content": []}
        if item.get("type") == "message"
        else item
        for index, item in enumerate(output)
    ]


def _rate(client: httpx.Client, chat_id: str, message_id: str, rating: int) -> str:
    """Thumb a reply up or down: update its feedback, then save the rating on the message."""
    history = copy.deepcopy(_history(client, chat_id))
    message = history["messages"][message_id]
    annotation = {**(message.get("annotation") or {}), "rating": rating, "reason": None}
    feedback = {
        "type": "rating",
        "data": {**annotation, "model_id": message.get("model")},
        "meta": {"chat_id": chat_id, "message_id": message_id, "model_id": message.get("model")},
        "snapshot": {"chat": {"history": history}},
    }
    if message.get("feedbackId"):
        sent = client.post(f"/api/v1/evaluations/feedback/{message['feedbackId']}", json=feedback)
    else:
        sent = client.post("/api/v1/evaluations/feedback", json=feedback)
    assert sent.status_code == 200, f"saving the feedback failed: {sent.text}"
    message.update(annotation=annotation, feedbackId=sent.json()["id"])
    _save(client, chat_id, history)
    return sent.json()["id"]


def _completion(client: httpx.Client, chat_id: str, question: dict, reply_id: str, **extra):
    sent = client.post(
        "/api/chat/completions",
        json={
            "model": MOCK_MODEL_ID,
            "messages": [{"role": "user", "content": "x"}],
            "stream": True,
            "chat_id": chat_id,
            "parent_id": question["parentId"],
            "id": reply_id,
            "user_message": question,
            "session_id": f"harness-{uuid.uuid4().hex[:8]}",
            "background_tasks": NO_BACKGROUND_TASKS,
            **extra,
        },
    )
    assert sent.status_code == 200, f"the request was refused: {sent.text}"


def _regenerate(client: httpx.Client, chat_id: str, question: dict) -> str:
    reply_id = str(uuid.uuid4())
    asked = {**question, "childrenIds": [*question["childrenIds"], reply_id]}
    _completion(client, chat_id, asked, reply_id)
    wait_for_reply(client, ChatTurn(chat_id, question["id"], reply_id), timeout=90)
    return reply_id


def _continue_reply(client: httpx.Client, chat_id: str, question: dict, reply_id: str) -> None:
    _completion(client, chat_id, question, reply_id, assistant_message_id=reply_id)


def _wait_for(client: httpx.Client, chat_id: str, message_id: str, done: Callable[[dict], bool]):
    deadline = time.monotonic() + 90
    message: dict = {}
    while time.monotonic() < deadline:
        message = _history(client, chat_id)["messages"].get(message_id, {})
        if message.get("done") and done(message):
            return message
        time.sleep(0.2)
    raise AssertionError(f"the reply never finished; last stored state: {message}")


def _sent_texts(request: dict) -> list[str]:
    texts = []
    for message in request["messages"]:
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if message["role"] in ("user", "assistant"):
            texts.append(content or "")
    return texts


# --- the journeys ---------------------------------------------------------------------------


def test_edits_a_rating_and_a_deleted_version_stick_on_an_upgraded_chat(upgraded):
    branched = upgraded.manifest["chats"]["branched"]
    chat_id, ids = branched["id"], _ids(branched)
    tag = _tag()
    question = f"And for a rainy evening? {tag}"
    answer = f"A raincoat and boots. {tag}"
    old_question = branched["messages"][ids["question"]]["content"]
    old_answer = branched["messages"][ids["answer"]]["content"]

    with upgraded.client("alice") as client:
        before = _history(client, chat_id)
        times = {key: message["timestamp"] for key, message in before["messages"].items()}
        # the older shapes made it through the upgrade, so the edits below start from them
        assert isinstance(before["messages"][ids["question"]]["content"], list)
        assert before["messages"][ids["version"]]["content"] is None
        assert all(value > 10**12 for value in times.values()), f"times not in ms: {times}"

        def edit_question(message: dict) -> None:
            message["content"] = question

        _edit(client, chat_id, ids["question"], edit_question)
        edited = _history(client, chat_id)["messages"]
        assert edited[ids["question"]]["content"] == question, (
            f"the edited question reads {edited[ids['question']]['content']!r} after a reload"
        )

        def edit_answer(message: dict) -> None:
            message["originalContent"] = message["content"]
            message["content"] = answer

        _edit(client, chat_id, ids["answer"], edit_answer)
        edited = _history(client, chat_id)["messages"]
        assert edited[ids["answer"]]["content"] == answer, (
            f"the edited reply reads {edited[ids['answer']]['content']!r} after a reload"
        )
        assert edited[ids["question"]]["content"] == question, (
            "editing the reply undid the question"
        )

        _rate(client, chat_id, ids["answer"], -1)
        rated = _history(client, chat_id)["messages"]
        assert (rated[ids["answer"]].get("annotation") or {}).get("rating") == -1, (
            f"the rating is gone after a reload: {rated[ids['answer']].get('annotation')}"
        )
        assert (rated[ids["question"]]["content"], rated[ids["answer"]]["content"]) == (
            question,
            answer,
        ), "rating the reply undid an edit"

        deleted = client.delete(f"/api/v1/chats/{chat_id}/messages/{ids['version']}")
        assert deleted.status_code == 200, f"deleting the version failed: {deleted.text}"
        after_delete = _history(client, chat_id)
        assert ids["version"] not in after_delete["messages"], "the deleted version came back"
        assert after_delete["messages"][ids["first"]]["childrenIds"] == [ids["picked"]], (
            "the version switcher still lists the deleted version: "
            f"{after_delete['messages'][ids['first']]['childrenIds']}"
        )

        follow_up = f"Thanks, anything else? {tag}"
        upgraded.provider.queue(reply.text(f"An umbrella. {tag}", match=reply.answering(follow_up)))
        turn = send_message(client, follow_up, chat_id=chat_id, parent_id=ids["answer"])
        wait_for_reply(client, turn, timeout=90)
        final = _history(client, chat_id)

    sent = _sent_texts(upgraded.provider.chat_requests()[-1])
    assert question in sent and answer in sent, f"the model was not sent the edits: {sent}"
    assert old_question not in sent and old_answer not in sent, (
        f"the model was sent the text before the edits: {sent}"
    )
    shown = [(message["role"], message["content"]) for message in _path(final)]
    assert shown[2:] == [
        ("user", question),
        ("assistant", answer),
        ("user", follow_up),
        ("assistant", f"An umbrella. {tag}"),
    ], f"after the new turn the chat reads {shown}"
    messages = final["messages"]
    assert ids["version"] not in messages, "the deleted version came back after the new turn"
    assert messages[ids["first"]]["childrenIds"] == [ids["picked"]]
    assert (messages[ids["answer"]].get("annotation") or {}).get("rating") == -1
    kept = {key: messages[key]["timestamp"] for key in times if key in messages}
    assert kept == {key: value for key, value in times.items() if key != ids["version"]}, (
        f"the old messages' times changed: {kept} (were {times})"
    )


def test_an_upgraded_reply_edited_rated_continued_and_regenerated_keeps_each_change(upgraded):
    live, feedback = upgraded.manifest["chats"]["live"], upgraded.manifest["feedback"]
    chat_id, rated_id, reply_id = live["id"], live["reply_id"], live["current_id"]
    question_id = live["messages"][reply_id]["parent"]
    tag = _tag()
    edited = f"Eat it with lemon and potato salad. {tag}"
    more = f" Lingonberry jam works too. {tag}"

    with upgraded.client("alice") as client:
        stored = _history(client, chat_id)["messages"]
        assert (stored[rated_id].get("annotation") or {}).get("rating") == feedback["rating"]
        # v0.10.2 kept a streamed reply's text only in its output
        old_reply = _output_text(stored[reply_id]) or stored[reply_id]["content"]
        assert old_reply, f"the reply shows no text before the edit: {stored[reply_id]}"

        def edit_reply(message: dict) -> None:
            if message.get("output"):
                message["output"] = _with_output_text(message["output"], edited)
            message["content"] = edited

        _edit(client, chat_id, reply_id, edit_reply)
        message = _history(client, chat_id)["messages"][reply_id]
        assert message["content"] == edited, f"the edited reply reads {message['content']!r}"
        if message.get("output"):
            assert _output_text(message) == edited, f"the reply shows {_output_text(message)!r}"
        assert old_reply not in json.dumps(message), (
            f"the old text is still in the reply: {message}"
        )

        assert _rate(client, chat_id, rated_id, -1) == feedback["id"]
        rerated = _history(client, chat_id)["messages"]
        assert (rerated[rated_id].get("annotation") or {}).get("rating") == -1, (
            f"the changed rating is gone after a reload: {rerated[rated_id].get('annotation')}"
        )
        stored_feedback = client.get(f"/api/v1/evaluations/feedback/{feedback['id']}")
        assert stored_feedback.status_code == 200, stored_feedback.text
        assert stored_feedback.json()["data"]["rating"] == -1, "the feedback kept the old rating"
        assert rerated[reply_id]["content"] == edited, "rating a reply undid the edit"

        question = rerated[question_id]
        upgraded.provider.queue(reply.text(more, match=reply.answering(question["content"])))
        _continue_reply(client, chat_id, question, reply_id)
        continued = _wait_for(
            client, chat_id, reply_id, lambda message: more.strip() in message["content"]
        )
        continued_request = upgraded.provider.chat_requests()[-1]

        upgraded.provider.queue(
            reply.text(f"Try it with fries. {tag}", match=reply.answering(question["content"]))
        )
        regenerated_id = _regenerate(
            client, chat_id, _history(client, chat_id)["messages"][question_id]
        )
        regenerated = _history(client, chat_id)

        switched = copy.deepcopy(regenerated)
        switched["currentId"] = reply_id
        _save(client, chat_id, switched)
        final = _history(client, chat_id)

    sent = _sent_texts(continued_request)
    assert sent[-1] == edited, f"continuing sent the model {sent[-1]!r} as the reply so far"
    assert continued["content"].startswith(edited), (
        f"the continued reply lost its edited start: {continued['content']!r}"
    )
    assert continued["content"].endswith(more.strip()), (
        f"the continued reply does not end with the continuation: {continued['content']!r}"
    )
    assert old_reply not in json.dumps(continued), (
        "continuing brought back the text before the edit"
    )

    assert regenerated["currentId"] == regenerated_id, "the chat does not show the new version"
    assert regenerated["messages"][question_id]["childrenIds"] == [reply_id, regenerated_id], (
        "the question does not list both versions in order"
    )
    assert final["currentId"] == reply_id, "switching back to the first version was not kept"
    kept = final["messages"][reply_id]
    assert kept["content"] == continued["content"], (
        f"the first version changed after the regenerate: {kept['content']!r}"
    )
    assert old_reply not in json.dumps(kept), "the text before the edit came back"
    assert (final["messages"][rated_id].get("annotation") or {}).get("rating") == -1
