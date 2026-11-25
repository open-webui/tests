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

Share links made on the old release keep working: the link shared with one account shows it the
conversation as it was when shared, on its branch and with its tool call, and so does a visitor
who is not signed in once the owner opens the link to anyone. After the owner adds a turn and
updates the link, it shows the whole current conversation. The v0.13.4 share that lets the
grantee reply opens the live chat with its tool output, reasoning and usage, still does after an
update, and the grantee's turn is sent with the conversation before it and shows for the owner.
A link made after the upgrade for an old chat answered from a knowledge base keeps its citation.

Discriminates: passes on dev f80a4f782 for all six data sets. In a backend copy whose message
save keeps the old message copy the migration stored instead of replacing it, both edit tests
fail on every data set: the new rating is gone after a reload, and the changed rating of the
rated reply reads its old value again. The share tests pass on dev 893584700 for all eight data
sets; with "Update link" keeping the old snapshot the update test fails, with a reply-allowed
share always served as a snapshot the continue test fails, and with sources dropped from a shared
chat the citation test fails.
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
from harness.access import grant, make_group
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


# --- share links made before the upgrade ------------------------------------------------------


def _open_share(upgraded: Upgraded, who: str | None, share_id: str) -> httpx.Response:
    """The share link opened as `who`, or signed out when `who` is None."""
    path = f"/api/v1/chats/share/{share_id}"
    if who is None:
        return httpx.get(f"{upgraded.backend.base_url}{path}", timeout=60.0)
    return upgraded.get(who, path)


def _shown_share(upgraded: Upgraded, who: str | None, share_id: str) -> dict:
    opened = _open_share(upgraded, who, share_id)
    assert opened.status_code == 200, f"{who or 'a signed-out visitor'}: {opened.text}"
    return opened.json()


def _shape(history: dict) -> dict:
    return {
        message_id: {
            "parent": message.get("parentId"),
            "children": message.get("childrenIds", []),
            "role": message["role"],
            "content": message["content"],
        }
        for message_id, message in history["messages"].items()
    }


def _refused(response: httpx.Response) -> bool:
    return response.status_code in (401, 403, 404)


def _save_share_access(upgraded: Upgraded, chat_id: str, grants: list[dict], mode) -> None:
    """Save the share dialog: who may open the link and whether they may reply."""
    with upgraded.client("alice") as client:
        saved = client.post(
            f"/api/v1/chats/shared/{chat_id}/access/update",
            json={"access_grants": grants, "share_mode": mode},
        )
    assert saved.status_code == 200, f"saving the share's access failed: {saved.text}"


def _update_link(upgraded: Upgraded, chat_id: str, share_id: str, mode) -> None:
    """Click "Update link" in the share dialog, which sends the mode the share has."""
    with upgraded.client("alice") as client:
        updated = client.post(f"/api/v1/chats/{chat_id}/share", json={"share_mode": mode})
    assert updated.status_code == 200, f"updating the link failed: {updated.text}"
    assert updated.json()["share_id"] == share_id, "updating the link made a new link"


def _add_turn(upgraded: Upgraded, who: str, chat_id: str, parent_id: str) -> tuple[str, str]:
    """`who` asks one more question after `parent_id`; returns the question and the reply."""
    tag = _tag()
    question, answer = f"One more stop? {tag}", f"The Prater. {tag}"
    upgraded.provider.queue(reply.text(answer, match=reply.answering(question)))
    with upgraded.client(who) as client:
        turn = send_message(client, question, chat_id=chat_id, parent_id=parent_id)
        stored = wait_for_reply(client, turn, timeout=90)
    assert stored["content"] == answer, f"the new turn was answered with {stored['content']!r}"
    return question, answer


def _owner_history(upgraded: Upgraded, chat_id: str) -> dict:
    with upgraded.client("alice") as client:
        return _history(client, chat_id)


def test_an_old_share_link_shows_the_chat_as_shared_and_updating_it_shows_the_whole_chat(upgraded):
    branched = upgraded.manifest["chats"]["branched"]
    chat_id, share_id, ids = branched["id"], branched["share_id"], _ids(branched)
    accounts = upgraded.manifest["accounts"]

    for who in ("carol", None):
        assert _refused(_open_share(upgraded, who, share_id)), (
            f"{who or 'a signed-out visitor'} opens a link shared with bob only"
        )
    make_group(
        upgraded.actor("admin"), [upgraded.actor("alice")], {"sharing": {"open_chats": True}}
    )
    _save_share_access(
        upgraded,
        chat_id,
        [grant("user", accounts["bob"]["id"], "read"), grant("anyone", "*", "read")],
        None,
    )

    for who in ("bob", None):
        shown = _shown_share(upgraded, who, share_id)
        viewer = who or "a signed-out visitor"
        assert shown["title"] == branched["title"], f"{viewer} sees the title {shown['title']!r}"
        assert shown["write_access"] is False, f"{viewer} may write to a clone only share"
        history = shown["chat"]["history"]
        assert _shape(history) == branched["messages"], (
            f"{viewer} does not see the conversation as it was shared: {_shape(history)}"
        )
        assert history["currentId"] == branched["current_id"], (
            f"{viewer} sees the branch ending at {history['currentId']}"
        )
        path = [message["id"] for message in _path(history)]
        assert path == [ids["first"], ids["picked"], ids["question"], ids["answer"]], (
            f"{viewer} sees the messages {path} on the shown branch"
        )
        tool_reply = history["messages"][ids["picked"]]["content"]
        assert "get_weather" in tool_reply and "Sunny in Vienna" in tool_reply, (
            f"{viewer} sees the tool reply without its call and output: {tool_reply!r}"
        )

    current_id = _owner_history(upgraded, chat_id)["currentId"]
    question, answer = _add_turn(upgraded, "alice", chat_id, current_id)
    frozen = _shown_share(upgraded, None, share_id)["chat"]["history"]
    assert _shape(frozen) == branched["messages"], "the link changed before it was updated"

    _update_link(upgraded, chat_id, share_id, None)
    owner_sees = _owner_history(upgraded, chat_id)
    for who in ("bob", None):
        viewer = who or "a signed-out visitor"
        history = _shown_share(upgraded, who, share_id)["chat"]["history"]
        assert _shape(history) == _shape(owner_sees), (
            f"after the update {viewer} sees {_shape(history)}, the owner {_shape(owner_sees)}"
        )
        assert history["currentId"] == owner_sees["currentId"]
        shown = [(message["role"], message["content"]) for message in _path(history)]
        assert shown[-2:] == [("user", question), ("assistant", answer)], (
            f"after the update {viewer} sees the branch {shown}"
        )


def test_an_old_continue_share_shows_the_live_chat_and_lets_the_grantee_continue(upgraded):
    if "tooled" not in upgraded.manifest["chats"]:
        pytest.skip("releases before v0.13 had no share others may continue")
    tooled = upgraded.manifest["chats"]["tooled"]
    chat_id, share_id, reply_id = tooled["id"], tooled["share_id"], tooled["reply_id"]
    stored_reply = tooled["stored"][reply_id]

    for who in ("carol", None):
        assert _refused(_open_share(upgraded, who, share_id)), (
            f"{who or 'a signed-out visitor'} opens a link shared with bob only"
        )
    shown = _shown_share(upgraded, "bob", share_id)
    assert (shown["id"], shown["write_access"], shown["chat"]["share_mode"]) == (
        chat_id,
        True,
        "continue",
    ), "bob does not get the chat to continue"
    history = shown["chat"]["history"]
    assert _shape(history) == tooled["messages"], f"bob sees {_shape(history)}"
    assert history["currentId"] == tooled["current_id"]
    shown_reply = history["messages"][reply_id]
    for key in ("output", "usage", "model"):
        assert shown_reply.get(key) == stored_reply.get(key), (
            f"bob sees the reply's {key} as {shown_reply.get(key)!r}"
        )
    written = json.dumps(shown_reply["output"])
    assert "Sunny in Graz" in written and tooled["reasoning"] in written

    question, answer = _add_turn(upgraded, "alice", chat_id, tooled["current_id"])
    _update_link(upgraded, chat_id, share_id, "continue")
    owner_sees = _owner_history(upgraded, chat_id)
    shown = _shown_share(upgraded, "bob", share_id)
    assert (shown["write_access"], shown["chat"]["share_mode"]) == (True, "continue"), (
        "updating the link took away bob's replies"
    )
    history = shown["chat"]["history"]
    assert _shape(history) == _shape(owner_sees), f"after the update bob sees {_shape(history)}"
    shown_path = [(message["role"], message["content"]) for message in _path(history)]
    assert shown_path[-2:] == [("user", question), ("assistant", answer)], (
        f"after the update bob sees the branch {shown_path}"
    )

    bobs_question, bobs_answer = _add_turn(upgraded, "bob", chat_id, owner_sees["currentId"])
    request = next(filter(reply.answering(bobs_question), upgraded.provider.chat_requests()))
    sent = [text for text in _sent_texts(request) if text]
    earlier = [tooled["messages"][key]["content"] for key in tooled["stored"]]
    assert sent == [*earlier, question, answer, bobs_question], f"bob's turn sent the model {sent}"
    assert "Sunny in Graz" in json.dumps(request["messages"]), (
        "bob's turn did not send the model the earlier tool output"
    )
    owner_sees = _owner_history(upgraded, chat_id)
    owner_path = _path(owner_sees)
    assert [(message["role"], message["content"]) for message in owner_path[-2:]] == [
        ("user", bobs_question),
        ("assistant", bobs_answer),
    ], "alice does not see bob's turn"
    bob = upgraded.manifest["accounts"]["bob"]
    assert owner_path[-2]["user_id"] == bob["id"], "bob's question is not shown as his"
    assert owner_path[-2]["user"]["name"] == bob["name"]
    seen_by_bob = _shown_share(upgraded, "bob", share_id)["chat"]["history"]
    assert _shape(seen_by_bob) == _shape(owner_sees)


def test_a_link_made_after_the_upgrade_for_an_old_cited_chat_keeps_its_sources(upgraded):
    cited = upgraded.manifest["chats"]["cited"]
    accounts = upgraded.manifest["accounts"]
    with upgraded.client("alice") as client:
        shared = client.post(f"/api/v1/chats/{cited['id']}/share", json={"share_mode": None})
        assert shared.status_code == 200, shared.text
    share_id = shared.json()["share_id"]
    _save_share_access(upgraded, cited["id"], [grant("user", accounts["bob"]["id"], "read")], None)

    history = _shown_share(upgraded, "bob", share_id)["chat"]["history"]
    assert _shape(history) == cited["messages"], f"bob sees {_shape(history)}"
    sources = history["messages"][cited["current_id"]].get("sources") or []
    names = {source.get("source", {}).get("name") for source in sources}
    assert set(cited["cited"]) <= names, f"bob sees the reply citing {names}"
