"""Journey: workers with the JSON codec switch off and on share one Redis and one database.

Flipping `ENABLE_ORJSON` across a deployment leaves workers of both values next to each other
for a while. They share the socket.io Redis manager's messages, the task command channel, the
response stream snapshots, the base model list and the tool server list, all of which the
switch writes in its own codec. One instance of each value boots on one real Redis and one
database, and every path runs both ways: written on the switched-off instance and read on the
switched-on one, and the reverse. A chat with mixed-script text streams through one and reaches
a tab signed in on the other, a stop sent to the other ends the reply, the other's read of the
chat shows the reply in flight, the model and tool server lists cached by one are served by the
other, and a note edited live on one reaches a tab on the other and is saved.

Discriminates: passes on dev 176d31d1d. In backend copies of dev 176d31d1d, the orjson codec
writing mojibake or orjson request parsing that mangles non-ASCII turns the crossings with an
orjson side red, the stdlib codec writing mojibake the ones with a stdlib side, and the orjson
writer for Redis task payloads truncating its output turns the stop test (every crossing with an
orjson side) and the in-flight test (an orjson writer) red.
"""

from __future__ import annotations

import dataclasses
import json
import time
import uuid
from typing import Iterator

import pytest

from harness import backends
from harness import upstream as reply
from harness.actors import Actor, create_user
from harness.chat import send_message, wait_for_reply
from harness.inflight import LAST_PIECE, start_slow_reply
from harness.json_codecs import CROSSINGS, MIXED_TEXT, codec_pair
from harness.socket_client import connected, note_text

pytestmark = [
    pytest.mark.journey,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

PIECES = [f"{text}\n" for text in MIXED_TEXT.values()]
ANSWER = "".join(PIECES)
CROSSING_IDS = [f"{writer}-to-{reader}" for writer, reader in CROSSINGS]
# how long a tab keeps listening after the sender's own tab saw the stream end
GRACE_SECONDS = 5.0
TOOL_SERVER_ID = "wetter"
TOOL_SERVER_NAME = f"Wetterdienst {MIXED_TEXT['chinese']} {MIXED_TEXT['emoji']}"
MODEL_IDS = ["Größe-模型", "modèle-été", "mock-model"]


@pytest.fixture(scope="module")
def shared_redis() -> Iterator[str]:
    with backends.redis_server() as url:
        yield url


@pytest.fixture(scope="module")
def pair(shared_redis, instance_with) -> dict:
    """One instance per codec, on one Redis and one database, the base model cache on."""
    extra = {
        "REDIS_URL": shared_redis,
        "WEBSOCKET_MANAGER": "redis",
        "ENABLE_BASE_MODELS_CACHE": "true",
    }
    return codec_pair(instance_with, extra)


def _on(actor: Actor, instance) -> Actor:
    """The same account, addressed to another instance."""
    return dataclasses.replace(actor, base_url=instance.base_url)


def _finished(events: list[dict]) -> bool:
    for event in events:
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        if event.get("type") == "chat:completion" and data.get("done") is True:
            return True
    return False


def _settled(events: list[dict]) -> bool:
    """Whether the chat's last event came: `chat:active` off, sent after the task ends."""
    return any(
        event.get("type") == "chat:active" and event["data"].get("active") is False
        for event in events
    )


def _streamed_text(events: list[dict]) -> str:
    payloads = [event.get("data") for event in events]
    return "".join(
        payload["delta"]
        for payload in payloads
        if isinstance(payload, dict) and isinstance(payload.get("delta"), str)
    )


def _output_text(output: list[dict]) -> str:
    """The text of the output items' `output_text` parts."""
    return "".join(
        part.get("text", "")
        for item in output
        for part in item.get("content") or []
        if part.get("type") == "output_text"
    )


def _content_so_far(client, turn) -> str:
    chat = client.get(f"/api/v1/chats/{turn.chat_id}").json()
    message = chat["chat"]["history"]["messages"].get(turn.assistant_message_id, {})
    return message.get("content") or ""


def _wait_until(condition, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.1)
    return condition()


# ---------------------------------------------------------------- live chat stream


@pytest.mark.parametrize(("writer", "reader"), CROSSINGS, ids=CROSSING_IDS)
def test_a_tab_on_the_other_codec_receives_the_mixed_text_stream(pair, writer, reader):
    sender, receiver = pair[writer], pair[reader]
    account = create_user(sender)
    prompt = f"Was gibt es Neues? ({writer} to {reader})"
    pair["stdlib"].upstream.queue(reply.text(PIECES, match=reply.answering(prompt)))

    with (
        connected(_on(account, receiver), in_order=True) as far_tab,
        connected(account, in_order=True) as near_tab,
        account.client() as client,
    ):
        turn = send_message(client, prompt)
        stored = wait_for_reply(client, turn)
        near_tab.wait_for(turn.chat_id, "chat:completion", done=True)
        tabs = (near_tab, far_tab)
        _wait_until(
            lambda: all(_settled(tab.events_of(turn.chat_id)) for tab in tabs), GRACE_SECONDS
        )

    far_events = far_tab.events_of(turn.chat_id)
    assert far_events, f"a tab on the {reader} instance saw nothing of a chat streamed on {writer}"
    assert _streamed_text(far_events) == ANSWER
    assert _finished(far_events)
    assert far_events == near_tab.events_of(turn.chat_id)
    assert stored["content"].strip() == ANSWER.strip()


# ---------------------------------------------------------------- stopping a reply


@pytest.mark.parametrize(("writer", "reader"), CROSSINGS, ids=CROSSING_IDS)
def test_a_stop_sent_to_the_other_codec_ends_the_reply(pair, writer, reader):
    runner, stopper = pair[writer], pair[reader]
    account = create_user(runner)
    upstream = pair["stdlib"].upstream

    with account.client() as client, _on(account, stopper).client() as other:
        turn = start_slow_reply(client, upstream, chunk_delay=0.3)
        _wait_until(lambda: "part-0" in _content_so_far(client, turn), timeout=10)
        listed_by_runner = client.get(f"/api/tasks/chat/{turn.chat_id}").json()["task_ids"]
        listed_by_stopper = other.get(f"/api/tasks/chat/{turn.chat_id}").json()["task_ids"]
        stopped = other.post(f"/api/tasks/chat/{turn.chat_id}/stop")
        assert stopped.status_code == 200, stopped.text
        content = wait_for_reply(client, turn, timeout=30)["content"]
        drained = _wait_until(
            lambda: (
                client.get(f"/api/tasks/chat/{turn.chat_id}").json()["task_ids"] == []
                and other.get(f"/api/tasks/chat/{turn.chat_id}").json()["task_ids"] == []
            ),
            timeout=10,
        )

    assert len(listed_by_runner) == 1
    assert listed_by_runner == listed_by_stopper, "the two instances list different tasks"
    assert content.startswith("part-0"), content
    assert LAST_PIECE not in content, f"the stop from {reader} never reached the reply on {writer}"
    assert drained, "the task lists of the two instances still show the stopped reply"


# ---------------------------------------------------------------- in-flight snapshot


@pytest.mark.parametrize(("writer", "reader"), CROSSINGS, ids=CROSSING_IDS)
def test_reading_the_chat_on_the_other_codec_shows_the_reply_in_flight(pair, writer, reader):
    runner, viewer = pair[writer], pair[reader]
    account = create_user(runner)
    upstream = pair["stdlib"].upstream
    prompt = f"Erzähl mir mehr ({writer} to {reader})"
    upstream.queue(reply.text(PIECES, chunk_delay=0.4, match=reply.answering(prompt)))

    with account.client() as client, _on(account, viewer).client() as other:
        turn = send_message(client, prompt)
        seen: dict = {}

        def snapshot_arrived() -> bool:
            chat = other.get(f"/api/v1/chats/{turn.chat_id}").json()
            message = chat["chat"]["history"]["messages"].get(turn.assistant_message_id, {})
            if not message.get("done") and len(message.get("content", "")) > len(PIECES[0]):
                seen.update(message)
                return True
            return False

        arrived = _wait_until(snapshot_arrived, timeout=30)
        stored = wait_for_reply(client, turn)

    assert arrived, f"the {reader} instance never showed the reply in flight"
    content = seen["content"]
    assert ANSWER.startswith(content), (
        f"the in-flight content is not a prefix of the reply: {content!r}"
    )
    assert seen["done"] is False
    assert seen["output"], "the in-flight response has no output"
    assert _output_text(seen["output"]) == content
    assert stored["content"].strip() == ANSWER.strip()


# ---------------------------------------------------------------- shared caches


def _listed_models(instance) -> list[dict]:
    """The models a fresh admin of `instance` is offered, of the ones this test serves."""
    with create_user(instance, role="admin").client() as client:
        listed = client.get("/api/models")
    assert listed.status_code == 200, listed.text
    return sorted(
        (
            {"id": model["id"], "name": model["name"]}
            for model in listed.json()["data"]
            if model["id"] in MODEL_IDS
        ),
        key=lambda model: model["id"],
    )


@pytest.mark.parametrize(("writer", "reader"), CROSSINGS, ids=CROSSING_IDS)
def test_the_base_model_list_one_codec_cached_is_served_by_the_other(pair, writer, reader):
    upstream = pair["stdlib"].upstream
    upstream.models = MODEL_IDS
    with create_user(pair[writer], role="admin").client() as client:
        refreshed = client.get("/api/models", params={"refresh": "true"})
    assert refreshed.status_code == 200, refreshed.text
    written = _listed_models(pair[writer])
    # a rebuilt list would show this one; the cached one does not
    upstream.models = ["mock-model"]

    read = _listed_models(pair[reader])

    assert [model["id"] for model in written] == sorted(MODEL_IDS)
    assert read == written, f"the {reader} instance did not serve the list the {writer} one cached"


def _openapi_spec() -> dict:
    operation = {"operationId": "wetter", "responses": {"200": {"description": "ok"}}}
    return {
        "openapi": "3.0.0",
        "info": {"title": "Wetter", "version": "1"},
        "paths": {"/wetter": {"get": operation}},
    }


def _tool_server_listing(instance) -> dict:
    with create_user(instance, role="admin").client() as client:
        listed = client.get("/api/v1/tools/")
    assert listed.status_code == 200, listed.text
    return next(tool for tool in listed.json() if tool["id"] == f"server:{TOOL_SERVER_ID}")


@pytest.mark.parametrize(("writer", "reader"), CROSSINGS, ids=CROSSING_IDS)
def test_the_tool_server_list_one_codec_cached_is_served_by_the_other(pair, writer, reader):
    connection = {
        "url": "http://127.0.0.1:9",
        "path": "openapi.json",
        "spec_type": "json",
        "spec": json.dumps(_openapi_spec()),
        "auth_type": "none",
        "key": "",
        "config": {"enable": True},
        "info": {
            "id": TOOL_SERVER_ID,
            "name": TOOL_SERVER_NAME,
            "description": MIXED_TEXT["markdown"],
        },
    }
    path = "/api/v1/configs/tool_servers"
    with create_user(pair[writer], role="admin").client() as client:
        saved = client.post(path, json={"TOOL_SERVER_CONNECTIONS": [connection]})
        assert saved.status_code == 200, saved.text
    try:
        written = _tool_server_listing(pair[writer])
        read = _tool_server_listing(pair[reader])
    finally:
        with create_user(pair[writer], role="admin").client() as client:
            client.post(path, json={"TOOL_SERVER_CONNECTIONS": []}).raise_for_status()

    assert written["name"] == TOOL_SERVER_NAME
    assert read["name"] == TOOL_SERVER_NAME
    assert read["meta"]["description"] == written["meta"]["description"] == MIXED_TEXT["markdown"]


# ---------------------------------------------------------------- live note edit


def _create_note(owner: Actor) -> str:
    with owner.client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={"title": f"live {uuid.uuid4().hex[:8]}", "data": {"content": {"md": ""}}},
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _saved_markdown(owner: Actor, note_id: str, expected: str) -> str | None:
    """The note's stored markdown once it reads `expected`, or as it is when the wait ends."""
    deadline = time.monotonic() + 10.0
    with owner.client() as client:
        while True:
            note = client.get(f"/api/v1/notes/{note_id}")
            assert note.status_code == 200, note.text
            markdown = ((note.json().get("data") or {}).get("content") or {}).get("md")
            if markdown == expected or time.monotonic() > deadline:
                return markdown
            time.sleep(0.2)


@pytest.mark.parametrize(("writer", "reader"), CROSSINGS, ids=CROSSING_IDS)
def test_a_note_edited_on_one_codec_reaches_a_tab_on_the_other(pair, writer, reader):
    owner = create_user(pair[writer])
    note_id = _create_note(owner)
    text = f"{MIXED_TEXT['chinese']} {MIXED_TEXT['emoji']} {MIXED_TEXT['pdf_paste']}"

    with connected(owner) as editing, connected(_on(owner, pair[reader])) as watching:
        editing.join_note(note_id)
        watching.join_note(note_id)
        editing.edit_note(note_id, text)
        passed_on = watching.note_update(note_id)
        saved = _saved_markdown(_on(owner, pair[reader]), note_id, text)

    assert note_text(passed_on["update"]) == f"<paragraph>{text}</paragraph>"
    assert saved == text
