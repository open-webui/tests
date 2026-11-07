"""Regression: a timer that failed before the model answered left a blank, unfinished reply.

Fix `b0650d04b` (open-webui/open-webui#31483, issue open-webui/open-webui#31481): when a timer
fired, it wrote its prompt and an empty assistant reply into the chat and then ran the
completion. If the completion failed before the model started, for example because the chat's
model had been deleted, only the timer's own state recorded the error; the reply stayed empty
and unfinished, which the chat shows as a reply still loading. The reply now carries the error
and is marked done.

The model sets a real timer through the `timer` tool (behind ENABLE_SUBAGENTS). The chat runs on
a preset that is deleted before the timer fires.

Also pins the older `f5a5a434b` (#27785, issue #27783), which caught the completion's error in
the scheduler at all. Twin of unit/models/test_automations_and_calendar.py.

`test_a_timer_on_a_working_model_still_gets_its_answer` is red on dev 62f70a844: since de73bb830 a
chat request whose reply message is already stored in the chat, the way automations, sub-agents and
timers prepare their reply, is refused with 409 and the reply is never written
(open-webui/open-webui#32066, fixed by 641c82df1).

Discriminates: passes on dev efe63bd34; with b0650d04b reverted in a backend copy the narrow test
fails (the timer's reply stays empty and not done). On dev ef67cc3fa, with the try/except around
the timer's completion removed, the error escapes the scheduler task and the narrow test fails
the same way. The nearby test passes on both.

`test_a_timer_run_has_no_client_address_for_the_ip_placeholders`: a fired timer filled the client
IP header placeholders (#33833) with 127.0.0.1, the address of the request timers build for
themselves (issue open-webui/open-webui#33882). Red on dev 1fc5b8f6d (sends 127.0.0.1), green with
fix 2ea13e650 (branch fix/user-ip-empty-for-automations).
"""

from __future__ import annotations

import time
import uuid

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.second_provider import OPENAI_CONFIG
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

SUBAGENTS = ("/api/v1/configs/subagents", "/api/v1/configs/subagents")
TIMER_PROMPT = "The tea has steeped."
# due after three seconds and polled every second
FIRE_WAIT = 30.0


@pytest.fixture
def timers_enabled(preserve, admin):
    preserve(SUBAGENTS)
    with admin.client() as client:
        current = client.get(SUBAGENTS[0]).json()
        client.post(SUBAGENTS[1], json={**current, "ENABLE_SUBAGENTS": True}).raise_for_status()


def _create_preset(client) -> str:
    model_id = f"timer-{uuid.uuid4().hex[:8]}"
    created = client.post(
        "/api/v1/models/create",
        json={
            "id": model_id,
            "base_model_id": MOCK_MODEL_ID,
            "name": "Timer model",
            "meta": {},
            "params": {},
        },
    )
    assert created.status_code == 200, created.text
    client.get("/api/models", params={"refresh": "true"}).raise_for_status()
    return model_id


def _set_timer(client, upstream, model: str) -> str:
    arguments = {"prompt": TIMER_PROMPT, "at": "3s"}
    upstream.queue(reply.tool_call("timer", arguments), reply.text("Timer set."))
    turn, message = ask(client, "tell me when the tea is ready", model=model)
    assert message["content"].endswith("Timer set."), message
    return turn.chat_id


def _timer_reply(client, chat_id: str) -> dict:
    """The reply to the timer's prompt once it is done or has an error, else as the wait ends."""
    deadline = time.monotonic() + FIRE_WAIT
    reply_message: dict = {}
    while time.monotonic() < deadline:
        messages = client.get(f"/api/v1/chats/{chat_id}").json()["chat"]["history"]["messages"]
        prompts = [
            message
            for message in messages.values()
            if message.get("role") == "user" and message.get("content") == TIMER_PROMPT
        ]
        if prompts and prompts[0].get("childrenIds"):
            reply_message = messages.get(prompts[0]["childrenIds"][0], {})
            if reply_message.get("done") or reply_message.get("error"):
                return reply_message
        time.sleep(0.5)
    assert reply_message, "the timer never fired"
    return reply_message


def test_a_timer_whose_model_was_deleted_shows_the_error(timers_enabled, make_user, upstream):
    owner = make_user(role="admin")
    with owner.client() as client:
        model_id = _create_preset(client)
        chat_id = _set_timer(client, upstream, model_id)
        deleted = client.post("/api/v1/models/model/delete", json={"id": model_id})
        assert deleted.status_code == 200, deleted.text
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()

        timer_reply = _timer_reply(client, chat_id)

    assert timer_reply.get("done") is True and timer_reply.get("error"), (
        "the timer failed before the model answered and left an empty, unfinished reply that "
        f"looks stuck: {timer_reply} (#31481)"
    )
    assert timer_reply["error"].get("content")


def test_a_timer_on_a_working_model_still_gets_its_answer(timers_enabled, make_user, upstream):
    owner = make_user(role="admin")
    upstream.queue(reply.text("Pour it now.", match=reply.answering(TIMER_PROMPT)))
    with owner.client() as client:
        chat_id = _set_timer(client, upstream, MOCK_MODEL_ID)
        timer_reply = _timer_reply(client, chat_id)

    assert timer_reply.get("error") is None, timer_reply
    assert timer_reply["content"] == "Pour it now."


def _sent_ip(entry) -> str:
    """What the provider received for the X-Real-IP header, empty when it was not sent."""
    return {name.lower(): value for name, value in entry.headers.items()}.get("x-real-ip", "")


def test_a_timer_run_has_no_client_address_for_the_ip_placeholders(
    timers_enabled, admin, preserve, make_user, upstream
):
    preserve(OPENAI_CONFIG)
    with admin.client() as client:
        config = client.get(OPENAI_CONFIG[0]).json()
        connection = config["OPENAI_API_CONFIGS"].get("0", {})
        config["OPENAI_API_CONFIGS"] = {
            "0": {**connection, "headers": {"X-Real-IP": "{{USER_IP}}"}}
        }
        client.post(OPENAI_CONFIG[1], json=config).raise_for_status()
    upstream.queue(reply.text("Pour it now.", match=reply.answering(TIMER_PROMPT)))
    owner = make_user(role="admin")
    with owner.client() as client:
        chat_id = _set_timer(client, upstream, MOCK_MODEL_ID)
        timer_reply = _timer_reply(client, chat_id)
    assert timer_reply["content"] == "Pour it now.", timer_reply

    requests = [entry for entry in upstream.requests_to("/chat/completions") if entry.body]
    fired = [_sent_ip(entry) for entry in requests if reply.answering(TIMER_PROMPT)(entry.body)]
    chatted = [
        _sent_ip(entry) for entry in requests if not reply.answering(TIMER_PROMPT)(entry.body)
    ]
    assert chatted and all(chatted), (
        f"the chat that set the timer sent no client address: {chatted}"
    )
    # a timer fires with no client, so no address may stand in for one (#33882)
    assert fired == [""], f"a timer run sent {{{{USER_IP}}}} as {fired}"
