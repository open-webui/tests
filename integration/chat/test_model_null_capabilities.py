"""Regression: a model whose `meta.capabilities` is null broke chats, automations and channels.

open-webui 0.10.2, issue #26412: `meta.get('capabilities', {})` returns None when the key is
present with a null value, and the next `.get` raised `'NoneType' object has no attribute 'get'`.
Fix `0016266c0` guarded `model_allows_memory`, which every chat with the memory feature on reaches
(the web client sends it by default), and `650b81792` guarded the default feature resolver
automations run through (now `_resolve_model_defaults`, which channel replies share). Both became
`meta.get('capabilities') or {}`.

Twin of unit/chat/test_model_null_capabilities.py.

Discriminates: passes on bbfa876af; fails with 0016266c0 reverted (the memory chat gets no reply)
and with 650b81792 reverted (the automation run records an error and the channel mention gets no
reply); the nearby tests pass on both.
"""

from __future__ import annotations

import time
import uuid

import pytest

from harness import upstream as reply
from harness.chat import ask, send_message, wait_for_reply
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


def _create_model(admin, capabilities: dict | None) -> str:
    model_id = f"capabilities-{uuid.uuid4().hex[:8]}"
    form = {
        "id": model_id,
        "base_model_id": MOCK_MODEL_ID,
        "name": "Report writer",
        "meta": {"capabilities": capabilities, "defaultFeatureIds": ["web_search"]},
        "params": {},
    }
    with admin.client() as client:
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
    assert created.json()["meta"]["capabilities"] == capabilities
    return model_id


@pytest.fixture
def model_id(admin):
    """An admin's preset on the scripted model with `capabilities: null`."""
    created = _create_model(admin, None)
    yield created
    with admin.client() as client:
        client.post("/api/v1/models/model/delete", json={"id": created})


def _poll(read, timeout: float = 30.0):
    """The first truthy value `read()` returns within the timeout, else None."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = read()
        if value:
            return value
        time.sleep(0.2)
    return None


def _finished_reply(client, chat_id: str) -> str | None:
    messages = client.get(f"/api/v1/chats/{chat_id}").json()["chat"]["history"]["messages"]
    finished = [
        entry["content"]
        for entry in messages.values()
        if entry["role"] == "assistant" and entry.get("done")
    ]
    return finished[0] if finished else None


# narrow (0016266c0): a chat with the memory feature on


def test_a_chat_with_memory_on_gets_its_reply(admin, upstream, model_id):
    upstream.queue(reply.text("remembered nothing"))

    with admin.client() as client:
        turn = send_message(client, "hello", model=model_id, features={"memory": True})
        message = wait_for_reply(client, turn, timeout=20)

    assert message["content"] == "remembered nothing", (
        f"a chat with a null-capabilities model and memory on failed (#26412): {message}"
    )


# narrow (650b81792): an automation on the model


def test_an_automation_on_the_model_runs(admin, upstream, model_id):
    upstream.queue(reply.text("automated answer"))
    form = {
        "name": "Daily report",
        "data": {"prompt": "write the report", "model_id": model_id, "rrule": "FREQ=DAILY"},
        "is_active": False,
    }
    with admin.client() as client:
        automation = client.post("/api/v1/automations/create", json=form)
        assert automation.status_code == 200, automation.text
        automation_id = automation.json()["id"]
        assert client.post(f"/api/v1/automations/{automation_id}/run").status_code == 200
        runs = _poll(lambda: client.get(f"/api/v1/automations/{automation_id}/runs").json())
        client.delete(f"/api/v1/automations/{automation_id}/delete")

    assert runs, "the automation never recorded a run"
    assert runs[0]["status"] == "success", (
        f"the automation on a null-capabilities model failed (#26412): {runs[0]['error']}"
    )
    with admin.client() as client:
        answer = _poll(lambda: _finished_reply(client, runs[0]["chat_id"]))
    assert answer == "automated answer"


# broad: the channel reply shares the automation's resolver


def test_a_channel_mention_of_the_model_gets_a_reply(admin, upstream, preserve, model_id):
    preserve("admin_config")
    with admin.client() as client:
        config = client.get("/api/v1/auths/admin/config").json()
        saved = client.post("/api/v1/auths/admin/config", json={**config, "ENABLE_CHANNELS": True})
        assert saved.status_code == 200, saved.text
        channel = client.post("/api/v1/channels/create", json={"name": "status"})
        assert channel.status_code == 200, channel.text
        channel_id = channel.json()["id"]
        posted = client.post(
            f"/api/v1/channels/{channel_id}/messages/post",
            json={"content": f"<@M:{model_id}|Report writer> ping"},
        )
        assert posted.status_code == 200, posted.text
        answered = _poll(lambda: upstream.chat_requests(), timeout=20)
        client.delete(f"/api/v1/channels/{channel_id}/delete")

    assert answered, "a channel mention of a null-capabilities model got no reply (#26412)"
    assert "ping" in str(answered[-1]["messages"])


# nearby: the same model without memory and a model with real capabilities


def test_a_chat_without_memory_gets_its_reply(admin, upstream, model_id):
    upstream.queue(reply.text("plain answer"))

    with admin.client() as client:
        _, message = ask(client, "hello", model=model_id)

    assert message["content"] == "plain answer"


def test_a_model_with_capabilities_chats_with_memory_on(admin, upstream):
    capable_id = _create_model(admin, {"web_search": True, "memory": True})
    upstream.queue(reply.text("capable answer"))
    try:
        with admin.client() as client:
            turn = send_message(client, "hello", model=capable_id, features={"memory": True})
            message = wait_for_reply(client, turn, timeout=20)
    finally:
        with admin.client() as client:
            client.post("/api/v1/models/model/delete", json={"id": capable_id})

    assert message["content"] == "capable answer"
