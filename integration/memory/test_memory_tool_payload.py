"""Memories written from a chat record the model id, and memory replies carry no stored meta.

0.11.4 `e9a0164690`: `/memories/update` stamped a new memory's meta with the chat metadata's
`model`, which on the chat path is the whole resolved model entry (name, info, params, access
control), and both that route and `/memories/path` returned every memory with its stored
`meta`. The memory tools hand those replies to the model. The fix records `model['id']` and
serializes memory rows without `meta`.

Twin of unit/memory/test_memory_tool_payload.py.

Discriminates: passes on dev bbfa876af; reverting `e9a0164690` fails the three narrow tests (the
stored meta holds the whole model entry and both replies carry `meta`).
"""

from __future__ import annotations

import json

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

PREFERENCE = {"action": "add", "content": "prefers tea over coffee", "path": "work/preferences"}


@pytest.fixture
def account(make_user):
    return make_user()


def _stored_memories(client) -> list[dict]:
    listed = client.get("/api/v1/memories/")
    assert listed.status_code == 200, listed.text
    return listed.json()


def test_a_memory_written_by_the_model_records_only_the_model_id(account, upstream):
    upstream.queue(
        reply.tool_call("update_memory", {"operations": [PREFERENCE]}), reply.text("noted")
    )
    with account.client() as client:
        turn, _ = ask(client, "remember that I prefer tea", features={"memory": True})
        [memory] = _stored_memories(client)

    assert memory["meta"]["model"] == MOCK_MODEL_ID, (
        f"the memory stored the whole model entry instead of its id: {memory['meta']['model']!r}"
    )
    assert memory["meta"]["created_by"] == "tool"
    assert memory["meta"]["chat_id"] == turn.chat_id

    tool_result = next(
        entry for entry in upstream.chat_requests()[-1]["messages"] if entry["role"] == "tool"
    )
    [written] = json.loads(tool_result["content"])
    assert "meta" not in written["memory"], "the tool reply handed the model the stored meta"


def test_the_update_route_returns_memories_without_their_meta(account):
    with account.client() as client:
        updated = client.post("/api/v1/memories/update", json={"operations": [PREFERENCE]})
    assert updated.status_code == 200, updated.text

    [result] = updated.json()
    assert result["status"] == "created"
    assert "meta" not in result["memory"], "the update reply carried the memory's stored meta"
    assert result["memory"]["content"] == "prefers tea over coffee"
    assert result["memory"]["path"] == "work/preferences"


def test_the_path_listing_returns_memories_without_their_meta(account):
    with account.client() as client:
        client.post("/api/v1/memories/update", json={"operations": [PREFERENCE]})
        listed = client.post("/api/v1/memories/path", json={"path": "work/preferences"})
    assert listed.status_code == 200, listed.text

    [memory] = listed.json()["memories"]
    assert "meta" not in memory, "the path listing carried the memory's stored meta"
    assert memory["content"] == "prefers tea over coffee"


@pytest.mark.parametrize("source", [None, "background_review"])
def test_a_memory_written_outside_a_chat_keeps_its_source_and_no_model(account, source):
    form = {"operations": [PREFERENCE], **({"source": source} if source else {})}
    with account.client() as client:
        client.post("/api/v1/memories/update", json=form).raise_for_status()
        [memory] = _stored_memories(client)

    assert memory["meta"]["created_by"] == (source or "tool")
    assert memory["meta"]["model"] is None
