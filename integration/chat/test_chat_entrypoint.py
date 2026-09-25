"""Journey: what the chat completion request accepts, the way the web client sends a message.

The request names the chat to continue, the models to answer and the user's own tool servers, so
the server decides what each part may do. Continuing another user's chat is refused and leaves
that chat as it was, one message sent to two models stores a reply from each, and tool servers
in the request reach the model only for an account with `features.direct_tool_servers`.

Discriminates: in a backend copy, dropping the ownership check on an existing chat in
`chat_completion` fails the foreign-chat test (the stranger's message is stored in the chat),
fanning out to the first model only fails the two-model test, and keeping `tool_servers` for
every caller fails the barred tool server test.
"""

from __future__ import annotations

import uuid
from typing import Iterator

import pytest

from harness import upstream as reply
from harness.chat import ChatTurn, ask, send_message, wait_for_reply
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

SECOND_MODEL_ID = "mock-model-second"
TOOL_SERVER = {
    "url": "http://127.0.0.1:9",
    "specs": [
        {
            "name": "lookup_weather",
            "description": "The weather in a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ],
}


def _answering_model(model_id: str):
    return lambda body: body.get("model") == model_id


def _offered_tools(upstream) -> set[str]:
    chat = next(request for request in upstream.chat_requests() if request.get("stream"))
    return {tool["function"]["name"] for tool in chat.get("tools") or []}


@pytest.fixture
def second_model(admin, upstream) -> Iterator[str]:
    """One more provider model every account may use, as the admin's visibility toggle makes it."""
    upstream.models = [MOCK_MODEL_ID, SECOND_MODEL_ID]
    grant = {"principal_type": "user", "principal_id": "*", "permission": "read"}
    with admin.client() as client:
        client.get("/api/models").raise_for_status()
        published = client.post(
            "/api/v1/models/model/access/update",
            json={"id": SECOND_MODEL_ID, "name": SECOND_MODEL_ID, "access_grants": [grant]},
        )
        assert published.status_code == 200, published.text
        yield SECOND_MODEL_ID
        client.delete(f"/api/v1/models/model/delete?id={SECOND_MODEL_ID}")
    upstream.reset()
    with admin.client() as client:
        client.get("/api/models")


@pytest.fixture
def tool_server_group(admin) -> Iterator[str]:
    with admin.client() as client:
        created = client.post(
            "/api/v1/groups/create",
            json={
                "name": f"tool servers {uuid.uuid4().hex[:8]}",
                "description": "may bring their own tool servers",
                "permissions": {"features": {"direct_tool_servers": True}},
            },
        )
        assert created.status_code == 200, created.text
        yield created.json()["id"]
        client.delete(f"/api/v1/groups/id/{created.json()['id']}/delete")


def test_continuing_another_users_chat_is_refused_and_changes_nothing(make_user, upstream):
    owner, stranger = make_user(), make_user()
    upstream.queue(reply.text("the owner's answer"))
    with owner.client() as client:
        turn, _ = ask(client, "the owner's question")
        before = client.get(f"/api/v1/chats/{turn.chat_id}").json()
    upstream.reset()

    with stranger.client() as client, pytest.raises(AssertionError, match="HTTP 404"):
        send_message(
            client,
            "a stranger's message",
            chat_id=turn.chat_id,
            parent_id=turn.assistant_message_id,
        )
    with owner.client() as client:
        after = client.get(f"/api/v1/chats/{turn.chat_id}").json()

    assert after["chat"]["history"] == before["chat"]["history"]
    assert after["updated_at"] == before["updated_at"]
    assert upstream.chat_requests() == [], "the stranger's message reached the model"


def test_one_message_to_two_models_stores_both_replies(make_user, upstream, second_model):
    account = make_user()
    upstream.queue(
        reply.text("first model's answer", match=_answering_model(MOCK_MODEL_ID)),
        reply.text("second model's answer", match=_answering_model(second_model)),
    )
    first_id, second_id = str(uuid.uuid4()), str(uuid.uuid4())
    message_ids = [
        {"model_id": MOCK_MODEL_ID, "message_id": first_id, "modelIdx": 0},
        {"model_id": second_model, "message_id": second_id, "modelIdx": 1},
    ]

    with account.client() as client:
        # the web client lists the models before it sends
        client.get("/api/models").raise_for_status()
        turn = send_message(client, "compare yourselves", message_ids=message_ids)
        first = wait_for_reply(client, ChatTurn(turn.chat_id, turn.user_message_id, first_id))
        second = wait_for_reply(client, ChatTurn(turn.chat_id, turn.user_message_id, second_id))
        stored = client.get(f"/api/v1/chats/{turn.chat_id}").json()["chat"]

    assert (first["model"], first["content"]) == (MOCK_MODEL_ID, "first model's answer")
    assert (second["model"], second["content"]) == (second_model, "second model's answer")
    user_message = stored["history"]["messages"][turn.user_message_id]
    assert set(user_message["childrenIds"]) >= {first_id, second_id}


def test_tool_servers_in_the_request_are_dropped_without_the_permission(make_user, upstream):
    account = make_user()

    with account.client() as client:
        ask(client, "what is the weather in Oslo?", tool_servers=[TOOL_SERVER])

    assert "lookup_weather" not in _offered_tools(upstream), (
        "an account without features.direct_tool_servers had its tool server offered to the model"
    )


def test_tool_servers_in_the_request_reach_the_model_with_the_permission(
    admin, make_user, upstream, tool_server_group
):
    account = make_user()
    with admin.client() as client:
        added = client.post(
            f"/api/v1/groups/id/{tool_server_group}/users/add", json={"user_ids": [account.id]}
        )
    assert added.status_code == 200, added.text

    with account.client() as client:
        ask(client, "what is the weather in Oslo?", tool_servers=[TOOL_SERVER])

    assert "lookup_weather" in _offered_tools(upstream)
