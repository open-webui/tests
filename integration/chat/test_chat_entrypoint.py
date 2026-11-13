"""Journey: what the chat completion request accepts, the way the web client sends a message.

The request names the chat to continue and the models to answer, so the server decides what
each part may do. Continuing another user's chat is refused and leaves that chat as it was, and
one message sent to two models stores a reply from each. Tool servers and terminals the request
names itself never reach the model, for a user or an admin: a person's own tool servers and
terminals, which the browser sent with each message, were removed from Open WebUI in 4720ff19e.

Discriminates: in a backend copy, dropping the ownership check on an existing chat in
`chat_completion` fails the foreign-chat test (the stranger's message is stored in the chat), and
fanning out to the first model only fails the two-model test. On dev 4720ff19e~1 with Direct
Integrations on, the request's tool server and terminal reach the model and the tool server test
fails for the admin; it passes on dev 4720ff19e.

The regenerate test pins a fixed bug (PR open-webui/open-webui#32326, fix 99a9f60b9): Regenerate
after adding a model resends the stored question with both models on it, and the server kept the
old model list, so a reload laid the replies out as versions of one. It passes on dev 5032e834e
and fails on a backend copy with 99a9f60b9 reverted (the question keeps one model).
"""

from __future__ import annotations

import uuid
from typing import Iterator

import pytest

from harness import upstream as reply
from harness.chat import NO_BACKGROUND_TASKS, ChatTurn, ask, send_message, wait_for_reply
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
PERSONAL_TERMINAL = {
    "url": "http://127.0.0.1:9/terminal",
    "is_terminal": True,
    "specs": [
        {
            "name": "run_personal_command",
            "description": "Run a command on my own terminal.",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
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


@pytest.mark.regression
def test_a_model_added_on_regenerate_is_stored_on_the_question(make_user, upstream, second_model):
    account = make_user()
    question = "compare yourselves again"
    upstream.queue(
        reply.text("the only answer", match=_answering_model(MOCK_MODEL_ID)),
        reply.text("first model's answer", match=_answering_model(MOCK_MODEL_ID)),
        reply.text("second model's answer", match=_answering_model(second_model)),
    )
    with account.client() as client:
        client.get("/api/models").raise_for_status()
        turn, _ = ask(client, question)
        first_id, second_id = str(uuid.uuid4()), str(uuid.uuid4())
        # Regenerate with a model added resends the stored question with both models on it
        regenerated = client.post(
            "/api/chat/completions",
            json={
                "model": MOCK_MODEL_ID,
                "messages": [{"role": "user", "content": question}],
                "stream": True,
                "chat_id": turn.chat_id,
                "parent_id": None,
                "id": first_id,
                "message_ids": [
                    {"model_id": MOCK_MODEL_ID, "message_id": first_id, "modelIdx": 0},
                    {"model_id": second_model, "message_id": second_id, "modelIdx": 1},
                ],
                "user_message": {
                    "id": turn.user_message_id,
                    "parentId": None,
                    "childrenIds": [turn.assistant_message_id, first_id, second_id],
                    "role": "user",
                    "content": question,
                    "models": [MOCK_MODEL_ID, second_model],
                },
                "session_id": f"harness-{uuid.uuid4().hex[:8]}",
                "background_tasks": NO_BACKGROUND_TASKS,
            },
        )
        assert regenerated.status_code == 200, regenerated.text
        wait_for_reply(client, ChatTurn(turn.chat_id, turn.user_message_id, first_id))
        wait_for_reply(client, ChatTurn(turn.chat_id, turn.user_message_id, second_id))
        stored = client.get(f"/api/v1/chats/{turn.chat_id}").json()["chat"]

    question_message = stored["history"]["messages"][turn.user_message_id]
    assert question_message["models"] == [MOCK_MODEL_ID, second_model], (
        "the question kept its old model list, so a reload shows one column of versions"
    )
    assert question_message["content"] == question


@pytest.mark.parametrize("role", ["user", "admin"])
def test_tool_servers_and_terminals_in_the_request_never_reach_the_model(make_user, upstream, role):
    account = make_user(role=role)

    with account.client() as client:
        ask(
            client,
            "what is the weather in Oslo?",
            tool_servers=[TOOL_SERVER, PERSONAL_TERMINAL],
        )

    offered = _offered_tools(upstream)
    assert not {"lookup_weather", "run_personal_command"} & offered, (
        f"the {role}'s request offered the model its own tool server or terminal: {offered}"
    )
