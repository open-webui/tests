"""Regression: a filter or action must not be pointed at someone else's chat.

open-webui 0.11.0 fix `c882222f6` (PR #27486): `/api/chat/completed` and
`/api/chat/actions/{action_id}` took `chat_id` from the request body and handed it to the event
emitter without checking the caller owned that chat. The emitter persists by chat id alone, so
an action or outlet filter wrote into whichever chat the caller named. The fix adds
`verify_chat_ownership` in front of both handlers: temporary (`temporary:`, `local:`) ids pass,
`channel:` ids get 400, and a non-admin who does not own the chat gets 404.

Twin of unit/security/test_chat_ownership_endpoints.py (its `ast` sweep over main.py stays
there).

Discriminates: passes on dev bbfa876af; with the two `verify_chat_ownership` calls removed a
user's action call rewrites the admin's stored reply and neither route answers 404.
"""

from __future__ import annotations

import uuid

import pytest

from harness.chat import ask

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

REWRITTEN = "rewritten by an action"

ACTION_SOURCE = f"""
class Action:
    async def action(self, body, __event_emitter__=None):
        await __event_emitter__({{"type": "replace", "data": {{"content": "{REWRITTEN}"}}}})
        return {{"rewritten": True}}
"""


@pytest.fixture
def rewriting_action(admin):
    """A global action that replaces the content of the message it is run on."""
    function_id = f"rewrite_{uuid.uuid4().hex[:8]}"
    with admin.client() as client:
        created = client.post(
            "/api/v1/functions/create",
            json={
                "id": function_id,
                "name": "Rewrite",
                "content": ACTION_SOURCE,
                "meta": {"description": "rewrites the message"},
            },
        )
        assert created.status_code == 200, created.text
        client.post(f"/api/v1/functions/id/{function_id}/toggle").raise_for_status()
        client.post(f"/api/v1/functions/id/{function_id}/toggle/global").raise_for_status()
        client.get("/api/models").raise_for_status()
    yield function_id
    with admin.client() as client:
        client.delete(f"/api/v1/functions/id/{function_id}/delete")
        client.get("/api/models")


def _body(chat_id: str | None, message_id: str) -> dict:
    return {
        "model": "mock-model",
        "messages": [{"id": message_id, "role": "assistant", "content": "a reply"}],
        "chat_id": chat_id,
        "id": message_id,
        "session_id": "harness-session",
    }


def _endpoint(name: str, action_id: str) -> str:
    return "/api/chat/completed" if name == "completed" else f"/api/chat/actions/{action_id}"


def _stored_content(client, turn) -> str:
    chat = client.get(f"/api/v1/chats/{turn.chat_id}").json()["chat"]
    return chat["history"]["messages"][turn.assistant_message_id]["content"]


@pytest.fixture
def admin_turn(admin, upstream):
    with admin.client() as client:
        turn, _ = ask(client, "the admin's private question")
    return turn


@pytest.mark.parametrize("endpoint", ["completed", "action"])
def test_a_user_naming_someone_elses_chat_is_not_found(
    endpoint, admin_turn, make_user, rewriting_action
):
    body = _body(admin_turn.chat_id, admin_turn.assistant_message_id)
    with make_user().client() as client:
        response = client.post(_endpoint(endpoint, rewriting_action), json=body)

    assert response.status_code == 404, (
        f"a non-owner got HTTP {response.status_code} instead of 404 from /api/chat/{endpoint} "
        f"with another user's chat_id (#27486): {response.text[:200]}"
    )


def test_a_users_action_cannot_rewrite_someone_elses_reply(
    admin, admin_turn, make_user, rewriting_action
):
    with admin.client() as admin_client:
        original = _stored_content(admin_client, admin_turn)
        body = _body(admin_turn.chat_id, admin_turn.assistant_message_id)
        with make_user().client() as client:
            client.post(f"/api/chat/actions/{rewriting_action}", json=body)

        assert _stored_content(admin_client, admin_turn) == original, (
            "a user's action call rewrote a reply in the admin's chat (#27486)"
        )


@pytest.mark.parametrize("endpoint", ["completed", "action"])
def test_a_channel_chat_id_is_refused(endpoint, make_user, rewriting_action):
    body = _body("channel:some-channel", "message-1")
    with make_user().client() as client:
        response = client.post(_endpoint(endpoint, rewriting_action), json=body)

    assert response.status_code == 400, response.text


def test_local_is_matched_as_a_prefix_not_a_substring(admin_turn, make_user, rewriting_action):
    """A saved chat id that merely contains 'local:' must still be checked."""
    body = _body(f"abc-local:{admin_turn.chat_id}", admin_turn.assistant_message_id)
    with make_user().client() as client:
        response = client.post("/api/chat/completed", json=body)

    assert response.status_code == 404, response.text


@pytest.mark.parametrize("endpoint", ["completed", "action"])
def test_the_owner_reaches_the_handler(endpoint, make_user, upstream, rewriting_action):
    with make_user().client() as client:
        turn, _ = ask(client, "my own question")
        body = _body(turn.chat_id, turn.assistant_message_id)
        response = client.post(_endpoint(endpoint, rewriting_action), json=body)

        assert response.status_code == 200, response.text
        if endpoint == "action":
            assert _stored_content(client, turn) == REWRITTEN


@pytest.mark.parametrize("endpoint", ["completed", "action"])
def test_an_admin_reaches_the_handler_on_any_chat(endpoint, admin, make_user, rewriting_action):
    with make_user().client() as client:
        turn, _ = ask(client, "a user's question")
    body = _body(turn.chat_id, turn.assistant_message_id)
    with admin.client() as client:
        response = client.post(_endpoint(endpoint, rewriting_action), json=body)

    assert response.status_code == 200, response.text


@pytest.mark.parametrize("chat_id", ["temporary:socket-1", "local:socket-1", "", None])
def test_an_unsaved_chat_reaches_the_handler(chat_id, make_user):
    with make_user().client() as client:
        response = client.post("/api/chat/completed", json=_body(chat_id, "message-1"))

    assert response.status_code == 200, response.text
