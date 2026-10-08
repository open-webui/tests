"""Journey: a chat a script runs over the API with the account's key, as its owner sees it.

The API docs have a script create a chat holding its question and an empty answer, then name both
in `/api/chat/completions` with a session id so the server answers into that chat. The owner,
with the chat already open in the browser, sees the answer arrive without reloading, and the chat
sits in the sidebar under the title the script gave it.

Red on dev 62f70a844: since de73bb830 a call into a chat whose empty answer was stored first, as the
docs describe, is refused with 409 (open-webui/open-webui#32066).

Discriminates: on dev 0f5a58f5f, in a backend copy, `get_current_user_by_api_key` finding no user
fails the test (the script is refused), and the OpenAI router stripping `data: ` from the relayed
stream fails it (the answer stays empty).
"""

from __future__ import annotations

import time
import uuid

import httpx
import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.chat import NO_BACKGROUND_TASKS
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import REPLY_TIMEOUT_MS, conversation, last_reply

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

ADMIN_CONFIG = "/api/v1/auths/admin/config"
DEFAULT_PERMISSIONS = "/api/v1/users/default/permissions"
TITLE = "Nightly ticket check"
QUESTION = "Which tickets are still open?"
ANSWER = "Only T-17 is still open."


@pytest.fixture
def key_holder(admin, make_user, preserve):
    """A fresh account and its API key, with API keys on and allowed to users."""
    preserve("admin_config", "permissions")
    with admin.client() as client:
        config = client.get(ADMIN_CONFIG).json()
        saved = client.post(
            ADMIN_CONFIG,
            json={
                **config,
                "ENABLE_API_KEYS": True,
                "ENABLE_API_KEYS_ENDPOINT_RESTRICTIONS": False,
            },
        )
        assert saved.status_code == 200, saved.text
        permissions = client.get(DEFAULT_PERMISSIONS).json()
        permissions["features"]["api_keys"] = True
        saved = client.post(DEFAULT_PERMISSIONS, json=permissions)
        assert saved.status_code == 200, saved.text
    account = make_user()
    with account.client() as client:
        generated = client.post("/api/v1/auths/api_key")
    assert generated.status_code == 200, generated.text
    return account, generated.json()["api_key"]


def _create_chat(client: httpx.Client) -> tuple[str, str]:
    """The chat as the docs create it, the question answered by an empty message."""
    question_id, answer_id = str(uuid.uuid4()), str(uuid.uuid4())
    now = int(time.time())
    question = {
        "id": question_id,
        "role": "user",
        "content": QUESTION,
        "timestamp": now,
        "models": [MOCK_MODEL_ID],
        "childrenIds": [answer_id],
    }
    answer = {
        "id": answer_id,
        "role": "assistant",
        "content": "",
        "parentId": question_id,
        "childrenIds": [],
        "model": MOCK_MODEL_ID,
        "modelIdx": 0,
        "done": False,
        "timestamp": now + 1,
    }
    history = {"currentId": answer_id, "messages": {question_id: question, answer_id: answer}}
    created = client.post(
        "/api/v1/chats/new",
        json={"chat": {"title": TITLE, "models": [MOCK_MODEL_ID], "history": history}},
    )
    assert created.status_code == 200, created.text
    return created.json()["id"], answer_id


def test_the_answer_a_script_asks_for_arrives_in_the_owners_open_chat(
    instance, page_for, key_holder, upstream
):
    account, api_key = key_holder
    upstream.queue(reply.text(ANSWER, match=reply.answering(QUESTION)))
    with instance.client(api_key) as client:
        chat_id, answer_id = _create_chat(client)
        page = page_for(account)
        page.goto(f"/c/{chat_id}")
        expect(conversation(page)).to_contain_text(QUESTION, timeout=REPLY_TIMEOUT_MS)

        accepted = client.post(
            "/api/chat/completions",
            json={
                "model": MOCK_MODEL_ID,
                "messages": [{"role": "user", "content": QUESTION}],
                "stream": True,
                "chat_id": chat_id,
                "id": answer_id,
                "session_id": f"api-{uuid.uuid4().hex[:8]}",
                "background_tasks": NO_BACKGROUND_TASKS,
            },
        )
        assert accepted.status_code == 200, accepted.text

        expect(last_reply(page)).to_contain_text(ANSWER, timeout=REPLY_TIMEOUT_MS)
    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    sidebar = page.get_by_role("navigation", name="Chat history")
    expect(sidebar.get_by_role("button", name=TITLE)).to_be_visible()
