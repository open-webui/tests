"""The Memories switch and the memories permission gate every memory surface of a chat.

Two 0.11.4 fixes:

* `702da1e47` (PR #30228, issue #30227): with Memories switched off in the admin settings, a chat
  still folded stored memories into the system prompt (only the narrower system-context switch
  was checked) and still offered the eight memory tools (no global switch was checked).
* `e8c26f839` (PR #30309): the background memory review drafted new memories after a turn even
  with Memories switched off or the account barred from them, spending a task-model call on
  writes the memories router then refuses.

Issue #34262 (open, fix on branch `fix-memory-review-temporary-chats`, 5679110c9): the background
review also read temporary chats and saved memories from them, so a temporary chat left memories
behind that later chats were given.

The review only runs with `ENABLE_MEMORY_BACKGROUND_REVIEW`, an environment-only setting, so
this module runs on an instance of its own that reviews after every turn.

Twin of unit/memory/test_memory_switch_gating.py.

Discriminates: passes on dev bbfa876af; dropping the switch from the prompt-context check or
from the memory-tool gate fails the switched-off test, and reverting `e8c26f839` (the review reads
neither switch nor permission) fails the switched-off and barred tests. The temporary-chat test
passes on 5679110c9 and fails on dev 0411edb27 (the temporary chat's fact is stored as a memory).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import pytest

from harness import upstream as reply
from harness.actors import admin_of, create_user
from harness.chat import ask, send_message
from harness.socket_client import connected

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

REVIEW_EVERY_TURN = {
    "ENABLE_MEMORY_BACKGROUND_REVIEW": "true",
    "MEMORIES_REVIEW_INTERVAL_TURNS": "1",
}
REVIEWER_PROMPT = "private memory reviewer"
REVIEW_WAIT = 5.0
STORED_MEMORY = "prefers green tea over coffee"
MEMORY_TOOLS = {
    "search_memories",
    "list_memory_paths",
    "read_memory_path",
    "list_memories",
    "update_memory",
    "add_memory",
    "replace_memory_content",
    "delete_memory",
}
ADMIN_CONFIG = "/api/v1/auths/admin/config"
DEFAULT_PERMISSIONS = "/api/v1/users/default/permissions"


@pytest.fixture
def review_instance(instance_with, preserve):
    launched = instance_with(REVIEW_EVERY_TURN)
    preserve("admin_config", "permissions", on=launched)
    return launched


def _set_admin_config(instance, **changes) -> None:
    with admin_of(instance).client() as client:
        current = client.get(ADMIN_CONFIG).json()
        client.post(ADMIN_CONFIG, json={**current, **changes}).raise_for_status()


def _permit_memories(instance, permitted: bool) -> None:
    with admin_of(instance).client() as client:
        current = client.get(DEFAULT_PERMISSIONS).json()
        features = {**current["features"], "memories": permitted}
        client.post(DEFAULT_PERMISSIONS, json={**current, "features": features}).raise_for_status()


@dataclass
class MemorySurfaces:
    prompt_context: bool
    tools: bool
    review: bool


def _chat_with_memory(instance, account) -> MemorySurfaces:
    """One memory-enabled turn; which memory surfaces the provider saw."""
    with account.client() as client:
        ask(client, "what do you remember about me?", features={"memory": True})
    chat = next(request for request in instance.upstream.chat_requests() if request.get("stream"))
    deadline = time.monotonic() + REVIEW_WAIT
    reviewed = False
    while not reviewed and time.monotonic() < deadline:
        reviewed = REVIEWER_PROMPT in str(instance.upstream.chat_requests())
        time.sleep(0.2)
    offered = {tool["function"]["name"] for tool in chat.get("tools") or []}
    return MemorySurfaces(
        prompt_context=STORED_MEMORY in str(chat["messages"]),
        tools=bool(offered & MEMORY_TOOLS),
        review=reviewed,
    )


def _user_with_a_memory(instance):
    account = create_user(instance)
    with account.client() as client:
        added = client.post("/api/v1/memories/add", json={"content": STORED_MEMORY, "type": "user"})
    assert added.status_code == 200, added.text
    return account


def test_switched_off_memories_reach_no_memory_surface(review_instance):
    account = _user_with_a_memory(review_instance)
    _set_admin_config(review_instance, ENABLE_MEMORIES=False)
    review_instance.upstream.reset()

    surfaces = _chat_with_memory(review_instance, account)

    assert surfaces == MemorySurfaces(prompt_context=False, tools=False, review=False), (
        "with Memories switched off a chat still used them (#30227, PR #30309)"
    )


def test_a_user_barred_from_memories_reaches_no_memory_surface(review_instance):
    account = _user_with_a_memory(review_instance)
    _permit_memories(review_instance, False)
    review_instance.upstream.reset()

    surfaces = _chat_with_memory(review_instance, account)

    assert surfaces == MemorySurfaces(prompt_context=False, tools=False, review=False), (
        "an account barred from memories still had them used in its chat (PR #30309)"
    )


def test_a_permitted_user_gets_every_memory_surface(review_instance):
    account = _user_with_a_memory(review_instance)
    review_instance.upstream.reset()

    assert _chat_with_memory(review_instance, account) == MemorySurfaces(True, True, True)


def test_the_system_context_switch_only_withholds_the_prompt_context(review_instance):
    account = _user_with_a_memory(review_instance)
    _set_admin_config(review_instance, ENABLE_MEMORY_SYSTEM_CONTEXT=False)
    review_instance.upstream.reset()

    assert _chat_with_memory(review_instance, account) == MemorySurfaces(False, True, True)


def test_an_admin_is_reviewed_even_when_the_default_permission_is_off(review_instance):
    _permit_memories(review_instance, False)
    review_instance.upstream.reset()

    surfaces = _chat_with_memory(review_instance, admin_of(review_instance))

    assert surfaces.tools and surfaces.review


TEMPORARY_PROMPT = "remember that I keep a tortoise named Ada"
SAVED_PROMPT = "remember that I play the cello"


def _reviewer_adds(prompt: str, fact: str) -> reply.Reply:
    """The review of the chat that asked `prompt` saves `fact`."""

    def reviews_prompt(body: dict) -> bool:
        return REVIEWER_PROMPT in str(body.get("messages")) and prompt in str(body.get("messages"))

    operations = {"operations": [{"action": "add", "type": "user", "path": "", "content": fact}]}
    return reply.text(json.dumps(operations), match=reviews_prompt)


def _memory_contents(client) -> list[str]:
    listed = client.get("/api/v1/memories/")
    assert listed.status_code == 200, listed.text
    return [memory["content"] for memory in listed.json()]


def test_a_temporary_chat_leaves_no_memory_behind(review_instance):
    account = create_user(review_instance)
    review_instance.upstream.reset()
    review_instance.upstream.queue(
        _reviewer_adds(TEMPORARY_PROMPT, "keeps a tortoise named Ada"),
        _reviewer_adds(SAVED_PROMPT, "plays the cello"),
    )

    with connected(account) as socket, account.client() as client:
        chat_id = socket.temporary_chat_id()
        send_message(
            client,
            TEMPORARY_PROMPT,
            chat_id=chat_id,
            session_id=socket.sid,
            features={"memory": True},
        )
        socket.wait_for(chat_id, "chat:completion", done=True)
        # a saved chat afterwards shows the review runs and its memory lands
        ask(client, SAVED_PROMPT, features={"memory": True})
        deadline = time.monotonic() + REVIEW_WAIT
        while "plays the cello" not in _memory_contents(client) and time.monotonic() < deadline:
            time.sleep(0.2)
        memories = _memory_contents(client)

    reviews = [
        str(body["messages"])
        for body in review_instance.upstream.chat_requests()
        if REVIEWER_PROMPT in str(body["messages"])
    ]
    assert "plays the cello" in memories, f"the saved chat's review stored nothing: {memories}"
    assert "keeps a tortoise named Ada" not in memories, (
        f"a temporary chat left a memory behind: {memories}"
    )
    assert not any(TEMPORARY_PROMPT in review for review in reviews), (
        "the memory review read a temporary chat"
    )
