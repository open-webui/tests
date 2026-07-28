"""Regression: a filter or action must not be pointed at someone else's chat.

open-webui 0.11.0 fix `c882222f6` (PR #27486): `/api/chat/completed` and
`/api/chat/actions/{action_id}` read `chat_id` straight out of the request body
and handed it to `get_event_emitter` without ever checking the caller owned that
chat. The emitter persists through `upsert_message_to_chat_by_id_and_message_id`,
which resolves by primary key and takes no owner argument, so an invoked filter
or action wrote into whichever chat the caller named. `/api/chat/completions`
already gated this; these two routes did not.

The fix adds `verify_chat_ownership`, awaited at the top of both handlers before
their `try` block (the `except Exception` there would otherwise swallow the
HTTPException and rewrite the 404 into a 400). Temporary chat ids pass through
(per-socket, never persisted), `channel:` ids are rejected with 400, and a
non-admin who fails `Chats.is_chat_owner` gets 404 rather than 403 so the
response does not confirm the chat exists.

Discriminates: passes on v0.11.0, fails on v0.10.2 (no ownership gate at all, so
the handler runs against another user's chat_id and nothing is raised).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

pytestmark = pytest.mark.regression

VICTIM_CHAT_ID = "victim-chat-id"


def _user(role: str = "user", id: str = "mallory") -> SimpleNamespace:
    return SimpleNamespace(id=id, role=role, email=f"{id}@example.com", name=id)


def _request() -> SimpleNamespace:
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(MODELS={})))


def _not_the_owner(main):
    """`verify_chat_ownership` resolves `Chats` from the singleton instance."""
    import open_webui.models.chats as chats_module

    return patch.object(chats_module.Chats, "is_chat_owner", AsyncMock(return_value=False))


def _the_owner(main):
    import open_webui.models.chats as chats_module

    return patch.object(chats_module.Chats, "is_chat_owner", AsyncMock(return_value=True))


def _stub_handlers(main):
    """Mock the downstream filter/action handlers: reaching them at all is the bug."""
    return (
        patch.object(main, "chat_completed_handler", AsyncMock(return_value={"reached": True})),
        patch.object(main, "chat_action_handler", AsyncMock(return_value={"reached": True})),
    )


async def _call_completed(main, form_data, user):
    return await main.chat_completed(_request(), form_data, user)


async def _call_action(main, form_data, user):
    return await main.chat_action(_request(), "my-action", form_data, user)


@pytest.fixture(scope="module")
def main_module(owui_module):
    return owui_module("open_webui.main")


@pytest.fixture(scope="module")
def main_source(main_module):
    return Path(main_module.__file__).read_text(encoding="utf-8")


# --- Narrow: the endpoints must refuse a chat the caller does not own ---------


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["completed", "action"])
async def test_endpoint_rejects_chat_owned_by_someone_else(main_module, endpoint):
    main = main_module
    call = _call_completed if endpoint == "completed" else _call_action
    completed_handler, action_handler = _stub_handlers(main)

    with _not_the_owner(main), completed_handler as completed, action_handler as action:
        with pytest.raises(HTTPException) as excinfo:
            await call(main, {"chat_id": VICTIM_CHAT_ID, "id": "msg-1"}, _user())

        reached = completed.await_count + action.await_count

    assert excinfo.value.status_code == 404, (
        "a non-owner got something other than 404; 403 or 400 tells the caller "
        "whether the chat exists, and #27486 chose 404 deliberately"
    )
    assert reached == 0, (
        "the filter/action handler ran against a chat the caller does not own, so "
        "it can write a message into another person's conversation (#27486)"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["completed", "action"])
async def test_endpoint_rejects_channel_chat_id(main_module, endpoint):
    """Channel ids reach the channel emitter, whose write-access gate lives only
    on /api/chat/completions."""
    main = main_module
    call = _call_completed if endpoint == "completed" else _call_action
    completed_handler, action_handler = _stub_handlers(main)

    with _not_the_owner(main), completed_handler as completed, action_handler as action:
        with pytest.raises(HTTPException) as excinfo:
            await call(main, {"chat_id": "channel:some-channel", "id": "msg-1"}, _user())

        reached = completed.await_count + action.await_count

    assert excinfo.value.status_code == 400
    assert reached == 0, (
        "a channel message was written through an endpoint that never checks "
        "channel membership or write access (#27486)"
    )


@pytest.mark.asyncio
async def test_temporary_prefix_is_matched_as_a_prefix_not_a_substring(main_module):
    """A saved chat whose id merely contains 'local:' must still be checked."""
    main = main_module
    completed_handler, _ = _stub_handlers(main)

    with _not_the_owner(main), completed_handler as completed:
        with pytest.raises(HTTPException) as excinfo:
            await _call_completed(main, {"chat_id": "abc-local:xyz", "id": "msg-1"}, _user())

    assert excinfo.value.status_code == 404
    assert completed.await_count == 0, (
        "an ordinary saved chat id containing 'local:' skipped the ownership "
        "check, which reopens #27486 with a crafted id"
    )


# --- Narrow: every branch of the helper itself -------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_id", [None, "", "local:sock-1", "temporary:sock-1"])
async def test_helper_allows_ids_with_no_owner(main_module, chat_id):
    with _not_the_owner(main_module) as is_chat_owner:
        await main_module.verify_chat_ownership(chat_id, _user())

    assert is_chat_owner.await_count == 0, (
        "an unsaved or absent chat id triggered a database ownership lookup it "
        "can never satisfy, which would break temporary chats"
    )


@pytest.mark.asyncio
async def test_helper_rejects_non_owner(main_module):
    with _not_the_owner(main_module):
        with pytest.raises(HTTPException) as excinfo:
            await main_module.verify_chat_ownership(VICTIM_CHAT_ID, _user())
    assert excinfo.value.status_code == 404


@pytest.mark.asyncio
async def test_helper_rejects_channel_chat(main_module):
    with _not_the_owner(main_module):
        with pytest.raises(HTTPException) as excinfo:
            await main_module.verify_chat_ownership("channel:general", _user())
    assert excinfo.value.status_code == 400


@pytest.mark.asyncio
async def test_helper_allows_owner(main_module):
    with _the_owner(main_module):
        await main_module.verify_chat_ownership(VICTIM_CHAT_ID, _user())


@pytest.mark.asyncio
async def test_helper_allows_admin_without_a_lookup(main_module):
    """Admins are exempt, matching /api/chat/completions, so deliberate
    cross-user operations keep working."""
    with _not_the_owner(main_module) as is_chat_owner:
        await main_module.verify_chat_ownership(VICTIM_CHAT_ID, _user(role="admin"))

    assert is_chat_owner.await_count == 0


# --- Broad: no route in main.py may take a body chat_id without a gate --------


def _route_functions(source: str):
    for node in ast.parse(source).body:
        if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            continue
        decorators = " ".join(ast.unparse(d) for d in node.decorator_list)
        if re.search(r"\bapp\.(get|post|put|patch|delete)\(", decorators):
            yield node


BODY_CHAT_ID = re.compile(r"form_data(\.get|\.pop)?\(?\[?'chat_id'")

# Handlers that resolve a chat_id out of the body they are handed and emit into it.
CHAT_ID_SINKS = ("chat_completed_handler", "chat_action_handler", "get_event_emitter")


def _takes_a_caller_chat_id(body: str) -> bool:
    """Either the route reads chat_id itself, or it forwards the raw body to
    something that will."""
    return bool(BODY_CHAT_ID.search(body)) or any(
        f"{sink}(" in body and "form_data" in body for sink in CHAT_ID_SINKS
    )


def _chat_id_routes(main_source: str):
    return [
        node for node in _route_functions(main_source) if _takes_a_caller_chat_id(ast.unparse(node))
    ]


def test_every_non_admin_route_taking_a_body_chat_id_checks_ownership(main_source):
    """The invariant #27486 is an instance of: if a route lets the caller name a
    chat, it must prove the caller may write to it."""
    ungated = []
    for node in _chat_id_routes(main_source):
        body = ast.unparse(node)
        if "get_admin_user" in body:
            continue
        if "verify_chat_ownership" not in body and "is_chat_owner" not in body:
            ungated.append(node.name)

    assert ungated == [], (
        f"{ungated} accept a caller-supplied chat_id in the request body but never "
        "check the caller owns it, so a filter or action can write into another "
        "person's conversation (#27486)"
    )


def test_the_known_body_chat_id_routes_are_still_covered(main_source):
    """Guards the sweep above against silently matching nothing."""
    covered = {node.name for node in _chat_id_routes(main_source)}
    assert {"chat_completion", "chat_completed", "chat_action"} <= covered, (
        f"the route sweep stopped seeing the known chat_id endpoints (saw {covered}); "
        "the invariant test above is no longer proving anything"
    )


# --- Nearby: the fix must not block legitimate callers ------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["completed", "action"])
async def test_owner_reaches_the_handler(main_module, endpoint):
    main = main_module
    call = _call_completed if endpoint == "completed" else _call_action
    completed_handler, action_handler = _stub_handlers(main)

    with _the_owner(main), completed_handler as completed, action_handler as action:
        result = await call(main, {"chat_id": VICTIM_CHAT_ID, "id": "msg-1"}, _user())

    assert result == {"reached": True}
    assert completed.await_count + action.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ["completed", "action"])
async def test_admin_reaches_the_handler_on_any_chat(main_module, endpoint):
    main = main_module
    call = _call_completed if endpoint == "completed" else _call_action
    completed_handler, action_handler = _stub_handlers(main)

    with _not_the_owner(main), completed_handler as completed, action_handler as action:
        result = await call(main, {"chat_id": VICTIM_CHAT_ID, "id": "msg-1"}, _user(role="admin"))

    assert result == {"reached": True}
    assert completed.await_count + action.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_id", ["local:sock-1", "", None])
async def test_unsaved_chat_reaches_the_handler(main_module, chat_id):
    main = main_module
    completed_handler, _ = _stub_handlers(main)

    with _not_the_owner(main), completed_handler as completed:
        result = await _call_completed(main, {"chat_id": chat_id, "id": "msg-1"}, _user())

    assert result == {"reached": True}
    assert completed.await_count == 1
