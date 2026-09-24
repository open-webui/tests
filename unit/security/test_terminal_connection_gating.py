"""A disabled terminal hands no tools to a chat, even from a spec cache that still lists it.

Fix `753798923` (0.11.0) refused a disabled terminal connection on every entry point that
resolves one by id, where before only the list had hidden it. The list, the HTTP proxy and the
WebSocket are pinned over HTTP by integration/security/test_terminal_connection_gating.py,
together with the admin-only rule for ungranted connections (#27581). The chat's tool entry
point stays here: without Redis every chat rebuilds the spec cache from the saved connections
and drops disabled ones, so only a Redis cache that still lists the terminal reaches the gate.

Discriminates: passes on dev bbfa876af, fails with the `enabled` check removed from
get_terminal_tools (the cached spec of the disabled terminal becomes a tool for the chat).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, create_autospec, patch

import pytest
from fastapi import FastAPI
from starlette.requests import Request

pytestmark = pytest.mark.regression

SERVER_ID = "terminal-1"
UNREACHABLE_URL = "http://127.0.0.1:9"
RUN_COMMAND_SPEC = {
    "name": "run_command",
    "description": "Run a shell command.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
}


@pytest.fixture
def terminal_tools(owui_module):
    """`terminal_tools(enabled=...)`: a granted terminal's tools while Redis caches its spec."""
    tools = owui_module("open_webui.utils.tools")
    users = owui_module("open_webui.models.users")
    config_model = owui_module("open_webui.models.config")
    groups_model = owui_module("open_webui.models.groups")
    redis_asyncio = owui_module("redis.asyncio")

    member = users.UserModel(
        id="member-id",
        email="member@example.com",
        name="Member",
        role="user",
        last_active_at=0,
        updated_at=0,
        created_at=0,
    )
    grant = {"principal_type": "user", "principal_id": member.id, "permission": "read"}
    cached_servers = [{"id": SERVER_ID, "url": UNREACHABLE_URL, "specs": [RUN_COMMAND_SPEC]}]
    app = FastAPI()
    app.state.redis = create_autospec(redis_asyncio.Redis, instance=True)
    app.state.redis.get.side_effect = AsyncMock(return_value=json.dumps(cached_servers))
    request = Request({"type": "http", "app": app, "headers": [], "method": "POST", "path": "/"})

    async def resolve(enabled: bool) -> dict:
        connection = {
            "id": SERVER_ID,
            "url": UNREACHABLE_URL,
            "enabled": enabled,
            "auth_type": "none",
            "config": {"access_grants": [grant]},
        }
        with (
            patch.object(config_model.Config, "get", AsyncMock(return_value=[connection])),
            patch.object(
                groups_model.Groups, "get_groups_by_member_id", AsyncMock(return_value=[])
            ),
        ):
            resolved = await tools.get_terminal_tools(
                request=request, terminal_id=SERVER_ID, user=member, extra_params={}
            )
        return resolved[0] if isinstance(resolved, tuple) else resolved

    return resolve


@pytest.mark.asyncio
async def test_a_disabled_terminal_offers_no_tools_while_its_spec_is_cached(terminal_tools):
    with pytest.raises(RuntimeError):
        await terminal_tools(enabled=False)


@pytest.mark.asyncio
async def test_an_enabled_terminal_offers_its_cached_tools(terminal_tools):
    assert "run_command" in await terminal_tools(enabled=True)
