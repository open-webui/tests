"""A `system_oauth` terminal connection gets the token of the caller's own OAuth session.

Fix `3a9b9a1a7` (PR #26719) stopped the terminal proxy from forwarding whatever token the
caller put in an `x-oauth-access-token` header; it resolves the token server-side from the
caller's OAuth session instead. That the header is ignored, and where every other forwarded
header comes from, is pinned over HTTP by
integration/security/test_terminal_sso_token_provenance.py. The positive half stays here
because an OAuth session needs an identity-provider sign-in the scratch instance cannot do:
the session store is a specced stand-in, the real router runs in a bare app and forwards to
a fake terminal server.

Discriminates: passes on dev bbfa876af, fails with 3a9b9a1a7 reverted (the proxy looks for
the header, finds none and sends the terminal no token).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, create_autospec, patch

import httpx
import pytest
from fastapi import FastAPI

from harness.listener import json_answer
from harness.terminal_server import read_grant, serving_terminal

pytestmark = pytest.mark.regression

SESSION_TOKEN = "token-of-the-callers-own-session"


@pytest.mark.asyncio
async def test_the_callers_own_oauth_session_token_reaches_the_terminal(owui_module):
    terminals = owui_module("open_webui.routers.terminals")
    auth = owui_module("open_webui.utils.auth")
    oauth = owui_module("open_webui.utils.oauth")
    users = owui_module("open_webui.models.users")
    config_model = owui_module("open_webui.models.config")
    groups_model = owui_module("open_webui.models.groups")

    caller = users.UserModel(
        id="caller-id",
        email="caller@example.com",
        name="Caller",
        role="user",
        last_active_at=0,
        updated_at=0,
        created_at=0,
    )
    app = FastAPI()
    app.include_router(terminals.router, prefix="/api/v1/terminals")
    app.dependency_overrides[auth.get_verified_user] = lambda: caller
    app.state.oauth_manager = create_autospec(oauth.OAuthManager, instance=True)
    app.state.oauth_manager.get_oauth_token.return_value = {"access_token": SESSION_TOKEN}

    with serving_terminal() as terminal:
        terminal.route("GET", "/probe", json_answer({"ok": True}))
        connection = terminal.connection(
            auth_type="system_oauth", config={"access_grants": [read_grant(caller.id)]}
        )
        with (
            patch.object(config_model.Config, "get", AsyncMock(return_value=[connection])),
            patch.object(
                groups_model.Groups, "get_groups_by_member_id", AsyncMock(return_value=[])
            ),
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://owui"
            ) as client:
                response = await client.get(
                    f"/api/v1/terminals/{connection['id']}/probe",
                    headers={"cookie": "oauth_session_id=callers-session"},
                )

    assert response.status_code == 200, response.text
    [received] = terminal.received
    assert received.headers.get("authorization") == f"Bearer {SESSION_TOKEN}", (
        "single sign-on to the terminal broke: the token of the caller's own OAuth session "
        "never reached it (#26719)"
    )
    app.state.oauth_manager.get_oauth_token.assert_awaited_once_with(caller.id, "callers-session")
