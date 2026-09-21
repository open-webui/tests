"""Regression: OAuth sign-in token payloads were written to the application log, and
comma-separated blocked groups typed into the admin form never blocked a group.

open-webui 0.11.4 fixes `39c1e86e9` (#29709) and `3fc1146c13d7b4b7a67c7fd058014436f80e0dfd`:

- #29709: two OAuth failure paths interpolated the raw token object into their log
  message. On the callback path that object is a live credential set, so a provider
  returning no user data wrote an access token, and usually a refresh token, straight
  into the application log. The token-exchange failure now logs the client_id and the
  provider's error description instead of the raw response, and the callback failure
  logs the provider name instead of the token.
- Blocked sign-in groups: the admin form saved `OAUTH_BLOCKED_GROUPS` through the
  comma-list field, so `['blocked-group']` was stored as the text `blocked-group`, while
  the sign-in check parsed the setting with `JSONCodec.loads`, which needs a JSON
  document, so comma-separated input read as a decode error and the blocked list came
  out empty, and a group name carrying a comma could not be saved at all. The form save
  now JSON-encodes the list, and the check accepts a JSON array, a stored list, or
  comma-separated text.

Discriminates: passes on v0.11.4 (344ea5306), fails on v0.11.4~ (39c1e86e9^ /
3fc1146c^): pre-fix the callback failure log carries the token object verbatim, the
stored blocked-groups value is plain comma text, and the sign-in check reads that text
as nothing (decode error, empty list).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.regression

ALICE = "alice-user-id"


@pytest.fixture(scope="module")
def oauth_module(owui_module):
    return owui_module("open_webui.utils.oauth")


@pytest.fixture(scope="module")
def auths_module(owui_module):
    return owui_module("open_webui.routers.auths")


# ── 1. tokens stay out of logs (39c1e86e9, #29709) ──────────────────────────


def _callback_manager(oauth_module, client):
    manager = oauth_module.OAuthManager(app=SimpleNamespace(state=SimpleNamespace()))
    manager._clients["oidc"] = client
    return manager


def _callback_request():
    return SimpleNamespace(
        base_url="https://owui.example/",
        app=SimpleNamespace(state=SimpleNamespace(redis=None)),
    )


@pytest.mark.asyncio
async def test_missing_user_data_log_does_not_leak_the_token(monkeypatch, oauth_module):
    """NARROW: a callback whose userinfo comes back empty must not log the credential set."""
    live_credentials = {
        "access_token": "leaky-access-token",
        "refresh_token": "leaky-refresh-token",
        "id_token": "header.leaky-id-token.sig",
        "token_type": "bearer",
    }

    async def _authorize_access_token(request, **kwargs):
        return dict(live_credentials)

    async def _userinfo(token=None):
        return None

    client = SimpleNamespace(
        authorize_access_token=_authorize_access_token,
        userinfo=_userinfo,
        server_metadata={},
    )
    config = SimpleNamespace(
        ENABLE_OAUTH=True,
        OAUTH_EMAIL_CLAIM="email",
        OAUTH_USERNAME_CLAIM="name",
        OAUTH_SUB_CLAIM="sub",
        OAUTH_ALLOWED_DOMAINS=["*"],
    )

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Capture(level=logging.WARNING)
    oauth_log = logging.getLogger("open_webui.utils.oauth")
    old_level = oauth_log.level
    oauth_log.addHandler(handler)
    oauth_log.setLevel(logging.DEBUG)
    try:
        manager = _callback_manager(oauth_module, client)
        request = _callback_request()
        with (
            patch.object(oauth_module, "get_oauth_runtime_config", AsyncMock(return_value=config)),
            patch.object(oauth_module, "OAUTH_PROVIDERS", {"oidc": {"sub_claim": "sub"}}),
            patch.object(oauth_module.Config, "get", AsyncMock(return_value="https://owui.example")),
        ):
            result = await manager.handle_callback(request, "oidc", SimpleNamespace(headers={}))
    finally:
        oauth_log.removeHandler(handler)
        oauth_log.setLevel(old_level)

    assert "error=" in result.headers["location"], "the failed callback redirected as a success"
    emitted = " ".join(record.getMessage() for record in records)
    for secret in (
        "leaky-access-token",
        "leaky-refresh-token",
        "leaky-id-token",
    ):
        assert secret not in emitted, f"the OAuth failure log leaked {secret}"
    assert "oidc" in emitted, "the failure log does not name the provider"


@pytest.mark.asyncio
async def test_callback_still_refuses_the_sign_in_without_user_data(oauth_module):
    """NEARBY: the missing-user-data path still raises the 400 it raised before."""
    async def _authorize_access_token(request, **kwargs):
        return {"access_token": "at", "token_type": "bearer"}

    async def _userinfo(token=None):
        return None

    client = SimpleNamespace(
        authorize_access_token=_authorize_access_token,
        userinfo=_userinfo,
        server_metadata={},
    )
    config = SimpleNamespace(
        ENABLE_OAUTH=True,
        OAUTH_EMAIL_CLAIM="email",
        OAUTH_USERNAME_CLAIM="name",
        OAUTH_SUB_CLAIM="sub",
        OAUTH_ALLOWED_DOMAINS=["*"],
    )
    manager = _callback_manager(oauth_module, client)
    with (
        patch.object(oauth_module, "get_oauth_runtime_config", AsyncMock(return_value=config)),
        patch.object(oauth_module, "OAUTH_PROVIDERS", {"oidc": {"sub_claim": "sub"}}),
        patch.object(oauth_module.Config, "get", AsyncMock(return_value="https://owui.example")),
    ):
        result = await manager.handle_callback(
            _callback_request(), "oidc", SimpleNamespace(headers={})
        )
    assert "error=" in result.headers["location"]


# ── 3. blocked sign-in groups (3fc1146c) ────────────────────────────────────


def test_comma_separated_blocked_groups_are_parsed_into_a_list(oauth_module):
    """NARROW: the sign-in check reads comma-separated admin text as the blocked groups."""
    assert oauth_module._parse_blocked_groups("blocked-group, other-group") == [
        "blocked-group",
        "other-group",
    ]


def test_json_saved_blocked_groups_round_trip(oauth_module):
    """NARROW: the JSON text the form save now writes parses back to the same list."""
    saved = oauth_module.JSONCodec.dumps(["blocked-group", "other, with comma"])
    assert oauth_module._parse_blocked_groups(saved) == ["blocked-group", "other, with comma"]


def test_blocked_groups_setting_as_a_live_list_is_accepted(oauth_module):
    """NEARBY: an already-persisted list value still reaches the check unchanged."""
    assert oauth_module._parse_blocked_groups(["a", "b"]) == ["a", "b"]


def test_empty_and_non_string_entries_are_dropped(oauth_module):
    """NEARBY: junk in the setting does not turn into patterns that block or match."""
    assert oauth_module._parse_blocked_groups(" , , ") == []
    assert oauth_module._parse_blocked_groups(["ok", 7, "", None]) == ["ok"]
    assert oauth_module._parse_blocked_groups(None) == []
    assert oauth_module._parse_blocked_groups("") == []


def test_blocked_group_actually_blocks_is_in_blocked_groups(oauth_module):
    """BROAD: end to end, the parsed setting must reach the match the check uses."""
    blocked = oauth_module._parse_blocked_groups("blocked-group")
    assert oauth_module.is_in_blocked_groups("blocked-group", blocked)


def test_a_group_name_with_a_comma_survives_the_save(auths_module):
    """NARROW: the form save JSON-encodes the list, so a comma inside a name survives."""
    saved = auths_module._format_oauth_form_value(
        "OAUTH_BLOCKED_GROUPS", ["team, with comma", "^vip-.*$"]
    )
    assert auths_module.JSONCodec.loads(saved) == ["team, with comma", "^vip-.*$"]


def test_comma_list_fields_still_save_as_comma_text(auths_module):
    """NEARBY: the ordinary comma-list fields keep their plain-text storage."""
    saved = auths_module._format_oauth_form_value("OAUTH_ADMIN_ROLES", ["admin", "owner"])
    assert saved == "admin,owner"
