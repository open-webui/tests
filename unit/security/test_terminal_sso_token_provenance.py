"""Regression: the terminal proxy must resolve the SSO token server-side.

open-webui 0.11.0 fix `3a9b9a1a7` (PR #26719): for a terminal connection with
`auth_type == "system_oauth"`, the proxy read the OAuth access token from the
client-supplied `x-oauth-access-token` request header and forwarded it verbatim
as a Bearer token to the terminal server. Any authenticated caller could
therefore substitute a token that was never issued to them. The fix resolves the
token from the caller's own OAuth session via
`oauth_manager.get_oauth_token(user.id, cookies["oauth_session_id"])` and ignores
the header entirely.

Assertions are taken at the HTTP-client boundary: `aiohttp.ClientSession.request`
is captured and the outbound headers are inspected.

Discriminates: passes on v0.11.0, fails on v0.10.2 (the attacker-supplied header
value is forwarded as the upstream Bearer token).
"""

from __future__ import annotations

import types

import pytest

pytestmark = pytest.mark.regression

ATTACKER_TOKEN = "attacker-supplied-token"
SESSION_TOKEN = "session-bound-token"
CONNECTION_KEY = "connection-configured-key"
JWT_CREDENTIALS = "verified-jwt-credentials"


class _CapturingSession:
    """Stands in for `aiohttp.ClientSession`, recording the outbound request."""

    def __init__(self, captured: list, **_kwargs):
        self._captured = captured

    async def request(self, **kwargs):
        self._captured.append(kwargs)
        return _UpstreamResponse()

    async def close(self):
        return None


class _UpstreamResponse:
    status = 200
    headers = {"content-type": "application/json"}

    async def read(self):
        return b"{}"

    async def release(self):
        return None


class _FakeRequest:
    def __init__(self, headers=None, cookies=None, oauth_token=None):
        self.method = "GET"
        self.headers = headers or {}
        self.cookies = cookies or {}
        self.query_params = {}
        self.state = types.SimpleNamespace(
            token=types.SimpleNamespace(credentials=JWT_CREDENTIALS)
        )
        self.app = types.SimpleNamespace(
            state=types.SimpleNamespace(oauth_manager=_FakeOAuthManager(oauth_token))
        )

    async def body(self):
        return b""


class _FakeOAuthManager:
    def __init__(self, token):
        self._token = token
        self.calls = []

    async def get_oauth_token(self, user_id, session_id, force_refresh=False):
        self.calls.append((user_id, session_id))
        return self._token


def _connection(auth_type):
    return {
        "id": "term-1",
        "url": "http://terminal.internal",
        "enabled": True,
        "auth_type": auth_type,
        "key": CONNECTION_KEY,
    }


@pytest.fixture
def run_proxy(terminals_router_module, monkeypatch):
    """Drive the real `proxy_terminal` and hand back the captured outbound request."""
    mod = terminals_router_module
    captured = []

    async def fake_config_get(key, default=None):
        return [_connection(fake_config_get.auth_type)]

    async def fake_get_groups(_user_id):
        return []

    async def fake_has_access(_user, _connection, _group_ids):
        return True

    monkeypatch.setattr(mod.Config, "get", fake_config_get)
    monkeypatch.setattr(mod.Groups, "get_groups_by_member_id", fake_get_groups)
    monkeypatch.setattr(mod, "has_connection_access", fake_has_access)
    monkeypatch.setattr(
        mod.aiohttp, "ClientSession", lambda **kwargs: _CapturingSession(captured, **kwargs)
    )

    async def _run(auth_type, request):
        fake_config_get.auth_type = auth_type
        user = types.SimpleNamespace(id="caller-user-id", role="user")
        response = await mod.proxy_terminal("term-1", "session/start", request, user)
        assert response.status_code == 200, "proxy did not reach the upstream request"
        return captured[-1]

    return _run


# --- narrow: the exact bug ------------------------------------------------


@pytest.mark.asyncio
async def test_attacker_header_never_becomes_the_upstream_bearer_token(run_proxy):
    request = _FakeRequest(
        headers={"x-oauth-access-token": ATTACKER_TOKEN},
        cookies={"oauth_session_id": "caller-oauth-session"},
        oauth_token={"access_token": SESSION_TOKEN},
    )

    outbound = await run_proxy("system_oauth", request)
    authorization = outbound["headers"].get("Authorization", "")

    assert ATTACKER_TOKEN not in authorization, (
        "the terminal server received a Bearer token the caller supplied in a "
        "request header, so any authenticated user could act upstream as the "
        "owner of a token that was never issued to them (#26719)"
    )
    assert authorization == f"Bearer {SESSION_TOKEN}", (
        "the forwarded token must be the one bound to the caller's own OAuth "
        "session (#26719)"
    )


@pytest.mark.asyncio
async def test_token_is_resolved_from_the_callers_own_session(run_proxy):
    request = _FakeRequest(
        headers={"x-oauth-access-token": ATTACKER_TOKEN},
        cookies={"oauth_session_id": "caller-oauth-session"},
        oauth_token={"access_token": SESSION_TOKEN},
    )

    await run_proxy("system_oauth", request)

    assert request.app.state.oauth_manager.calls == [
        ("caller-user-id", "caller-oauth-session")
    ], (
        "the proxy must look the token up server-side, keyed by the "
        "authenticated user and their own session cookie (#26719)"
    )


@pytest.mark.asyncio
async def test_legitimate_session_token_is_forwarded(run_proxy):
    request = _FakeRequest(
        cookies={"oauth_session_id": "caller-oauth-session"},
        oauth_token={"access_token": SESSION_TOKEN},
    )

    outbound = await run_proxy("system_oauth", request)

    assert outbound["headers"].get("Authorization") == f"Bearer {SESSION_TOKEN}", (
        "single sign-on must still work: the token Open WebUI issued for the "
        "caller has to reach the terminal server (#26719)"
    )


# --- broad: provenance of everything the proxy forwards -------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("auth_type", ["bearer", "session", "system_oauth", "none"])
async def test_no_forwarded_value_originates_from_a_caller_header(run_proxy, auth_type):
    """Every upstream header must be server-resolved, not echoed from the caller."""
    poisoned = f"poisoned-{auth_type}"
    request = _FakeRequest(
        headers={
            "x-oauth-access-token": poisoned,
            "authorization": f"Bearer {poisoned}",
            "x-user-id": poisoned,
            "x-forwarded-user": poisoned,
        },
        cookies={"oauth_session_id": "caller-oauth-session"},
        oauth_token={"access_token": SESSION_TOKEN},
    )

    outbound = await run_proxy(auth_type, request)
    echoed = {key: value for key, value in outbound["headers"].items() if poisoned in str(value)}

    assert echoed == {}, (
        f"auth_type={auth_type!r} forwarded {sorted(echoed)} straight from a "
        "caller-controlled request header, letting the caller choose what "
        "identity or credential the terminal server sees (#26719)"
    )


@pytest.mark.asyncio
async def test_upstream_identity_is_the_authenticated_user(run_proxy):
    request = _FakeRequest(headers={"x-user-id": "someone-else"})

    outbound = await run_proxy("none", request)

    assert outbound["headers"]["X-User-Id"] == "caller-user-id", (
        "the terminal server must be told who Open WebUI authenticated, not "
        "whoever the caller claimed to be (#26719)"
    )


# --- nearby: adjacent behaviour worth locking in --------------------------


@pytest.mark.asyncio
async def test_no_oauth_session_forwards_no_authorization(run_proxy):
    request = _FakeRequest(cookies={}, oauth_token=None)

    outbound = await run_proxy("system_oauth", request)

    assert "Authorization" not in outbound["headers"], (
        "a caller without an OAuth session must reach the terminal server "
        "unauthenticated rather than with an empty Bearer token"
    )


@pytest.mark.asyncio
async def test_bearer_auth_type_uses_the_configured_connection_key(run_proxy):
    request = _FakeRequest(headers={"x-oauth-access-token": ATTACKER_TOKEN})

    outbound = await run_proxy("bearer", request)

    assert outbound["headers"]["Authorization"] == f"Bearer {CONNECTION_KEY}"


@pytest.mark.asyncio
async def test_session_auth_type_uses_the_verified_jwt(run_proxy):
    request = _FakeRequest(headers={"x-oauth-access-token": ATTACKER_TOKEN})

    outbound = await run_proxy("session", request)

    assert outbound["headers"]["Authorization"] == f"Bearer {JWT_CREDENTIALS}"


@pytest.mark.asyncio
async def test_none_auth_type_sends_no_authorization(run_proxy):
    outbound = await run_proxy("none", _FakeRequest())

    assert "Authorization" not in outbound["headers"]


@pytest.mark.asyncio
async def test_benign_headers_are_still_passed_through(run_proxy):
    request = _FakeRequest(
        headers={"x-session-id": "term-session-42", "content-type": "application/json"}
    )

    outbound = await run_proxy("none", request)

    assert outbound["headers"]["X-Session-Id"] == "term-session-42"
    assert outbound["headers"]["Content-Type"] == "application/json"


@pytest.mark.asyncio
async def test_cookies_are_forwarded_for_oauth_connections(run_proxy):
    cookies = {"oauth_session_id": "caller-oauth-session", "token": "jwt-cookie"}
    request = _FakeRequest(cookies=cookies, oauth_token={"access_token": SESSION_TOKEN})

    outbound = await run_proxy("system_oauth", request)

    assert outbound["cookies"] == cookies
