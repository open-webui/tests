"""Tool server auth headers and the terminal server cache, open-webui v0.11.2.

Two fixes in `open_webui/utils/tools.py`:

* `26f37426b` - `build_tool_server_headers` formatted `Bearer {key}` unconditionally, so a
  connection saved without a key was sent a bare `Authorization: Bearer ` header. Some servers
  reject a malformed credential outright rather than treating it as anonymous. All four bearer
  branches now go through the existing `utils.headers.bearer_auth_header`, which strips the token
  and omits the header when nothing is left.
* `81b9afb73` - `get_terminal_servers` accepted an empty list from the Redis cache as a hit and
  wrote it over `app.state.TERMINAL_SERVERS`, so configured terminal servers vanished whenever
  another worker had cached an empty result. The empty list is now only trusted when no enabled
  connection with a url is configured.

Discriminates: passes on v0.11.2, fails on v0.11.1 (pre-fix sends `Bearer ` with no credential,
and lets an empty cached terminal list overwrite configured servers instead of rebuilding).
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.regression


@pytest.fixture(scope="module")
def tools_module(owui_module):
    return owui_module("open_webui.utils.tools")


def _request(**state):
    app_state = SimpleNamespace(redis=None, TERMINAL_SERVERS=[], oauth_client_manager=None, **state)
    return SimpleNamespace(
        app=SimpleNamespace(state=app_state),
        state=SimpleNamespace(token=SimpleNamespace(credentials="")),
        cookies={},
    )


async def _headers(tools_module, connection, request=None, user=None, **kwargs):
    headers, _ = await tools_module.build_tool_server_headers(
        connection, request or _request(), user, **kwargs
    )
    return headers


# ═════════════════════════════════════════════════════════════════════════════
# 26f37426b. A keyless connection sends no Authorization header at all
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "connection", [{}, {"auth_type": "bearer"}, {"auth_type": "bearer", "key": ""}]
)
async def test_bearer_connection_without_a_key_sends_no_authorization(tools_module, connection):
    """Narrow. Pre-fix this was `{'Authorization': 'Bearer '}`, which some tool servers refuse."""
    assert "Authorization" not in await _headers(tools_module, connection)


@pytest.mark.asyncio
async def test_bearer_connection_with_a_whitespace_only_key_sends_no_authorization(tools_module):
    """Narrow. A key of blanks is no credential; pre-fix it was forwarded verbatim."""
    headers = await _headers(tools_module, {"auth_type": "bearer", "key": "   "})
    assert "Authorization" not in headers


@pytest.mark.asyncio
async def test_session_auth_without_a_token_sends_no_authorization(tools_module):
    """Broad. The session branch shares the rule."""
    assert "Authorization" not in await _headers(tools_module, {"auth_type": "session"})


@pytest.mark.asyncio
async def test_system_oauth_without_an_access_token_sends_no_authorization(tools_module):
    """Broad. An OAuth token dict carrying no access_token is not a credential either."""
    headers = await _headers(
        tools_module,
        {"auth_type": "system_oauth"},
        extra_params={"__oauth_token__": {"access_token": ""}},
    )
    assert "Authorization" not in headers


@pytest.mark.asyncio
async def test_oauth_21_without_an_access_token_sends_no_authorization(tools_module, monkeypatch):
    """Broad. Same rule on the oauth_2.1 branch, which reads the token from the client manager."""
    monkeypatch.setattr(tools_module, "ENABLE_FORWARD_USER_INFO_HEADERS", False)

    class TokenManager:
        async def get_oauth_token(self, _user_id, _key):
            return {"access_token": ""}

    request = _request()
    request.app.state.oauth_client_manager = TokenManager()

    headers = await _headers(
        tools_module,
        {"auth_type": "oauth_2.1"},
        request=request,
        user=SimpleNamespace(id="alice"),
        server_id="srv",
    )
    assert "Authorization" not in headers


def test_no_tool_server_header_is_built_by_raw_bearer_formatting(open_webui_backend):
    """Broad. A new auth branch that formats the header itself reintroduces `Bearer ` with no
    credential; every site must route through `bearer_auth_header`."""
    source = (open_webui_backend / "open_webui" / "utils" / "tools.py").read_text(encoding="utf-8")
    offenders = [
        line.strip()
        for line in source.splitlines()
        if re.search(r"\[.Authorization.\]\s*=\s*f.Bearer", line)
    ]
    assert offenders == [], f"these sites format an unguarded bearer header: {offenders}"


@pytest.mark.asyncio
async def test_bearer_connection_with_a_key_still_sends_it(tools_module):
    """Nearby. The positive path is unchanged."""
    headers = await _headers(tools_module, {"auth_type": "bearer", "key": "server-key"})
    assert headers["Authorization"] == "Bearer server-key"


@pytest.mark.asyncio
async def test_session_auth_with_a_token_still_sends_it(tools_module):
    """Nearby."""
    request = _request()
    request.state.token.credentials = "user-jwt"
    headers = await _headers(tools_module, {"auth_type": "session"}, request=request)
    assert headers["Authorization"] == "Bearer user-jwt"


@pytest.mark.asyncio
async def test_custom_connection_headers_survive_a_keyless_connection(tools_module):
    """Broad. Dropping the auth header must not drop the custom ones next to it."""
    headers = await _headers(tools_module, {"auth_type": "bearer", "headers": {"X-Tenant": "acme"}})
    assert headers == {"X-Tenant": "acme"}


# ═════════════════════════════════════════════════════════════════════════════
# 81b9afb73. An empty cached terminal list must not erase configured servers
# ═════════════════════════════════════════════════════════════════════════════


ENABLED_CONNECTION = {"id": "term-1", "name": "Term", "url": "http://term.invalid", "enabled": True}
REBUILT_SERVER = {"id": "term-1", "url": "http://term.invalid", "specs": []}


class FakeRedis:
    def __init__(self, value):
        self.value = value
        self.writes = []

    async def get(self, _key):
        return self.value

    async def set(self, key, value):
        self.writes.append((key, value))


@pytest.fixture
def terminal_env(tools_module, monkeypatch):
    """Terminal connections from config, with the two network boundaries stubbed."""
    connections = []

    async def config_get(_key, default=None):
        return connections

    async def fetch_servers_data(server_configs):
        return [REBUILT_SERVER for _ in server_configs]

    async def fetch_system_prompt(_url, _headers):
        return None

    monkeypatch.setattr(tools_module, "Config", SimpleNamespace(get=config_get))
    monkeypatch.setattr(tools_module, "get_tool_servers_data", fetch_servers_data)
    monkeypatch.setattr(tools_module, "get_terminal_system_prompt", fetch_system_prompt)
    return connections


def _request_with_cache(cached):
    request = _request()
    request.app.state.redis = FakeRedis(cached)
    return request


@pytest.mark.asyncio
async def test_empty_cached_list_does_not_erase_configured_terminal_servers(
    tools_module, terminal_env
):
    """Narrow. Pre-fix the cached `[]` was written straight into app state and returned, so a
    configured terminal server disappeared from every worker that read the stale cache."""
    terminal_env.append(dict(ENABLED_CONNECTION))
    request = _request_with_cache("[]")

    result = await tools_module.get_terminal_servers(request)

    assert result == [REBUILT_SERVER]
    assert request.app.state.TERMINAL_SERVERS == [REBUILT_SERVER]


@pytest.mark.asyncio
async def test_rebuild_after_an_empty_cache_repopulates_the_cache(tools_module, terminal_env):
    """Narrow. The rebuild has to write back, otherwise every request rebuilds forever."""
    terminal_env.append(dict(ENABLED_CONNECTION))
    request = _request_with_cache("[]")

    await tools_module.get_terminal_servers(request)

    assert [key for key, _ in request.app.state.redis.writes] == [
        f"{tools_module.REDIS_KEY_PREFIX}:terminal_servers"
    ]


@pytest.mark.asyncio
async def test_connection_without_a_url_does_not_force_a_rebuild(tools_module, terminal_env):
    """Nearby. Only a connection that could actually yield a server justifies distrusting the
    cached empty list."""
    terminal_env.append({"id": "term-1", "name": "Term", "enabled": True})
    request = _request_with_cache("[]")

    assert await tools_module.get_terminal_servers(request) == []
    assert request.app.state.redis.writes == []


@pytest.mark.asyncio
async def test_disabled_connection_does_not_force_a_rebuild(tools_module, terminal_env):
    """Nearby. A disabled connection produces no server, so the empty cache is correct."""
    terminal_env.append({**ENABLED_CONNECTION, "enabled": False})
    request = _request_with_cache("[]")

    assert await tools_module.get_terminal_servers(request) == []
    assert request.app.state.TERMINAL_SERVERS == []


@pytest.mark.asyncio
async def test_empty_cache_with_no_connections_is_still_a_hit(tools_module, terminal_env):
    """Nearby. The 0.11.1 fix that made an empty cached list a hit must survive."""
    request = _request_with_cache("[]")

    assert await tools_module.get_terminal_servers(request) == []
    assert request.app.state.redis.writes == []


@pytest.mark.asyncio
async def test_populated_cache_is_returned_without_rebuilding(tools_module, terminal_env):
    """Nearby. A non-empty cached list is trusted whatever the connections say."""
    terminal_env.append(dict(ENABLED_CONNECTION))
    cached = '[{"id": "term-1", "url": "http://term.invalid", "specs": [], "system_prompt": "hi"}]'
    request = _request_with_cache(cached)

    result = await tools_module.get_terminal_servers(request)

    assert result[0]["system_prompt"] == "hi"
    assert request.app.state.redis.writes == []


@pytest.mark.asyncio
async def test_absent_cache_key_rebuilds(tools_module, terminal_env):
    """Nearby. `None` means unpopulated, which is a miss on both refs."""
    terminal_env.append(dict(ENABLED_CONNECTION))
    request = _request_with_cache(None)

    assert await tools_module.get_terminal_servers(request) == [REBUILT_SERVER]
