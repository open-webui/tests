"""Tool server plumbing regressions fixed in Open WebUI 0.11.1.

Four independent defects in `open_webui/utils/tools.py` and `open_webui/routers/terminals.py`:

* `parse_docstring` matched `:param x: text` line by line with `(.+)` and kept no cursor, so
  only the first line of a multi-line parameter description ever reached the model, and a
  description that started on the following line was dropped entirely (commit b606e13da3).
* `make_tool_function` in `get_tools` took `headers` but not `cookies`, so every OpenAPI tool
  closure read whichever `cookies` the enclosing loop had bound last. A bearer server called
  after a session server sent that session's browser cookies upstream (PR 28630, cd9db21c5).
* `get_tool_servers` / `get_terminal_servers` decoded the Redis cache unconditionally, so an
  unpopulated key hit `loads(None)` and logged a decode error for a cache that was merely
  empty, and an empty-but-valid cached list was treated as a miss and rebuilt on every single
  request (commit f1a64ccfc2, issue 28568).
* `get_tool_server_data` and `proxy_terminal` logged `log.exception` for plain connectivity
  failures, so an unreachable server wrote a full traceback per attempt; `proxy_terminal` also
  read the request body outside its try block, so a client disconnect escaped the handler
  (PR 27755 / 6b4131d1d, PR 27757 / fd7024f19).

Discriminates: passes on v0.11.1, fails on v0.11.0 (pre-fix drops continuation lines, leaks the
last-bound cookies into every closure, rebuilds and mis-logs an empty cache, and emits tracebacks
plus an escaping ClientDisconnect for unreachable terminal servers).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from types import SimpleNamespace

import aiohttp
import pytest
from starlette.requests import ClientDisconnect

pytestmark = pytest.mark.regression


@pytest.fixture(scope="module")
def tools_module(owui_module):
    return owui_module("open_webui.utils.tools")


@pytest.fixture(scope="module")
def terminals_module(owui_module):
    return owui_module("open_webui.routers.terminals")


@contextmanager
def capture_records(module):
    """Records straight off the module logger; caplog cannot see it past loguru's intercept."""
    records: list[logging.LogRecord] = []

    class Collector(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Collector()
    logger = module.log
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def errors(records):
    return [r for r in records if r.levelno >= logging.ERROR]


# ---------------------------------------------------------------------------
# 44. Full tool parameter descriptions
# ---------------------------------------------------------------------------


def test_multiline_param_description_keeps_continuation_lines(tools_module):
    docstring = """
    Search something.

    :param query: The search query.
        Keep it short.
        Quote exact phrases.
    :param limit: How many results.
    """

    descriptions = tools_module.parse_docstring(docstring)

    assert descriptions["query"] == "The search query.\nKeep it short.\nQuote exact phrases."
    assert descriptions["limit"] == "How many results."


def test_param_description_starting_on_the_next_line_is_kept(tools_module):
    docstring = """
    :param mode:
        One of fast or thorough.
    """

    assert tools_module.parse_docstring(docstring) == {"mode": "One of fast or thorough."}


def test_continuation_stops_at_the_next_field(tools_module):
    docstring = """
    :param path: A file path.
    :return: The file contents.
        Decoded as UTF-8.
    :raises OSError: When unreadable.
    """

    descriptions = tools_module.parse_docstring(docstring)

    assert descriptions == {"path": "A file path."}


def test_dunder_param_and_its_continuation_stay_out(tools_module):
    docstring = """
    :param topic: What to write about.
    :param __user__: Injected user.
        Never shown to the model.
    """

    descriptions = tools_module.parse_docstring(docstring)

    assert descriptions == {"topic": "What to write about."}


@pytest.mark.parametrize(
    "docstring,expected",
    [
        ("", {}),
        (None, {}),
        ("Just a summary with no fields.", {}),
        (":param a: first\n:param b: second", {"a": "first", "b": "second"}),
    ],
)
def test_parse_docstring_simple_shapes(tools_module, docstring, expected):
    assert tools_module.parse_docstring(docstring) == expected


# ---------------------------------------------------------------------------
# 93. Signed-in tool servers: each closure binds its own cookies
# ---------------------------------------------------------------------------


BROWSER_COOKIES = {"token": "session-secret"}


def openapi_spec(name):
    return {
        "name": name,
        "description": name,
        "parameters": {"type": "object", "properties": {}, "required": []},
    }


def fake_request():
    state = SimpleNamespace(
        redis=None,
        TOOL_SERVERS=[],
        TERMINAL_SERVERS=[],
        oauth_client_manager=None,
    )
    app = SimpleNamespace(state=state)
    return SimpleNamespace(
        app=app,
        cookies=dict(BROWSER_COOKIES),
        state=SimpleNamespace(token=SimpleNamespace(credentials="user-jwt")),
    )


@pytest.fixture
def tool_server_env(tools_module, monkeypatch):
    """Two OpenAPI tool servers, one bearer and one session, and a recorded call boundary."""
    calls = []

    async def record_execute(**kwargs):
        calls.append(kwargs)
        return {}, {}

    servers = [
        {
            "id": "bearer-srv",
            "idx": 0,
            "url": "http://bearer.invalid",
            "specs": [openapi_spec("fn_bearer")],
        },
        {
            "id": "session-srv",
            "idx": 1,
            "url": "http://session.invalid",
            "specs": [openapi_spec("fn_session")],
        },
    ]
    connections = [
        {"auth_type": "bearer", "key": "server-key"},
        {"auth_type": "session"},
    ]

    async def get_servers(_request):
        return servers

    async def config_get(key, default=None):
        return connections if key == "tool_server.connections" else default

    async def no_groups(_user_id):
        return []

    async def no_tools(_ids):
        return {}

    async def allow(*_args, **_kwargs):
        return True

    monkeypatch.setattr(tools_module, "ENABLE_PLUGINS", True)
    monkeypatch.setattr(tools_module, "execute_tool_server", record_execute)
    monkeypatch.setattr(tools_module, "get_tool_servers", get_servers)
    monkeypatch.setattr(tools_module, "Config", SimpleNamespace(get=config_get))
    monkeypatch.setattr(tools_module, "Groups", SimpleNamespace(get_groups_by_member_id=no_groups))
    monkeypatch.setattr(tools_module, "Tools", SimpleNamespace(get_tools_by_ids=no_tools))
    monkeypatch.setattr(tools_module, "has_connection_access", allow)
    return calls


@pytest.fixture
def tool_user():
    return SimpleNamespace(
        id="u1", name="U", email="u@example.com", role="user", profile_image_url=""
    )


@pytest.mark.asyncio
async def test_bearer_tool_server_does_not_inherit_session_cookies(
    tools_module, tool_server_env, tool_user
):
    request = fake_request()

    tools = await tools_module.get_tools(
        request, ["server:bearer-srv", "server:session-srv"], tool_user, {"__metadata__": {}}
    )

    await tools["fn_bearer"]["callable"]()

    assert tool_server_env[0]["cookies"] == {}


@pytest.mark.asyncio
async def test_session_tool_server_still_forwards_the_caller_cookies(
    tools_module, tool_server_env, tool_user
):
    request = fake_request()

    tools = await tools_module.get_tools(
        request, ["server:bearer-srv", "server:session-srv"], tool_user, {"__metadata__": {}}
    )

    await tools["fn_session"]["callable"]()

    assert tool_server_env[0]["cookies"] == BROWSER_COOKIES
    assert tool_server_env[0]["headers"]["Authorization"] == "Bearer user-jwt"


@pytest.mark.asyncio
async def test_each_tool_server_keeps_its_own_authorization_header(
    tools_module, tool_server_env, tool_user
):
    request = fake_request()

    tools = await tools_module.get_tools(
        request, ["server:bearer-srv", "server:session-srv"], tool_user, {"__metadata__": {}}
    )

    await tools["fn_bearer"]["callable"]()

    assert tool_server_env[0]["headers"]["Authorization"] == "Bearer server-key"
    assert tool_server_env[0]["url"] == "http://bearer.invalid"


# ---------------------------------------------------------------------------
# 118 + 119. An empty server cache is a hit, not a miss, and not an error
# ---------------------------------------------------------------------------


class FakeRedis:
    def __init__(self, value):
        self.value = value
        self.writes = []

    async def get(self, _key):
        return self.value

    async def set(self, key, value):
        self.writes.append((key, value))


@pytest.fixture
def cache_env(tools_module, monkeypatch):
    """Counts every rebuild by counting reads of the connections config."""
    reads = []

    async def config_get(key, default=None):
        reads.append(key)
        return []

    monkeypatch.setattr(tools_module, "Config", SimpleNamespace(get=config_get))
    return reads


def request_with_cache(value):
    request = fake_request()
    request.app.state.redis = FakeRedis(value)
    return request


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "loader,cached", [("get_tool_servers", "[]"), ("get_terminal_servers", "[]")]
)
async def test_empty_cached_server_list_is_not_rebuilt(tools_module, cache_env, loader, cached):
    request = request_with_cache(cached)

    result = await getattr(tools_module, loader)(request)

    assert result == []
    assert cache_env == []


@pytest.mark.asyncio
@pytest.mark.parametrize("loader", ["get_tool_servers", "get_terminal_servers"])
async def test_unpopulated_cache_rebuilds_without_logging_an_error(tools_module, cache_env, loader):
    request = request_with_cache(None)

    with capture_records(tools_module) as records:
        result = await getattr(tools_module, loader)(request)

    assert result == []
    assert cache_env, "an absent cache key must still trigger a rebuild"
    assert errors(records) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("loader", ["get_tool_servers", "get_terminal_servers"])
async def test_populated_cache_is_returned_without_rebuilding(tools_module, cache_env, loader):
    request = request_with_cache('[{"id": "srv", "url": "http://srv.invalid"}]')

    result = await getattr(tools_module, loader)(request)

    assert result == [{"id": "srv", "url": "http://srv.invalid"}]
    assert cache_env == []


@pytest.mark.asyncio
@pytest.mark.parametrize("loader", ["get_tool_servers", "get_terminal_servers"])
async def test_corrupt_cache_still_logs_and_rebuilds(tools_module, cache_env, loader):
    request = request_with_cache("{not json")

    with capture_records(tools_module) as records:
        result = await getattr(tools_module, loader)(request)

    assert result == []
    assert cache_env, "a genuinely broken cache value must fall back to a rebuild"
    assert errors(records), "a broken cache value is worth an error line"


@pytest.mark.asyncio
@pytest.mark.parametrize("loader", ["get_tool_servers", "get_terminal_servers"])
async def test_no_redis_rebuilds_every_call(tools_module, cache_env, loader):
    request = fake_request()

    assert await getattr(tools_module, loader)(request) == []
    assert cache_env


# ---------------------------------------------------------------------------
# 165. An unreachable server logs one line, not a stack trace
# ---------------------------------------------------------------------------


class RaisingSession:
    """Stand-in for aiohttp.ClientSession that fails the way an unreachable host does."""

    def __init__(self, error):
        self.error = error
        self.closed = False

    def __call__(self, *_args, **_kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    def get(self, *_args, **_kwargs):
        raise self.error

    async def request(self, *_args, **_kwargs):
        raise self.error

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_unreachable_tool_server_spec_logs_without_a_traceback(tools_module, monkeypatch):
    session = RaisingSession(
        aiohttp.ClientConnectionError("Cannot connect to host tools.invalid:443")
    )
    monkeypatch.setattr(tools_module.aiohttp, "ClientSession", session)

    with capture_records(tools_module) as records:
        with pytest.raises(Exception, match="Cannot connect to host"):
            await tools_module.get_tool_server_data("http://tools.invalid/openapi.json", None)

    logged = errors(records)
    assert len(logged) == 1
    assert logged[0].exc_info is None


@pytest.mark.asyncio
async def test_unexpected_tool_server_failure_still_logs_a_traceback(tools_module, monkeypatch):
    session = RaisingSession(ValueError("something structural"))
    monkeypatch.setattr(tools_module.aiohttp, "ClientSession", session)

    with capture_records(tools_module) as records:
        with pytest.raises(Exception, match="something structural"):
            await tools_module.get_tool_server_data("http://tools.invalid/openapi.json", None)

    logged = errors(records)
    assert len(logged) == 1
    assert logged[0].exc_info is not None


def terminal_proxy_request(body_error=None):
    async def body():
        if body_error is not None:
            raise body_error
        return b""

    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(oauth_manager=None)),
        method="POST",
        headers={},
        query_params="",
        cookies={},
        state=SimpleNamespace(token=SimpleNamespace(credentials="user-jwt")),
        body=body,
    )


@pytest.fixture
def terminal_proxy_env(terminals_module, monkeypatch):
    connection = {
        "id": "term-1",
        "url": "http://terminal.invalid",
        "enabled": True,
        "auth_type": "none",
    }

    async def config_get(key, default=None):
        return [connection] if key == "terminal_server.connections" else default

    async def no_groups(_user_id):
        return []

    async def allow(*_args, **_kwargs):
        return True

    monkeypatch.setattr(terminals_module, "Config", SimpleNamespace(get=config_get))
    monkeypatch.setattr(
        terminals_module, "Groups", SimpleNamespace(get_groups_by_member_id=no_groups)
    )
    monkeypatch.setattr(terminals_module, "has_connection_access", allow)
    return connection


@pytest.mark.asyncio
async def test_terminal_proxy_connection_error_logs_without_a_traceback(
    terminals_module, terminal_proxy_env, monkeypatch
):
    session = RaisingSession(
        aiohttp.ClientConnectionError("Cannot connect to host terminal.invalid:443")
    )
    monkeypatch.setattr(terminals_module.aiohttp, "ClientSession", session)

    with capture_records(terminals_module) as records:
        response = await terminals_module.proxy_terminal(
            "term-1", "api/terminals", terminal_proxy_request(), SimpleNamespace(id="u1")
        )

    assert response.status_code == 502
    logged = errors(records)
    assert len(logged) == 1
    assert logged[0].exc_info is None
    assert session.closed


@pytest.mark.asyncio
async def test_terminal_proxy_unexpected_error_still_logs_a_traceback(
    terminals_module, terminal_proxy_env, monkeypatch
):
    session = RaisingSession(ValueError("something structural"))
    monkeypatch.setattr(terminals_module.aiohttp, "ClientSession", session)

    with capture_records(terminals_module) as records:
        response = await terminals_module.proxy_terminal(
            "term-1", "api/terminals", terminal_proxy_request(), SimpleNamespace(id="u1")
        )

    assert response.status_code == 502
    logged = errors(records)
    assert len(logged) == 1
    assert logged[0].exc_info is not None


@pytest.mark.asyncio
async def test_terminal_proxy_client_disconnect_returns_499(
    terminals_module, terminal_proxy_env, monkeypatch
):
    session = RaisingSession(RuntimeError("upstream should never be reached"))
    monkeypatch.setattr(terminals_module.aiohttp, "ClientSession", session)

    request = terminal_proxy_request(body_error=ClientDisconnect())
    response = await terminals_module.proxy_terminal(
        "term-1", "api/terminals", request, SimpleNamespace(id="u1")
    )

    assert response.status_code == 499


@pytest.mark.asyncio
async def test_terminal_proxy_still_rejects_an_unknown_server(terminals_module, terminal_proxy_env):
    response = await terminals_module.proxy_terminal(
        "nope", "api/terminals", terminal_proxy_request(), SimpleNamespace(id="u1")
    )

    assert response.status_code == 404
