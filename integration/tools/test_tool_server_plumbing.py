"""Tool and tool server plumbing regressions fixed in Open WebUI 0.11.1, seen through the API.

* `b606e13da3`: `parse_docstring` matched `:param x: text` one line at a time, so only the first
  line of a multi-line parameter description reached the tool's spec, and a description that
  started on the next line was dropped.
* PR 28630 (`cd9db21c5`): each OpenAPI tool function read the `cookies` the enclosing loop bound
  last, so a bearer tool server called after a session server was sent the browser's cookies.
* PR 27757 (`fd7024f19`) and PR 27755 (`6b4131d1d`): an unreachable tool server or terminal
  server logged a full traceback per attempt, and the terminal proxy read the request body
  outside its `try`, so a client that hung up mid-body escaped the handler.
* `f1a64ccfc2` (issue 28568): the tool and terminal server lists cached in Redis were decoded
  unconditionally and rebuilt whenever the result was falsy. An absent key logged a decode
  error, and an empty cached list was taken for a miss, so every request fetched every server's
  spec again. These run on an instance of their own with a real Redis, whose MONITOR stream shows
  each rebuild as the list being written back.
* PR 32229 (`ee1d9255f`, issue 32213): an MCP server that failed to connect during a chat (a
  refused connection, a 401) was logged at debug level only, so at the default log level the
  chat lost its tools with nothing in the log saying why. The failure is now an error line
  naming the server, with the reason.
* Issue 34170: once an MCP server dropped the connection during a reply (a restart, say), every
  later call to it in that reply failed with "MCP client is not connected.", even after the
  server was back. The model kept retrying a server it could never reach again.

Twin of unit/tools/test_tool_server_plumbing.py. The empty cached terminal list with no terminal
able to serve stays in unit/tools/test_tool_server_auth_and_terminal_cache.py: a chat on a
terminal without a url fails before the list is read.

Discriminates: passes on dev `bbfa876af`; each narrow test fails with its fix reverted (the
continuation parsing, the per-connection cookies, the connection-error arm of the spec fetch,
the proxy's connection-error arm and the body read inside the `try`, one mutation each). On dev
ef67cc3fa, with `f1a64ccfc2` reverted the empty-list and absent-key cases fail for both lists
(the empty tool list is written back, the absent key logs a decode error); the populated and
corrupt cases pass on both. The MCP failure lines pass on dev 5032e834e and are missing with
`ee1d9255f` reverted. The MCP restart test is red on dev 782f4b705 (the call after the
restart gets "MCP client is not connected.") and green with the fix 7e88a1e7a.
"""

from __future__ import annotations

import contextlib
import json
import secrets
import socket
import threading
import time
import uuid
from urllib.parse import urlsplit

import pytest
import redis

from harness import backends
from harness import upstream as reply
from harness.actors import admin_of
from harness.chat import ask, send_message, wait_for_reply
from harness.instance import free_port
from harness.listener import json_answer
from harness.mcp_server import mcp_connection, serving_mcp
from harness.redis_monitor import recording
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    configure_terminals,
    read_grant,
    serving_terminal,
)

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

TOOL_SERVERS_CONFIG = ("/api/v1/configs/tool_servers", "/api/v1/configs/tool_servers")
DEAD_URL = "http://127.0.0.1:9"


# --- b606e13da3: every line of a parameter's description reaches the spec ----------------

DOCUMENTED_TOOL = '''
class Tools:
    def search(self, query: str, mode: str = "fast", topic: str = "", __user__: dict = None) -> str:
        """
        Search something.

        :param query: The search query.
            Keep it short.
            Quote exact phrases.
        :param mode:
            One of fast or thorough.
        :param topic: What to search about.
        :param __user__: Injected user.
            Never shown to the model.
        :return: Results.
            As text.
        """
        return "x"
'''


@pytest.fixture
def documented_tool_spec(admin):
    tool_id = f"documented_{uuid.uuid4().hex[:8]}"
    form = {"id": tool_id, "name": "Documented", "content": DOCUMENTED_TOOL, "meta": {}}
    with admin.client() as client:
        created = client.post("/api/v1/tools/create", json=form)
        assert created.status_code == 200, created.text
        yield client.get(f"/api/v1/tools/id/{tool_id}").json()["specs"][0]
        client.delete(f"/api/v1/tools/id/{tool_id}/delete")


def _description(spec: dict, parameter: str) -> str:
    return spec["parameters"]["properties"][parameter].get("description", "")


def test_a_multi_line_parameter_description_keeps_every_line(documented_tool_spec):
    assert _description(documented_tool_spec, "query") == (
        "The search query.\nKeep it short.\nQuote exact phrases."
    ), "only the first line of a multi-line :param description reached the model"


def test_a_description_that_starts_on_the_next_line_is_kept(documented_tool_spec):
    assert _description(documented_tool_spec, "mode") == "One of fast or thorough."


def test_continuation_lines_stop_at_the_next_field(documented_tool_spec):
    assert _description(documented_tool_spec, "topic") == "What to search about."
    assert "__user__" not in documented_tool_spec["parameters"]["properties"]


# --- cd9db21c5: each tool server gets its own cookies -------------------------------------

BROWSER_COOKIE = "browser-session-secret"


def _openapi(operation_id: str) -> dict:
    operation = {"operationId": operation_id, "responses": {"200": {"description": "ok"}}}
    return {
        "openapi": "3.0.0",
        "info": {"title": operation_id, "version": "1"},
        "paths": {"/lookup": {"post": operation}},
    }


@pytest.fixture
def two_tool_servers(admin, preserve, listener):
    """A bearer server, then a session server that forwards cookies, both on the listener."""
    connections = []
    for server_id, auth in (
        ("bearer-srv", {"auth_type": "bearer", "key": "server-key"}),
        ("session-srv", {"auth_type": "session", "forward_cookies": True}),
    ):
        listener.route(
            "GET", f"/{server_id}/openapi.json", json_answer(_openapi(server_id.replace("-", "_")))
        )
        listener.route("POST", f"/{server_id}/lookup", json_answer({"found": server_id}))
        connections.append(
            {
                "url": f"{listener.base_url}/{server_id}",
                "path": "openapi.json",
                "key": "",
                "config": {"enable": True},
                "info": {"id": server_id, "name": server_id},
                **auth,
            }
        )
    preserve(TOOL_SERVERS_CONFIG)
    with admin.client() as client:
        saved = client.post(TOOL_SERVERS_CONFIG[1], json={"TOOL_SERVER_CONNECTIONS": connections})
        assert saved.status_code == 200, saved.text
    return listener


def _call_both_tool_servers(owner, upstream) -> None:
    upstream.queue(
        reply.tool_call("bearer_srv", {}),
        reply.tool_call("session_srv", {}),
        reply.text("both looked up"),
    )
    with owner.client() as client:
        client.cookies.set("browser_session", BROWSER_COOKIE)
        _, message = ask(
            client, "look it up twice", tool_ids=["server:bearer-srv", "server:session-srv"]
        )
    assert message["content"].endswith("both looked up"), message


def test_a_bearer_tool_server_is_not_sent_the_browser_cookies(
    make_user, upstream, two_tool_servers
):
    _call_both_tool_servers(make_user(role="admin"), upstream)

    (bearer_call,) = two_tool_servers.requests_to("/bearer-srv/lookup")
    assert BROWSER_COOKIE not in bearer_call.headers.get("Cookie", ""), (
        "the bearer tool server was sent the browser's cookies, bound for the session server "
        "configured after it (#28630)"
    )
    assert bearer_call.headers.get("Authorization") == "Bearer server-key"


def test_a_session_tool_server_still_gets_the_cookies_and_the_token(
    make_user, upstream, two_tool_servers
):
    owner = make_user(role="admin")
    _call_both_tool_servers(owner, upstream)

    (session_call,) = two_tool_servers.requests_to("/session-srv/lookup")
    assert BROWSER_COOKIE in session_call.headers.get("Cookie", "")
    assert session_call.headers.get("Authorization") == f"Bearer {owner.token}"


# --- fd7024f19 and 6b4131d1d: an unreachable server is one log line ------------------------


def _log_lines(instance, offset: int, text: str) -> list[str]:
    return [line for line in instance.log_since(offset).splitlines() if text in line]


def test_an_unreachable_tool_server_logs_one_line_and_no_traceback(instance, admin, preserve):
    preserve(TOOL_SERVERS_CONFIG)
    dead = {
        "url": DEAD_URL,
        "path": "openapi.json",
        "auth_type": "none",
        "key": "",
        "config": {"enable": True},
        "info": {"id": "dead-srv", "name": "dead"},
    }
    offset = instance.log_size()
    with admin.client() as client:
        saved = client.post(TOOL_SERVERS_CONFIG[1], json={"TOOL_SERVER_CONNECTIONS": [dead]})
    assert saved.status_code == 200, saved.text

    assert len(_log_lines(instance, offset, "Could not fetch tool server spec")) == 1
    assert _log_lines(instance, offset, "Traceback") == [], (
        "a plain connection failure to a tool server logged a full traceback (#27757)"
    )


def test_a_tool_server_that_answers_an_error_still_logs_a_traceback(
    instance, admin, preserve, listener
):
    listener.route("GET", "/broken/openapi.json", json_answer({"detail": "boom"}, status=500))
    preserve(TOOL_SERVERS_CONFIG)
    broken = {
        "url": f"{listener.base_url}/broken",
        "path": "openapi.json",
        "auth_type": "none",
        "key": "",
        "config": {"enable": True},
        "info": {"id": "broken-srv", "name": "broken"},
    }
    offset = instance.log_size()
    with admin.client() as client:
        saved = client.post(TOOL_SERVERS_CONFIG[1], json={"TOOL_SERVER_CONNECTIONS": [broken]})
    assert saved.status_code == 200, saved.text

    assert _log_lines(instance, offset, "Traceback"), "only connection failures lose the traceback"


@pytest.fixture
def dead_terminal(instance, admin, preserve):
    """A saved terminal connection whose server is down; returns its id."""
    preserve(TERMINAL_SERVERS_CONFIG)
    terminal_id = f"terminal-{uuid.uuid4().hex[:8]}"
    connection = {
        "id": terminal_id,
        "name": "Dead terminal",
        "enabled": True,
        "url": DEAD_URL,
        "path": "/openapi.json",
        "key": "",
        "auth_type": "none",
        "forward_cookies": False,
        "config": {"access_grants": []},
    }
    with admin.client() as client:
        configure_terminals(client, connection)
    return terminal_id


def test_an_unreachable_terminal_logs_one_line_and_no_traceback(instance, admin, dead_terminal):
    offset = instance.log_size()
    with admin.client() as client:
        proxied = client.post(f"/api/v1/terminals/{dead_terminal}/api/files", json={})

    assert proxied.status_code == 502, proxied.text
    assert len(_log_lines(instance, offset, "Terminal proxy error")) == 1
    assert _log_lines(instance, offset, "Traceback") == [], (
        "a plain connection failure to a terminal server logged a full traceback (#27755)"
    )


def test_a_client_that_hangs_up_mid_body_is_handled(instance, admin, dead_terminal):
    offset = instance.log_size()
    address = urlsplit(instance.base_url)
    head = (
        f"POST /api/v1/terminals/{dead_terminal}/api/files HTTP/1.1\r\n"
        f"Host: {address.netloc}\r\nAuthorization: Bearer {admin.token}\r\n"
        "Content-Type: application/json\r\nContent-Length: 4096\r\n\r\n"
    )
    with socket.create_connection((address.hostname, address.port), timeout=10) as client:
        client.sendall(head.encode() + b'{"partial": "bo')
    # nothing announces the end of an abandoned request; two seconds bound the handler
    time.sleep(2)

    assert _log_lines(instance, offset, "Traceback") == [], (
        "the proxy read the body outside its try, so a client that hung up mid-upload raised "
        "ClientDisconnect out of the handler (#27755)"
    )


def test_an_unknown_terminal_is_still_refused(admin, dead_terminal):
    with admin.client() as client:
        proxied = client.post("/api/v1/terminals/no-such-terminal/api/files", json={})

    assert proxied.status_code == 404


# --- ee1d9255f: an MCP server that fails to connect in a chat is logged with the reason -----

# the reason as the traceback under the error line names it. mcp 2 carries its traffic over the
# vendored httpx2, and words a rejected handshake as an MCPError rather than echoing the 4xx.
MCP_FAILURES = {
    "refused": "httpx2.ConnectError",
    "unauthorized": "MCPError: Server returned an error response",
}


@contextlib.contextmanager
def _failing_mcp_url(failure: str):
    if failure == "refused":
        yield f"http://127.0.0.1:{free_port()}/mcp"
        return
    with serving_mcp(bearer_key=secrets.token_hex(8)) as url:
        yield url


@pytest.mark.parametrize("failure", list(MCP_FAILURES))
def test_an_mcp_server_that_fails_in_a_chat_is_logged_with_the_reason(
    instance, admin, make_user, preserve, upstream, failure
):
    person = make_user()
    server_id = f"lighthouse_{secrets.token_hex(4)}"
    preserve(TOOL_SERVERS_CONFIG)
    with _failing_mcp_url(failure) as url:
        connection = mcp_connection(url, server_id, [read_grant(person.id)])
        with admin.client() as client:
            saved = client.post(
                TOOL_SERVERS_CONFIG[1], json={"TOOL_SERVER_CONNECTIONS": [connection]}
            )
        assert saved.status_code == 200, saved.text

        offset = instance.log_size()
        upstream.queue(
            reply.text("answered without the tools", match=reply.answering("light the lamp"))
        )
        with person.client() as client:
            _, message = ask(client, "light the lamp", tool_ids=[f"server:mcp:{server_id}"])

    assert message["content"] == "answered without the tools", message
    log = instance.log_since(offset)
    marker = f"Failed to connect to MCP server '{server_id}'"
    error_lines = [line for line in log.splitlines() if marker in line and "ERROR" in line]
    assert error_lines, (
        f"a {failure} MCP connection in a chat left no error line at the default level "
        f"(#32213): {log}"
    )
    assert MCP_FAILURES[failure] in log[log.index(marker) :], (
        f"the error line does not say why the connection failed: {log}"
    )


# --- issue 34170: an MCP server that comes back during a reply is called again -------------


def _wait_for_chat_requests(upstream, count: int) -> None:
    deadline = time.monotonic() + 30
    while len(upstream.chat_requests()) < count:
        assert time.monotonic() < deadline, f"the model was not asked {count} times"
        time.sleep(0.05)


def test_an_mcp_server_that_restarts_mid_reply_is_called_again(
    admin, make_user, preserve, upstream
):
    person = make_user()
    server_id = f"beacon_{secrets.token_hex(4)}"
    echo = f"{server_id}_echo"
    port = free_port()
    stopped, restarted = threading.Event(), threading.Event()
    preserve(TOOL_SERVERS_CONFIG)
    asked = len(upstream.chat_requests())
    upstream.queue(
        reply.tool_call(echo, {"text": "lit"}, call_id="call_1"),
        reply.tool_call(echo, {"text": "while down"}, call_id="call_2", hold=stopped),
        reply.tool_call(echo, {"text": "lit again"}, call_id="call_3", hold=restarted),
        reply.text("the beacon is lit"),
    )
    try:
        with serving_mcp(port=port) as url:
            connection = mcp_connection(url, server_id, [read_grant(person.id)])
            with admin.client() as client:
                saved = client.post(
                    TOOL_SERVERS_CONFIG[1], json={"TOOL_SERVER_CONNECTIONS": [connection]}
                )
            assert saved.status_code == 200, saved.text
            with person.client() as client:
                turn = send_message(
                    client, "light the beacon", tool_ids=[f"server:mcp:{server_id}"]
                )
            _wait_for_chat_requests(upstream, asked + 2)
        stopped.set()
        _wait_for_chat_requests(upstream, asked + 3)
        with serving_mcp(port=port):
            restarted.set()
            with person.client() as client:
                message = wait_for_reply(client, turn)
    finally:
        stopped.set()
        restarted.set()

    assert message["content"] == "the beacon is lit", message
    results = {
        entry["tool_call_id"]: entry["content"]
        for entry in reply.conversation(upstream.chat_requests()[-1])
        if entry["role"] == "tool"
    }
    assert results["call_1"] == "lit", results
    assert results["call_3"] == "lit again", (
        f"the call after the MCP server came back did not reach it (#34170): {results}"
    )


# --- f1a64ccfc2: an empty cache is a hit, an absent one is no error ------------------------

TOOL_CACHE_KEY = "open-webui:tool_servers"
TERMINAL_CACHE_KEY = "open-webui:terminal_servers"
SAVED_SERVERS = {TOOL_CACHE_KEY: 0, TERMINAL_CACHE_KEY: 1}
DECODE_ERRORS = {
    TOOL_CACHE_KEY: "Error fetching tool_servers from Redis",
    TERMINAL_CACHE_KEY: "Error fetching terminal_servers from Redis",
}


@pytest.fixture(scope="module")
def redis_url():
    with backends.redis_server() as url:
        yield url


@pytest.fixture(scope="module")
def cached_instance(instance_with, redis_url):
    return instance_with({"REDIS_URL": redis_url})


@pytest.fixture
def cache(cached_instance, redis_url, preserve):
    """The instance's Redis, with no tool server and one terminal saved; yields (store, id)."""
    preserve(TOOL_SERVERS_CONFIG, TERMINAL_SERVERS_CONFIG, on=cached_instance)
    with serving_terminal() as terminal:
        terminal.route("GET", "/openapi.json", json_answer(_openapi("run_command")))
        connection = terminal.connection()
        with cached_instance.client() as client:
            saved = client.post(TOOL_SERVERS_CONFIG[1], json={"TOOL_SERVER_CONNECTIONS": []})
            assert saved.status_code == 200, saved.text
            configure_terminals(client, connection)
        with redis.Redis.from_url(redis_url, decode_responses=True) as store:
            yield store, connection["id"]


def list_tools(instance) -> list[str]:
    with instance.client() as client:
        listed = client.get("/api/v1/tools/")
    assert listed.status_code == 200, listed.text
    return [tool["id"] for tool in listed.json()]


def chat_on_terminal(instance, terminal_id: str) -> None:
    instance.upstream.queue(reply.text("files listed"))
    with admin_of(instance).client() as client:
        _, message = ask(client, "list my files", terminal_id=terminal_id)
    assert message["content"] == "files listed", f"the chat on the terminal failed: {message}"


def load(instance, redis_url, key: str, terminal_id: str) -> tuple[int, list[str]]:
    """Read the list behind `key` the way a request does; (times it was written back, errors)."""
    offset = instance.log_size()
    with recording(redis_url) as commands:
        if key == TOOL_CACHE_KEY:
            list_tools(instance)
        else:
            chat_on_terminal(instance, terminal_id)
    return len(commands.sent("SET", key)), _log_lines(instance, offset, DECODE_ERRORS[key])


@pytest.mark.slow
def test_an_empty_cached_tool_list_is_not_rebuilt(cached_instance, redis_url, cache):
    store, terminal_id = cache
    store.set(TOOL_CACHE_KEY, "[]")

    rebuilds, _ = load(cached_instance, redis_url, TOOL_CACHE_KEY, terminal_id)

    assert rebuilds == 0, "an empty tool server list was taken for a miss and rebuilt (#28568)"
    assert store.get(TOOL_CACHE_KEY) == "[]"


@pytest.mark.slow
@pytest.mark.parametrize("key", [TOOL_CACHE_KEY, TERMINAL_CACHE_KEY], ids=["tools", "terminals"])
def test_an_absent_cache_key_is_rebuilt_without_an_error(cached_instance, redis_url, cache, key):
    store, terminal_id = cache
    store.delete(key)

    rebuilds, errors = load(cached_instance, redis_url, key, terminal_id)

    assert rebuilds == 1
    assert errors == [], f"an absent {key} key logged a decode error (#28568): {errors}"
    assert len(json.loads(store.get(key))) == SAVED_SERVERS[key]


@pytest.mark.slow
@pytest.mark.parametrize("key", [TOOL_CACHE_KEY, TERMINAL_CACHE_KEY], ids=["tools", "terminals"])
def test_a_corrupt_cache_is_logged_and_rebuilt(cached_instance, redis_url, cache, key):
    store, terminal_id = cache
    store.set(key, "{not json")

    rebuilds, errors = load(cached_instance, redis_url, key, terminal_id)

    assert rebuilds == 1
    assert errors, "a broken cache value is worth an error line"
    assert len(json.loads(store.get(key))) == SAVED_SERVERS[key]


@pytest.mark.slow
def test_a_cached_tool_server_is_listed_without_fetching_its_spec(
    cached_instance, redis_url, cache, listener
):
    store, terminal_id = cache
    listener.route("GET", "/cached-srv/openapi.json", json_answer(_openapi("cached_lookup")))
    connection = {
        "url": f"{listener.base_url}/cached-srv",
        "path": "openapi.json",
        "auth_type": "none",
        "key": "",
        "config": {"enable": True},
        "info": {"id": "cached-srv", "name": "Cached"},
    }
    with cached_instance.client() as client:
        saved = client.post(TOOL_SERVERS_CONFIG[1], json={"TOOL_SERVER_CONNECTIONS": [connection]})
    assert saved.status_code == 200, saved.text
    cached = json.loads(store.get(TOOL_CACHE_KEY))
    assert [server["id"] for server in cached] == ["cached-srv"]
    fetches_before = len(listener.requests_to("/cached-srv/openapi.json"))

    rebuilds, _ = load(cached_instance, redis_url, TOOL_CACHE_KEY, terminal_id)

    assert "server:cached-srv" in list_tools(cached_instance)
    assert rebuilds == 0
    assert len(listener.requests_to("/cached-srv/openapi.json")) == fetches_before
