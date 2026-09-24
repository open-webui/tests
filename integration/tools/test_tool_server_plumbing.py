"""Tool and tool server plumbing regressions fixed in Open WebUI 0.11.1, seen through the API.

* `b606e13da3`: `parse_docstring` matched `:param x: text` one line at a time, so only the first
  line of a multi-line parameter description reached the tool's spec, and a description that
  started on the next line was dropped.
* PR 28630 (`cd9db21c5`): each OpenAPI tool function read the `cookies` the enclosing loop bound
  last, so a bearer tool server called after a session server was sent the browser's cookies.
* PR 27757 (`fd7024f19`) and PR 27755 (`6b4131d1d`): an unreachable tool server or terminal
  server logged a full traceback per attempt, and the terminal proxy read the request body
  outside its `try`, so a client that hung up mid-body escaped the handler.

Twin of unit/tools/test_tool_server_plumbing.py.

Discriminates: passes on dev `bbfa876af`; each narrow test fails with its fix reverted (the
continuation parsing, the per-connection cookies, the connection-error arm of the spec fetch,
the proxy's connection-error arm and the body read inside the `try`, one mutation each).
"""

from __future__ import annotations

import socket
import time
import uuid
from urllib.parse import urlsplit

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.listener import json_answer
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals

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
