"""Journey: ENABLE_TOOLS, ENABLE_FUNCTIONS and ENABLE_TOOL_SERVERS each switch off one plugin kind.

Since 0.12.0 an admin can turn off workspace Tools, Functions or external tool servers (OpenAPI,
MCP and terminal servers) on their own, under the older ENABLE_PLUGINS. Each switch hides and
withholds its own kind and leaves the other two working. One instance per switch joins the
shared instance's database, so the tool, the global filter and the servers the shared admin
saved are all there for it to hide or keep. `/api/config` reports the switches to the web client.

Discriminates: passes on dev 948432b43. In backend copies of it: with the ENABLE_TOOLS checks
dropped from the tools router, the chat's tool resolution and the tool loader, the Tools test
fails on the listed workspace tool (and on the offered one with the router check kept); with the
ENABLE_FUNCTIONS checks dropped from the functions router, the filter run and the plugin loader,
the three functions lists and the filter test fail; with the ENABLE_TOOL_SERVERS checks dropped
from the terminals, configs and tools routers and the tool server resolution, the listed and
offered, terminal and six config route tests fail; with `/api/config` reporting all three on,
the three switched-off config tests fail. Every other test passes under each of these.
"""

from __future__ import annotations

import contextlib
import dataclasses
import secrets
from dataclasses import dataclass
from typing import Callable

import pytest

from harness import upstream as reply
from harness.actors import Actor
from harness.chat import ask
from harness.instance import LaunchedInstance
from harness.listener import json_answer
from harness.mcp_server import TOOL_SERVERS, mcp_connection, serving_mcp
from harness.openapi_server import calls_to, openapi_connection, serve_openapi
from harness.plugins import installed_function
from harness.python_tools import EVERYONE_READS, python_tool
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, serving_terminal
from harness.tool_calls import offered_tools, run_tool

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source, pytest.mark.slow]

TOOLS, FUNCTIONS, TOOL_SERVERS_SWITCH = "ENABLE_TOOLS", "ENABLE_FUNCTIONS", "ENABLE_TOOL_SERVERS"
SWITCHES = [TOOLS, FUNCTIONS, TOOL_SERVERS_SWITCH]
# None is the shared instance, with every switch on
EVERY_INSTANCE = [None, *SWITCHES]

TIDE_MARK = "high water at 12:04"
TOOL_SOURCE = f'''
class Tools:
    def tide_mark(self) -> str:
        """Read the harbour's tide mark."""
        return "{TIDE_MARK}"
'''
FILTER_MARK = "[seen by the harbour filter]"
FILTER_SOURCE = f"""
class Filter:
    def inlet(self, body: dict) -> dict:
        body["messages"][-1]["content"] += " {FILTER_MARK}"
        return body
"""


@pytest.fixture(scope="module")
def switched_off(instance, instance_with) -> Callable[[str | None], LaunchedInstance]:
    """`switched_off(TOOLS)` is an instance with that switch off, on the shared database."""

    def boot(switch: str | None) -> LaunchedInstance:
        if switch is None:
            return instance
        return instance_with(
            {
                switch: "false",
                "DATABASE_URL": instance.database_url,
                # the shared provider answers whichever connection setting the database holds
                "OPENAI_API_BASE_URL": instance.upstream.base_url,
                "OPENAI_API_BASE_URLS": instance.upstream.base_url,
            }
        )

    return boot


def _on(target: LaunchedInstance, actor: Actor) -> Actor:
    return dataclasses.replace(actor, base_url=target.base_url)


@dataclass(frozen=True)
class Plugins:
    tool_id: str
    filter_id: str
    openapi_id: str
    mcp_id: str
    mcp_tool: str
    terminal_id: str
    terminal: object


@pytest.fixture
def plugins(admin, preserve, listener) -> Plugins:
    """A workspace tool, a global filter and an OpenAPI, MCP and terminal server, all shared."""
    preserve(TOOL_SERVERS, TERMINAL_SERVERS_CONFIG)
    suffix = secrets.token_hex(3)
    title = f"Tide Table {suffix}"
    serve_openapi(listener, title, {"get_tides": json_answer({"high": "12:04"})})
    openapi = openapi_connection(listener.base_url, title, [EVERYONE_READS])
    mcp_id = f"harbour_{suffix}"
    with contextlib.ExitStack() as stack:
        mcp = mcp_connection(stack.enter_context(serving_mcp()), mcp_id, [EVERYONE_READS])
        terminal = stack.enter_context(serving_terminal())
        terminal_connection = terminal.connection(config={"access_grants": [EVERYONE_READS]})
        terminal.route("GET", "/x", json_answer({"reached": True}))
        terminal.route("GET", "/api/config", json_answer({}))
        with admin.client() as client:
            saved = client.post(TOOL_SERVERS[1], json={"TOOL_SERVER_CONNECTIONS": [openapi, mcp]})
            assert saved.status_code == 200, saved.text
            configure_terminals(client, terminal_connection)
        tool_id = stack.enter_context(python_tool(admin, TOOL_SOURCE, name=f"Tides {suffix}"))
        filter_id = stack.enter_context(installed_function(admin, FILTER_SOURCE, is_global=True))
        terminal.clear()
        yield Plugins(
            tool_id=tool_id,
            filter_id=filter_id,
            openapi_id=f"server:{openapi['info']['id']}",
            mcp_id=f"server:mcp:{mcp_id}",
            mcp_tool=f"{mcp_id}_echo",
            terminal_id=terminal_connection["id"],
            terminal=terminal,
        )


# --- what /api/config tells the web client ----------------------------------------------


@pytest.mark.parametrize("switch", EVERY_INSTANCE)
def test_the_config_reports_which_plugin_kinds_are_on(switched_off, admin, plugins, switch):
    target = switched_off(switch)
    with _on(target, admin).client() as client:
        features = client.get("/api/config").json()["features"]

    reported = {
        name: features.get(name)
        for name in (
            "enable_plugins",
            "enable_tools",
            "enable_functions",
            "enable_tool_servers",
            "enable_terminal_servers",
        )
    }
    assert reported == {
        "enable_plugins": True,
        "enable_tools": switch != TOOLS,
        "enable_functions": switch != FUNCTIONS,
        "enable_tool_servers": switch != TOOL_SERVERS_SWITCH,
        "enable_terminal_servers": switch != TOOL_SERVERS_SWITCH,
    }, f"with {switch or 'every switch on'} the web client is told {reported}"


# --- workspace tools ---------------------------------------------------------------------


@pytest.mark.parametrize("switch", EVERY_INSTANCE)
def test_workspace_tools_are_listed_and_run_unless_tools_are_off(
    switched_off, admin, upstream, plugins, switch
):
    target = switched_off(switch)
    with _on(target, admin).client() as client:
        listed = client.get("/api/v1/tools/list")
        assert listed.status_code == 200, listed.text
        listed_ids = {tool["id"] for tool in listed.json()}
        if switch == TOOLS:
            offered = offered_tools(client, upstream, tool_ids=[plugins.tool_id])
        else:
            result = run_tool(client, upstream, "tide_mark", {}, tool_ids=[plugins.tool_id])

    if switch == TOOLS:
        assert plugins.tool_id not in listed_ids, "Tools are off, yet the workspace tool is listed"
        assert "tide_mark" not in offered, (
            f"Tools are off, yet the model was offered the workspace tool: {sorted(offered)}"
        )
    else:
        assert plugins.tool_id in listed_ids, (
            f"with {switch or 'every switch on'} the workspace tool is not listed"
        )
        assert TIDE_MARK in result, (
            f"with {switch or 'every switch on'} the tool returned {result!r}"
        )


# --- functions ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path", ["/api/v1/functions/", "/api/v1/functions/list", "/api/v1/functions/export"]
)
@pytest.mark.parametrize("switch", EVERY_INSTANCE)
def test_functions_are_listed_unless_functions_are_off(switched_off, admin, plugins, switch, path):
    with _on(switched_off(switch), admin).client() as client:
        listed = client.get(path)

    assert listed.status_code == 200, listed.text
    listed_ids = {function["id"] for function in listed.json()}
    if switch == FUNCTIONS:
        assert listed_ids == set(), f"Functions are off, yet {path} lists {sorted(listed_ids)}"
    else:
        assert plugins.filter_id in listed_ids, (
            f"with {switch or 'every switch on'} {path} leaves out the filter: {sorted(listed_ids)}"
        )


@pytest.mark.parametrize("switch", EVERY_INSTANCE)
def test_a_global_filter_runs_unless_functions_are_off(
    switched_off, admin, upstream, plugins, switch
):
    prompt = f"how is the tide? ({secrets.token_hex(4)})"
    upstream.queue(reply.text("calm", match=reply.answering(prompt)))
    with _on(switched_off(switch), admin).client() as client:
        ask(client, prompt)

    [sent] = [request for request in upstream.chat_requests() if reply.answering(prompt)(request)]
    asked = [entry for entry in sent["messages"] if entry["role"] == "user"][-1]["content"]
    if switch == FUNCTIONS:
        assert FILTER_MARK not in asked, f"Functions are off, yet the filter ran: {asked!r}"
    else:
        assert FILTER_MARK in asked, (
            f"with {switch or 'every switch on'} the global filter did not run: {asked!r}"
        )


# --- tool servers ------------------------------------------------------------------------


@pytest.mark.parametrize("switch", EVERY_INSTANCE)
def test_tool_servers_are_listed_and_offered_unless_tool_servers_are_off(
    switched_off, admin, upstream, plugins, switch
):
    with _on(switched_off(switch), admin).client() as client:
        listed = client.get("/api/v1/tools/")
        assert listed.status_code == 200, listed.text
        offered = offered_tools(client, upstream, tool_ids=[plugins.openapi_id, plugins.mcp_id])

    listed_ids = {tool["id"] for tool in listed.json()}
    servers = {plugins.openapi_id, plugins.mcp_id}
    tools = {"get_tides", plugins.mcp_tool}
    if switch == TOOL_SERVERS_SWITCH:
        assert not servers & listed_ids, f"tool servers are off, yet {servers & listed_ids} listed"
        assert not tools & offered, (
            f"tool servers are off, yet the model was offered {tools & offered}"
        )
    else:
        state = switch or "every switch on"
        assert servers <= listed_ids, (
            f"with {state} the tools list leaves out {servers - listed_ids}"
        )
        assert tools <= offered, f"with {state} the model was not offered {tools - offered}"


@pytest.mark.parametrize("switch", [None, TOOLS, FUNCTIONS])
def test_tool_server_tools_run_with_another_switch_off(
    switched_off, admin, upstream, listener, plugins, switch
):
    with _on(switched_off(switch), admin).client() as client:
        tides = run_tool(client, upstream, "get_tides", {}, tool_ids=[plugins.openapi_id])
        echoed = run_tool(
            client, upstream, plugins.mcp_tool, {"text": "ahoy"}, tool_ids=[plugins.mcp_id]
        )

    state = switch or "every switch on"
    assert "12:04" in tides, f"with {state} the OpenAPI tool returned {tides!r}"
    assert calls_to(listener, "get_tides"), f"with {state} the OpenAPI server never got the call"
    assert "ahoy" in echoed, f"with {state} the MCP tool returned {echoed!r}"


@pytest.mark.parametrize("switch", EVERY_INSTANCE)
def test_terminal_servers_are_listed_and_reached_unless_tool_servers_are_off(
    switched_off, admin, plugins, switch
):
    with _on(switched_off(switch), admin).client() as client:
        listed = client.get("/api/v1/terminals/")
        proxied = client.get(f"/api/v1/terminals/{plugins.terminal_id}/x")

    assert listed.status_code == 200, listed.text
    listed_ids = {terminal["id"] for terminal in listed.json()}
    reached = plugins.terminal.requests_to("/x")
    if switch == TOOL_SERVERS_SWITCH:
        assert listed_ids == set(), f"tool servers are off, yet terminals are listed: {listed_ids}"
        assert proxied.status_code == 403, f"the terminal proxy answered {proxied.status_code}"
        assert reached == [], "tool servers are off, yet the proxy reached the terminal"
    else:
        state = switch or "every switch on"
        assert plugins.terminal_id in listed_ids, f"with {state} the terminal is not listed"
        assert proxied.status_code == 200, f"with {state} the proxy answered {proxied.text}"
        assert len(reached) == 1, f"with {state} the proxy did not reach the terminal"


def _calls_to_servers(plugins: Plugins, listener) -> int:
    return len(plugins.terminal.received) + len(listener.requests_to("/openapi.json"))


def _server_routes(plugins: Plugins, listener) -> dict[str, dict]:
    terminal_url = plugins.terminal.base_url
    return {
        "/api/v1/configs/tool_servers/verify": openapi_connection(listener.base_url, "Verify"),
        "/api/v1/configs/terminal_servers/verify": {"url": terminal_url, "auth_type": "none"},
        "/api/v1/configs/terminal_servers/policy": {"url": terminal_url, "policy_id": "pol1"},
        "/api/v1/configs/terminal_servers/lifecycle": {"url": terminal_url, "policy_id": "pol1"},
        "/api/v1/configs/terminal_servers/refresh": {"url": terminal_url},
        "/api/v1/configs/oauth/clients/register": {
            "url": listener.base_url,
            "client_id": "harbour",
        },
    }


SERVER_ROUTES = [
    "/api/v1/configs/tool_servers/verify",
    "/api/v1/configs/terminal_servers/verify",
    "/api/v1/configs/terminal_servers/policy",
    "/api/v1/configs/terminal_servers/lifecycle",
    "/api/v1/configs/terminal_servers/refresh",
    "/api/v1/configs/oauth/clients/register",
]


@pytest.mark.parametrize("route", SERVER_ROUTES)
def test_the_server_config_routes_refuse_while_tool_servers_are_off(
    switched_off, admin, listener, plugins, route
):
    form = _server_routes(plugins, listener)[route]
    with _on(switched_off(TOOL_SERVERS_SWITCH), admin).client() as client:
        before = _calls_to_servers(plugins, listener)
        refused = client.post(route, json=form)

    assert refused.status_code == 403, f"{route} answered {refused.status_code}: {refused.text}"
    assert "Tool servers are disabled" in refused.text
    assert _calls_to_servers(plugins, listener) == before, "a refused route still called a server"


@pytest.mark.parametrize("switch", [None, TOOLS, FUNCTIONS])
@pytest.mark.parametrize(
    "route", ["/api/v1/configs/tool_servers/verify", "/api/v1/configs/terminal_servers/verify"]
)
def test_the_server_config_routes_answer_with_another_switch_off(
    switched_off, admin, listener, plugins, switch, route
):
    form = _server_routes(plugins, listener)[route]
    with _on(switched_off(switch), admin).client() as client:
        before = _calls_to_servers(plugins, listener)
        verified = client.post(route, json=form)

    state = switch or "every switch on"
    assert verified.status_code == 200, f"with {state} {route} answered {verified.text}"
    assert _calls_to_servers(plugins, listener) > before, f"with {state} {route} called no server"
