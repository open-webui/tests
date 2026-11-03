"""Journey: an MCP app shown in a chat reaches its server only within the bounds it was given.

New in v0.13.0 (0e0ba872a, 4ab90a6c1, "MCP Apps in chat"). A tool that declares an app page has
its result shown as that page inside the chat, and the page reaches its server through one route
of Open WebUI. The chat stores, with the tool result, a binding the server signed for that one
server and tool; the route refuses any request whose binding does not match, lets the page call
only the tools the MCP server offers to apps, and serves the page with a content policy that
allows only the sites the server declared for it.

The confirmation the chat asks for before the page calls a tool is front end only and is not
covered here.

Discriminates: passes on v0.13.0 and dev a01b2ed55; fails on v0.12.5 (no app is stored with the
tool result). On v0.13.0 in a backend copy: the binding test turns red with the binding check
removed from the route; the tool test with the app visibility check removed; the policy test with
the other site lists merged into `connect-src`, with the site check matching only a prefix (the
smuggled directive is served) and with `connect-src` open to any site when none is declared.
"""

from __future__ import annotations

import re
import secrets

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.mcp_server import (
    BERTHS_REFRESHED,
    DASHBOARD_PAGE,
    DASHBOARD_SITES,
    HARBOUR_CLOSED,
    LEDGER_PAGE,
    SMUGGLED_SITE,
    TOOL_SERVERS,
    mcp_connection,
    serving_mcp,
)
from harness.terminal_server import read_grant

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

SITE = re.compile(r"(?:https?|wss?)://[^\s;]+")


@pytest.fixture
def harbour(admin, make_user, preserve):
    """(person, server ids): two connections to one MCP app server, both usable by `person`."""
    preserve(TOOL_SERVERS)
    person = make_user()
    tag = secrets.token_hex(4)
    server_ids = [f"harbour_{tag}", f"marina_{tag}"]
    with serving_mcp(apps=True) as url:
        connections = [mcp_connection(url, sid, [read_grant(person.id)]) for sid in server_ids]
        with admin.client() as client:
            saved = client.post(TOOL_SERVERS[1], json={"TOOL_SERVER_CONNECTIONS": connections})
        assert saved.status_code == 200, saved.text
        yield person, server_ids


def apps_shown(person, upstream, calls: list[tuple[str, str]]) -> dict[tuple[str, str], dict]:
    """Have the model run each (server id, tool) in one chat; the app each result stored."""
    prompt = f"show the harbour apps {secrets.token_hex(4)}"
    upstream.queue(
        *[
            reply.tool_call(f"{server_id}_{tool}", {}, call_id=f"call_{index}")
            for index, (server_id, tool) in enumerate(calls)
        ],
        reply.text("Here they are."),
    )
    tool_ids = sorted({f"server:mcp:{server_id}" for server_id, _ in calls})
    with person.client() as client:
        _, message = ask(client, prompt, tool_ids=tool_ids)
    embeds = [
        embed
        for item in message.get("output") or []
        for embed in item.get("embeds") or []
        if isinstance(embed, dict) and embed.get("type") == "mcp_app"
    ]
    apps = {(embed["server_id"], embed["tool_name"]): embed for embed in embeds}
    assert set(apps) == set(calls), f"the chat did not store one app per tool result: {message}"
    return apps


def app_request(person, server_id: str, tool_name: str, binding: str, method: str, params: dict):
    """What the app page sends through the chat to its server."""
    form = {"tool_name": tool_name, "binding": binding, "method": method, "params": params}
    with person.client() as client:
        return client.post(f"/api/v1/tools/id/server:mcp:{server_id}/mcp/app/rpc", json=form)


def page_for(person, app: dict, server_id: str | None = None, tool_name: str | None = None):
    return app_request(
        person,
        server_id or app["server_id"],
        tool_name or app["tool_name"],
        app["binding"],
        "ui/resource",
        {},
    )


def directives(policy: str) -> dict[str, list[str]]:
    parsed = {}
    for directive in policy.split(";"):
        name, *sources = directive.split()
        parsed[name] = sources
    return parsed


def test_an_app_reaches_only_the_server_and_tool_its_result_came_from(harbour, upstream, make_user):
    person, (harbour_id, marina_id) = harbour
    apps = apps_shown(
        person, upstream, [(harbour_id, "dashboard"), (harbour_id, "ledger"), (marina_id, "ledger")]
    )
    dashboard = apps[(harbour_id, "dashboard")]

    shown = page_for(person, dashboard)
    assert shown.status_code == 200, f"the app's own page did not load: {shown.text}"
    assert shown.json()["html"] == DASHBOARD_PAGE, shown.text

    other_tool = page_for(person, dashboard, tool_name="ledger")
    assert other_tool.status_code == 403, (
        f"the dashboard's binding opened the ledger: HTTP {other_tool.status_code} "
        f"{other_tool.text}"
    )
    other_server = page_for(person, apps[(harbour_id, "ledger")], server_id=marina_id)
    assert other_server.status_code == 403, (
        "a ledger bound to one server opened on another: "
        f"HTTP {other_server.status_code} {other_server.text}"
    )
    marina_ledger = page_for(person, apps[(marina_id, "ledger")])
    assert marina_ledger.status_code == 200, marina_ledger.text
    assert marina_ledger.json()["html"] == LEDGER_PAGE, marina_ledger.text

    forged = page_for(person, {**dashboard, "binding": "0" * 64})
    assert forged.status_code == 403, (
        f"a made-up binding was accepted: HTTP {forged.status_code} {forged.text}"
    )

    stranger = make_user()
    for method, params in [("ui/resource", {}), ("tools/call", {"name": "refresh_berths"})]:
        refused = app_request(
            stranger, harbour_id, "dashboard", dashboard["binding"], method, params
        )
        assert refused.status_code in (403, 404), (
            f"{method} from an account without access to the server was answered: "
            f"HTTP {refused.status_code} {refused.text}"
        )
        assert BERTHS_REFRESHED not in refused.text, refused.text


def test_an_app_calls_only_the_tools_its_server_offers_to_apps(harbour, upstream):
    person, (harbour_id, _) = harbour
    shown = apps_shown(person, upstream, [(harbour_id, "dashboard")])
    binding = shown[(harbour_id, "dashboard")]["binding"]
    offered = {tool["function"]["name"] for tool in upstream.chat_requests()[-1]["tools"]}
    assert f"{harbour_id}_close_harbour" in offered, offered
    assert f"{harbour_id}_refresh_berths" not in offered, (
        f"the model was offered a tool meant only for the app: {offered}"
    )

    def call(tool: str, arguments: dict | None = None):
        params = {"name": tool, "arguments": arguments or {}}
        return app_request(person, harbour_id, "dashboard", binding, "tools/call", params)

    refreshed = call("refresh_berths")
    assert refreshed.status_code == 200, refreshed.text
    assert BERTHS_REFRESHED in refreshed.text, refreshed.text
    echoed = call("echo", {"text": "ahoy"})
    assert echoed.status_code == 200, f"a tool offered to both was refused: {echoed.text}"
    assert "ahoy" in echoed.text, echoed.text

    closed = call("close_harbour")
    assert closed.status_code == 403, (
        f"the app ran a tool its server offers to the model only: "
        f"HTTP {closed.status_code} {closed.text}"
    )
    assert HARBOUR_CLOSED not in closed.text, closed.text
    missing = call("scuttle_fleet")
    assert missing.status_code == 403, (
        f"the app reached a tool its server lacks: HTTP {missing.status_code} {missing.text}"
    )


def test_an_app_page_may_load_only_from_the_sites_its_server_declared(harbour, upstream):
    person, (harbour_id, _) = harbour
    apps = apps_shown(
        person, upstream, [(harbour_id, "dashboard"), (harbour_id, "ledger"), (harbour_id, "chart")]
    )

    dashboard = page_for(person, apps[(harbour_id, "dashboard")])
    assert dashboard.status_code == 200, dashboard.text
    policy = directives(dashboard.json()["csp"])
    assert policy.get("default-src") == ["'none'"], policy
    declared = {site for sites in DASHBOARD_SITES.values() for site in sites}
    allowed = set(SITE.findall(dashboard.json()["csp"]))
    assert allowed == declared, f"the page may reach sites other than those declared: {policy}"
    [resource_site] = DASHBOARD_SITES["resourceDomains"]
    for directive in ("script-src", "style-src", "img-src", "media-src", "font-src"):
        assert resource_site in policy.get(directive, []), (directive, policy)
    for directive, key in [
        ("connect-src", "connectDomains"),
        ("frame-src", "frameDomains"),
        ("base-uri", "baseUriDomains"),
    ]:
        sites = [source for source in policy.get(directive, []) if SITE.fullmatch(source)]
        assert sites == DASHBOARD_SITES[key], f"{directive} allows {sites}: {policy}"

    ledger = page_for(person, apps[(harbour_id, "ledger")])
    assert ledger.status_code == 200, ledger.text
    bare = ledger.json()["csp"]
    assert SITE.findall(bare) == [], f"a page that declared no sites may reach some: {bare}"
    for directive in ("default-src", "connect-src", "frame-src"):
        assert directives(bare).get(directive) == ["'none'"], (directive, bare)

    chart = page_for(person, apps[(harbour_id, "chart")])
    assert chart.status_code != 200, f"a page smuggling in a directive was served: {chart.text}"
    assert "evil.example" not in chart.text, chart.text
    assert SMUGGLED_SITE not in chart.text, chart.text
