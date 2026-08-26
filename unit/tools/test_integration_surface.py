"""Integration-surface regressions fixed in Open WebUI 0.11.0.

Four independent defects across the actions, tool-server, user-preview and terminal paths:

* `chat_action` in `open_webui/utils/actions.py` ran the resolved Function with no availability
  check at all, so a disabled action, an action the model does not surface, and an action on a
  model the caller cannot read were all reachable by POSTing a raw `action_id` (PR 27243,
  commit 9a6d16849).
* `resolve_schema` in `open_webui/utils/tools.py` dropped the visited-ref set when descending
  into `properties`/`items`, so a tool server whose OpenAPI description defines mutually
  referencing types recursed forever and the whole server failed to load (PR 27413,
  commit 301bf519a, issue 27239).
* `get_user_preview` in `open_webui/routers/users.py` listed only the models, knowledge bases
  and tools SHARED with the target user and omitted the ones they own (PR 27423, commit a9a3e5b,
  discussion 27407).
* Terminal call-sites each built the orchestrator policy route themselves and several did not,
  so a connection carrying a policy but a different `server_type` hit the root route and the
  orchestrator started a second unintended container. `open_webui/utils/terminals.py` now owns
  that routing (issue 26945, commit 7088d24).

Discriminates: passes on v0.11.0, fails on v0.10.2 (pre-fix runs unavailable actions, recurses
without bound on circular `$ref`s, hides owned resources from the preview, and routes a
non-orchestrator policy connection to the bare base URL).
"""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import aiohttp
import pytest

pytestmark = pytest.mark.regression


# ---------------------------------------------------------------------------
# 1. chat_action availability gate
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def actions_module(owui_module):
    return owui_module("open_webui.utils.actions")


@pytest.fixture(scope="module")
def models_utils_module(owui_module):
    """Home of the real `check_model_access`, whose DB edge the harness stubs."""
    return owui_module("open_webui.utils.models")


class _FakeFunctions:
    """Stands in for the Functions DB accessor only."""

    def __init__(self, function):
        self._function = function

    async def get_function_by_id(self, _id):
        return self._function

    async def get_function_valves_by_id(self, _id):
        return {}

    async def get_user_valves_by_id_and_user_id(self, _id, _user_id):
        return {}


def _make_request(models, direct=False, direct_model=None):
    state = SimpleNamespace(direct=direct)
    if direct_model is not None:
        state.model = direct_model
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(MODELS=models)), state=state)


async def _noop_emit(_payload):
    return None


@contextmanager
def _action_harness(
    monkeypatch, module, models_utils, *, active=True, action_type="action", model_owner="u1"
):
    """Patch every I/O boundary around `chat_action` and record action executions.

    `model_owner` decides what the real `check_model_access` sees in the models
    table, so the access hop runs for real instead of hitting a database miss.
    """
    executions: list[dict] = []

    def action(body):
        executions.append(body)
        return {"executed": True}

    function_module = SimpleNamespace(action=action)
    function_row = SimpleNamespace(id="act", type=action_type, is_active=active)

    async def get_function_module_from_cache(_request, _action_id):
        return function_module, None, None

    async def get_event_emitter(_metadata, **_kwargs):
        return _noop_emit

    async def get_event_call(_metadata, **_kwargs):
        return _noop_emit

    async def process_tool_result(_request, _action_id, data, _kind):
        return data, None, []

    monkeypatch.setattr(module, "Functions", _FakeFunctions(function_row))
    monkeypatch.setattr(module, "get_function_module_from_cache", get_function_module_from_cache)
    monkeypatch.setattr(module, "get_event_emitter", get_event_emitter)
    monkeypatch.setattr(module, "get_event_call", get_event_call)
    monkeypatch.setattr(module, "process_tool_result", process_tool_result)

    model_row = SimpleNamespace(id="m1", user_id=model_owner, base_model_id=None)
    monkeypatch.setattr(models_utils.Models, "get_model_by_id", _const(model_row))
    monkeypatch.setattr(models_utils.Groups, "get_groups_by_member_id", _const([]))
    monkeypatch.setattr(models_utils.AccessGrants, "has_access", _const(False))
    yield executions


def _form_data(model_id="m1"):
    return {"model": model_id, "chat_id": "c1", "id": "msg1", "session_id": "s1"}


@pytest.mark.asyncio
async def test_inactive_action_is_refused(actions_module, models_utils_module, monkeypatch):
    """Narrow: a disabled action must not execute."""
    user = SimpleNamespace(id="u1", role="user")
    model = {"id": "m1", "actions": [{"id": "act"}]}
    request = _make_request({"m1": model})

    # The caller owns the model, so the access hop passes and is_active decides.
    with _action_harness(
        monkeypatch, actions_module, models_utils_module, active=False
    ) as executions:
        with pytest.raises(Exception, match="Action not available"):
            await actions_module.chat_action(request, "act", _form_data(), user)

    assert executions == []


@pytest.mark.asyncio
async def test_action_not_surfaced_by_model_is_refused(
    actions_module, models_utils_module, monkeypatch
):
    """Narrow: an action the model does not list must not execute."""
    user = SimpleNamespace(id="u1", role="user")
    model = {"id": "m1", "actions": [{"id": "other"}]}
    request = _make_request({"m1": model})

    with _action_harness(monkeypatch, actions_module, models_utils_module) as executions:
        with pytest.raises(Exception, match="Action not available"):
            await actions_module.chat_action(request, "act", _form_data(), user)

    assert executions == []


@pytest.mark.asyncio
async def test_action_on_inaccessible_model_is_refused(
    actions_module, models_utils_module, monkeypatch
):
    """Narrow: the real access check runs for a model the caller cannot read.

    The model row exists and is surfaced; only its owner differs, so the refusal
    can only come from the access check.
    """
    user = SimpleNamespace(id="u1", role="user")
    model = {"id": "m1", "actions": [{"id": "act"}]}
    request = _make_request({"m1": model})

    with _action_harness(
        monkeypatch, actions_module, models_utils_module, model_owner="u2"
    ) as executions:
        with pytest.raises(Exception, match="Model not found"):
            await actions_module.chat_action(request, "act", _form_data(), user)

    assert executions == []


@pytest.mark.asyncio
async def test_actions_refused_when_plugins_disabled(
    actions_module, models_utils_module, monkeypatch
):
    """Narrow: ENABLE_PLUGINS=false takes the whole route out."""
    user = SimpleNamespace(id="u1", role="admin")
    model = {"id": "m1", "actions": [{"id": "act"}]}
    request = _make_request({"m1": model})

    with _action_harness(monkeypatch, actions_module, models_utils_module) as executions:
        monkeypatch.setattr(actions_module, "ENABLE_PLUGINS", False)
        with pytest.raises(Exception, match="Plugins are disabled"):
            await actions_module.chat_action(request, "act", _form_data(), user)

    assert executions == []


@pytest.mark.asyncio
async def test_surfaced_active_action_runs(actions_module, models_utils_module, monkeypatch):
    """Nearby: the happy path still executes the Function."""
    user = SimpleNamespace(id="u1", role="user")
    model = {"id": "m1", "actions": [{"id": "act"}]}
    request = _make_request({"m1": model})

    with _action_harness(monkeypatch, actions_module, models_utils_module) as executions:
        result = await actions_module.chat_action(request, "act", _form_data(), user)

    assert result == {"executed": True}
    assert len(executions) == 1


@pytest.mark.asyncio
async def test_sub_action_of_surfaced_function_runs(
    actions_module, models_utils_module, monkeypatch
):
    """Nearby: `foo.bar` is surfaced by the `foo` prefix."""
    user = SimpleNamespace(id="u1", role="user")
    model = {"id": "m1", "actions": [{"id": "act.sub"}]}
    request = _make_request({"m1": model})

    with _action_harness(monkeypatch, actions_module, models_utils_module) as executions:
        await actions_module.chat_action(request, "act.sub", _form_data(), user)

    assert len(executions) == 1


@pytest.mark.asyncio
async def test_admin_is_not_blocked_by_model_bound_checks(
    actions_module, models_utils_module, monkeypatch
):
    """Nearby: admins bypass the access and surfaced-id hops."""
    user = SimpleNamespace(id="admin1", role="admin")
    model = {"id": "unowned-model", "actions": []}
    request = _make_request({"unowned-model": model})

    with _action_harness(
        monkeypatch, actions_module, models_utils_module, model_owner="u2"
    ) as executions:
        await actions_module.chat_action(request, "act", _form_data("unowned-model"), user)

    assert len(executions) == 1


@pytest.mark.asyncio
async def test_direct_connection_skips_model_bound_checks(
    actions_module, models_utils_module, monkeypatch
):
    """Nearby: a direct connection carries a client-owned model."""
    user = SimpleNamespace(id="u1", role="user")
    direct_model = {"id": "direct-model", "actions": []}
    request = _make_request({"placeholder": {}}, direct=True, direct_model=direct_model)

    with _action_harness(
        monkeypatch, actions_module, models_utils_module, model_owner="u2"
    ) as executions:
        await actions_module.chat_action(request, "act", _form_data("direct-model"), user)

    assert len(executions) == 1


# ---------------------------------------------------------------------------
# 2. Circular OpenAPI $ref resolution
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tools_module(owui_module):
    return owui_module("open_webui.utils.tools")


@contextmanager
def _bounded_recursion(extra=250):
    """Cap recursion just above the current depth so the pre-fix loop dies fast."""
    depth = 0
    frame = sys._getframe()
    while frame is not None:
        depth += 1
        frame = frame.f_back

    previous = sys.getrecursionlimit()
    sys.setrecursionlimit(min(previous, depth + extra))
    try:
        yield
    finally:
        sys.setrecursionlimit(previous)


CIRCULAR_COMPONENTS = {
    "schemas": {
        "Node": {
            "type": "object",
            "properties": {"leaf": {"$ref": "#/components/schemas/Leaf"}},
        },
        "Leaf": {
            "type": "object",
            "properties": {"node": {"$ref": "#/components/schemas/Node"}},
        },
    }
}


def _circular_spec():
    return {
        "paths": {
            "/nodes": {
                "post": {
                    "operationId": "create_node",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Node"}
                            }
                        }
                    },
                }
            }
        },
        "components": CIRCULAR_COMPONENTS,
    }


def _max_depth(value, depth=0):
    if isinstance(value, dict):
        return max((_max_depth(v, depth + 1) for v in value.values()), default=depth)
    if isinstance(value, list):
        return max((_max_depth(v, depth + 1) for v in value), default=depth)
    return depth


def test_circular_ref_schema_terminates(tools_module):
    """Narrow: two types referring to each other resolve once instead of forever."""
    with _bounded_recursion():
        resolved = tools_module.resolve_schema(
            {"$ref": "#/components/schemas/Node"}, CIRCULAR_COMPONENTS
        )

    assert resolved["properties"]["leaf"]["properties"]["node"] == {}
    assert _max_depth(resolved) < 20


def test_circular_ref_server_still_yields_a_tool_list(tools_module):
    """Narrow: the server keeps loading instead of disappearing from tool selection."""
    with _bounded_recursion():
        payload = tools_module.convert_openapi_to_tool_payload(_circular_spec())

    assert [tool["name"] for tool in payload] == ["create_node"]
    assert "leaf" in payload[0]["parameters"]["properties"]


def test_non_circular_ref_still_resolves_fully(tools_module):
    """Nearby: an ordinary nested $ref is inlined."""
    components = {
        "schemas": {
            "Outer": {
                "type": "object",
                "properties": {"inner": {"$ref": "#/components/schemas/Inner"}},
            },
            "Inner": {"type": "object", "properties": {"name": {"type": "string"}}},
        }
    }

    resolved = tools_module.resolve_schema({"$ref": "#/components/schemas/Outer"}, components)

    assert resolved["properties"]["inner"]["properties"]["name"] == {"type": "string"}


@pytest.mark.parametrize("schema", [{}, None])
def test_empty_schema_resolves_to_empty_dict(tools_module, schema):
    """Nearby: empty and None inputs stay empty."""
    assert tools_module.resolve_schema(schema, CIRCULAR_COMPONENTS) == {}


# ---------------------------------------------------------------------------
# 3. Admin access preview includes owned resources
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def users_router(owui_module):
    return owui_module("open_webui.routers.users")


def _resource(resource_id, owner_id, is_active=True):
    return SimpleNamespace(id=resource_id, name=resource_id, user_id=owner_id, is_active=is_active)


@contextmanager
def _preview_harness(monkeypatch, module, *, models, knowledge, tools, shared=None):
    """Patch the preview's DB accessors; the union logic stays production's."""
    shared = shared or {}

    async def get_accessible_resource_ids(**kwargs):
        allowed = shared.get(kwargs["resource_type"], set())
        return {rid for rid in kwargs["resource_ids"] if rid in allowed}

    monkeypatch.setattr(
        module.Users, "get_user_by_id", _const(SimpleNamespace(id="u1", name="Target"))
    )
    monkeypatch.setattr(module.Groups, "get_groups_by_member_id", _const([]))
    monkeypatch.setattr(module.Models, "get_all_models", _const(models))
    monkeypatch.setattr(module.Knowledges, "get_knowledge_bases", _const(knowledge))
    monkeypatch.setattr(module.Tools, "get_tools", _const(tools))
    monkeypatch.setattr(
        module.AccessGrants, "get_accessible_resource_ids", get_accessible_resource_ids
    )
    yield


def _const(value):
    async def _call(*_args, **_kwargs):
        return value

    return _call


@pytest.mark.asyncio
async def test_preview_lists_resources_the_user_owns(users_router, monkeypatch):
    """Narrow: owned models, knowledge bases and tools appear with nothing shared."""
    admin = SimpleNamespace(id="admin1", role="admin")

    with _preview_harness(
        monkeypatch,
        users_router,
        models=[_resource("m-owned", "u1")],
        knowledge=[_resource("k-owned", "u1")],
        tools=[_resource("t-owned", "u1")],
    ):
        preview = await users_router.get_user_preview("u1", user=admin, db=None)

    assert [item["id"] for item in preview["models"]["items"]] == ["m-owned"]
    assert [item["id"] for item in preview["knowledge"]["items"]] == ["k-owned"]
    assert [item["id"] for item in preview["tools"]["items"]] == ["t-owned"]


@pytest.mark.asyncio
async def test_preview_lists_an_owned_and_shared_resource_once(users_router, monkeypatch):
    """Nearby: the union must not duplicate."""
    admin = SimpleNamespace(id="admin1", role="admin")

    with _preview_harness(
        monkeypatch,
        users_router,
        models=[_resource("m-owned", "u1")],
        knowledge=[],
        tools=[],
        shared={"model": {"m-owned"}},
    ):
        preview = await users_router.get_user_preview("u1", user=admin, db=None)

    assert [item["id"] for item in preview["models"]["items"]] == ["m-owned"]


@pytest.mark.asyncio
async def test_preview_excludes_other_peoples_unshared_resources(users_router, monkeypatch):
    """Nearby: the preview stays a preview, not a listing of everything."""
    admin = SimpleNamespace(id="admin1", role="admin")

    with _preview_harness(
        monkeypatch,
        users_router,
        models=[_resource("m-someone-else", "u2")],
        knowledge=[_resource("k-someone-else", "u2")],
        tools=[_resource("t-someone-else", "u2")],
    ):
        preview = await users_router.get_user_preview("u1", user=admin, db=None)

    assert preview["models"]["items"] == []
    assert preview["knowledge"]["items"] == []
    assert preview["tools"]["items"] == []
    assert preview["models"]["total"] == 1


@pytest.mark.asyncio
async def test_preview_excludes_an_owned_but_inactive_model(users_router, monkeypatch):
    """Nearby: inactive models stay out even when owned."""
    admin = SimpleNamespace(id="admin1", role="admin")

    with _preview_harness(
        monkeypatch,
        users_router,
        models=[_resource("m-owned", "u1", is_active=False)],
        knowledge=[],
        tools=[],
    ):
        preview = await users_router.get_user_preview("u1", user=admin, db=None)

    assert preview["models"]["items"] == []


# ---------------------------------------------------------------------------
# 4. Terminal policy routing
# ---------------------------------------------------------------------------

TERMINAL_CALL_SITES = (
    "open_webui/utils/automations.py",
    "open_webui/routers/terminals.py",
    "open_webui/utils/tools.py",
)


def _load_terminals_module(open_webui_backend):
    """Import directly: on a checkout without the module this must fail, not skip."""
    if str(open_webui_backend) not in sys.path:
        sys.path.insert(0, str(open_webui_backend))
    return importlib.import_module("open_webui.utils.terminals")


@pytest.mark.parametrize("server_type", ["orchestrator", "local", None])
def test_policy_route_is_used_regardless_of_server_type(open_webui_backend, server_type):
    """Narrow: the policy route no longer hinges on server_type."""
    terminals = _load_terminals_module(open_webui_backend)

    url = terminals.get_terminal_server_url(
        {"url": "https://term.example", "policy_id": "pol1", "server_type": server_type}
    )

    assert url == "https://term.example/p/pol1"


@pytest.mark.parametrize("policy_id", [None, "", "   "])
def test_missing_policy_keeps_the_bare_base_url(open_webui_backend, policy_id):
    """Nearby: connections without a policy keep their root route."""
    terminals = _load_terminals_module(open_webui_backend)

    url = terminals.get_terminal_server_url({"url": "https://term.example", "policy_id": policy_id})

    assert url == "https://term.example"


@pytest.mark.parametrize(
    ("policy_id", "expected"),
    [("a/b", "a%2Fb"), ("a?b", "a%3Fb"), ("../root", "..%2Froot")],
)
def test_policy_id_cannot_escape_its_path_segment(open_webui_backend, policy_id, expected):
    """Narrow: a policy id is one encoded segment."""
    terminals = _load_terminals_module(open_webui_backend)

    connection = {"url": "https://term.example/", "policy_id": policy_id}
    url = terminals.get_terminal_server_url(connection)

    assert url == f"https://term.example/p/{expected}"


def test_trailing_slashes_on_the_base_url_are_stripped(open_webui_backend):
    """Nearby: no doubled separator in the built URL."""
    terminals = _load_terminals_module(open_webui_backend)

    assert (
        terminals.get_terminal_server_url({"url": "https://term.example///", "policy_id": "pol1"})
        == "https://term.example/p/pol1"
    )


class _FakeResponse:
    status = 200

    async def text(self):
        return ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


def _fake_session_class(recorded_urls):
    class _FakeSession:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        def post(self, url, **_kwargs):
            recorded_urls.append(url)
            return _FakeResponse()

    return _FakeSession


@pytest.mark.asyncio
async def test_cwd_call_routes_through_the_policy_for_a_non_orchestrator(
    automations_module, monkeypatch
):
    """Narrow: pre-fix this connection hit the root route and started a second container."""
    recorded: list[str] = []
    monkeypatch.setattr(aiohttp, "ClientSession", _fake_session_class(recorded))

    connection = {
        "id": "srv1",
        "url": "https://term.example",
        "policy_id": "pol1",
        "server_type": "local",
    }
    app = SimpleNamespace(
        state=SimpleNamespace(config=SimpleNamespace(TERMINAL_SERVER_CONNECTIONS=[connection]))
    )

    await automations_module._set_terminal_cwd(
        app, "srv1", SimpleNamespace(id="u1"), "/work", "chat1"
    )

    assert recorded == ["https://term.example/p/pol1/files/cwd"]


@pytest.mark.asyncio
async def test_cwd_call_routes_through_the_policy_for_an_orchestrator(
    automations_module, monkeypatch
):
    """Nearby: the case that already worked keeps working."""
    recorded: list[str] = []
    monkeypatch.setattr(aiohttp, "ClientSession", _fake_session_class(recorded))

    connection = {
        "id": "srv1",
        "url": "https://term.example",
        "policy_id": "pol1",
        "server_type": "orchestrator",
    }
    app = SimpleNamespace(
        state=SimpleNamespace(config=SimpleNamespace(TERMINAL_SERVER_CONNECTIONS=[connection]))
    )

    await automations_module._set_terminal_cwd(
        app, "srv1", SimpleNamespace(id="u1"), "/work", "chat1"
    )

    assert recorded == ["https://term.example/p/pol1/files/cwd"]


@pytest.mark.asyncio
async def test_cwd_call_without_a_policy_uses_the_root_route(automations_module, monkeypatch):
    """Nearby: a policy-less connection is unchanged."""
    recorded: list[str] = []
    monkeypatch.setattr(aiohttp, "ClientSession", _fake_session_class(recorded))

    connection = {"id": "srv1", "url": "https://term.example", "server_type": "orchestrator"}
    app = SimpleNamespace(
        state=SimpleNamespace(config=SimpleNamespace(TERMINAL_SERVER_CONNECTIONS=[connection]))
    )

    await automations_module._set_terminal_cwd(
        app, "srv1", SimpleNamespace(id="u1"), "/work", "chat1"
    )

    assert recorded == ["https://term.example/files/cwd"]


def test_no_terminal_call_site_builds_the_policy_route_by_hand(open_webui_backend):
    """Broad: policy routing lives in one helper, so the next call-site cannot forget it."""
    offenders = [
        path
        for path in TERMINAL_CALL_SITES
        if "/p/" in (open_webui_backend / path).read_text(encoding="utf-8")
    ]

    assert offenders == []
