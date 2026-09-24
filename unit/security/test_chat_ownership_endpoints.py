"""Regression: no route in main.py may take a chat id from the body without an ownership gate.

open-webui 0.11.0 fix `c882222f6` (PR #27486): `/api/chat/completed` and
`/api/chat/actions/{action_id}` handed a body `chat_id` to the event emitter without checking
the caller owned that chat, so a filter or action wrote into another user's conversation. The
two known routes are pinned over HTTP by integration/security/test_chat_ownership_endpoints.py;
this `ast` sweep covers the class, so the next route that reads a body `chat_id` (or forwards
the body to a handler that does) needs `verify_chat_ownership`, `is_chat_owner` or an
admin-only dependency.

Discriminates: passes on dev bbfa876af; fails with either `verify_chat_ownership` call removed
from main.py (the route shows up as ungated).
"""

from __future__ import annotations

import ast

import pytest

pytestmark = pytest.mark.regression

KNOWN_BODY_CHAT_ID_ROUTES = {"chat_completion", "chat_completed", "chat_action"}

# Handlers that pull `chat_id` out of the body they are given and emit into it.
CHAT_ID_SINKS = {"chat_completed_handler", "chat_action_handler", "get_event_emitter"}

OWNERSHIP_GATES = {"verify_chat_ownership", "is_chat_owner"}


def _called_name(call: ast.Call) -> str | None:
    return getattr(call.func, "attr", getattr(call.func, "id", None))


def _is_route(function: ast.AST) -> bool:
    return any(
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and isinstance(decorator.func.value, ast.Name)
        and decorator.func.value.id == "app"
        and decorator.func.attr in {"get", "post", "put", "patch", "delete"}
        for decorator in getattr(function, "decorator_list", [])
    )


def _reads_body_chat_id(function: ast.AST) -> bool:
    for node in ast.walk(function):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if node.slice.value == "chat_id" and getattr(node.value, "id", None) == "form_data":
                return True
        if not isinstance(node, ast.Call):
            continue
        receiver = getattr(node.func, "value", None)
        first_argument = node.args[0] if node.args else None
        if (
            getattr(receiver, "id", None) == "form_data"
            and _called_name(node) in {"get", "pop"}
            and isinstance(first_argument, ast.Constant)
            and first_argument.value == "chat_id"
        ):
            return True
        forwarded = [*node.args, *(keyword.value for keyword in node.keywords)]
        if _called_name(node) in CHAT_ID_SINKS and any(
            getattr(argument, "id", None) == "form_data" for argument in forwarded
        ):
            return True
    return False


def _is_gated(function: ast.AST) -> bool:
    names = {_called_name(node) for node in ast.walk(function) if isinstance(node, ast.Call)}
    admin_only = any(
        getattr(node, "id", None) == "get_admin_user" for node in ast.walk(function.args)
    )
    return admin_only or bool(names & OWNERSHIP_GATES)


@pytest.fixture(scope="module")
def body_chat_id_routes(open_webui_backend):
    source = (open_webui_backend / "open_webui" / "main.py").read_text(encoding="utf-8")
    routes = [node for node in ast.parse(source).body if _is_route(node)]
    return [route for route in routes if _reads_body_chat_id(route)]


def test_the_sweep_still_sees_the_known_chat_id_routes(body_chat_id_routes):
    seen = {route.name for route in body_chat_id_routes}
    assert KNOWN_BODY_CHAT_ID_ROUTES <= seen, (
        f"the sweep no longer recognises {sorted(KNOWN_BODY_CHAT_ID_ROUTES - seen)} as taking "
        "a body chat_id; retarget it before trusting the invariant below"
    )


def test_every_route_taking_a_body_chat_id_checks_ownership(body_chat_id_routes):
    ungated = [route.name for route in body_chat_id_routes if not _is_gated(route)]
    assert ungated == [], (
        f"{ungated} take a caller-supplied chat_id but never check the caller owns it, so a "
        "filter or action can write into another person's conversation (#27486)"
    )
