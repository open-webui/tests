"""Regression: a password write that leaves the earlier sessions working.

open-webui 0.11.1 fix `21e390561` (#28725): neither password route revoked the sessions issued
under the old password, so a stolen token kept working for the whole JWT lifetime. The routes
called `revoke_user_tokens` after a successful write; since `24e30d1cb` the password write itself
rotates the session stamp every token is checked against (`revoke=True`). The behaviour is pinned
over HTTP in integration/security/test_password_change_revokes_sessions.py. This audit is the
broad layer: every backend function that writes a password must revoke, itself or through the
write it calls, so the next one cannot ship without it.

Discriminates: passes on dev b5a20423e and f6cbeb1a1, fails with `revoke=True` dropped from the
password write (the audit names both routes that then write without revoking).
"""

from __future__ import annotations

import ast

import pytest

pytestmark = pytest.mark.regression

PASSWORD_WRITE = "update_user_password_by_id"
REVOCATIONS = {"revoke_user_tokens", "revoke_sessions_by_user_id"}


def _callee_name(callee: ast.expr) -> str:
    if isinstance(callee, ast.Attribute):
        return callee.attr
    return getattr(callee, "id", "")


def _calls(function: ast.AST) -> list[ast.Call]:
    return [node for node in ast.walk(function) if isinstance(node, ast.Call)]


def _revokes(function: ast.AST) -> bool:
    return any(
        _callee_name(call.func) in REVOCATIONS
        or any(
            keyword.arg == "revoke"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in call.keywords
        )
        for call in _calls(function)
    )


def _updates_password_column(function: ast.AST) -> bool:
    return any(
        _callee_name(call.func) == "values"
        and any(keyword.arg == "password" for keyword in call.keywords)
        for call in _calls(function)
    )


def test_every_password_write_revokes_the_earlier_sessions(open_webui_backend):
    package = open_webui_backend / "open_webui"
    functions = []
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for function in ast.walk(tree):
            if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                location = f"{path.relative_to(package)}:{function.lineno} {function.name}"
                functions.append((location, function))

    write_revokes = any(
        function.name == PASSWORD_WRITE and _revokes(function) for _, function in functions
    )
    writers, unrevoked = [], []
    for location, function in functions:
        calls_write = PASSWORD_WRITE in {_callee_name(call.func) for call in _calls(function)}
        writes_password = calls_write or _updates_password_column(function)
        if function.name == PASSWORD_WRITE or not writes_password:
            continue
        writers.append(location)
        if not (_revokes(function) or (calls_write and write_revokes)):
            unrevoked.append(location)

    assert writers, (
        f"no function calls {PASSWORD_WRITE} any more: the password write was renamed or moved, "
        "retarget this audit at its replacement"
    )
    assert not unrevoked, (
        f"{unrevoked} writes a new password without revoking the earlier sessions, so every "
        "device signed in under the old password keeps working (#28725)"
    )
