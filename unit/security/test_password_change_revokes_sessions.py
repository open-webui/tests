"""Regression: a password write that leaves the earlier sessions working.

open-webui 0.11.1 fix `21e390561` (#28725): neither password route revoked the sessions issued
under the old password, so a stolen token kept working for the whole JWT lifetime. Both routes
now call `revoke_user_tokens` after a successful write; their behaviour is pinned over HTTP in
integration/security/test_password_change_revokes_sessions.py. This audit is the broad layer:
every backend function that writes a password must revoke too, so the next one cannot ship
without it.

Discriminates: passes on dev bbfa876af, fails with `revoke_user_tokens` removed from either
password route (the audit names the function that writes without revoking).
"""

from __future__ import annotations

import ast

import pytest

pytestmark = pytest.mark.regression

PASSWORD_WRITE = "update_user_password_by_id"
REVOCATION = "revoke_user_tokens"


def _callee_name(callee: ast.expr) -> str:
    if isinstance(callee, ast.Attribute):
        return callee.attr
    return getattr(callee, "id", "")


def _called_names(function: ast.AST) -> set[str]:
    return {_callee_name(node.func) for node in ast.walk(function) if isinstance(node, ast.Call)}


def test_every_password_write_revokes_the_earlier_sessions(open_webui_backend):
    package = open_webui_backend / "open_webui"
    writers, unrevoked = [], []
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            called = _called_names(function)
            if PASSWORD_WRITE not in called:
                continue
            location = f"{path.relative_to(package)}:{function.lineno} {function.name}"
            writers.append(location)
            if REVOCATION not in called:
                unrevoked.append(location)

    assert writers, (
        f"no function calls {PASSWORD_WRITE} any more: the password write was renamed or moved, "
        "retarget this audit at its replacement"
    )
    assert not unrevoked, (
        f"{unrevoked} writes a new password without calling {REVOCATION}, so every device signed "
        "in under the old password keeps working (#28725)"
    )
