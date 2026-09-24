"""Regression: a reply that fails while it continues must say so, in every continuation loop.

open-webui 0.11.0 `8ab44ed` (#27426 / #27411): the `except` arms around the follow-up completion
after a tool call and after a code-interpreter run logged at debug and broke out of the loop, so
the turn ended mid-answer with nothing said. The fix reports the error through
`emit_message_error`, which stores it on the message and tells the page.

The tool-call arm is pinned end to end by integration/chat/test_middleware_stream_assembly.py and
its e2e twin. The code-interpreter arm needs a code execution engine to reach, so this audit holds
the class: every arm that abandons a follow-up completion reports the error first. The other
stream assembly regressions this file once covered moved to those two twins.

Discriminates: passes on dev `bbfa876af`; fails with either arm reverted to `log.debug(e)` plus
`break` (the tool-call arm and the code-interpreter arm, one mutation each).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression

FOLLOW_UP_CALL = "generate_chat_completion"
REPORTER = "emit_message_error"


def _calls(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(inner, ast.Call)
        and name in (getattr(inner.func, "id", None), getattr(inner.func, "attr", None))
        for inner in ast.walk(node)
    )


def _abandoning_arms(tree: ast.AST) -> list[ast.ExceptHandler]:
    """`except` arms that give up on a follow-up completion by breaking out of their loop."""
    return [
        handler
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        and any(_calls(statement, FOLLOW_UP_CALL) for statement in node.body)
        for handler in node.handlers
        if isinstance(handler.body[-1], ast.Break)
    ]


def test_every_abandoned_follow_up_reports_its_error(open_webui_backend: Path):
    source = (open_webui_backend / "open_webui" / "utils" / "middleware.py").read_text("utf-8")
    arms = _abandoning_arms(ast.parse(source))
    assert arms, (
        f"no `except` arm around a {FOLLOW_UP_CALL} follow-up breaks out of its loop any more: "
        "retarget this audit at wherever tool-call and code-interpreter continuations moved"
    )

    silent = [f"middleware.py:{arm.lineno}" for arm in arms if not _calls(arm, REPORTER)]
    assert silent == [], (
        f"these arms abandon a follow-up completion without {REPORTER}, so the reply stops "
        f"mid-answer with nothing said and nothing stored (#27411): {silent}"
    )
