"""The initial title task must keep logging its failures above DEBUG.

open-webui/open-webui#30339 (every new chat ended its title task with `KeyError: 'model'`) is
pinned over HTTP in integration/chat/test_initial_title_background_context.py, and the only
symptom that test can see is the "Error generating initial chat title" line in the server log.
That line exists because PR #30106 raised the handler from `log.debug` to `log.exception`; if it
sinks back to DEBUG the crash is invisible again and the integration test passes on a broken
build. So this audit keeps the handler logging with the traceback.

Discriminates: passes on dev bbfa876af, fails when `run_initial_title_generation` logs with
`log.debug` again (as before PR #30106).
"""

from __future__ import annotations

import ast

import pytest

pytestmark = pytest.mark.regression


def test_initial_title_failures_are_logged_with_their_traceback(open_webui_backend):
    tree = ast.parse((open_webui_backend / "open_webui" / "main.py").read_text(encoding="utf-8"))
    handlers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_initial_title_generation"
    ]
    assert handlers, "run_initial_title_generation is gone from main.py; retarget this audit"

    logging_calls = {
        call.func.attr
        for call in ast.walk(handlers[0])
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    }
    assert "exception" in logging_calls, (
        "run_initial_title_generation no longer logs failures with log.exception, so a broken "
        "title path goes unnoticed (#29533, #30106, #30339)"
    )
