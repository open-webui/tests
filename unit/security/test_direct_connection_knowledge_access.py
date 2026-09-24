"""Regression: a direct-connection request could name any knowledge it liked.

open-webui 0.11.0 fix `305880f2e` (#26723): on a direct connection the model object arrives in
the request body as `model_item`. `/api/chat/completions`, `/api/chat/completed` and
`/api/chat/actions/{id}` copied it straight onto `request.state.model`, so the knowledge it
listed (collections, files, notes) was used without a read check. The fix routes all three
through `_set_direct_model`, which runs the claimed knowledge through
`get_accessible_folder_files` before installing the model.

Kept as an `ast` audit of main.py: a direct-connection completion is answered by the browser
over socket.io, so an HTTP twin needs a socket client standing in for it. Which entries the
filter keeps is pinned by unit/security/test_folder_notes_access_filter.py.

Discriminates: passes on dev bbfa876af; fails with the filtering lines removed from
`_set_direct_model`, or with any of the three routes assigning `request.state.model` itself.
"""

from __future__ import annotations

import ast

import pytest

pytestmark = pytest.mark.regression

DIRECT_MODEL_ROUTES = {"chat_completion", "chat_completed", "chat_action"}


@pytest.fixture(scope="module")
def main_functions(open_webui_backend):
    source = (open_webui_backend / "open_webui" / "main.py").read_text(encoding="utf-8")
    return {
        node.name: node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _assigns_state_model(function: ast.AST) -> bool:
    return any(
        isinstance(target, ast.Attribute)
        and target.attr == "model"
        and getattr(target.value, "attr", None) == "state"
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        for target in node.targets
    )


def _calls(function: ast.AST, name: str) -> bool:
    return any(
        isinstance(node, ast.Call) and getattr(node.func, "id", None) == name
        for node in ast.walk(function)
    )


def test_only_the_filtering_helper_installs_a_client_model(main_functions):
    installers = {name for name, node in main_functions.items() if _assigns_state_model(node)}
    assert installers == {"_set_direct_model"}, (
        f"{sorted(installers - {'_set_direct_model'})} put a browser-supplied model on "
        "request.state without filtering the knowledge it claims (#26723)"
    )

    unrouted = {
        name
        for name in DIRECT_MODEL_ROUTES
        if not _calls(main_functions[name], "_set_direct_model")
    }
    assert not unrouted, f"{sorted(unrouted)} no longer go through _set_direct_model (#26723)"


def test_the_helper_filters_the_claimed_knowledge_before_installing_it(main_functions):
    helper = main_functions.get("_set_direct_model")
    assert helper is not None, "_set_direct_model is gone from main.py; retarget this audit"

    filtered_knowledge = any(
        isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Await)
        and _calls(node.value, "get_accessible_folder_files")
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "knowledge"
            for target in node.targets
        )
        for node in ast.walk(helper)
    )
    assert filtered_knowledge, (
        "_set_direct_model no longer replaces the claimed knowledge with what "
        "get_accessible_folder_files lets the caller read, so a direct connection can name any "
        "collection, file or note (#26723)"
    )
