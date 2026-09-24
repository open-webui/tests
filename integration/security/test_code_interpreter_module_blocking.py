"""Regression: the code interpreter's module blocklist let every import through.

open-webui 0.11.0 fix `dc4b82885` (#27245): the preamble `utils/middleware.py` prepends to
interpreter code when `CODE_INTERPRETER_BLOCKED_MODULES` is set defined its
`builtins.__import__` replacement as `async def`. The import machinery calls the hook
synchronously and got back a coroutine that never ran: blocked modules were not blocked, and
every other import bound that coroutine instead of the module. The fix makes the hook a plain
`def`. `tools/builtin.py` carries its own copy of the preamble for the `execute_code` tool, so
every test runs on the `<code_interpreter>` tag path and, where it differs, the tool path.

The code interpreter runs on its Jupyter engine, served by `harness.code_interpreter`, whose
kernel runs each cell in a fresh Python process; the tests read what the cell printed from the
stored reply.

Twin of unit/security/test_code_interpreter_module_blocking.py.

Discriminates: passes on dev bbfa876af; with the hook made `async def` again in
`utils/middleware.py` the tag-path narrow and broad tests fail, and in `tools/builtin.py` the
tool-path ones (the blocked import goes through and `import json` binds a coroutine).
"""

from __future__ import annotations

import json
import re

import pytest

from harness import upstream as reply
from harness.actors import create_user
from harness.chat import ask
from harness.code_interpreter import fake_jupyter

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

CODE_EXECUTION_CONFIG = "/api/v1/configs/code_execution"
BLOCKLIST = {"CODE_INTERPRETER_BLOCKED_MODULES": "socket,xml,o,js,email.message"}
TAG_FORMAT = '<code_interpreter type="code" lang="python">'
REFUSED = "is restricted"
PATHS = ["tag", "tool"]


@pytest.fixture(scope="module")
def kernel():
    with fake_jupyter() as jupyter:
        yield jupyter


def _use_kernel(launched, kernel) -> None:
    with launched.client() as client:
        current = client.get(CODE_EXECUTION_CONFIG).json()
        on_kernel = {
            **current,
            "ENABLE_CODE_INTERPRETER": True,
            "CODE_INTERPRETER_ENGINE": "jupyter",
            "CODE_INTERPRETER_JUPYTER_URL": kernel.base_url,
            "CODE_INTERPRETER_JUPYTER_AUTH": "",
        }
        client.post(CODE_EXECUTION_CONFIG, json=on_kernel).raise_for_status()


@pytest.fixture
def blocking(instance_with, kernel):
    """An instance of this module's own with a blocklist, on the fake kernel."""
    launched = instance_with(BLOCKLIST)
    _use_kernel(launched, kernel)
    return launched


@pytest.fixture
def unblocked(instance, upstream, preserve, kernel):
    """The shared instance, which has no blocklist, on the fake kernel."""
    preserve((CODE_EXECUTION_CONFIG, CODE_EXECUTION_CONFIG))
    _use_kernel(instance, kernel)
    return instance


def _run(launched, path: str, code: str) -> str:
    """The stored reply after the model ran `code` through the tag or the execute_code tool."""
    if path == "tag":
        model_reply = reply.text(f"{TAG_FORMAT}\n{code}\n</code_interpreter>")
        function_calling = "legacy"
    else:
        model_reply = reply.tool_call("execute_code", {"code": code})
        function_calling = "native"
    launched.upstream.queue(model_reply, reply.text("done"))
    with create_user(launched).client() as client:
        _, message = ask(
            client,
            "run it",
            features={"code_interpreter": True},
            params={"function_calling": function_calling},
        )
    return json.dumps(message)


def _import_outcomes(launched, path: str, statements: list[str]) -> dict[str, str]:
    """Each statement run in the user's cell: `imported`, or the ImportError it raised."""
    cell = []
    for statement in statements:
        cell += [
            "try:",
            f"    {statement}",
            f"    print({statement!r}, '=>', 'imported', '<=')",
            "except ImportError as error:",
            f"    print({statement!r}, '=>', error, '<=')",
        ]
    stored = _run(launched, path, "\n".join(cell))
    outcomes = {}
    for statement in statements:
        reported = re.search(re.escape(statement) + r" => (.*?) <=", stored)
        assert reported, f"the cell never reported on {statement!r}: {stored[-1500:]}"
        outcomes[statement] = reported.group(1)
    return outcomes


# Narrow


@pytest.mark.parametrize("path", PATHS)
def test_a_blocked_import_is_refused(path, blocking):
    outcome = _import_outcomes(blocking, path, ["import socket"])["import socket"]
    assert REFUSED in outcome, (
        f"`import socket` with socket on the blocklist was not refused on the {path} path "
        f"({outcome!r}); the import hook never runs (#27245)"
    )


@pytest.mark.parametrize("path", PATHS)
def test_an_allowed_import_binds_the_real_module(path, blocking):
    stored = _run(blocking, path, "import json\nprint('json =>', json.dumps([1, 2]), '<=')")
    assert "json => [1, 2] <=" in stored, (
        f"`import json`, which is not blocked, gave no usable module on the {path} path, so "
        f"every legitimate import in the interpreter is broken (#27245): {stored[-1500:]}"
    )


# Broad


@pytest.mark.parametrize("path", PATHS)
def test_every_import_form_of_a_blocked_module_is_refused(path, blocking):
    statements = [
        "import socket as network",
        "from socket import create_connection",
        "import xml.dom.minidom",
        "from xml.dom import minidom",
    ]
    outcomes = _import_outcomes(blocking, path, statements)
    let_through = {statement: seen for statement, seen in outcomes.items() if REFUSED not in seen}
    assert not let_through, f"blocked modules imported on the {path} path (#27245): {let_through}"


# Nearby


def test_a_blocklist_entry_matches_whole_module_names_only(blocking):
    """`o`, `js` and `email.message` are on the list; `os`, `json` and `email` are not."""
    outcomes = _import_outcomes(blocking, "tag", ["import os", "import json", "import email"])
    refused = {statement: seen for statement, seen in outcomes.items() if seen != "imported"}
    assert not refused, f"refused for sharing a prefix with a blocklist entry: {refused}"


def test_library_code_may_still_import_a_blocked_module(blocking):
    """Only the user's own code is refused; `http.client` imports socket itself."""
    outcome = _import_outcomes(blocking, "tag", ["import http.client"])["import http.client"]
    assert outcome == "imported", outcome


def test_without_a_blocklist_nothing_is_blocked(unblocked, kernel):
    outcome = _import_outcomes(unblocked, "tag", ["import socket"])["import socket"]
    assert outcome == "imported", outcome
    assert "restricted_import" not in kernel.executed_cells()[-1]
