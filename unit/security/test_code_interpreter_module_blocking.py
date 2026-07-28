"""Regression: the code interpreter must block the modules an admin configured, and
the Python it generates must always be syntactically valid.

Two 0.11.0 fixes, both about source the backend/frontend generates for the interpreter:

`dc4b82885` (#27245) - the blocklist preamble that `utils/middleware.py` prepends to
interpreter code defined its `builtins.__import__` replacement as `async def`. Python's
import machinery calls the hook synchronously, so it got back a coroutine that never ran:
blocked modules were not blocked, and every other import bound a dangling coroutine
instead of the module. The fix makes the hook a plain `def`.

`b940cd529` (#26800, issue #26660) - `pyodideSandboxHost.ts` (the default path, used when
ENABLE_PYODIDE_FILE_PERSISTENCE is off) embeds its script in a `String.raw` template but
wrote the matplotlib `show()` override with `\\t` escapes. `String.raw` keeps both
characters, so the sandbox's JS parser produced a literal backslash-t and Pyodide refused
to compile any chart-drawing code. The fix uses single `\t`.

Both generators live inside large functions or inside a JS template, so the tests pull the
real generated source out of the shipped files: the middleware preamble via `ast` (the
actual f-string node, evaluated with a chosen blocklist), the matplotlib patch by decoding
the shipped JS string literals the way the browser's parser would.

Discriminates: passes on v0.11.0, fails on v0.10.2 (blocked imports are not refused,
unblocked imports bind coroutines, and the default chart preamble does not compile).
"""

from __future__ import annotations

import ast
import builtins
import re
import textwrap
from contextlib import contextmanager
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression

CHART_CODE = "import matplotlib.pyplot as plt\nplt.plot([1, 2, 3])\nplt.show()\n"

JS_ESCAPES = {"t": "\t", "n": "\n", "r": "\r", "\\": "\\", "'": "'", '"': '"', "0": "\0"}


def _repo_root(open_webui_backend: Path) -> Path:
    root = open_webui_backend.parent
    if not (root / "src" / "lib" / "pyodide").is_dir():
        pytest.skip("frontend sources not present next to the backend checkout")
    return root


def _blocking_preamble(open_webui_backend: Path, blocked_modules: list[str]) -> str:
    """The real `blocking_code` f-string from middleware.py, rendered for a blocklist."""
    source = (open_webui_backend / "open_webui" / "utils" / "middleware.py").read_text(
        encoding="utf-8"
    )
    assignments = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "blocking_code"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1, (
        f"expected exactly one `blocking_code` assignment in middleware.py, found "
        f"{len(assignments)}; the extraction below no longer matches the shipped code (#27245)"
    )
    expression = ast.Expression(body=assignments[0].value)
    ast.fix_missing_locations(expression)
    return eval(
        compile(expression, "middleware.py", "eval"),
        {"textwrap": textwrap, "CODE_INTERPRETER_BLOCKED_MODULES": blocked_modules},
    )


@contextmanager
def _interpreter_namespace(open_webui_backend: Path, blocked_modules: list[str]):
    """Run the generated preamble, yield a `__main__` namespace, restore the importer."""
    real_import = builtins.__import__
    namespace = {"__name__": "__main__"}
    try:
        exec(_blocking_preamble(open_webui_backend, blocked_modules), namespace)
        yield namespace
    finally:
        builtins.__import__ = real_import


def _decode_js_string_literal(body: str) -> str:
    decoded = []
    index = 0
    while index < len(body):
        if body[index] == "\\":
            escape = body[index + 1]
            assert escape in JS_ESCAPES, f"unhandled JS escape \\{escape} in the shipped literal"
            decoded.append(JS_ESCAPES[escape])
            index += 2
        else:
            decoded.append(body[index])
            index += 1
    return "".join(decoded)


def _sandbox_host_chart_patch(repo_root: Path) -> str:
    """Python the iframe sandbox builds for matplotlib (file persistence off, the default)."""
    source = (repo_root / "src" / "lib" / "pyodide" / "pyodideSandboxHost.ts").read_text(
        encoding="utf-8"
    )
    array = re.search(r"runPythonAsync\(\[(.*?)\]\.join\('\\n'\)\)", source, re.S)
    assert array, "could not find the matplotlib patch array in pyodideSandboxHost.ts (#26800)"

    lines = []
    for raw_entry in array.group(1).splitlines():
        entry = raw_entry.strip().rstrip(",").strip()
        if not entry or entry.startswith("//"):
            continue
        literal = re.fullmatch(r"'((?:[^'\\]|\\.)*)'", entry)
        assert literal, f"unexpected entry in the matplotlib patch array: {entry!r}"
        lines.append(_decode_js_string_literal(literal.group(1)))
    return "\n".join(lines)


def _worker_chart_patch(repo_root: Path) -> str:
    """Python the dedicated worker builds for matplotlib (file persistence on)."""
    source = (repo_root / "src" / "lib" / "workers" / "pyodide.worker.ts").read_text(
        encoding="utf-8"
    )
    template = re.search(r"runPythonAsync\(`(.*?)`\)", source, re.S)
    assert template, "could not find the matplotlib patch template in pyodide.worker.ts"
    return template.group(1)


def _chart_patch(repo_root: Path, file_persistence: bool) -> str:
    if file_persistence:
        return _worker_chart_patch(repo_root)
    return _sandbox_host_chart_patch(repo_root)


# Narrow


def test_blocked_module_import_is_refused(open_webui_backend):
    with _interpreter_namespace(open_webui_backend, ["socket", "subprocess"]) as namespace:
        with pytest.raises(ImportError):
            exec("import socket", namespace)


def test_unblocked_module_import_binds_the_real_module(open_webui_backend):
    with _interpreter_namespace(open_webui_backend, ["socket"]) as namespace:
        exec("import json", namespace)
        imported = namespace["json"]
    assert imported.dumps({"a": 1}) == '{"a": 1}', (
        "an import the admin never blocked did not produce a usable module, so all "
        "legitimate interpreter imports are broken (#27245)"
    )


def test_default_chart_patch_compiles(open_webui_backend):
    """The sandbox host path is the default (ENABLE_PYODIDE_FILE_PERSISTENCE off)."""
    source = _sandbox_host_chart_patch(_repo_root(open_webui_backend))
    compile(source, "<matplotlib-patch>", "exec")


# Broad


@pytest.mark.parametrize(
    "file_persistence", [False, True], ids=["persistence-off", "persistence-on"]
)
@pytest.mark.parametrize("blocklist_set", [False, True], ids=["no-blocklist", "blocklist"])
@pytest.mark.parametrize("chart_code", [False, True], ids=["no-chart", "chart"])
def test_generated_interpreter_source_compiles_for_every_toggle(
    open_webui_backend, file_persistence, blocklist_set, chart_code
):
    repo_root = _repo_root(open_webui_backend)
    user_code = CHART_CODE if chart_code else "print(sum(range(10)))\n"

    fragments = []
    if chart_code:
        fragments.append(_chart_patch(repo_root, file_persistence))
    if blocklist_set:
        fragments.append(
            _blocking_preamble(open_webui_backend, ["os", "socket"]) + "\n" + user_code
        )
    else:
        fragments.append(user_code)

    for position, fragment in enumerate(fragments):
        try:
            compile(fragment, f"<interpreter-{position}>", "exec")
        except SyntaxError as error:
            pytest.fail(
                f"fragment {position} of the generated interpreter source does not parse "
                f"({error}); the interpreter refuses this code outright (#26800, #27245)"
            )


@pytest.mark.parametrize(
    "statement",
    ["import email", "from email import message", "import email.message as message_module"],
)
def test_every_import_form_of_a_blocked_module_is_refused(open_webui_backend, statement):
    with _interpreter_namespace(open_webui_backend, ["email"]) as namespace:
        with pytest.raises(ImportError):
            exec(statement, namespace)


# Nearby


@pytest.mark.parametrize(
    "blocked_modules, statement",
    [
        (["o"], "import os"),
        (["js"], "import json"),
        (["email.message"], "import email"),
    ],
)
def test_blocklist_matches_whole_module_names_not_prefixes(
    open_webui_backend, blocked_modules, statement
):
    with _interpreter_namespace(open_webui_backend, blocked_modules) as namespace:
        exec(statement, namespace)


def test_empty_blocklist_blocks_nothing(open_webui_backend):
    with _interpreter_namespace(open_webui_backend, []) as namespace:
        exec("import os", namespace)


def test_blocked_module_stays_importable_for_library_code(open_webui_backend):
    """Only `__main__` (the user's own code) is refused; libraries may still import it."""
    with _interpreter_namespace(open_webui_backend, ["socket"]) as namespace:
        exec("import socket", dict(namespace, __name__="some_library"))
