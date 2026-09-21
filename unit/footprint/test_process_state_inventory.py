"""Guard: process-lifetime containers in the backend are on record with their bound.

The first test makes two passes over the backend source. One lists state that starts empty
and lives for the life of the process: module-level `{}` / `[]` / `dict()` / `list()` /
`set()` (top level, or inside a top-level `if`/`try` including its handlers), the same on
plain classes, and functions memoised with `cache` or `lru_cache(maxsize=None)`. The other
lists the same shapes assigned to `app.state.X` anywhere, since those are process-wide
wherever they are written. A second test covers the caches `utils/plugin.py` creates lazily
on `app.state`. Something that starts empty is an accumulator, so anyone adding one has to
say here what empties it; the recorded bounds are documentation, the test compares names
only. Out of scope: `config.py` and `env.py` (static tables populated once at import), the
one-shot scripts under `migrations/`, containers built by a constructor call such as
`RedisDict(...)`, non-ClassVar attributes of pydantic models, memoised methods, and TTL
caches such as aiocache's `@cached`.

Unpinned: bounds as read on upstream dev at v0.11.3 (a253bf0c3). The rate limiter, the task
registry, the warned-URL set and the plugin source caches have behavioural tests in
`test_unbounded_process_state.py`; the two lock maps do not. Unmarked: nothing to pin.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

STATIC_MODULES = {"config.py", "env.py"}
SKIP_DIRS = {"migrations"}
DATA_CLASS_BASES = {"BaseModel"}
EMPTY_CALLS = {"dict", "list", "set"}

# path::name -> what bounds it
KNOWN = {
    "utils/subagents.py::_parent_locks": "unbounded: one Lock per chat id that ran a subagent",
    "utils/timers.py::_timer_locks": "unbounded: one Lock per timer id ever executed",
    "tasks.py::item_tasks": "unbounded for a falsy id; otherwise in-flight tasks",
    "tasks.py::tasks": "in-flight tasks, popped by cleanup_task",
    "tasks.py::response_streams": "in-flight tasks, popped by cleanup_task",
    "socket/main.py::SESSION_POOL": "live sockets, reaped by periodic_session_pool_cleanup",
    "socket/main.py::USAGE_POOL": "models x live sockets, reaped on disconnect",
    "utils/subagents.py::_background_active": "discarded when done; capped unless max_async is -1",
    "socket/main.py::MODELS": "never written after import; app.state.MODELS is rebound elsewhere",
    "socket/main.py::EVENT_QUEUES": "one queue per active stream channel, popped on channel end",
    "utils/redis.py::_CONNECTION_POOL": "one entry per distinct connection parameter tuple",
    "utils/plugin.py::_installed_requirements": "distinct requirement strings, never removed",
    "models/config.py::Config.DEFAULTS": "one entry per config key, replaced at boot",
    "main.py::app.state.OLLAMA_MODELS": "replaced wholesale with the upstream model list",
    "main.py::app.state.OPENAI_MODELS": "replaced wholesale with the upstream model list",
    "main.py::app.state.BASE_MODELS": "replaced wholesale with the model list",
    "main.py::app.state.TOOL_SERVERS": "replaced wholesale with configured connections",
    "main.py::app.state.TERMINAL_SERVERS": "replaced wholesale with configured connections",
    "utils/models.py::app.state.BASE_MODELS": "reset before rebuild",
    "routers/openai.py::app.state.OPENAI_MODELS": "reset before rebuild",
    "routers/openai.py::app.state.BASE_MODELS": "reset before rebuild",
    "routers/openai.py::app.state.MODELS": "reset before rebuild",
    "routers/ollama.py::app.state.OLLAMA_MODELS": "reset before rebuild",
    "routers/ollama.py::app.state.BASE_MODELS": "reset before rebuild",
    "routers/ollama.py::app.state.MODELS": "reset before rebuild",
    "utils/tools.py::get_builtin_function_introspection()": "one entry per builtin tool",
    "utils/tools.py::build_builtin_tool_spec_json()": "one entry per builtin tool",
}

# created on first use by `_state_cache(request, name)` -> what empties it
LAZY_CACHES = {
    "TOOLS": "installed tools, popped on delete",
    "FUNCTIONS": "installed functions, popped on delete",
    "TOOL_CONTENTS": "unbounded: source kept after delete",
    "FUNCTION_CONTENTS": "unbounded: source kept after delete",
}


def _empty_container_kind(node) -> str | None:
    if isinstance(node, ast.Dict) and not node.keys:
        return "dict"
    if isinstance(node, ast.List) and not node.elts:
        return "list"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in EMPTY_CALLS and not node.args and not node.keywords:
            return node.func.id
    return None


def _assigned_names(stmt):
    if isinstance(stmt, ast.Assign):
        return [t.id for t in stmt.targets if isinstance(t, ast.Name)], stmt.value
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return [stmt.target.id], stmt.value
    return [], None


def _flattened_statements(body):
    for stmt in body:
        if isinstance(stmt, ast.If):
            yield from _flattened_statements(stmt.body)
            yield from _flattened_statements(stmt.orelse)
        elif isinstance(stmt, ast.Try):
            for block in (stmt.body, stmt.orelse, stmt.finalbody, *(h.body for h in stmt.handlers)):
                yield from _flattened_statements(block)
        else:
            yield stmt


def _data_classes(trees) -> set[str]:
    """Pydantic models and every class deriving from one anywhere in the package, matched by
    bare class name: their attributes are per-instance fields."""
    bases: dict[str, set[str]] = {}
    for node in (n for tree in trees for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        bases.setdefault(node.name, set()).update(
            b.id if isinstance(b, ast.Name) else getattr(b, "attr", "") for b in node.bases
        )

    def is_data(name: str, seen: frozenset = frozenset()) -> bool:
        if name in DATA_CLASS_BASES:
            return True
        return any(is_data(b, seen | {name}) for b in bases.get(name, set()) - seen)

    return {name for name, parents in bases.items() if any(is_data(base) for base in parents)}


def _is_class_var(stmt) -> bool:
    return isinstance(stmt, ast.AnnAssign) and "ClassVar" in ast.unparse(stmt.annotation)


def _app_state_target(node: ast.Assign) -> str | None:
    """`app.state.X` or `request.app.state.X`; `request.state.X` is per request."""
    target = node.targets[0] if len(node.targets) == 1 else None
    if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Attribute)):
        return None
    state, owner = target.value, target.value.value
    owner_name = owner.id if isinstance(owner, ast.Name) else getattr(owner, "attr", "")
    return target.attr if state.attr == "state" and owner_name == "app" else None


def _unbounded_memoiser(func) -> bool:
    """`@cache`, or `@lru_cache` with `maxsize=None`; a bare `@lru_cache` keeps 128."""
    for deco in func.decorator_list:
        call = deco if isinstance(deco, ast.Call) else None
        name = call.func if call else deco
        name = name.id if isinstance(name, ast.Name) else getattr(name, "attr", "")
        if name == "cache":
            return True
        if name == "lru_cache" and call:
            keyword = next((k.value for k in call.keywords if k.arg == "maxsize"), None)
            maxsize = call.args[0] if call.args else keyword
            if isinstance(maxsize, ast.Constant) and maxsize.value is None:
                return True
    return False


def _find_process_state(package_root: Path) -> dict[str, str]:
    trees = {
        path.relative_to(package_root).as_posix(): ast.parse(path.read_text(encoding="utf-8"))
        for path in sorted(package_root.rglob("*.py"))
    }
    data_classes = _data_classes(trees.values())

    found: dict[str, str] = {}
    for rel, tree in trees.items():
        if rel in STATIC_MODULES or SKIP_DIRS & set(rel.split("/")):
            continue
        for stmt in _flattened_statements(tree.body):
            names, value = _assigned_names(stmt)
            kind = _empty_container_kind(value)
            if kind:
                for name in names:
                    found[f"{rel}::{name}"] = kind
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and _unbounded_memoiser(
                stmt
            ):
                found[f"{rel}::{stmt.name}()"] = "cache"
            if isinstance(stmt, ast.ClassDef):
                is_data_class = stmt.name in data_classes
                for attr in _flattened_statements(stmt.body):
                    if is_data_class and not _is_class_var(attr):
                        continue
                    attr_names, attr_value = _assigned_names(attr)
                    attr_kind = _empty_container_kind(attr_value)
                    if attr_kind:
                        for name in attr_names:
                            found[f"{rel}::{stmt.name}.{name}"] = attr_kind

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                attr = _app_state_target(node)
                kind = _empty_container_kind(node.value)
                if attr and kind:
                    found[f"{rel}::app.state.{attr}"] = kind
    return found


def test_every_scanned_container_has_a_recorded_bound(open_webui_backend: Path):
    found = _find_process_state(open_webui_backend / "open_webui")
    unknown = sorted(set(found) - set(KNOWN))
    stale = sorted(set(KNOWN) - set(found))
    problems = []
    if unknown:
        problems.append(
            "new process-lifetime containers; record what empties each in KNOWN:\n  "
            + "\n  ".join(f"{key} ({found[key]})" for key in unknown)
        )
    if stale:
        problems.append(f"no longer in the backend, remove from KNOWN: {stale}")
    assert not problems, "\n".join(problems)


def test_lazily_created_app_state_caches_are_recorded(open_webui_backend: Path):
    source = (open_webui_backend / "open_webui" / "utils" / "plugin.py").read_text(encoding="utf-8")
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_state_cache"
    ]
    if not calls or any(
        len(call.args) != 2 or not isinstance(call.args[1], ast.Constant) for call in calls
    ):
        pytest.fail("_state_cache is gone or no longer names its cache with a literal")
    names = {call.args[1].value for call in calls}
    assert names == set(LAZY_CACHES), "record what empties each lazily created cache in LAZY_CACHES"
