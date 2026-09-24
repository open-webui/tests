"""Guard: process-lifetime containers in the backend are on record with their bound.

One pass over the backend source lists state that starts empty and lives for the life of the
process: module-level `{}` / `[]` / `dict()` / `list()` / `set()` (top level, or inside a
top-level `if`/`try` including its handlers), the same on plain classes, functions memoised with
`cache` or `lru_cache(maxsize=None)`, and every cache created on `app.state`, either assigned
(`app.state.X = {}`) or created on first use by a helper that does `setattr(<...>.state, name,
{})` and is called with a literal name. Something that starts empty is an accumulator, so anyone
adding one has to say here what empties it; the recorded bounds are documentation, the test
compares names only. Out of scope: `config.py` and `env.py` (static tables populated once at
import), the one-shot scripts under `migrations/`, containers built by a constructor call such
as `RedisDict(...)`, non-ClassVar attributes of pydantic models, memoised methods, and TTL
caches such as aiocache's `@cached`.

The ratchet is one-directional: a new name fails, a name upstream removes passes. Names are
recorded without their file, so moving a container to another module is not a new one.

Unpinned: bounds as read on upstream dev at v0.11.3 (a253bf0c3), updated for #29983. The task
registry, the warned-URL set and the plugin source caches have behavioural tests in
`test_unbounded_process_state.py` and `integration/footprint/`; the two lock maps do not.
Discriminates: a module-level `{}` or a new lazily created `app.state` cache added to a copy of
dev bbfa876af fails; deleting a recorded container from the copy passes.
"""

from __future__ import annotations

import ast
from pathlib import Path

STATIC_MODULES = {"config.py", "env.py"}
SKIP_DIRS = {"migrations"}
DATA_CLASS_BASES = {"BaseModel"}
EMPTY_CALLS = {"dict", "list", "set"}

# name -> where it lives and what bounds it
KNOWN = {
    "_parent_locks": "utils/subagents.py, unbounded: one Lock per chat id that ran a subagent",
    "_timer_locks": "utils/timers.py, unbounded: one Lock per timer id ever executed",
    "item_tasks": "tasks.py, in-flight tasks per item (#29980 dropped the empty id)",
    "tasks": "tasks.py, in-flight tasks, popped by cleanup_task",
    "response_streams": "tasks.py, in-flight tasks, popped by cleanup_task",
    "SESSION_POOL": "socket/main.py, live sockets, reaped by periodic_session_pool_cleanup",
    "USAGE_POOL": "socket/main.py, models x live sockets, reaped on disconnect",
    "_background_active": "utils/subagents.py, discarded when done; capped unless max_async -1",
    "MODELS": "socket/main.py, never written after import; app.state.MODELS is rebound",
    "EVENT_QUEUES": "socket/main.py, one queue per active stream channel, popped on its end",
    "_CONNECTION_POOL": "utils/redis.py, one entry per distinct connection parameter tuple",
    "_installed_requirements": "utils/plugin.py, distinct requirement strings, never removed",
    "Config.DEFAULTS": "models/config.py, one entry per config key, replaced at boot",
    "get_builtin_function_introspection()": "utils/tools.py, one entry per builtin tool",
    "build_builtin_tool_spec_json()": "utils/tools.py, one entry per builtin tool",
    "app.state.OLLAMA_MODELS": "replaced wholesale with the upstream model list",
    "app.state.OPENAI_MODELS": "replaced wholesale with the upstream model list",
    "app.state.BASE_MODELS": "replaced wholesale with the model list",
    "app.state.MODELS": "reset before every rebuild",
    "app.state.TOOL_SERVERS": "replaced wholesale with configured connections",
    "app.state.TERMINAL_SERVERS": "replaced wholesale with configured connections",
    "app.state.TOOLS": "utils/plugin.py, installed tools, popped on delete",
    "app.state.FUNCTIONS": "utils/plugin.py, installed functions, popped on delete",
    "app.state.TOOL_CONTENTS": "utils/plugin.py, installed tool sources, popped on delete",
    "app.state.FUNCTION_CONTENTS": "utils/plugin.py, installed function sources, popped on delete",
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


def _is_app_state(node) -> bool:
    """`app.state` or `request.app.state`; `request.state` is per request."""
    if not (isinstance(node, ast.Attribute) and node.attr == "state"):
        return False
    owner = node.value
    return (owner.id if isinstance(owner, ast.Name) else getattr(owner, "attr", "")) == "app"


def _call_name(call: ast.Call) -> str:
    return call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", "")


def _state_setattr(node) -> ast.Call | None:
    """`setattr(<...>.app.state, name, <empty container>)`."""
    if isinstance(node, ast.Call) and _call_name(node) == "setattr" and len(node.args) == 3:
        target, _, value = node.args
        if _is_app_state(target) and _empty_container_kind(value):
            return node
    return None


def _lazy_cache_factories(trees) -> dict[str, int]:
    """Functions that create an `app.state` cache named by one of their parameters, with the
    position of that parameter."""
    factories = {}
    for function in (n for tree in trees for n in ast.walk(tree)):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        parameters = [argument.arg for argument in function.args.args]
        for node in ast.walk(function):
            call = _state_setattr(node)
            name = call.args[1] if call else None
            if isinstance(name, ast.Name) and name.id in parameters:
                factories[function.name] = parameters.index(name.id)
    return factories


def _app_state_caches(tree, factories: dict[str, int]) -> set[str]:
    caches = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _empty_container_kind(node.value):
            caches |= {
                target.attr
                for target in node.targets
                if isinstance(target, ast.Attribute) and _is_app_state(target.value)
            }
        if not isinstance(node, ast.Call):
            continue
        direct = _state_setattr(node)
        position = 1 if direct else factories.get(_call_name(node))
        name = node.args[position] if position is not None and len(node.args) > position else None
        if isinstance(name, ast.Constant) and isinstance(name.value, str):
            caches.add(name.value)
    return {f"app.state.{name}" for name in caches}


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


def _module_state(tree, data_classes: set[str]) -> set[str]:
    found = set()
    for stmt in _flattened_statements(tree.body):
        names, value = _assigned_names(stmt)
        if _empty_container_kind(value):
            found.update(names)
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and _unbounded_memoiser(stmt):
            found.add(f"{stmt.name}()")
        if isinstance(stmt, ast.ClassDef):
            for attr in _flattened_statements(stmt.body):
                if stmt.name in data_classes and not _is_class_var(attr):
                    continue
                attr_names, attr_value = _assigned_names(attr)
                if _empty_container_kind(attr_value):
                    found.update(f"{stmt.name}.{name}" for name in attr_names)
    return found


def _find_process_state(package_root: Path) -> dict[str, str]:
    """Every process-lifetime container, by name, with the first file it was seen in."""
    trees = {
        path.relative_to(package_root).as_posix(): ast.parse(path.read_text(encoding="utf-8"))
        for path in sorted(package_root.rglob("*.py"))
        if not SKIP_DIRS & set(path.relative_to(package_root).parts)
    }
    data_classes = _data_classes(trees.values())
    factories = _lazy_cache_factories(trees.values())

    found: dict[str, str] = {}
    for rel, tree in trees.items():
        names = _app_state_caches(tree, factories)
        if rel not in STATIC_MODULES:
            names |= _module_state(tree, data_classes)
        for name in names:
            found.setdefault(name, rel)
    return found


def test_every_process_lifetime_container_has_a_recorded_bound(open_webui_backend: Path):
    found = _find_process_state(open_webui_backend / "open_webui")
    assert found.keys() & KNOWN.keys(), "the scan no longer finds any recorded container"

    unknown = sorted(set(found) - set(KNOWN))
    assert not unknown, (
        "new process-lifetime containers; record what empties each in KNOWN:\n  "
        + "\n  ".join(f"{name} ({found[name]})" for name in unknown)
    )
