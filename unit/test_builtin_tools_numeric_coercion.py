"""Tests for numeric-parameter coercion in builtin tool functions.

Regression for open-webui/open-webui#25641.

Builtin tools (backend/open_webui/tools/builtin.py) declare int / Optional[int]
parameters, but a model using native function calling can emit those values
as JSON strings ("count": "1" instead of "count": 1). The middleware passes
the raw values straight through to the tool, so any comparison / arithmetic /
slicing on a string-typed numeric raises e.g.

    TypeError: '<' not supported between instances of 'int' and 'str'

For tools wrapped in `except Exception` (search_web) the model receives an
error JSON instead of results; for tools without that wrapper
(calculate_timestamp) the TypeError propagates and crashes the tool call.

Fix (PR #25638): every builtin tool coerces its numeric params with int()
at the top of the function, preserving None for Optional params.

Two layers:
  - behavioral: drive search_web and calculate_timestamp with string numerics
    and assert they work (the exact repro + a mock-free pure case)
  - broad/static: audit EVERY public async tool function — each int /
    Optional[int] / int|None parameter must be int()-coerced in the body.
    Catches the whole class, including any future tool that forgets.
"""

from __future__ import annotations

import ast
import json
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


# =============================================================================
# Behavioral — the documented repro
# =============================================================================


def _request_with_config(engine: str = "searxng", result_count=5) -> SimpleNamespace:
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                config=SimpleNamespace(
                    WEB_SEARCH_ENGINE=engine,
                    WEB_SEARCH_RESULT_COUNT=result_count,
                )
            )
        )
    )


def _fake_results(n: int) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(title=f"t{i}", link=f"https://e/{i}", snippet=f"s{i}")
        for i in range(n)
    ]


@pytest.mark.regression
@pytest.mark.asyncio
async def test_search_web_coerces_string_count(builtin_tools_module: ModuleType) -> None:
    """Regression for open-webui/open-webui#25641.

    search_web(count="3") must not blow up on min(count, max_count); it
    should coerce "3" -> 3, query, and return at most 3 results.
    """
    mod = builtin_tools_module
    search = AsyncMock(return_value=_fake_results(10))

    with patch.object(mod, "_search_web", search):
        out = await mod.search_web(
            query="portland weather",
            count="3",  # native function calling delivered a string
            __request__=_request_with_config(),
            __user__=None,
        )

    data = json.loads(out)
    assert isinstance(data, list), f"expected results list, got error JSON: {data!r}"
    assert "error" not in (data[0] if data else {}), data
    assert len(data) == 3, f"string count='3' should yield 3 results, got {len(data)}"


@pytest.mark.regression
@pytest.mark.asyncio
async def test_search_web_string_count_does_not_typeerror(
    builtin_tools_module: ModuleType,
) -> None:
    """The specific symptom: the min(count, max_count) comparison must not
    surface the '<' not supported between int and str TypeError."""
    mod = builtin_tools_module
    with patch.object(mod, "_search_web", AsyncMock(return_value=_fake_results(2))):
        out = await mod.search_web(
            query="q", count="1", __request__=_request_with_config(), __user__=None
        )
    data = json.loads(out)
    err = data.get("error") if isinstance(data, dict) else None
    assert not (err and "not supported between" in err), (
        f"Regression of #25641: search_web surfaced the coercion TypeError: {err!r}"
    )


@pytest.mark.regression
@pytest.mark.asyncio
async def test_calculate_timestamp_coerces_string_offsets(
    builtin_tools_module: ModuleType,
) -> None:
    """calculate_timestamp is pure (no request/DB). With string offsets it
    hits `weeks_ago * 7` and `months_ago > 0`; uncoerced that raises
    TypeError. Coerced, it returns a valid timestamp JSON."""
    mod = builtin_tools_module
    out = await mod.calculate_timestamp(
        days_ago="3",
        weeks_ago="1",
        months_ago="0",
        years_ago="0",
        __request__=None,
        __user__=None,
    )
    data = json.loads(out)
    assert "calculated_timestamp" in data, data
    assert isinstance(data["calculated_timestamp"], int), data
    # 3 days + 1 week = 10 days earlier than current.
    assert data["current_timestamp"] - data["calculated_timestamp"] == 10 * 86400, data


# =============================================================================
# Broad/static — every numeric tool param must be coerced
# =============================================================================


_INT_ANNOTATIONS = {"int", "Optional[int]", "int | None", "None | int"}


def _is_int_annotation(annotation: ast.expr | None) -> bool:
    if annotation is None:
        return False
    return ast.unparse(annotation).replace(" ", "") in {
        a.replace(" ", "") for a in _INT_ANNOTATIONS
    }


def _coerced_params(func: ast.AsyncFunctionDef) -> set[str]:
    """Param names that get an int(<param>) call somewhere in the body."""
    coerced: set[str] = set()
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "int"
            and node.args
            and isinstance(node.args[0], ast.Name)
        ):
            coerced.add(node.args[0].id)
    return coerced


def _tool_int_params(func: ast.AsyncFunctionDef) -> list[str]:
    """int / Optional[int] params of a tool fn, excluding __dunder__ ctx args."""
    params = []
    args = func.args
    for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
        if a.arg.startswith("__"):
            continue
        if _is_int_annotation(a.annotation):
            params.append(a.arg)
    return params


@pytest.mark.regression
def test_every_builtin_tool_coerces_its_numeric_params(
    open_webui_backend,
) -> None:
    """Regression for open-webui/open-webui#25641 (broad).

    Audit backend/open_webui/tools/builtin.py: for every public async tool
    function, each int / Optional[int] / int|None parameter must be
    int()-coerced in the body. A tool that declares a numeric param but
    passes the raw value into a comparison/arithmetic/slice will crash
    when a native-function-calling model sends it as a string.
    """
    builtin = (
        open_webui_backend / "open_webui" / "tools" / "builtin.py"
    )
    assert builtin.is_file(), builtin
    tree = ast.parse(builtin.read_text(encoding="utf-8"))

    offenders: dict[str, list[str]] = {}
    audited = 0
    for node in tree.body:
        # Public tool functions only (helpers are _-prefixed and receive
        # already-coerced values).
        if not isinstance(node, ast.AsyncFunctionDef) or node.name.startswith("_"):
            continue
        int_params = _tool_int_params(node)
        if not int_params:
            continue
        audited += 1
        coerced = _coerced_params(node)
        missing = [p for p in int_params if p not in coerced]
        if missing:
            offenders[node.name] = missing

    assert audited > 0, "found no tool functions with int params — audit logic broke?"
    assert not offenders, (
        "Regression of open-webui/open-webui#25641: these builtin tool "
        "function(s) declare int/Optional[int] parameters that are never "
        "int()-coerced, so a native-function-calling model sending them as "
        f"strings will crash the tool:\n{json.dumps(offenders, indent=2)}"
    )
