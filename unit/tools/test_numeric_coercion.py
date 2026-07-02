"""Tests for numeric-parameter coercion of builtin tool functions.

Regression for open-webui/open-webui#25641.

Builtin tools (backend/open_webui/tools/builtin.py) declare int / Optional[int]
parameters, but a model using native function calling can emit those values as
JSON strings ("count": "1" instead of "count": 1). Uncoerced, any comparison /
arithmetic / slicing on a string-typed numeric raises e.g.

    TypeError: '<' not supported between instances of 'int' and 'str'

Where coercion actually happens — the wrapper, not the tool
--------------------------------------------------------------
The tool bodies do NOT int()-coerce their own params, and they must not have to:
production never calls a builtin directly. The middleware invokes every tool
through the wrapper returned by

    open_webui.utils.tools.get_async_tool_function_and_apply_extra_params(func, extra_params)

whose inner `coerce_kwargs` inspects the target's type hints and, for a str
passed to an `int` / `Optional[int]` param, does `int(value)` BEFORE the tool
runs (tools.py, `coerce_kwargs`). middleware calls `tool['callable'](**params)`
where `callable` IS that wrapper, so `search_web(count="3")` reaches the tool
body as `count=3`. Coercion is centralized there, once, for every tool.

These tests therefore exercise the REAL path: wrap a builtin via the production
wrapper and drive the wrapper with STRING numerics, asserting the tool sees the
coerced int and behaves correctly. `calculate_timestamp` is pure (no request /
DB), so it needs no mocks; `search_web` gets its network + config dependencies
stubbed so the assertion is offline and deterministic.
"""

from __future__ import annotations

import json
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from open_webui.utils.tools import get_async_tool_function_and_apply_extra_params

pytestmark = [pytest.mark.regression, pytest.mark.asyncio]


def _fake_results(n: int) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(title=f"t{i}", link=f"https://e/{i}", snippet=f"s{i}") for i in range(n)
    ]


async def test_wrapper_coerces_string_offsets_for_calculate_timestamp(
    builtin_tools_module: ModuleType,
) -> None:
    """The production wrapper coerces string int params before the tool runs.

    calculate_timestamp is pure. Its offsets are bare `int`, hitting
    `weeks_ago * 7` and `months_ago > 0`; a raw string there would TypeError.
    Wrapped and called with string offsets, coerce_kwargs turns "3"/"1" into
    3/1 first, so the tool returns a valid timestamp JSON.
    """
    wrapped = await get_async_tool_function_and_apply_extra_params(
        builtin_tools_module.calculate_timestamp, {}
    )

    out = await wrapped(
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


async def test_wrapper_leaves_unset_optional_offsets_alone(
    builtin_tools_module: ModuleType,
) -> None:
    """coerce_kwargs skips None, so omitted offsets keep their int defaults and
    the wrapped tool still returns the current timestamp unchanged."""
    wrapped = await get_async_tool_function_and_apply_extra_params(
        builtin_tools_module.calculate_timestamp, {}
    )

    out = await wrapped(__request__=None, __user__=None)

    data = json.loads(out)
    assert data["current_timestamp"] == data["calculated_timestamp"], data


async def test_wrapper_coerces_string_count_for_search_web(
    builtin_tools_module: ModuleType,
) -> None:
    """Regression for open-webui/open-webui#25641.

    search_web(count="3") must not blow up on `min(count, max_count)`. The
    wrapper coerces "3" -> 3 before the tool runs, so the query succeeds and at
    most 3 results come back. `_search_web` and Config are stubbed for offline
    determinism; count comes back as a real string from native function calling.
    """
    mod = builtin_tools_module
    wrapped = await get_async_tool_function_and_apply_extra_params(mod.search_web, {})

    search = AsyncMock(return_value=_fake_results(10))
    config = AsyncMock(return_value=None)  # engine + result_count both unset -> defaults
    with (
        patch.object(mod, "_search_web", search),
        patch.object(mod.Config, "get", config),
    ):
        out = await wrapped(
            query="portland weather",
            count="3",  # native function calling delivered a string
            __request__=SimpleNamespace(),
            __user__=None,
        )

    data = json.loads(out)
    assert isinstance(data, list), f"expected results list, got error JSON: {data!r}"
    assert "error" not in (data[0] if data else {}), data
    assert len(data) == 3, f"string count='3' should yield 3 results, got {len(data)}"


async def test_wrapper_search_web_string_count_does_not_typeerror(
    builtin_tools_module: ModuleType,
) -> None:
    """The specific #25641 symptom: `min(count, max_count)` must never surface
    the '<' not supported between int and str TypeError once the wrapper has
    coerced count."""
    mod = builtin_tools_module
    wrapped = await get_async_tool_function_and_apply_extra_params(mod.search_web, {})

    with (
        patch.object(mod, "_search_web", AsyncMock(return_value=_fake_results(2))),
        patch.object(mod.Config, "get", AsyncMock(return_value=None)),
    ):
        out = await wrapped(query="q", count="1", __request__=SimpleNamespace(), __user__=None)

    data = json.loads(out)
    err = data.get("error") if isinstance(data, dict) else None
    assert not (err and "not supported between" in err), (
        f"Regression of #25641: search_web surfaced the coercion TypeError: {err!r}"
    )
