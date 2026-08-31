"""Regression tests for two 0.11.2 repairs in `open_webui/utils/filter.py`.

* Active filter ids fetched per stage (commit 7d694570a). `resolve_filter_pipeline` called
  `Functions.get_active_filter_ids()` directly, so every filter stage in a single chat turn
  (inlet, the new request stage, stream, outlet) re-queried the function table. `FilterContext`
  gained an `active_filters` cache and `get_filter_context(request)` parks one context on
  `request.state`, so the ids are fetched once per request.
* Non-callable handler attributes (commit 2a4ef46ac). `process_filter_function` accepted any
  truthy `inlet`/`stream`/`outlet` attribute as a handler, so a filter module exposing e.g. a
  string named `inlet` reached `inspect.signature` and blew up the request. The guard is now
  `if not callable(handler)`.

Discriminates: passes on v0.11.3, fails on v0.11.1 (the active filter ids are re-fetched for
every stage, and a non-callable handler attribute raises instead of being skipped).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.regression


@pytest.fixture(scope="session")
def filter_utils(owui_module):
    return owui_module("open_webui.utils.filter")


class FakeFunction:
    def __init__(self, function_id: str):
        self.id = function_id
        self.name = function_id
        self.type = "filter"
        self.is_active = True
        self.is_global = True


def make_request() -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace(), cookies={}, headers={})


# =============================================================================
# 7d694570a -- one active-filter-id fetch per request
# =============================================================================


@pytest.fixture
def counting_functions(filter_utils, monkeypatch):
    """Count the table hits `resolve_filter_pipeline` makes for the active filter ids."""
    calls = {"active": 0}

    class FakeFunctions:
        @staticmethod
        async def get_active_filter_ids():
            calls["active"] += 1
            return [("f1", True)]

        @staticmethod
        async def get_functions_by_ids(filter_ids):
            return [FakeFunction(fid) for fid in filter_ids]

        @staticmethod
        async def get_function_valves_by_ids(filter_ids):
            return {}

    async def fake_loader(request, function_id, function=None, load_from_db=True):
        return SimpleNamespace(), "filter", {}

    monkeypatch.setattr(filter_utils, "Functions", FakeFunctions)
    monkeypatch.setattr(filter_utils, "get_function_module_from_cache", fake_loader)
    return calls


MODEL = {"id": "m"}


@pytest.mark.asyncio
async def test_active_filter_ids_are_fetched_once_per_request(filter_utils, counting_functions):
    request = make_request()

    # Four resolutions, the shape of one chat turn: inlet, request, stream, outlet.
    for _ in range(4):
        await filter_utils.get_filter_functions(request, MODEL, [])

    assert counting_functions["active"] == 1, (
        "the active filter ids were re-queried for every filter stage of a single request "
        "(7d694570a)"
    )


@pytest.mark.asyncio
async def test_sorted_ids_and_functions_share_the_request_cache(filter_utils, counting_functions):
    request = make_request()

    await filter_utils.get_sorted_filter_ids(request, MODEL, [])
    await filter_utils.get_filter_functions(request, MODEL, [])

    assert counting_functions["active"] == 1


@pytest.mark.asyncio
async def test_the_filter_context_is_parked_on_the_request(filter_utils, counting_functions):
    request = make_request()

    await filter_utils.get_sorted_filter_ids(request, MODEL, [])

    context = request.state.filter_context
    assert context.active_filters == [("f1", True)], (
        "resolving a pipeline left no cached active filter ids on the request (7d694570a)"
    )
    assert filter_utils.get_filter_context(request) is context


@pytest.mark.asyncio
async def test_cached_ids_still_resolve_the_same_filters(filter_utils, counting_functions):
    request = make_request()

    first = await filter_utils.get_sorted_filter_ids(request, MODEL, [])
    second = await filter_utils.get_sorted_filter_ids(request, MODEL, [])

    assert first == ["f1"]
    assert second == first, "the cached active filter ids resolved to a different pipeline"


# --- Nearby: caching is per request, and stays off when plugins are disabled -------------


@pytest.mark.asyncio
async def test_separate_requests_do_not_share_the_cache(filter_utils, counting_functions):
    first_request, second_request = make_request(), make_request()

    await filter_utils.get_filter_functions(first_request, MODEL, [])
    await filter_utils.get_filter_functions(second_request, MODEL, [])

    assert counting_functions["active"] >= 2, "a second request reused the first request's ids"


@pytest.mark.asyncio
async def test_disabled_plugins_resolve_nothing(filter_utils, counting_functions, monkeypatch):
    monkeypatch.setattr(filter_utils, "ENABLE_PLUGINS", False)

    filter_ids, filter_functions = await filter_utils.resolve_filter_pipeline(
        make_request(), MODEL, []
    )

    assert (filter_ids, filter_functions) == ([], [])
    assert counting_functions["active"] == 0


# =============================================================================
# 2a4ef46ac -- a non-callable handler attribute is skipped, not called
# =============================================================================


@pytest.fixture
def patch_module_loader(filter_utils, monkeypatch):
    """Swap the plugin loader, the only I/O boundary process_filter_function has."""

    def _install(module):
        async def fake_loader(request, function_id, function=None, load_from_db=True):
            return module, "filter", {}

        monkeypatch.setattr(filter_utils, "get_function_module_from_cache", fake_loader)
        return module

    return _install


async def run_filter(filter_utils, filter_type, form_data, filter_id="acme_filter"):
    return await filter_utils.process_filter_function(
        request=make_request(),
        function=FakeFunction(filter_id),
        filter_type=filter_type,
        form_data=form_data,
        extra_params={"__user__": {"id": "user-1"}},
        filter_context=None,
        valves_by_id=None,
        filter_ids=[filter_id],
    )


NON_CALLABLE_HANDLERS = [
    pytest.param("inlet not implemented", id="str"),
    pytest.param({"enabled": True}, id="dict"),
    pytest.param([1, 2], id="list"),
    pytest.param(7, id="int"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("handler", NON_CALLABLE_HANDLERS)
async def test_non_callable_inlet_is_skipped(filter_utils, patch_module_loader, handler):
    patch_module_loader(SimpleNamespace(inlet=handler))
    form_data = {"messages": [{"role": "user", "content": "hi"}]}

    result, valves_by_id, skip_files = await run_filter(filter_utils, "inlet", form_data)

    assert result is form_data, (
        "a filter module exposing a non-callable 'inlet' was treated as a handler and crashed "
        "the request (2a4ef46ac)"
    )
    assert valves_by_id is None
    assert skip_files is None


# --- Broad: the guard holds for every filter stage --------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("filter_type", ["inlet", "request", "stream", "outlet"])
async def test_non_callable_handler_is_skipped_for_every_stage(
    filter_utils, patch_module_loader, filter_type
):
    patch_module_loader(SimpleNamespace(**{filter_type: "not a handler"}))
    form_data = {"messages": []}

    result, _, _ = await run_filter(filter_utils, filter_type, form_data)

    assert result is form_data


# --- Nearby: real handlers, absent handlers and falsy attributes -------------------------


class WorkingModule:
    async def inlet(self, body):
        return {**body, "inlet_ran": True}

    def stream(self, event):
        return {**event, "stream_ran": True}

    async def outlet(self, body):
        return {**body, "outlet_ran": True}

    async def request(self, body):
        return {**body, "request_ran": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("filter_type", ["inlet", "request", "stream", "outlet"])
async def test_callable_handler_still_runs(filter_utils, patch_module_loader, filter_type):
    patch_module_loader(WorkingModule())

    result, _, _ = await run_filter(filter_utils, filter_type, {"messages": []})

    assert result[f"{filter_type}_ran"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("filter_type", ["inlet", "request", "stream", "outlet"])
async def test_module_with_no_handlers_is_a_no_op(filter_utils, patch_module_loader, filter_type):
    patch_module_loader(SimpleNamespace())
    form_data = {"messages": []}

    result, valves_by_id, skip_files = await run_filter(filter_utils, filter_type, form_data)

    assert (result, valves_by_id, skip_files) == (form_data, None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("handler", [None, "", 0, [], {}])
async def test_falsy_handler_attribute_is_skipped(filter_utils, patch_module_loader, handler):
    patch_module_loader(SimpleNamespace(inlet=handler))
    form_data = {"messages": []}

    result, _, _ = await run_filter(filter_utils, "inlet", form_data)

    assert result is form_data


@pytest.mark.asyncio
async def test_handler_exception_still_propagates(filter_utils, patch_module_loader):
    class RaisingModule:
        async def inlet(self, body):
            raise RuntimeError("inlet exploded")

    patch_module_loader(RaisingModule())

    with pytest.raises(RuntimeError):
        await run_filter(filter_utils, "inlet", {"messages": []})


@pytest.mark.asyncio
async def test_file_handler_flag_survives_the_callable_guard(filter_utils, patch_module_loader):
    module = WorkingModule()
    module.file_handler = True
    patch_module_loader(module)

    _, _, skip_files = await run_filter(filter_utils, "inlet", {"messages": []})

    assert skip_files is True
