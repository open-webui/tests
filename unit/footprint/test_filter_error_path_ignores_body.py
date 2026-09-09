"""Guard: the filter pipeline's error path does not reference the request body.

An except block that renders, copies or pickles the failed request costs a whole message
history per failure, so a burst of upstream errors becomes a burst of allocations on the
event loop. A poisoned body raises on any such attempt, at any log level: unlike the audit
in `test_eager_payload_logging.py`, this path is held to the stricter rule that even a lazy
DEBUG argument must not name the body.

Unpinned: read on upstream dev at v0.11.3 (a253bf0c3), where the path logs only the filter
type and id. Unmarked: nothing to pin.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="session")
def filter_utils(owui_module):
    return owui_module("open_webui.utils.filter")


class Poison:
    """Raises when rendered; the copy and pickle arms guard paths that do not exist today. A JSON
    encoder rejects it on its own."""

    def _refuse(self, *args):
        raise AssertionError("the request body was rendered, copied or pickled")

    __repr__ = __str__ = __format__ = __deepcopy__ = __reduce_ex__ = _refuse


@pytest.mark.asyncio
@pytest.mark.parametrize("filter_type", ["inlet", "stream", "outlet"])
async def test_failing_filter_does_not_reference_the_body(
    filter_utils, monkeypatch, caplog, filter_type
):
    class Broken:
        async def inlet(self, body):
            raise ValueError("boom")

        async def stream(self, event):
            raise ValueError("boom")

        outlet = inlet

    async def fake_loader(request, function_id, function=None, load_from_db=True):
        return Broken(), "filter", {}

    monkeypatch.setattr(filter_utils, "get_function_module_from_cache", fake_loader)
    body = {"messages": [{"role": "user", "content": Poison()}], "metadata": {"m": Poison()}}
    caplog.set_level(logging.DEBUG, logger="open_webui")

    with pytest.raises(ValueError):
        await filter_utils.process_filter_function(
            request=None,
            function=SimpleNamespace(id="broken"),
            filter_type=filter_type,
            form_data=body,
            extra_params={},
            filter_context=None,
            valves_by_id={},
            filter_ids=["broken"],
        )
    assert any(r.name == "open_webui.utils.filter" for r in caplog.records), (
        "the failure was not logged, so the poison never had a chance to fire"
    )
