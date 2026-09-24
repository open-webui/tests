"""The active filter ids were read from the database once per filter stage, not once per request.

Fix commit `7d694570a` in `open_webui/utils/filter.py`. `resolve_filter_pipeline` called
`Functions.get_active_filter_ids()` directly, so every filter stage of a single chat turn
(inlet, request, stream, outlet) queried the function table again. The ids are now fetched once
and kept on the request. Only the query count shows it, so it stays here, driven through the
public resolvers against the real scratch database. The non-callable handler fix from the same
file (2a4ef46ac) is pinned over HTTP in integration/chat/test_filter_context_and_handlers.py.

Discriminates: passes on dev bbfa876af; with 7d694570a reverted both tests fail (every request
reads four times).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from starlette.requests import Request

pytestmark = pytest.mark.regression

MODEL = {"id": "m"}


@pytest.fixture(scope="session")
def filter_utils(owui_module):
    owui_module("open_webui.config")  # runs the migrations, so the function table exists
    return owui_module("open_webui.utils.filter")


@pytest.fixture
def active_filter_reads(filter_utils):
    """A spy on the real table read; the scratch database answers it."""
    table = filter_utils.Functions
    with patch.object(table, "get_active_filter_ids", wraps=table.get_active_filter_ids) as read:
        yield read


async def resolve_one_turn(filter_utils, request: Request) -> None:
    """The four resolutions one chat turn makes: inlet, request, stream and outlet."""
    await filter_utils.get_sorted_filter_ids(request, MODEL, [])
    await filter_utils.get_filter_functions(request, MODEL, [])
    await filter_utils.get_filter_functions(request, MODEL, [])
    await filter_utils.get_sorted_filter_ids(request, MODEL, [])


@pytest.mark.asyncio
async def test_one_request_reads_the_active_filters_once(filter_utils, active_filter_reads):
    await resolve_one_turn(filter_utils, Request({"type": "http"}))

    assert active_filter_reads.await_count == 1, (
        "the active filter ids were re-read for every filter stage of one request (7d694570a)"
    )


@pytest.mark.asyncio
async def test_each_request_reads_its_own_active_filters(filter_utils, active_filter_reads):
    await resolve_one_turn(filter_utils, Request({"type": "http"}))
    await resolve_one_turn(filter_utils, Request({"type": "http"}))

    assert active_filter_reads.await_count == 2
