"""Guard: per-chunk and per-filter code hands the payload on by reference.

A helper that rebuilds every dict and list it walks turns shared references into real heap,
and on a path that runs per stream chunk, per filter or per tool round that is quadratic
copying over a response. These contracts pin the places the filter and streaming paths share
references, so a deepcopy or a structural rebuild creeping in fails immediately:

* `get_filter_params` runs per chunk per stream filter, and once per request per inlet
  filter, and must hand over the event or body and the extra params themselves;
* `handle_responses_streaming_event` runs per chunk and is copy-on-write: only the touched
  item, its content list and the touched part are new objects, the rest is the same objects
  (that the input is left unchanged is pinned in `unit/chat/test_stream_event_handling.py`).

Stays a unit test: `integration/footprint/test_streaming_cost_stays_linear.py` sees the same
contract from outside, but only as a cost ratio that another copy on the path already breaks.

Unpinned: read on upstream dev at v0.11.3 (a253bf0c3), where both hold. Unmarked: nothing
to pin.
Discriminates: passes on dev bbfa876af, fails once a copy of it deep-copies `form_data` in
`get_filter_params` or rebuilds every output item per delta.
"""

from __future__ import annotations

import inspect

import pytest


@pytest.fixture(scope="session")
def filter_utils(owui_module):
    return owui_module("open_webui.utils.filter")


@pytest.fixture(scope="session")
def middleware_module(owui_module):
    return owui_module("open_webui.utils.middleware")


def stream_handler(event, __body__, __metadata__): ...


def inlet_handler(body, __body__, __metadata__): ...


@pytest.mark.parametrize(
    "filter_type, key, handler",
    [("stream", "event", stream_handler), ("inlet", "body", inlet_handler)],
)
def test_filter_params_hand_over_the_payload_by_reference(filter_utils, filter_type, key, handler):
    payload = {"messages": [{"role": "user", "content": "x"}]}
    extra = {"__body__": {"messages": []}, "__metadata__": {"chat_id": "c1"}}

    params = filter_utils.get_filter_params(
        sig=inspect.signature(handler),
        filter_id="f1",
        filter_type=filter_type,
        form_data=payload,
        extra_params=extra,
    )

    assert params[key] is payload
    assert params["__body__"] is extra["__body__"]
    assert params["__metadata__"] is extra["__metadata__"]


def test_text_delta_leaves_untouched_output_items_and_parts_shared(middleware_module):
    reasoning = {"type": "reasoning", "id": "r1"}
    second_part = {"type": "text", "text": "tail"}
    message = {
        "type": "message",
        "id": "m1",
        "content": [{"type": "text", "text": "Hello"}, second_part],
    }
    output = [reasoning, message]

    delta = {"type": "response.output_text.delta", "output_index": 1, "content_index": 0}
    new_output, _ = middleware_module.handle_responses_streaming_event(
        data={**delta, "delta": "!"}, current_output=output
    )

    assert new_output[0] is reasoning
    assert new_output[1] is not message
    assert new_output[1]["content"][1] is second_part
