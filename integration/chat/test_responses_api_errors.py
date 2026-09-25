"""Regression: a failed reply on a Responses API connection was lost.

Issue open-webui/open-webui#31433, fix PR open-webui/open-webui#31439. A `response.failed`
event showed its error while streaming but was never stored, so the reply was empty after a
reload; a plain `error` event was dropped altogether. A normal reply must stay error-free.

Discriminates: fails on dev ac00d40e3 (both stored replies carry no error), passes with #31439
applied.
"""

from __future__ import annotations

import json

import pytest

from harness import responses_provider as responses_api
from harness.chat import ask
from harness.second_provider import OPENAI_CONFIG

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

MODEL = responses_api.RESPONSES_MODEL
CREATED = {"type": "response.created", "response": {"id": "resp_1", "status": "in_progress"}}


@pytest.fixture
def responses(admin, preserve, listener):
    preserve(OPENAI_CONFIG)
    with admin.client() as client:
        provider = responses_api.connect_responses(client, listener)
        yield provider, client


def test_a_failed_response_is_stored_as_the_replys_error(responses):
    provider, client = responses
    provider.answer(responses_api.events_stream(CREATED, responses_api.failed("quota exceeded")))

    _, message = ask(client, "hello?", model=MODEL)

    assert "quota exceeded" in json.dumps(message.get("error")), message


def test_an_error_event_is_stored_as_the_replys_error(responses):
    provider, client = responses
    error = {"type": "error", "code": "server_error", "message": "upstream exploded"}
    provider.answer(responses_api.events_stream(CREATED, error))

    _, message = ask(client, "hello?", model=MODEL)

    assert "upstream exploded" in json.dumps(message.get("error")), message


def test_a_completed_response_carries_no_error(responses):
    provider, client = responses
    provider.answer(
        responses_api.events_stream(
            CREATED, *responses_api.message("All good."), responses_api.completed()
        )
    )

    _, message = ask(client, "hello?", model=MODEL)

    assert message["content"] == "All good."
    assert not message.get("error")
