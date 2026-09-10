"""Guard: a request its provider fails is logged without its body.

Same contract as `unit/footprint/test_eager_payload_logging.py`, checked on a running instance
at its default log level: the prompt carries a canary string, the mock provider fails the
request, and the server log written since the request began must not contain the canary. The
filter error path stays with its unit form (`test_filter_error_path_ignores_body.py`): its only
render site is at DEBUG, which the default level never writes. The recorded URL leak in
`get_web_loader` is only reachable through web search results, which need a search engine.

Unpinned: read on upstream dev at 4948842be (2026-09-09), where it passes. Unmarked: nothing to
pin.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from integration.conftest import MOCK_MODEL_ID

pytestmark = [pytest.mark.slow, pytest.mark.api, pytest.mark.requires_source]


def _chat(client: httpx.Client, content: str) -> httpx.Response:
    return client.post(
        "/api/chat/completions",
        json={
            "model": MOCK_MODEL_ID,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
        },
    )


def test_failed_upstream_request_keeps_the_prompt_out_of_the_log(launched_instance):
    launched_instance.upstream.reset(mode="error")
    canary = f"canary-{uuid.uuid4().hex}"
    offset = launched_instance.log_size()

    with launched_instance.client() as client:
        response = _chat(client, canary)

    assert response.status_code != 200, "the mock upstream did not fail the request"
    logged = launched_instance.log_since(offset)
    assert logged, "the failure was not logged, so nothing could have leaked"
    assert canary not in logged, "prompt text reached the log"
