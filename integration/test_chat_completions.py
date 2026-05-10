"""API integration tests for /api/chat/completions.

These exercise the chat-completion entrypoint via httpx (no browser) and
target known regressions on the direct-API path (callers that pass no
chat_id, no session_id, etc. — i.e. `curl` users and OpenAI-SDK clients).
"""

import httpx
import pytest


def _pick_model_id(api_client: httpx.Client) -> str | None:
    """Return the first model id available to the test user, or None."""
    resp = api_client.get("/api/models")
    if resp.status_code != 200:
        return None
    data = resp.json().get("data") or []
    return data[0]["id"] if data else None


@pytest.mark.api
@pytest.mark.auth_required
@pytest.mark.regression
def test_direct_chat_completion_no_chat_id_does_not_crash(api_client: httpx.Client):
    """Regression for open-webui/open-webui#24553.

    Direct API calls to /api/chat/completions with no chat_id in the payload
    crashed in v0.9.5 with HTTP 400 ``'NoneType' object has no attribute
    'startswith'`` because the channels-streaming branch added to
    socket.main.get_event_emitter assumed chat_id was a string. The fix
    coalesces None to '' before the .startswith check.

    The assertion is intentionally narrow: we don't require the upstream LLM
    to actually answer (the test env may lack credits / connectivity), only
    that the request doesn't surface the specific NoneType crash.
    """
    model_id = _pick_model_id(api_client)
    if not model_id:
        pytest.skip("No models configured on this Open WebUI instance")

    resp = api_client.post(
        "/api/chat/completions",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
        },
    )

    if resp.status_code == 400:
        detail = ""
        try:
            detail = resp.json().get("detail", "")
        except ValueError:
            detail = resp.text
        assert "'NoneType' object has no attribute 'startswith'" not in detail, (
            f"Regression of #24553: {detail!r}"
        )


@pytest.mark.api
@pytest.mark.auth_required
def test_direct_chat_completion_returns_openai_shape(api_client: httpx.Client):
    """Happy-path sanity: a minimal direct API call returns an OpenAI-shaped
    chat-completion object. Skipped if no model is configured or the
    upstream provider rejects the call.
    """
    model_id = _pick_model_id(api_client)
    if not model_id:
        pytest.skip("No models configured on this Open WebUI instance")

    resp = api_client.post(
        "/api/chat/completions",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            "stream": False,
        },
    )

    if resp.status_code != 200:
        pytest.skip(
            f"Upstream provider did not respond cleanly "
            f"(HTTP {resp.status_code}); skipping happy-path assertion"
        )

    body = resp.json()
    assert body.get("object") == "chat.completion", body
    assert body.get("choices"), body
    assert body["choices"][0].get("message", {}).get("role") == "assistant", body
