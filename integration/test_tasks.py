"""API integration tests for /api/v1/tasks/*.

Targets known regressions in the task-orchestration endpoints — title
generation, follow-ups, tags, etc.
"""

import httpx
import pytest


@pytest.mark.api
@pytest.mark.auth_required
@pytest.mark.regression
def test_generate_title_does_not_404_on_empty_model(api_client: httpx.Client):
    """Regression for open-webui/open-webui#24604.

    In v0.9.4-v0.9.5 (before e5c8f8110), the Sidebar's title-regenerate
    button could send an empty `model` to /api/v1/tasks/title/completions
    when the active branch of an edited chat referenced a model the
    sidebar couldn't resolve. The backend then fell through to the
    `if model_id not in models:` check and returned

        404 Not Found  {"detail": "Model '' was not found"}

    which surfaced as an unhelpful toast in the UI.

    Fixed in e5c8f8110 by gating the empty-model case earlier with a
    400 + a specific message ("No model specified for title generation
    ..."). The frontend separately got smarter about which model to
    send, but the backend gate is the one that locks this regression
    out for any client.

    The test:
      - posts with `model=""` and a minimal messages payload
      - asserts the response detail (whatever the status code) doesn't
        contain the original "Model '' was not found" stringification
      - if title generation is disabled on this instance, the endpoint
        returns 200 with a "disabled" detail — that's fine, skip the
        assertion.
    """
    resp = api_client.post(
        "/api/v1/tasks/title/completions",
        json={
            "model": "",
            "messages": [{"role": "user", "content": "ping"}],
        },
    )

    try:
        body = resp.json()
    except ValueError:
        body = {}
    detail = body.get("detail", "") if isinstance(body, dict) else ""

    # Endpoint short-circuits with 200 when title-gen is disabled —
    # nothing to regress.
    if resp.status_code == 200 and "disabled" in detail.lower():
        pytest.skip("Title generation disabled on this instance")

    # The original symptom from rotemdan's report.
    assert "Model '' was not found" not in detail, (
        f"Regression of open-webui/open-webui#24604: empty model_id still "
        f"falls through to the model-not-found check.\n"
        f"HTTP {resp.status_code}: {detail!r}"
    )

    # Stronger positive check: the fix returns 400 with a message that
    # makes the actual problem clear. Skipped if some other failure
    # mode is in the way (e.g. permissions, pipeline filters).
    if resp.status_code == 400 and detail:
        assert "No model specified" in detail or "model" in detail.lower(), (
            f"400 response detail should explain the missing-model case, got: {detail!r}"
        )
