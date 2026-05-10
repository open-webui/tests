"""API integration tests for /api/v1/notes/*.

Targets known regressions where the `is_pinned` attribute was
out-of-sync between the Pydantic NoteModel and the SQLAlchemy Note
mapping, breaking both note creation and note retrieval.
"""

import uuid

import httpx
import pytest


def _delete_note_silently(api_client: httpx.Client, note_id: str) -> None:
    """Best-effort cleanup — never fail the test on cleanup errors."""
    try:
        api_client.delete(f"/api/v1/notes/{note_id}/delete")
    except httpx.HTTPError:
        pass


@pytest.mark.api
@pytest.mark.auth_required
@pytest.mark.regression
def test_note_create_and_get_roundtrip_does_not_crash(api_client: httpx.Client):
    """Regression for open-webui/open-webui#24484.

    In v0.9.3-v0.9.4:
      * POST /api/v1/notes/create returned HTTP 400 with
        ``'is_pinned' is an invalid keyword argument for Note`` because
        the SQLAlchemy Note() constructor was being passed the Pydantic
        is_pinned field.
      * GET  /api/v1/notes/{id} returned HTTP 500 with
        ``NoteResponse() got multiple values for keyword argument 'is_pinned'``
        because is_pinned was both inside model_dump() and passed
        explicitly to NoteResponse().

    Fixed in v0.9.5 (PR #24486). The test exercises both endpoints and
    asserts neither regression resurfaces.
    """
    # 1. Create
    title = f"regression-24484-{uuid.uuid4().hex[:8]}"
    create_resp = api_client.post(
        "/api/v1/notes/create",
        json={"title": title, "data": {"content": {"md": "ping"}}},
    )

    if create_resp.status_code == 401:
        pytest.skip("Test user lacks features.notes permission")

    if create_resp.status_code == 400:
        detail = ""
        try:
            detail = create_resp.json().get("detail", "")
        except ValueError:
            detail = create_resp.text
        assert "is_pinned" not in detail, (
            f"Regression of #24484 (create path): HTTP 400 {detail!r}"
        )

    assert create_resp.status_code == 200, (
        f"Note create failed unexpectedly: HTTP {create_resp.status_code} {create_resp.text}"
    )
    note = create_resp.json()
    note_id = note["id"]

    try:
        # 2. Get by id
        get_resp = api_client.get(f"/api/v1/notes/{note_id}")

        if get_resp.status_code == 500:
            detail = ""
            try:
                detail = get_resp.json().get("detail", "")
            except ValueError:
                detail = get_resp.text
            assert "is_pinned" not in detail, (
                f"Regression of #24484 (get path): HTTP 500 {detail!r}"
            )

        assert get_resp.status_code == 200, (
            f"Note get failed unexpectedly: HTTP {get_resp.status_code} {get_resp.text}"
        )
        fetched = get_resp.json()
        assert fetched["id"] == note_id
        assert fetched["title"] == title
        # Field is part of the response contract — guard against accidental removal.
        assert "is_pinned" in fetched
    finally:
        _delete_note_silently(api_client, note_id)
