"""Regression: a rejected profile image URL must not be retained or logged.

open-webui 0.11.4 fix `38a8dc9f3` (PR #29971): the `ModelMeta` validator that clears an
invalid `profile_image_url` added every distinct rejected value to a module-level set and
logged a warning the first time it saw each one. FastAPI validates the body before the route
checks permissions, so any signed-in user could post model entries with distinct invalid URLs
and grow that set without bound (8.9 MB after 2,000 values of 4 KB). The fix drops the set and
the warning; the value is still cleared.

The warning was written exactly when a value entered the set, so a server log with no line for
the posted values is a log with no retained entry.

Twin of unit/security/test_v0114_profile_image_retention.py.

Discriminates: passes on dev bbfa876af, fails with the set and warning restored in
`ModelMeta.check_image_url` (one warning per distinct posted value).
"""

from __future__ import annotations

import uuid

import pytest

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

DISTINCT_VALUES = 20
VALID_URL = "https://example.com/avatar.png"


def _model_form(model_id: str, **meta) -> dict:
    return {"id": model_id, "name": model_id, "meta": meta, "params": {}}


def _create(actor, **meta):
    with actor.client() as client:
        return client.post(
            "/api/v1/models/create", json=_model_form(f"preset-{uuid.uuid4().hex[:8]}", **meta)
        )


def _read_back(actor, model_id: str) -> dict:
    with actor.client() as client:
        stored = client.get("/api/v1/models/model", params={"id": model_id})
    stored.raise_for_status()
    return stored.json()["meta"]


def test_distinct_rejected_urls_leave_nothing_behind(instance, make_user):
    marker = f"javascript:alert(1)//{uuid.uuid4().hex}"
    user = make_user()
    offset = instance.log_size()

    # A user without model permissions: the body is validated before the 401.
    refused = [
        _create(user, profile_image_url=f"{marker}-{index}").status_code
        for index in range(DISTINCT_VALUES)
    ]

    assert set(refused) == {401}
    assert marker not in instance.log_since(offset), (
        "each distinct rejected profile image URL was logged, which is the moment it was added "
        "to a process-lifetime set any signed-in user can grow without bound (#29971)"
    )


def test_a_rejected_url_is_still_cleared(admin):
    created = _create(admin, profile_image_url="javascript:alert(1)")

    assert created.status_code == 200, created.text
    assert _read_back(admin, created.json()["id"])["profile_image_url"] is None


def test_a_valid_url_is_kept(admin):
    created = _create(admin, profile_image_url=VALID_URL)

    assert created.status_code == 200, created.text
    assert _read_back(admin, created.json()["id"])["profile_image_url"] == VALID_URL


def test_an_invalid_background_image_is_still_refused(admin):
    assert _create(admin, background_image_url="javascript:alert(1)").status_code == 422
