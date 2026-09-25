"""Regression: with Notes switched off, the notes API keeps working.

Issue #31414, open on dev ac00d40e3: turning Notes off (`ENABLE_NOTES=false`, or the Notes switch
in the admin settings, both the `notes.enable` config) only hides notes in the interface. The
notes routes check the user's `features.notes` permission and never the switch, so a user still
creates and lists notes over the API. Channels, folders and memories refuse their routes when
switched off.

Discriminates: fails on dev ac00d40e3 (with Notes off a user's create and list answer 200); passes
with a `notes.enable` check in front of the notes routes, as `check_folders_permission` does for
folders. The switched-on test passes on both.
"""

from __future__ import annotations

import uuid

import pytest

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

ADMIN_CONFIG = "/api/v1/auths/admin/config"


def _switch_notes(admin, enabled: bool) -> None:
    with admin.client() as client:
        current = client.get(ADMIN_CONFIG).json()
        client.post(ADMIN_CONFIG, json={**current, "ENABLE_NOTES": enabled}).raise_for_status()
        reported = client.get("/api/config").json()["features"]["enable_notes"]
    assert reported is enabled


def _create_note(account, title: str):
    with account.client() as client:
        return client.post(
            "/api/v1/notes/create", json={"title": title, "data": {"content": {"md": "hello"}}}
        )


def _listed_titles(account) -> list[str]:
    with account.client() as client:
        listed = client.get("/api/v1/notes/")
    assert listed.status_code == 200, listed.text
    return [note["title"] for note in listed.json()]


@pytest.fixture
def notes_switch(admin, preserve):
    preserve("admin_config")
    return lambda enabled: _switch_notes(admin, enabled)


def test_switched_off_notes_refuse_creating_a_note(make_user, notes_switch):
    account = make_user()
    title = f"while off {uuid.uuid4().hex[:8]}"
    notes_switch(False)

    created = _create_note(account, title)
    notes_switch(True)

    assert created.status_code in (401, 403), (
        f"with Notes switched off a user still created a note: HTTP {created.status_code} (#31414)"
    )
    assert title not in _listed_titles(account)


def test_switched_off_notes_refuse_listing_notes(make_user, notes_switch):
    account = make_user()
    title = f"kept {uuid.uuid4().hex[:8]}"
    assert _create_note(account, title).status_code == 200
    notes_switch(False)

    with account.client() as client:
        listed = client.get("/api/v1/notes/")

    assert listed.status_code in (401, 403), (
        f"with Notes switched off a user still listed their notes: HTTP {listed.status_code} "
        "(#31414)"
    )
    assert title not in listed.text


def test_switched_on_notes_are_created_and_listed(make_user, notes_switch):
    account = make_user()
    title = f"while on {uuid.uuid4().hex[:8]}"
    notes_switch(True)

    created = _create_note(account, title)

    assert created.status_code == 200, created.text
    assert title in _listed_titles(account)
