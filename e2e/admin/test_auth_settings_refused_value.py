"""The admin authentication page kept showing a value the server had refused, #30293.

Fix commit `963071768` (open-webui/open-webui#30433). Saving the page posts the admin config,
and the server quietly keeps the stored value for a setting it refuses, such as a JWT expiration
without a time unit, answering with what it stored. The page ignored that answer, so the field
still read the refused value, as if it had been saved. It now shows the stored settings.

Discriminates: passes on dev efe63bd34; with 963071768 reverted the field still reads the refused
"10" after the save; the accepted value passes on both.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.actors import Actor
from utils.admin_settings import save

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]


def _stored_jwt_expiry(admin: Actor) -> str:
    with admin.client() as client:
        return client.get("/api/v1/auths/admin/config").json()["JWT_EXPIRES_IN"]


def _save_jwt_expiry(page: Page, value: str) -> Locator:
    page.goto("/admin/settings/authentication")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_text("JWT Expiration", exact=True)).to_be_visible()
    # the field's label is not tied to it, so find it by its example text
    field = settings.get_by_placeholder('e.g.) "30m","1h", "10d".')
    field.fill(value)
    save(settings)
    return field


def test_a_refused_jwt_expiration_is_replaced_by_the_stored_one(
    page_for, make_user, admin, preserve
):
    preserve("admin_config")
    stored = _stored_jwt_expiry(admin)
    assert stored != "10"

    field = _save_jwt_expiry(page_for(make_user(role="admin")), "10")

    expect(field).to_have_value(stored)
    assert _stored_jwt_expiry(admin) == stored


# ---------------------------------------------------------------- nearby


def test_an_accepted_jwt_expiration_is_saved_and_shown(page_for, make_user, admin, preserve):
    preserve("admin_config")

    field = _save_jwt_expiry(page_for(make_user(role="admin")), "12h")

    expect(field).to_have_value("12h")
    assert _stored_jwt_expiry(admin) == "12h"
