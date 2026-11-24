"""Journey: the background limit sits right under the switch that turns sub-agents on.

Every sub-agent now works on while the lead reply ends and reports back by itself, so the
Sub-agents tab has no separate switch for background sub-agents any more. Turning sub-agents on
shows the limit on how many may run at once in the background straight away, and the limit an
admin types in is saved, shown again after a reload and read back from the settings API.

Discriminates: written on dev 9b626e4e5 (the switch removed in 8522af2d6).
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from utils.admin_settings import save

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

SUBAGENTS = ("/api/v1/configs/subagents", "/api/v1/configs/subagents")


def _open_tab(page: Page) -> Locator:
    page.goto("/admin/settings/subagents")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("switch", name="Enable sub-agents")).to_be_visible()
    return settings


def test_turning_sub_agents_on_shows_the_background_limit_and_it_saves(
    make_user, page_for, admin, preserve
):
    preserve(SUBAGENTS)
    with admin.client() as client:
        current = client.get(SUBAGENTS[0]).json()
        started = client.post(
            SUBAGENTS[1],
            json={**current, "ENABLE_SUBAGENTS": False, "SUBAGENTS_MAX_ASYNC": None},
        )
    assert started.status_code == 200, started.text
    page = page_for(make_user(role="admin"))
    settings = _open_tab(page)
    limit = settings.get_by_label("Max background")
    expect(limit).to_have_count(0)

    settings.get_by_role("switch", name="Enable sub-agents").click()

    expect(limit, "the background limit did not show once sub-agents were on").to_be_visible()
    expect(limit).to_have_value("")
    expect(settings.get_by_role("switch", name="Enable background sub-agents")).to_have_count(0)
    limit.fill("3")
    save(settings)

    with admin.client() as client:
        stored = client.get(SUBAGENTS[0]).json()
    assert stored["ENABLE_SUBAGENTS"] is True, stored
    assert stored["SUBAGENTS_MAX_ASYNC"] == 3, stored
    expect(_open_tab(page).get_by_label("Max background")).to_have_value("3")
