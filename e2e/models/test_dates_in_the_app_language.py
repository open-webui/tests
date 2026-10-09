"""Regression: schedules and dates stayed in English for a person using the app in another language.

Fix 7d205a86d: an automation's schedule was worded in English with an English clock ("Daily at
9:00 AM") on the Automations list and on its own page whatever language the person chose, and
dates written through dayjs (the shared chat's date among them) stayed English after a reload,
since the app handed dayjs a language code such as `de-DE` it does not know. Schedules are now
worded through the app's translations and clock, and dayjs follows the chosen language.

The browser itself runs in English here, so only the language picked in Settings can make the
page German.

Discriminates: passes on dev 1c010b438; in a frontend build with 7d205a86d reverted all four
fail (English schedules, "March 3, 2026 10:45 AM" on the shared chat).
"""

from __future__ import annotations

import datetime as dt
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness.actors import Actor
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

ENGLISH_BROWSER = {"timezone_id": "UTC", "locale": "en-US"}
SCHEDULES = {
    "daily": ("DTSTART:20990101T090000\nRRULE:FREQ=DAILY;BYHOUR=9;BYMINUTE=0", "Täglich um 9:00"),
    "weekly": (
        "DTSTART:20990101T081500\nRRULE:FREQ=WEEKLY;BYDAY=MO,TH;BYHOUR=8;BYMINUTE=15",
        "Mo, Do um 8:15",
    ),
    "once": ("DTSTART:20990303T104500\nRRULE:FREQ=DAILY;COUNT=1", "Einmalig · 3. März 10:45"),
}
SHARED_AT = dt.datetime(2026, 3, 3, 10, 45, tzinfo=dt.timezone.utc)


def in_german(page: Page) -> Page:
    """Pick Deutsch in Settings > General, then reload as a returning visit would."""
    page.goto("/?settings=general")
    tab = page.locator("#tab-general")
    tab.get_by_role("combobox", name="Language").select_option("de-DE")
    expect(page.locator("html")).to_have_attribute("lang", "de-DE")
    page.reload()
    expect(page.locator("html")).to_have_attribute("lang", "de-DE")
    return page


def create_automation(owner: Actor, rrule: str) -> dict:
    form = {
        "name": f"Hafenbericht {uuid.uuid4().hex[:6]}",
        "is_active": True,
        "data": {
            "prompt": f"Write the harbour report, batch {uuid.uuid4().hex[:6]}.",
            "model_id": MOCK_MODEL_ID,
            "rrule": rrule,
            "target": {"type": "chat"},
        },
    }
    with owner.client() as client:
        created = client.post("/api/v1/automations/create", json=form)
    assert created.status_code == 200, created.text
    return created.json()


@pytest.mark.parametrize("kind", list(SCHEDULES))
def test_a_schedule_is_worded_in_the_chosen_language(page_for, scheduler, kind):
    rrule, worded = SCHEDULES[kind]
    automation = create_automation(scheduler, rrule)
    page = in_german(page_for(scheduler, **ENGLISH_BROWSER))

    page.goto("/automations")
    page.get_by_role("textbox", name="Automatisierungen suchen").fill(automation["name"])
    row = page.get_by_role("button", name="Automatisierung öffnen").filter(
        has_text=automation["name"]
    )
    expect(row).to_contain_text(f"{worded} · Neuer Chat")

    row.click()
    schedule = page.get_by_role("main").get_by_text("Zeitplan", exact=True).locator("xpath=..")
    expect(schedule).to_contain_text(worded)


def test_a_shared_chat_shows_its_date_in_the_chosen_language(page_for, make_user):
    owner = make_user()
    chat = {
        "title": "Fährplan",
        "models": [MOCK_MODEL_ID],
        "timestamp": int(SHARED_AT.timestamp() * 1000),
        "history": {"currentId": None, "messages": {}},
    }
    with owner.client() as client:
        created = client.post("/api/v1/chats/new", json={"chat": chat})
        assert created.status_code == 200, created.text
        shared = client.post(f"/api/v1/chats/{created.json()['id']}/share")
    assert shared.status_code == 200, shared.text
    page = in_german(page_for(owner, **ENGLISH_BROWSER))

    page.goto(f"/s/{shared.json()['share_id']}")

    expect(page.locator("header time")).to_have_text("3. März 2026 10:45")
