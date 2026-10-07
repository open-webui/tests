"""Journey: an automation made in the Create dialog, by schedule kind, model, folder and calendar.

A fresh admin creates an automation with each kind the Schedule menu offers (once, hourly,
daily, weekly on chosen days, monthly on a chosen day and a custom rule): each is stored as the
rule the menu built and its page and list entry describe it, and the daily one's next run is at
the time chosen, in the browser's own zone. An automation on a model preset runs with the
preset's system prompt and its tool, and its run's chat shows the tool's result. One filed in a
folder from the Destination menu runs into that folder: its chat is listed there and the
folder's system prompt reaches the model. The calendar's Scheduled Tasks show an active
automation's coming runs, which open the automation; once paused they are gone, and a run made
with Run now shows on today and opens its chat.

Renaming an automation in its Edit dialog keeps every schedule the menu can show as it was.

Bugs, both red on dev ebc6add67:

* The Once schedule fills its date with today's date in UTC and its time with the browser's
  local time, so in a browser whose day differs from UTC's the dialog starts on the wrong day
  (yesterday east of UTC, where Create is refused as "Scheduled time must be in the future", and
  tomorrow west of it, where the run comes a day late). The browser's zone is picked from the
  clock so the two days always differ. `test_the_once_schedule_starts_on_the_browsers_own_day`.
* The Edit dialog reads a stored rule into the menu's fields and builds it again on Save, which
  keeps only frequency, interval, days and time. A rule that ends, after a number of runs
  (`COUNT` with its `DTSTART`, as an API client or the model's automation tool may save one) or
  on a date (`UNTIL`, typed as a Custom rule), loses its end when the automation is only renamed,
  and then runs forever. `test_renaming_an_automation_keeps_the_end_of_its_schedule`.

Discriminates: passes on dev ebc6add67 apart from the two bugs. In a frontend copy building every
weekly rule without its days the weekly create and rename cases go red, and with the Once date
taken from the local day the Once day test turns green. In backend copies: the create route
storing each rule with a stray trailing separator turns every schedule and rename case red; the
next run computed in UTC turns the daily next-run test red; the middleware's folder lookup
returning no folder turns the folder test red (no prompt); the Scheduled Tasks left out of the
calendar listing turn the calendar test red; a run built without its model's defaults turns the
preset test red (the tool is not found).
"""

from __future__ import annotations

import datetime as dt
import re
import uuid
from zoneinfo import ZoneInfo

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.python_tools import EVERYONE_READS, python_tool
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import chat_input, conversation, expect_reply, last_reply

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

RUN_TIMEOUT_MS = 30_000
ZONE = "Asia/Tokyo"
IN_ZONE = {"timezone_id": ZONE, "locale": "en-US"}
NEXT_YEAR = dt.date.today().year + 1
TIDE_TOOL = '''
class Tools:
    def get_tide(self, port: str) -> str:
        """Read the tide table of a port."""
        return f"High water at {port}: 11:42"
'''
CLERK_PROMPT = "You are the tide clerk of Port Ellen."
HARBOUR_PROMPT = "Answer as the harbour master of Port Ellen."


def unique(label: str) -> str:
    return f"{label} {uuid.uuid4().hex[:6]}"


def open_create_dialog(page: Page) -> Locator:
    page.goto("/automations")
    page.get_by_role("main").get_by_role("button", name="Create", exact=True).click()
    dialog = page.get_by_role("dialog")
    expect(dialog.get_by_role("textbox", name="Enter prompt here.")).to_be_visible()
    return dialog


def pick_model(page: Page, dialog: Locator, name: str) -> None:
    dialog.get_by_role("button", name="Select model").first.click()
    picker = page.get_by_role("menu")
    picker.get_by_role("textbox", name="Search a model").fill(name)
    picker.get_by_role("button", name=name).click()


def choose_schedule(page: Page, dialog: Locator, current: str, kind: str) -> None:
    dialog.get_by_role("button", name=current, exact=True).first.click()
    page.get_by_role("combobox").filter(
        has=page.get_by_role("option", name="Custom")
    ).select_option(label=kind)


def close_schedule(dialog: Locator, kind: str) -> None:
    dialog.get_by_role("button", name=kind, exact=True).first.click()


def finish(page: Page, dialog: Locator, prompt: str, title: str) -> str:
    """Fill the prompt and title, press Create and return the new automation's id."""
    dialog.get_by_role("textbox", name="Enter prompt here.").fill(prompt)
    # the dialog clears the title once its folders and channels load, so it goes in last
    dialog.get_by_role("textbox", name="Automation title").fill(title)
    dialog.get_by_role("button", name="Create", exact=True).click()
    expect(page).to_have_url(re.compile(r"/automations/[0-9a-f-]+$"))
    expect(page.get_by_role("main")).to_contain_text(prompt)
    return page.url.rsplit("/", 1)[-1]


def stored(owner, automation_id: str) -> dict:
    with owner.client() as client:
        found = client.get(f"/api/v1/automations/{automation_id}")
    assert found.status_code == 200, found.text
    return found.json()


def detail(page: Page, label: str) -> Locator:
    """The value beside `label` in the automation page's summary."""
    return page.get_by_role("main").get_by_text(label, exact=True).locator("xpath=..")


def set_daily(page: Page) -> None:
    page.locator('input[type="time"]').fill("07:30")


def set_weekly(page: Page) -> None:
    page.locator('input[type="time"]').fill("08:15")
    page.get_by_role("button", name="Mo", exact=True).click()
    page.get_by_role("button", name="Th", exact=True).click()


def set_monthly(page: Page) -> None:
    page.locator('input[type="time"]').fill("18:00")
    page.locator('input[type="number"]').fill("15")


def set_custom(page: Page) -> None:
    page.get_by_placeholder("RRULE:FREQ=DAILY;BYHOUR=9;BYMINUTE=0").fill(
        "RRULE:FREQ=YEARLY;BYMONTH=6;BYMONTHDAY=1;BYHOUR=9;BYMINUTE=0"
    )


def set_once(page: Page) -> None:
    page.locator('input[type="date"]').fill(f"{NEXT_YEAR}-03-03")
    page.locator('input[type="time"]').fill("10:45")


SCHEDULE_KINDS = {
    "Once": (
        set_once,
        f"DTSTART:{NEXT_YEAR}0303T104500\nRRULE:FREQ=DAILY;COUNT=1",
        "Once · Mar 3 10:45 AM",
    ),
    "Hourly": (lambda page: None, "RRULE:FREQ=HOURLY;BYMINUTE=0", "Hourly"),
    "Daily": (set_daily, "RRULE:FREQ=DAILY;BYHOUR=7;BYMINUTE=30", "Daily at 7:30 AM"),
    "Weekly": (
        set_weekly,
        "RRULE:FREQ=WEEKLY;BYDAY=MO,TH;BYHOUR=8;BYMINUTE=15",
        "MO,TH at 8:15 AM",
    ),
    "Monthly": (
        set_monthly,
        "RRULE:FREQ=MONTHLY;BYMONTHDAY=15;BYHOUR=18;BYMINUTE=0",
        "Monthly 15 at 6:00 PM",
        "Monthly on the 15th at 6:00 PM",
    ),
    "Custom": (
        set_custom,
        "RRULE:FREQ=YEARLY;BYMONTH=6;BYMONTHDAY=1;BYHOUR=9;BYMINUTE=0",
        "RRULE:FREQ=YEARLY;BYMONTH=6;BYMONTHDAY=1;BYHOUR=9;BYMINUTE=0",
    ),
}


@pytest.mark.parametrize("kind", list(SCHEDULE_KINDS))
def test_each_schedule_kind_is_saved_as_chosen_and_described(page_for, scheduler, kind):
    fill, rule, described, *listed = SCHEDULE_KINDS[kind]
    title = unique(f"{kind} report")
    page = page_for(scheduler, **IN_ZONE)
    dialog = open_create_dialog(page)
    pick_model(page, dialog, MOCK_MODEL_ID)

    choose_schedule(page, dialog, "Daily", kind)
    fill(page)
    close_schedule(dialog, kind)
    automation_id = finish(page, dialog, f"Write the {kind.lower()} report.", title)

    assert stored(scheduler, automation_id)["data"]["rrule"] == rule
    expect(detail(page, "Schedule")).to_contain_text(described)
    page.goto("/automations")
    page.get_by_role("textbox", name="Search Automations").fill(title)
    row = page.get_by_role("button", name="Open automation").filter(has_text=title)
    # the list words a monthly rule its own way
    expect(row).to_contain_text(listed[0] if listed else described)


def test_a_daily_run_is_due_at_the_chosen_time_in_the_browsers_zone(page_for, scheduler):
    page = page_for(scheduler, **IN_ZONE)
    dialog = open_create_dialog(page)
    pick_model(page, dialog, MOCK_MODEL_ID)
    choose_schedule(page, dialog, "Daily", "Daily")
    set_daily(page)
    close_schedule(dialog, "Daily")
    automation_id = finish(page, dialog, "Read the morning tides.", unique("Tides"))

    next_run_ns = stored(scheduler, automation_id)["next_run_at"]
    next_run = dt.datetime.fromtimestamp(next_run_ns / 1e9, ZoneInfo(ZONE))
    assert (next_run.hour, next_run.minute) == (7, 30), next_run
    expect(detail(page, "Next run")).to_contain_text(re.compile(r"7:30\s?AM"))


def zone_on_another_day() -> tuple[str, dt.date]:
    """A zone whose date differs from UTC's right now, away from its midnight, and that date."""
    now = dt.datetime.now(dt.timezone.utc)
    zone = "Etc/GMT-14" if now.hour >= 10 else "Etc/GMT+12"
    return zone, now.astimezone(ZoneInfo(zone)).date()


@pytest.mark.regression
def test_the_once_schedule_starts_on_the_browsers_own_day(page_for, scheduler):
    zone, local_today = zone_on_another_day()
    page = page_for(scheduler, timezone_id=zone, locale="en-US")
    dialog = open_create_dialog(page)

    choose_schedule(page, dialog, "Daily", "Once")

    expect(
        page.locator('input[type="date"]'),
        "the Once date starts on today's date in UTC, not the browser's own day",
    ).to_have_value(local_today.isoformat())


@pytest.fixture
def tide_clerk(admin):
    """A preset of the scripted model with a system prompt and a tide tool; yields its name."""
    name = unique("Tide clerk")
    model_id = name.lower().replace(" ", "-")
    with python_tool(admin, TIDE_TOOL) as tool_id, admin.client() as client:
        created = client.post(
            "/api/v1/models/create",
            json={
                "id": model_id,
                "base_model_id": MOCK_MODEL_ID,
                "name": name,
                "meta": {"toolIds": [tool_id]},
                "params": {"system": CLERK_PROMPT},
                "access_grants": [EVERYONE_READS],
            },
        )
        assert created.status_code == 200, created.text
        yield name
        client.post("/api/v1/models/model/delete", json={"id": model_id})


def run_now_and_open_the_chat(page: Page) -> None:
    details = page.get_by_role("main")
    details.get_by_role("button", name="Run now").click()
    view_chat = details.get_by_role("button", name="View chat")
    expect(view_chat).to_be_visible(timeout=RUN_TIMEOUT_MS)
    view_chat.click()
    expect(page).to_have_url(re.compile(r"/c/[0-9a-f-]+$"))


def requests_for(upstream, prompt: str) -> list[dict]:
    return [body for body in upstream.chat_requests() if reply.answering(prompt)(body)]


def system_text(request: dict) -> str:
    return "\n".join(
        str(message["content"]) for message in request["messages"] if message["role"] == "system"
    )


def test_an_automation_on_a_preset_runs_with_its_prompt_and_tool(
    page_for, scheduler, upstream, tide_clerk
):
    prompt = f"Read the tide at Port Ellen, batch {uuid.uuid4().hex[:6]}."
    upstream.queue(
        reply.tool_call("get_tide", {"port": "Port Ellen"}, match=reply.answering(prompt)),
        reply.text("The tide is in at 11:42.", match=reply.answering(prompt)),
    )
    page = page_for(scheduler, **IN_ZONE)
    dialog = open_create_dialog(page)
    pick_model(page, dialog, tide_clerk)
    finish(page, dialog, prompt, unique("Tide check"))
    expect(detail(page, "Model")).to_contain_text(tide_clerk.lower().replace(" ", "-"))

    run_now_and_open_the_chat(page)
    expect_reply(page, "The tide is in at 11:42.")
    last_reply(page).get_by_text("View Result from get_tide").click()
    expect(last_reply(page)).to_contain_text("High water at Port Ellen: 11:42")

    first, *_ = requests_for(upstream, prompt)
    assert system_text(first).startswith(CLERK_PROMPT), system_text(first)
    offered = {tool["function"]["name"] for tool in first.get("tools") or []}
    assert "get_tide" in offered, offered


def test_an_automation_filed_in_a_folder_runs_into_it_under_its_prompt(
    page_for, scheduler, upstream
):
    folder = unique("Harbour")
    with scheduler.client() as client:
        created = client.post(
            "/api/v1/folders/", json={"name": folder, "data": {"system_prompt": HARBOUR_PROMPT}}
        )
    assert created.status_code == 200, created.text
    folder_id = created.json()["id"]
    prompt = f"List the ships in port, batch {uuid.uuid4().hex[:6]}."
    title = unique("Harbour log")
    upstream.queue(reply.text("Two trawlers and a ferry.", match=reply.answering(prompt)))
    page = page_for(scheduler, **IN_ZONE)
    dialog = open_create_dialog(page)
    pick_model(page, dialog, MOCK_MODEL_ID)

    dialog.get_by_role("button", name="New chat", exact=True).last.click()
    destinations = page.get_by_role("menu")
    destinations.get_by_role("button", name="Folder", exact=True).click()
    destinations.get_by_placeholder("Search folders").fill(folder)
    destinations.get_by_role("button", name=folder).click()
    expect(dialog.get_by_role("button", name=folder).last).to_be_visible()
    automation_id = finish(page, dialog, prompt, title)

    assert stored(scheduler, automation_id)["folder_id"] == folder_id
    expect(detail(page, "Destination")).to_contain_text(f"Folder: {folder}")
    run_now_and_open_the_chat(page)
    expect_reply(page, "Two trawlers and a ferry.")
    chat_id = page.url.rsplit("/", 1)[-1]
    with scheduler.client() as client:
        assert client.get(f"/api/v1/chats/{chat_id}").json()["folder_id"] == folder_id

    page.goto(f"/folders/{folder_id}")
    expect(chat_input(page)).to_be_visible()
    expect(page.get_by_role("link", name=re.compile(re.escape(title)))).to_have_attribute(
        "href", f"/c/{chat_id}"
    )
    [request] = requests_for(upstream, prompt)
    assert system_text(request).startswith(HARBOUR_PROMPT), system_text(request)


def calendar_chips(page: Page, name: str) -> Locator:
    return page.get_by_role("button", name=name).filter(has_not=page.get_by_role("button"))


def test_the_calendar_shows_an_active_automations_runs_and_opens_them(
    page_for, scheduler, upstream
):
    title = unique("Ferry report")
    prompt = f"Report the ferry times, batch {uuid.uuid4().hex[:6]}."
    tomorrow = dt.datetime.now(ZoneInfo(ZONE)) + dt.timedelta(days=1)
    weekday = ["MO", "TU", "WE", "TH", "FR", "SA", "SU"][tomorrow.weekday()]
    with scheduler.client() as client:
        created = client.post(
            "/api/v1/automations/create",
            json={
                "name": title,
                "is_active": True,
                "data": {
                    "prompt": prompt,
                    "model_id": MOCK_MODEL_ID,
                    "rrule": f"RRULE:FREQ=WEEKLY;BYDAY={weekday};BYHOUR=9;BYMINUTE=0",
                    "target": {"type": "chat"},
                },
            },
        )
    assert created.status_code == 200, created.text
    automation_id = created.json()["id"]
    upstream.queue(reply.text("Ferries at ten and four.", match=reply.answering(prompt)))
    page = page_for(scheduler, **IN_ZONE)

    page.goto("/calendar")
    upcoming = calendar_chips(page, title)
    expect(upcoming.first).to_contain_text(re.compile(r"9:00\s?AM"))
    upcoming.first.click()
    expect(page).to_have_url(re.compile(f"/automations/{automation_id}$"))

    page.get_by_role("switch", name="Active").click()
    expect(detail(page, "Status")).to_contain_text("Paused")
    page.get_by_role("main").get_by_role("button", name="Run now").click()
    expect(page.get_by_role("main").get_by_role("button", name="View chat")).to_be_visible(
        timeout=RUN_TIMEOUT_MS
    )

    page.goto("/calendar")
    expect(calendar_chips(page, title)).to_have_count(1)
    calendar_chips(page, title).click()
    expect(page).to_have_url(re.compile(r"/c/[0-9a-f-]+$"))
    expect(conversation(page).get_by_text(prompt)).to_be_visible()
    expect_reply(page, "Ferries at ten and four.")


KEPT_ON_RENAME = {
    "weekly": "RRULE:FREQ=WEEKLY;BYDAY=MO,TH;BYHOUR=8;BYMINUTE=15",
    "monthly": "RRULE:FREQ=MONTHLY;BYMONTHDAY=15;BYHOUR=18;BYMINUTE=0",
    "hourly": "RRULE:FREQ=HOURLY;BYMINUTE=30",
    "custom": "RRULE:FREQ=YEARLY;BYMONTH=6;BYMONTHDAY=1;BYHOUR=9;BYMINUTE=0",
    "once": f"DTSTART:{NEXT_YEAR}0303T104500\nRRULE:FREQ=DAILY;COUNT=1",
    "every other day": "RRULE:FREQ=DAILY;INTERVAL=2;BYHOUR=6;BYMINUTE=0",
}
ENDING = {
    "after four runs": f"DTSTART:{NEXT_YEAR}0303T090000\nRRULE:FREQ=DAILY;COUNT=4",
    "on a date": f"RRULE:FREQ=DAILY;BYHOUR=9;BYMINUTE=0;UNTIL={NEXT_YEAR}0601T000000",
}


def rename_in_the_edit_dialog(page_for, scheduler, rule: str) -> str:
    """Save `rule` as an automation, rename it in its Edit dialog and return the stored rule."""
    title = unique("Rota")
    with scheduler.client() as client:
        created = client.post(
            "/api/v1/automations/create",
            json={
                "name": title,
                "is_active": False,
                "data": {"prompt": "Post the rota.", "model_id": MOCK_MODEL_ID, "rrule": rule},
            },
        )
    assert created.status_code == 200, created.text
    automation_id = created.json()["id"]
    page = page_for(scheduler, **IN_ZONE)
    page.goto(f"/automations/{automation_id}")
    expect(page.get_by_role("main")).to_contain_text("Post the rota.")

    page.get_by_role("button", name="Edit", exact=True).click()
    editing = page.get_by_role("dialog")
    name = editing.get_by_role("textbox", name="Automation title")
    expect(name).to_have_value(title)
    name.fill(f"{title} renamed")
    editing.get_by_role("button", name="Save", exact=True).click()
    expect(editing).to_be_hidden()
    expect(page.get_by_text(f"{title} renamed")).to_be_visible()
    return stored(scheduler, automation_id)["data"]["rrule"]


@pytest.mark.parametrize("label", list(KEPT_ON_RENAME))
def test_renaming_an_automation_keeps_its_schedule(page_for, scheduler, label):
    rule = KEPT_ON_RENAME[label]
    assert rename_in_the_edit_dialog(page_for, scheduler, rule) == rule


@pytest.mark.regression
@pytest.mark.parametrize("label", list(ENDING))
def test_renaming_an_automation_keeps_the_end_of_its_schedule(page_for, scheduler, label):
    rule = ENDING[label]
    saved = rename_in_the_edit_dialog(page_for, scheduler, rule)
    assert saved == rule, (
        f"renaming the automation rewrote its schedule {rule!r} as {saved!r}: it no longer ends"
    )
