"""Journey: a person keeps their automations on the Automations pages, and only if allowed to.

The list shows each automation with its schedule and opens it. On its own page, which is its
editor since dev 9723b3a6a, an edit to the title, prompt and schedule is saved with the header's
Save and still there after a reload, the new title on its page tab as well (since dev da850248d
the page opens as a tab named after the automation); pausing and
resuming from either page sticks the same way; deleting it from the page or from the list's menu
takes it off the list. A plain account sees no Automations entry and is refused the page until
a group grants the automations permission, and the admin's global switch takes them away from an
admin too. Since dev da850248d a refused page keeps its address and shows a permission message in
place of the list; it sent the account home before.

Bug, open-webui/open-webui#34427: the page rebuilds a stored schedule in its own form and compared
that with the stored text, so an automation the model's tool saved with a start date opened with
Save already showing. `test_an_automation_made_in_a_chat_opens_without_unsaved_changes` pins it
(fix PR branch fix-automation-page-save-on-open).

Discriminates: passes on dev 176d31d1d; one frontend copy turns each test red: the list's daily
schedule without its time, the edit dialog sending the old title, the page's switch not saving,
both deletes not calling the server, the user menu offering Automations to everyone (the
no-permission and admin switch tests) and the list page letting only admins in (the group test).
Retargeted to the editor page on dev 2fdf4ce6b, where a frontend copy saving the old title turns
the edit test red and one whose switch does not save the pausing test. The unsaved changes test
is red on dev 2fdf4ce6b and passes on the fix branch.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page, expect

from harness.access import make_group
from utils.automations_page import (
    automation_page,
    automation_row,
    expect_no_runs,
    instructions,
    name_box,
    open_automation,
    open_page_menu_entry,
    property_row,
    save_changes,
)
from utils.page_tabs import expect_no_permission, page_tab
from utils.user_menu import settings_item

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


def _list_row(page: Page, name: str):
    page.goto("/automations")
    page.get_by_role("textbox", name="Search Automations").fill(name)
    return automation_row(page, name)


def _open_user_menu(page: Page) -> None:
    page.get_by_role("navigation", name="Chat history").get_by_label("User menu").click()
    expect(settings_item(page)).to_be_visible()


def test_an_automation_is_listed_with_its_schedule_and_opens_from_the_list(
    page_for, scheduler, make_automation
):
    automation = make_automation(scheduler)
    page = page_for(scheduler)

    row = _list_row(page, automation["name"])
    expect(row).to_contain_text("Daily at 9:00 AM · New chat")
    row.click()

    expect(page).to_have_url(re.compile(f"/automations/{automation['id']}$"))
    expect(instructions(page)).to_have_value(automation["data"]["prompt"])
    expect(property_row(page, "Schedule")).to_contain_text("Daily")
    expect(property_row(page, "Next run")).to_contain_text(re.compile(r"9:00\s?AM"))
    expect(property_row(page, "Status")).to_contain_text("Active")


def test_an_edit_to_title_prompt_and_schedule_survives_a_reload(
    page_for, scheduler, make_automation
):
    automation = make_automation(scheduler)
    new_title = f"{automation['name']} (hourly)"
    new_prompt = "List the open incidents."
    page = page_for(scheduler)
    open_automation(page, automation)

    name_box(page).fill(new_title)
    instructions(page).fill(new_prompt)
    automation_page(page).get_by_role("button", name="Daily", exact=True).click()
    page.get_by_role("combobox").select_option(label="Hourly")
    automation_page(page).get_by_role("button", name="Hourly", exact=True).click()
    save_changes(page)

    page.reload()
    expect(instructions(page)).to_have_value(new_prompt)
    expect(name_box(page)).to_have_value(new_title)
    expect(property_row(page, "Schedule")).to_contain_text("Hourly")
    expect(page_tab(page, new_title).get_by_text(new_title, exact=True)).to_be_visible()


@pytest.mark.regression
def test_an_automation_made_in_a_chat_opens_without_unsaved_changes(
    page_for, scheduler, make_automation
):
    # the schedule as the model's automation tool writes it, after the tool's own example
    automation = make_automation(scheduler, rrule="DTSTART:20990101T090000\nRRULE:FREQ=DAILY")
    page = page_for(scheduler)
    open_automation(page, automation)
    expect_no_runs(page)

    save = automation_page(page).get_by_role("button", name="Save", exact=True)
    expect(
        save, "the page offers Save before anything was changed (open-webui/open-webui#34427)"
    ).to_have_count(0)
    name_box(page).fill(f"{automation['name']} renamed")
    expect(save).to_be_visible()


def test_pausing_and_resuming_sticks_on_both_pages(page_for, scheduler, make_automation):
    automation = make_automation(scheduler)
    page = page_for(scheduler)
    open_automation(page, automation)

    automation_page(page).get_by_role("switch", name="Active").click()
    expect(property_row(page, "Status")).to_contain_text("Paused")
    page.reload()
    expect(automation_page(page).get_by_role("switch", name="Paused")).not_to_be_checked()
    expect(property_row(page, "Status")).to_contain_text("Paused")

    row = _list_row(page, automation["name"])
    expect(row.get_by_role("switch")).not_to_be_checked()
    row.get_by_role("switch").click()
    expect(row.get_by_role("switch")).to_be_checked()

    open_automation(page, automation)
    expect(automation_page(page).get_by_role("switch", name="Active")).to_be_checked()
    expect(property_row(page, "Status")).to_contain_text("Active")


def test_deleting_from_its_page_takes_it_off_the_list(page_for, scheduler, make_automation):
    automation = make_automation(scheduler)
    page = page_for(scheduler)
    open_automation(page, automation)

    open_page_menu_entry(page, "Delete")
    confirm = page.get_by_role("dialog")
    expect(confirm).to_contain_text(automation["name"])
    confirm.get_by_role("button", name="Confirm").click()

    expect(page).to_have_url(re.compile(r"/automations$"))
    _list_row(page, automation["name"])
    expect(page.get_by_text("No results found")).to_be_visible()


def test_deleting_from_the_list_menu_takes_it_off_the_list(page_for, scheduler, make_automation):
    automation = make_automation(scheduler)
    kept = make_automation(scheduler)
    page = page_for(scheduler)

    row = _list_row(page, automation["name"])
    row.get_by_role("button", name="Automation Menu").first.click()
    page.get_by_role("menu").get_by_role("menuitem", name="Delete").click()
    page.get_by_role("dialog").get_by_role("button", name="Confirm").click()
    expect(page.get_by_text(f"Deleted {automation['name']}")).to_be_visible()

    page.reload()
    _list_row(page, automation["name"])
    expect(page.get_by_text("No results found")).to_be_visible()
    expect(_list_row(page, kept["name"])).to_be_visible()


def test_a_user_without_the_permission_sees_no_automations(page_for, make_user):
    page = page_for(make_user())
    _open_user_menu(page)
    expect(page.get_by_role("menu").get_by_role("menuitem", name="Automations")).to_have_count(0)

    page.goto("/automations")
    expect_no_permission(page, "/automations")
    expect(page.get_by_role("textbox", name="Search Automations")).to_have_count(0)


def test_a_group_permission_opens_automations_to_a_user(page_for, make_user, admin):
    member = make_user()
    make_group(admin, [member], {"features": {"automations": True}})
    page = page_for(member)

    _open_user_menu(page)
    page.get_by_role("menu").get_by_role("menuitem", name="Automations").click()

    expect(page).to_have_url(re.compile(r"/automations$"))
    expect(page.get_by_role("textbox", name="Search Automations")).to_be_visible()
    expect(page.get_by_text("No automations found")).to_be_visible()


def test_the_admin_switch_takes_automations_away_from_an_admin(
    page_for, make_user, admin, preserve
):
    preserve("admin_config")
    with admin.client() as client:
        current = client.get("/api/v1/auths/admin/config").json()
        switched = client.post(
            "/api/v1/auths/admin/config", json={**current, "ENABLE_AUTOMATIONS": False}
        )
    assert switched.status_code == 200, switched.text
    page = page_for(make_user(role="admin"))

    _open_user_menu(page)
    expect(page.get_by_role("menu").get_by_role("menuitem", name="Automations")).to_have_count(0)
    page.goto("/automations")
    expect_no_permission(page, "/automations")
    expect(page.get_by_role("textbox", name="Search Automations")).to_have_count(0)
