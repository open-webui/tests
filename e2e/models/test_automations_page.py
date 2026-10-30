"""Journey: a person keeps their automations on the Automations pages, and only if allowed to.

The list shows each automation with its schedule and opens it. On its own page an edit to the
title, prompt and schedule is saved and still there after a reload; pausing and resuming from
either page sticks the same way; deleting it from the page or from the list's menu takes it off
the list. A plain account sees no Automations entry and is sent home from the page until a group
grants the automations permission, and the admin's global switch takes them away from an admin
too.

Discriminates: passes on dev 176d31d1d; one frontend copy turns each test red: the list's daily
schedule without its time, the edit dialog sending the old title, the page's switch not saving,
both deletes not calling the server, the user menu offering Automations to everyone (the
no-permission and admin switch tests) and the list page letting only admins in (the group test).
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page, expect

from harness.access import make_group
from utils.automations_page import automation_row
from utils.user_menu import settings_item

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


def _open(page: Page, automation: dict) -> None:
    page.goto(f"/automations/{automation['id']}")
    expect(page.get_by_role("main")).to_contain_text(automation["data"]["prompt"])


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
    details = page.get_by_role("main")
    expect(details).to_contain_text(automation["data"]["prompt"])
    expect(details).to_contain_text("Daily at 9:00 AM")
    expect(details).to_contain_text("Active")


def test_an_edit_to_title_prompt_and_schedule_survives_a_reload(
    page_for, scheduler, make_automation
):
    automation = make_automation(scheduler)
    new_title = f"{automation['name']} (hourly)"
    new_prompt = "List the open incidents."
    page = page_for(scheduler)
    _open(page, automation)

    page.get_by_role("button", name="Edit", exact=True).click()
    editing = page.get_by_role("dialog")
    title = editing.get_by_role("textbox", name="Automation title")
    # the dialog fills its fields once its folders and channels load
    expect(title).to_have_value(automation["name"])
    title.fill(new_title)
    editing.get_by_role("textbox", name="Instructions").fill(new_prompt)
    editing.get_by_role("button", name="Daily").first.click()
    page.get_by_role("combobox").select_option(label="Hourly")
    editing.get_by_role("button", name="Hourly").first.click()
    editing.get_by_role("button", name="Save", exact=True).click()
    expect(editing).to_be_hidden()

    page.reload()
    details = page.get_by_role("main")
    expect(details).to_contain_text(new_prompt)
    expect(details).to_contain_text("Hourly")
    expect(page.get_by_text(new_title)).to_be_visible()
    expect(details).not_to_contain_text(automation["data"]["prompt"])


def test_pausing_and_resuming_sticks_on_both_pages(page_for, scheduler, make_automation):
    automation = make_automation(scheduler)
    page = page_for(scheduler)
    _open(page, automation)

    page.get_by_role("switch", name="Active").click()
    expect(page.get_by_role("main")).to_contain_text("Paused")
    page.reload()
    expect(page.get_by_role("switch", name="Paused")).not_to_be_checked()
    expect(page.get_by_role("main")).to_contain_text("Paused")

    row = _list_row(page, automation["name"])
    expect(row.get_by_role("switch")).not_to_be_checked()
    row.get_by_role("switch").click()
    expect(row.get_by_role("switch")).to_be_checked()

    _open(page, automation)
    expect(page.get_by_role("switch", name="Active")).to_be_checked()
    expect(page.get_by_role("main")).to_contain_text("Active")


def test_deleting_from_its_page_takes_it_off_the_list(page_for, scheduler, make_automation):
    automation = make_automation(scheduler)
    page = page_for(scheduler)
    _open(page, automation)

    page.get_by_role("button", name="Delete", exact=True).click()
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
    expect(page).to_have_url(re.compile(r"/$"))


def test_a_group_permission_opens_automations_to_a_user(page_for, make_user, admin):
    member = make_user()
    make_group(admin, [member], {"features": {"automations": True}})
    page = page_for(member)

    _open_user_menu(page)
    page.get_by_role("menu").get_by_role("menuitem", name="Automations").click()

    expect(page).to_have_url(re.compile(r"/automations$"))
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
    expect(page).to_have_url(re.compile(r"/$"))
