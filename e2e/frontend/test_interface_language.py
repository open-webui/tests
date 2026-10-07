"""Journey: a language picked in Settings relabels the whole open app at once, dates included.

A fresh account picks Deutsch in Settings > General. Without a reload the sidebar, the chat
input's placeholder and the Settings dialog speak German, and moving on to the workspace and
the admin panel through the sidebar finds them in German too; a reload keeps all of it. Dates
are written in the picked language as well: the admin's user list gives an account's creation
date and last activity in English, in German after the pick and in Arabic after the next one.
That the Settings tab and the user menu turn German and stay after a reload is
e2e/config/test_settings_general.py, and how the first language is chosen (the browser's,
DEFAULT_LOCALE, a `?lang=` link) is e2e/frontend/test_browser_language_detection.py.

Discriminates: passes on the dev ebc6add67 build; in a frontend copy whose i18n store is not
refreshed when the language changes, the at-once tests go red (the sidebar and the input stay
English until a reload), and in one that never hands the language to the date library, the
date test goes red (the German list still reads the English month and "ago").
"""

from __future__ import annotations

import datetime as dt
import re

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.actors import Actor
from utils.chat_ui import chat_input

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

ENGLISH_MONTHS = [dt.date(2000, month, 1).strftime("%B") for month in range(1, 13)]
GERMAN_MONTHS = (
    "Januar Februar März April Mai Juni Juli August September Oktober November Dezember".split()
)
ARABIC_MONTHS = "يناير فبراير مارس أبريل مايو يونيو يوليو أغسطس سبتمبر أكتوبر نوفمبر ديسمبر".split()
ENGLISH_PLACEHOLDER = "How can I help you today?"
GERMAN_PLACEHOLDER = "Wie kann ich Ihnen heute helfen?"
IN_UTC = {"timezone_id": "UTC", "locale": "en-US"}


def pick_language(page: Page, code: str) -> None:
    page.goto("/?settings=general")
    settings = page.get_by_role("dialog")
    # found by what it offers, since its own label changes with the language
    picker = settings.locator("select").filter(has=page.locator("option[value='en-US']"))
    picker.select_option(code)
    expect(page.locator("html")).to_have_attribute("lang", code)


def close_settings(page: Page) -> None:
    page.keyboard.press("Escape")
    expect(page.get_by_role("dialog")).to_be_hidden()


def sidebar(page: Page, label: str) -> Locator:
    return page.get_by_role("navigation", name=label, exact=True)


def input_placeholder(page: Page) -> Locator:
    return chat_input(page).locator("[data-placeholder]")


def test_the_sidebar_input_and_settings_turn_german_without_a_reload(page_for, make_user):
    page = page_for(make_user())
    expect(input_placeholder(page)).to_have_attribute("data-placeholder", ENGLISH_PLACEHOLDER)

    pick_language(page, "de-DE")

    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("tab", name="Benutzeroberfläche", exact=True)).to_be_visible()
    close_settings(page)
    expect(sidebar(page, "Chatverlauf").get_by_role("link", name="Neuer Chat")).to_be_visible()
    expect(sidebar(page, "Chatverlauf").get_by_label("Suchen", exact=True)).to_be_visible()
    expect(page.get_by_role("link", name="New Chat", exact=True)).to_have_count(0)
    expect(input_placeholder(page)).to_have_attribute("data-placeholder", GERMAN_PLACEHOLDER)

    page.reload()
    expect(sidebar(page, "Chatverlauf").get_by_role("link", name="Neuer Chat")).to_be_visible()
    expect(input_placeholder(page)).to_have_attribute("data-placeholder", GERMAN_PLACEHOLDER)


def test_the_workspace_and_admin_panel_follow_the_pick_without_a_reload(page_for, make_user):
    page = page_for(make_user(role="admin"))
    pick_language(page, "de-DE")
    close_settings(page)

    sidebar(page, "Chatverlauf").get_by_role("link", name="Arbeitsbereich").click()
    workspace = page.get_by_role("main")
    expect(workspace.get_by_role("link", name="Modelle")).to_be_visible()
    expect(workspace.get_by_role("link", name="Werkzeuge")).to_be_visible()

    sidebar(page, "Chatverlauf").get_by_label("Benutzermenü").click()
    page.get_by_role("menu").get_by_text("Admin-Bereich").click()
    admin_panel = page.get_by_role("main")
    expect(admin_panel.get_by_role("link", name="Benutzer", exact=True)).to_be_visible()
    expect(admin_panel.get_by_role("link", name="Einstellungen", exact=True)).to_be_visible()
    expect(admin_panel.get_by_role("button", name="Benutzer hinzufügen")).to_be_visible()

    page.reload()
    add_user = page.get_by_role("main").get_by_role("button", name="Benutzer hinzufügen")
    expect(add_user).to_be_visible()


def created_on(admin: Actor, account: Actor) -> dt.date:
    with admin.client() as client:
        stored = client.get(f"/api/v1/users/{account.id}").json()
    return dt.datetime.fromtimestamp(stored["created_at"], dt.timezone.utc).date()


def user_row(page: Page, account: Actor) -> Locator:
    page.get_by_role("main").locator("input").first.fill(account.email)
    row = page.get_by_role("main").locator("tr", has_text=account.email)
    expect(row).to_have_count(1)
    return row


def test_dates_in_the_user_list_follow_the_picked_language(page_for, make_user, admin):
    viewer = make_user(role="admin")
    created = created_on(admin, viewer)
    page = page_for(viewer, **IN_UTC)
    page.goto("/admin/users")
    english = f"{ENGLISH_MONTHS[created.month - 1]} {created.day}, {created.year}"
    expect(user_row(page, viewer)).to_contain_text(english)

    pick_language(page, "de-DE")
    close_settings(page)
    page.goto("/admin/users")
    row = user_row(page, viewer)
    expect(row).to_contain_text(f"{created.day}. {GERMAN_MONTHS[created.month - 1]} {created.year}")
    expect(row).to_contain_text(re.compile(r"vor (ein|einer|\d)"))
    expect(row).not_to_contain_text("ago")

    pick_language(page, "ar")
    close_settings(page)
    page.goto("/admin/users")
    row = user_row(page, viewer)
    expect(row).to_contain_text(ARABIC_MONTHS[created.month - 1])
    expect(row).not_to_contain_text(ENGLISH_MONTHS[created.month - 1])
