"""Journey: an admin backs up and restores the instance from Admin Settings > Database.

Export Config downloads every stored setting as JSON, with what the admin saved elsewhere in it.
Import Config takes such a file back: an exported file with one setting changed is applied, so
a feature it switches off leaves every user's menu, while the admin stays signed in. The export
section downloads every account's chats as JSON, the accounts as CSV and, on SQLite, the
database file itself.

Discriminates: passes on dev 30f3f6a8f; in a frontend build whose Export Config saves an empty
object, whose All Chats export saves an empty list, whose Users export drops the rows and whose
Import Config posts an empty config, every test but the database download fails.
"""

from __future__ import annotations

import csv
import io
import json
import uuid
from pathlib import Path

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import backends
from utils.chat_ui import chat_input

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

ADMIN_CONFIG = "/api/v1/auths/admin/config"


def open_database_settings(page: Page) -> Locator:
    page.goto("/admin/settings/db")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_text("Export Config", exact=True)).to_be_visible()
    return settings


def row_button(settings: Locator, row_label: str, button: str) -> Locator:
    label = settings.get_by_text(row_label, exact=True)
    row = label.locator(f"xpath=ancestor::div[.//button[normalize-space()='{button}']][1]")
    return row.get_by_role("button", name=button, exact=True)


def download(page: Page, button: Locator) -> bytes:
    with page.expect_download() as downloaded:
        button.click()
    return Path(downloaded.value.path()).read_bytes()


def save_admin_config(admin, **values) -> None:
    with admin.client() as client:
        current = client.get(ADMIN_CONFIG).json()
        saved = client.post(ADMIN_CONFIG, json={**current, **values})
    saved.raise_for_status()


def test_export_config_downloads_what_the_admin_saved(page_for, admin, preserve):
    preserve("admin_config")
    watermark = f"Exported watermark {uuid.uuid4().hex[:6]}"
    save_admin_config(admin, RESPONSE_WATERMARK=watermark)
    page = page_for(admin)
    settings = open_database_settings(page)

    exported = json.loads(download(page, row_button(settings, "Export Config", "Export")))

    assert exported["ui.watermark"] == watermark
    assert exported["notes.enable"] is True


def test_an_imported_config_switches_notes_off_for_users(
    page_for, admin, make_user, preserve, tmp_path
):
    preserve("admin_config")
    save_admin_config(admin, ENABLE_NOTES=True)
    page = page_for(admin)
    settings = open_database_settings(page)
    exported = json.loads(download(page, row_button(settings, "Export Config", "Export")))
    edited = tmp_path / "config.json"
    edited.write_text(json.dumps({**exported, "notes.enable": False}))

    with page.expect_file_chooser() as chooser:
        row_button(settings, "Import Config", "Import").click()
    chooser.value.set_files(edited)

    expect(page.get_by_text("Config imported successfully")).to_be_visible()
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    user_page = page_for(make_user())
    expect(chat_input(user_page)).to_be_visible()
    user_page.get_by_role("button", name="User menu").first.click()
    menu = user_page.get_by_role("menu")
    expect(menu.get_by_role("button", name="Settings")).to_be_visible()
    expect(menu.get_by_role("link", name="Notes", exact=True)).to_have_count(0)


def test_all_chats_export_holds_every_accounts_chat(page_for, admin, make_user):
    title = f"Lighthouse log {uuid.uuid4().hex[:6]}"
    with make_user().client() as client:
        created = client.post("/api/v1/chats/new", json={"chat": {"title": title}})
    assert created.status_code == 200, created.text
    page = page_for(admin)
    settings = open_database_settings(page)

    exported = json.loads(download(page, row_button(settings, "All Chats", "Export")))

    assert title in [chat["title"] for chat in exported]


def test_users_export_lists_every_account(page_for, admin, make_user):
    account = make_user(name="Keeper Export")
    page = page_for(admin)
    settings = open_database_settings(page)

    exported = download(page, row_button(settings, "Users", "Export")).decode()

    rows = list(csv.DictReader(io.StringIO(exported)))
    expected = {"id": account.id, "name": "Keeper Export", "email": account.email, "role": "user"}
    assert expected in rows


@pytest.mark.skipif(backends.DATABASE != "sqlite", reason="only a SQLite database is downloaded")
def test_the_database_download_is_the_sqlite_file(page_for, admin):
    page = page_for(admin)
    settings = open_database_settings(page)

    downloaded = download(page, row_button(settings, "Database", "Database"))

    assert downloaded.startswith(b"SQLite format 3\x00")
