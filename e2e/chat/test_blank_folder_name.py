"""Regression: a folder renamed to spaces was saved with a blank name and vanished from the sidebar.

Issue open-webui/open-webui#31379, fix da36d149b (PR open-webui/open-webui#31380). The rename and
create-subfolder handlers of the sidebar folder and of the folder page checked for an empty name
before trimming it, so a name of only spaces passed the check, was trimmed to nothing and saved.
The folder and its chats then disappeared from the sidebar. The name is now trimmed first, and
a blank one is refused with "Folder name cannot be empty.".

The dialogs are submitted with one click on Save, as a person does. On dev efe63bd34 that click was
swallowed because the folder menus left themselves open behind the dialog (the bug of
open-webui/open-webui#31486), until ef67cc3fa (open-webui/open-webui#31496) closed them first.

Discriminates: passes on dev 176d31d1d and a5bc78300 and fails with da36d149b reverted (the blank
name is saved and "Folder updated successfully" shows).
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from utils.sidebar_folders import click_folder_menu_button, folder_link, folder_row

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

NAME = "Work"
EMPTY_NAME_ERROR = "Folder name cannot be empty."


@pytest.fixture
def owner(make_user):
    return make_user()


@pytest.fixture
def folder_id(owner) -> str:
    with owner.client() as client:
        created = client.post("/api/v1/folders/", json={"name": NAME})
    assert created.status_code == 200, created.text
    return created.json()["id"]


def stored_folders(owner) -> list[dict]:
    with owner.client() as client:
        listed = client.get("/api/v1/folders/")
    assert listed.status_code == 200, listed.text
    return listed.json()


def open_sidebar(page: Page) -> Locator:
    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    sidebar = page.get_by_role("navigation", name="Chat history")
    show_folders(sidebar)
    return sidebar


def show_folders(sidebar: Locator) -> None:
    section = sidebar.get_by_role("button", name="Folders", exact=True)
    expect(section).to_be_visible()
    if section.get_attribute("aria-expanded") != "true":
        section.click()


def folder_menu(sidebar: Locator, name: str) -> Locator:
    """Open the "More" menu on the folder's row, a tooltip-only button shown on hover."""
    row = folder_row(sidebar, name)
    click_folder_menu_button(row)
    return sidebar.page.get_by_role("menu")


def submit_folder_modal(page: Page, name: str) -> None:
    dialog = page.get_by_role("dialog")
    dialog.get_by_placeholder("Enter folder name").fill(name)
    dialog.get_by_role("button", name="Save").click()


def expect_refused(page: Page, owner) -> None:
    expect(page.get_by_text(EMPTY_NAME_ERROR).first).to_be_visible()
    expect(page.get_by_text("Folder updated successfully")).to_have_count(0)
    assert [folder["name"] for folder in stored_folders(owner)] == [NAME]


@pytest.fixture
def sidebar(page_for, owner, folder_id) -> Locator:
    side = open_sidebar(page_for(owner))
    expect(folder_row(side, NAME)).to_be_visible()
    return side


def test_renaming_a_folder_to_spaces_is_refused(sidebar, owner):
    folder_menu(sidebar, NAME).get_by_role("menuitem", name="Edit").click()
    submit_folder_modal(sidebar.page, "   ")

    expect_refused(sidebar.page, owner)
    sidebar.page.reload()
    show_folders(sidebar)
    expect(folder_row(sidebar, NAME)).to_be_visible()


def test_renaming_a_folder_in_place_to_spaces_is_refused(sidebar, owner):
    folder_link(sidebar, NAME).dblclick()
    rename_input = sidebar.get_by_role("textbox")
    rename_input.fill(" ")
    rename_input.press("Enter")

    expect_refused(sidebar.page, owner)


def test_a_subfolder_named_with_spaces_is_refused(sidebar, owner):
    folder_menu(sidebar, NAME).get_by_role("menuitem", name="Create Folder").click()
    submit_folder_modal(sidebar.page, "  ")

    expect_refused(sidebar.page, owner)


def test_renaming_on_the_folder_page_to_spaces_is_refused(page_for, owner, folder_id):
    page = page_for(owner, path=f"/folders/{folder_id}")
    page.get_by_label("Folder options").click()
    page.get_by_role("menu").get_by_role("menuitem", name="Edit").click()
    submit_folder_modal(page, "   ")

    expect_refused(page, owner)


def test_surrounding_spaces_are_trimmed_from_a_new_name(sidebar, owner):
    folder_menu(sidebar, NAME).get_by_role("menuitem", name="Edit").click()
    submit_folder_modal(sidebar.page, "  Projects  ")

    expect(folder_row(sidebar, "Projects")).to_be_visible()
    assert [folder["name"] for folder in stored_folders(owner)] == ["Projects"]


def test_an_empty_name_is_still_refused(sidebar, owner):
    folder_menu(sidebar, NAME).get_by_role("menuitem", name="Edit").click()
    submit_folder_modal(sidebar.page, "")

    expect_refused(sidebar.page, owner)
