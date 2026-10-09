"""Journey: the admin function editor keeps, compares and restores a function's versions.

An admin edits a pipe's code in Admin Panel > Functions and saves with a message typed beside
Save; the version dropdown in the editor header then lists that message and the server holds it
as Production. The older version shows its code read only and Compare to current shows the
removed and added lines. Set as Production puts the older code back in the editor and in the
stored function, and the next chat with the pipe is answered by that code. An older version can
be deleted from its menu and the Production entry offers no delete.

Discriminates: passes on the dev 206bf9723 build; one frontend build turns all three tests red: the
editor's restore no longer copying the restored code in (the editor and chat keep the new code),
the save sending no commit message (the dropdown lists no message), the compare asking for a
version against itself (no diff lines) and the version delete skipping its server call (the
version stays listed).
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from harness.actors import Actor
from harness.plugins import installed_function
from utils.chat_ui import expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


def clock_pipe(answer: str) -> str:
    return f'class Pipe:\n    def pipe(self, body: dict) -> str:\n        return "{answer}"\n'


OLD_SOURCE = clock_pipe("High tide at nine.")
NEW_SOURCE = clock_pipe("High tide at ten.")


def _function(admin: Actor, function_id: str) -> dict:
    with admin.client() as client:
        return client.get(f"/api/v1/functions/id/{function_id}").json()


def _history(admin: Actor, function_id: str) -> list[dict]:
    with admin.client() as client:
        return client.get(f"/api/v1/functions/id/{function_id}/history").json()


def _version(page: Page, label: str):
    return page.get_by_role("menuitemradio", name=label)


def _save_a_new_version(page: Page, function_id: str) -> None:
    page.goto(f"/admin/functions/edit?id={function_id}")
    code = page.get_by_role("main").locator(".cm-content")
    expect(code).to_contain_text("class Pipe")
    expect(page.get_by_label("Select version")).to_have_text("Production")
    code.click()
    page.keyboard.press("ControlOrMeta+A")
    page.keyboard.insert_text(NEW_SOURCE)
    page.get_by_label("Describe this change").fill("Later tide")
    page.get_by_role("main").get_by_role("button", name="Save", exact=True).click()
    expect(page.get_by_text("Function updated successfully")).to_be_visible()
    expect(page.get_by_label("Describe this change")).to_have_value("")


def test_a_saved_edit_is_listed_by_its_message_and_is_production(page_for, admin):
    with installed_function(admin, OLD_SOURCE) as function_id:
        first = _function(admin, function_id)["version_id"]
        page = page_for(admin)
        _save_a_new_version(page, function_id)

        page.get_by_label("Select version").click()
        expect(_version(page, "Production")).to_contain_text("Later tide")
        expect(_version(page, first[:7])).to_be_visible()
        stored = _function(admin, function_id)
        assert stored["content"] == NEW_SOURCE
        versions = _history(admin, function_id)
        assert [(v["commit_message"], v["id"] == stored["version_id"]) for v in versions] == [
            ("Later tide", True),
            (None, False),
        ]


def test_the_older_version_compares_to_production_and_set_as_production_restores_it(
    page_for, admin
):
    with installed_function(admin, OLD_SOURCE) as function_id:
        first = _function(admin, function_id)["version_id"]
        page = page_for(admin)
        _save_a_new_version(page, function_id)

        page.get_by_label("Select version").click()
        _version(page, first[:7]).click()
        code = page.get_by_role("main").locator(".cm-content:visible")
        expect(code).to_contain_text("High tide at nine.")
        expect(code).to_have_attribute("contenteditable", "false")
        page.get_by_role("button", name="Compare to current").click()
        diff = page.get_by_role("region", name="Compare to current")
        expect(diff.get_by_text('return "High tide at nine."')).to_be_visible()
        expect(diff.get_by_text('return "High tide at ten."')).to_be_visible()

        page.get_by_role("button", name="Set as Production").click()
        expect(page.get_by_text("Production version updated")).to_be_visible()
        expect(page.get_by_role("main").locator(".cm-content")).to_contain_text(
            "High tide at nine."
        )
        stored = _function(admin, function_id)
        assert stored["content"] == OLD_SOURCE
        assert stored["version_id"] == first

        page.goto(f"/?models={function_id}")
        send(page, "when is high tide?")
        expect_reply(page, "High tide at nine.")


def test_an_older_version_can_be_deleted_but_the_production_one_cannot(page_for, admin):
    with installed_function(admin, OLD_SOURCE) as function_id:
        first = _function(admin, function_id)["version_id"]
        page = page_for(admin)
        _save_a_new_version(page, function_id)

        page.get_by_label("Select version").click()
        expect(_version(page, first[:7])).to_be_visible()
        production_row = _version(page, "Production").locator("..")
        expect(production_row.get_by_label("More Options")).to_have_count(0)

        page.get_by_label("More Options").click()
        page.get_by_role("button", name="Delete", exact=True).click()
        dialog = page.get_by_role("dialog", name="Delete Version")
        dialog.get_by_role("button", name="Delete").click()
        expect(page.get_by_text("Version deleted")).to_be_visible()

        page.get_by_label("Select version").click()
        expect(_version(page, "Production")).to_be_visible()
        expect(_version(page, first[:7])).to_have_count(0)
        assert [v["commit_message"] for v in _history(admin, function_id)] == ["Later tide"]
