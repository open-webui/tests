"""Journey: a workspace tool's versions in the editor, saved with a message, compared and restored.

A tool's owner edits its code in the workspace editor and saves with a message typed beside Save;
the version dropdown in the header then lists that message and the server holds it as Production.
Choosing the older version shows its code read only, and Compare to current shows the removed and
added lines. Set as Production puts the older code back in the editor and in the stored tool.
An older version can be deleted from its menu and the Production entry offers no delete. Save
stays disabled until something changes, and a save with a description of only spaces is refused with
nothing stored. A user who may only read the tool is sent away from its editor.

Discriminates: passes on the dev 206bf9723 build; one frontend build with these edits turns every
test red: the editor's restore no longer copying the restored code in (the restore tests keep the
new code), the save sending no commit message (the dropdown lists the short id, not the message),
Save never disabled (the unchanged-save test), the compare asking for a version against itself (no
diff lines), the version delete skipping its server call (the version stays listed) and the edit
page no longer sending a reader away (the reader test). The unrelated test_workspace_tools.py
passes on that build.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness.access import grant, make_group
from harness.actors import Actor

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

TOOL_BUILDER = {"workspace": {"tools": True}}
OLD_SOURCE = 'class Tools:\n    def ping(self) -> str:\n        return "pong"\n'
NEW_SOURCE = 'class Tools:\n    def ping(self) -> str:\n        return "ping back"\n'


@pytest.fixture
def toolsmith(admin, make_user):
    """A fresh user who may build tools; the tools they made are deleted afterwards."""
    account = make_user()
    make_group(admin, [account], TOOL_BUILDER)
    yield account
    with account.client() as client:
        for tool in client.get("/api/v1/tools/").json():
            if tool["user_id"] == account.id:
                client.delete(f"/api/v1/tools/id/{tool['id']}/delete")


def _new_tool(owner: Actor) -> dict:
    tool_id = f"history_{uuid.uuid4().hex[:8]}"
    form = {
        "id": tool_id,
        "name": f"Ping {tool_id}",
        "content": OLD_SOURCE,
        "meta": {"description": "answers a ping"},
        "commit_message": "First draft",
    }
    with owner.client() as client:
        created = client.post("/api/v1/tools/create", json=form)
    assert created.status_code == 200, created.text
    return created.json()


def _tool(owner: Actor, tool_id: str) -> dict:
    with owner.client() as client:
        return client.get(f"/api/v1/tools/id/{tool_id}").json()


def _history(owner: Actor, tool_id: str) -> list[dict]:
    with owner.client() as client:
        return client.get(f"/api/v1/tools/id/{tool_id}/history").json()


def _open_editor(page: Page, tool_id: str) -> None:
    page.goto(f"/workspace/tools/edit?id={tool_id}")
    expect(page.get_by_role("main").locator(".cm-content")).to_contain_text("class Tools")
    expect(page.get_by_label("Select version")).to_have_text("Production")


def _replace_code(page: Page, source: str) -> None:
    page.get_by_role("main").locator(".cm-content").click()
    page.keyboard.press("ControlOrMeta+A")
    page.keyboard.insert_text(source)


def _save(page: Page, message: str) -> None:
    page.get_by_label("Describe this change").fill(message)
    page.get_by_role("main").get_by_role("button", name="Save", exact=True).click()
    expect(page.get_by_text("Tool updated successfully")).to_be_visible()


def _version_menu(page: Page) -> None:
    page.get_by_label("Select version").click()


def _version(page: Page, label: str):
    return page.get_by_role("menuitemradio", name=label)


def _save_a_new_version(page: Page, tool_id: str) -> None:
    _open_editor(page, tool_id)
    _replace_code(page, NEW_SOURCE)
    _save(page, "Answer back")
    expect(page.get_by_label("Describe this change")).to_have_value("")


def test_a_saved_edit_is_listed_by_its_message_and_is_production(page_for, toolsmith):
    tool = _new_tool(toolsmith)
    page = page_for(toolsmith)
    _save_a_new_version(page, tool["id"])

    _version_menu(page)
    expect(_version(page, "Production")).to_contain_text("Answer back")
    expect(_version(page, "First draft")).to_be_visible()
    stored = _tool(toolsmith, tool["id"])
    assert stored["content"] == NEW_SOURCE
    versions = _history(toolsmith, tool["id"])
    assert [(v["commit_message"], v["id"] == stored["version_id"]) for v in versions] == [
        ("Answer back", True),
        ("First draft", False),
    ]


def test_the_older_version_shows_its_code_read_only_and_compares_to_production(page_for, toolsmith):
    tool = _new_tool(toolsmith)
    page = page_for(toolsmith)
    _save_a_new_version(page, tool["id"])

    _version_menu(page)
    _version(page, "First draft").click()
    code = page.get_by_role("main").locator(".cm-content:visible")
    expect(code).to_contain_text('return "pong"')
    expect(code).to_have_attribute("contenteditable", "false")
    expect(page.get_by_role("button", name="Set as Production")).to_be_visible()

    page.get_by_role("button", name="Compare to current").click()
    diff = page.get_by_role("region", name="Compare to current")
    expect(diff.get_by_text('return "pong"')).to_be_visible()
    expect(diff.get_by_text('return "ping back"')).to_be_visible()


def test_set_as_production_puts_the_older_code_back_in_the_editor_and_the_tool(page_for, toolsmith):
    tool = _new_tool(toolsmith)
    page = page_for(toolsmith)
    _save_a_new_version(page, tool["id"])

    _version_menu(page)
    _version(page, "First draft").click()
    page.get_by_role("button", name="Set as Production").click()

    expect(page.get_by_text("Production version updated")).to_be_visible()
    expect(page.get_by_label("Select version")).to_have_text("Production")
    code = page.get_by_role("main").locator(".cm-content")
    expect(code).to_contain_text('return "pong"')
    expect(code).not_to_contain_text("ping back")
    stored = _tool(toolsmith, tool["id"])
    assert stored["content"] == OLD_SOURCE
    assert stored["version_id"] == tool["version_id"]


def test_set_as_production_asks_before_discarding_unsaved_edits(page_for, toolsmith):
    tool = _new_tool(toolsmith)
    page = page_for(toolsmith)
    _save_a_new_version(page, tool["id"])
    _replace_code(page, NEW_SOURCE.replace("ping back", "unsaved words"))

    _version_menu(page)
    _version(page, "First draft").click()
    page.get_by_role("button", name="Set as Production").click()
    dialog = page.get_by_role("dialog", name="Discard unsaved changes?")
    dialog.get_by_role("button", name="Set as Production").click()

    expect(page.get_by_text("Production version updated")).to_be_visible()
    expect(page.get_by_role("main").locator(".cm-content")).to_contain_text('return "pong"')
    assert _tool(toolsmith, tool["id"])["content"] == OLD_SOURCE


def test_an_older_version_can_be_deleted_but_the_production_one_cannot(page_for, toolsmith):
    tool = _new_tool(toolsmith)
    page = page_for(toolsmith)
    _save_a_new_version(page, tool["id"])

    _version_menu(page)
    expect(_version(page, "First draft")).to_be_visible()
    production_row = page.get_by_role("menuitemradio", name="Production").locator("..")
    expect(production_row.get_by_label("More Options")).to_have_count(0)

    page.get_by_label("More Options").click()
    page.get_by_role("button", name="Delete", exact=True).click()
    page.get_by_role("dialog", name="Delete Version").get_by_role("button", name="Delete").click()
    expect(page.get_by_text("Version deleted")).to_be_visible()

    _version_menu(page)
    expect(_version(page, "Production")).to_be_visible()
    expect(_version(page, "First draft")).to_have_count(0)
    assert [v["commit_message"] for v in _history(toolsmith, tool["id"])] == ["Answer back"]


def test_save_waits_for_a_change_and_a_blank_description_is_refused(page_for, toolsmith):
    tool = _new_tool(toolsmith)
    page = page_for(toolsmith)
    _open_editor(page, tool["id"])
    save = page.get_by_role("main").get_by_role("button", name="Save", exact=True)
    expect(save).to_be_disabled()

    page.get_by_role("textbox", name="Tool Description").fill("   ")
    expect(save).to_be_enabled()
    save.click()

    expect(page.get_by_text("Name and description are required")).to_be_visible()
    assert _tool(toolsmith, tool["id"])["meta"]["description"] == "answers a ping"
    assert len(_history(toolsmith, tool["id"])) == 1


def test_a_user_who_may_only_read_the_tool_is_sent_away_from_its_editor(
    page_for, admin, make_user, toolsmith
):
    tool = _new_tool(toolsmith)
    reader = make_user()
    make_group(admin, [reader], TOOL_BUILDER)
    with toolsmith.client() as client:
        shared = client.post(
            f"/api/v1/tools/id/{tool['id']}/access/update",
            json={"access_grants": [grant("user", reader.id, "read")]},
        )
    assert shared.status_code == 200, shared.text
    page = page_for(reader)
    page.goto(f"/workspace/tools/edit?id={tool['id']}")

    expect(page.get_by_text("You do not have permission to edit this tool")).to_be_visible()
    expect(page).to_have_url(re.compile(r"/workspace/tools$"))
    expect(page.get_by_label("Select version")).to_have_count(0)
