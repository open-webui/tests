"""Journey: editing, versioning, switching off and sharing a prompt, as the chat's `/` menu sees it.

An old version edited in the prompt editor and saved with Set as Production unticked adds a version
to the editor's history without going live: the `/` menu still inserts the production text until
that version is set as production from the history. A prompt switched off in the workspace list
leaves the `/` menu and returns when switched back on. A prompt shared with a group is offered to
its members, who see it read-only in the editor, while an account outside the group is only offered
what was made public. A prompt shared with a group in the editor's Access dialog is offered to the
group's member and not to an outsider, and taking the group out of that dialog withdraws it.

Discriminates: passes on the 176d31d1d build. In a backend copy where saving a version ignores
`is_production`, the draft test goes red (the draft goes live at once); where the prompt list
ignores `is_active`, the switched-off prompt stays offered; where the prompt list skips the
read-grant check, the stranger is offered the group's prompt; where the access update route stores
no grants, both sharing tests go red (the member is never offered the prompt). Retargeted for
37138282f and b130fec73, where the text is edited in place, versions are picked from a menu and
only an old version taken up again can be saved without going live: the draft and group tests pass
on b130fec73, and the draft test goes red in a frontend build that saves every edit as production.
Retargeted for 6a7678ac7, which named the version picker History and the live version Live: the
draft test passes on dev 206bf9723.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.access import grant, make_group
from harness.actors import Actor
from utils.chat_ui import chat_input

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

PROMPT_USER = {"workspace": {"prompts": True}}


@pytest.fixture
def librarian(make_user):
    """A fresh admin; `add(command, content)` saves a prompt of theirs and returns its id."""
    account = make_user(role="admin")
    created: list[str] = []

    def add(command: str, content: str) -> str:
        form = {"command": command, "name": f"Prompt {command}", "content": content}
        with account.client() as client:
            response = client.post("/api/v1/prompts/create", json=form)
        assert response.status_code == 200, response.text
        created.append(response.json()["id"])
        return created[-1]

    yield account, add
    with account.client() as client:
        for prompt_id in created:
            client.delete(f"/api/v1/prompts/id/{prompt_id}/delete")


def _share(owner: Actor, prompt_id: str, grants: list[dict]) -> None:
    with owner.client() as client:
        shared = client.post(
            f"/api/v1/prompts/id/{prompt_id}/access/update", json={"access_grants": grants}
        )
    assert shared.status_code == 200, shared.text


def _slash_menu(page: Page, typed: str) -> Locator:
    """Type `/typed` into a fresh chat; returns the prompt list it opens."""
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    chat_input(page).click()
    # the chat keeps text inserted earlier as a draft
    page.keyboard.press("ControlOrMeta+A")
    page.keyboard.press("Delete")
    page.keyboard.type(f"/{typed}")
    return page.get_by_role("tooltip")


def _offered(menu: Locator, command: str) -> Locator:
    return menu.get_by_role("button", name=command)


def _inserted_text(page: Page, command: str) -> Locator:
    _offered(_slash_menu(page, command), command).click()
    return chat_input(page)


def test_an_edit_saved_as_a_draft_goes_live_only_when_set_as_production(page_for, librarian):
    account, add = librarian
    command = f"brief{uuid.uuid4().hex[:8]}"
    prompt_id = add(command, "Summarise this in three lines.")
    with account.client() as client:
        form = {"command": command, "name": f"Prompt {command}"}
        live = client.post(
            f"/api/v1/prompts/id/{prompt_id}/update",
            json={**form, "content": "Summarise this.", "commit_message": "Plain"},
        )
        assert live.status_code == 200, live.text
        history = client.get(f"/api/v1/prompts/id/{prompt_id}/history").json()
    [first] = [entry for entry in history if entry["snapshot"]["content"].endswith("three lines.")]
    page = page_for(account)

    page.goto(f"/workspace/prompts/{prompt_id}")
    content = page.get_by_role("textbox", name="Prompt Content")
    expect(content).to_have_value("Summarise this.")
    picker = page.get_by_label("History", exact=True)
    picker.click()
    page.get_by_role("menuitemradio", name=first["commit_message"]).click()
    page.get_by_role("button", name="Edit as new version").click()
    content.fill("Summarise this in one line.")
    page.get_by_role("textbox", name="Commit Message").fill("Shorter summary")
    expect(page.get_by_role("checkbox", name="Set as Production")).not_to_be_checked()
    page.get_by_role("button", name="Save", exact=True).click()
    expect(picker).to_have_text("Shorter summary")

    expect(_inserted_text(page, command)).to_have_text("Summarise this.")

    page.goto(f"/workspace/prompts/{prompt_id}")
    expect(content).to_have_value("Summarise this.")
    picker.click()
    page.get_by_role("menuitemradio", name="Shorter summary").click()
    page.get_by_role("button", name="Set as Production", exact=True).click()
    expect(page.get_by_text("Production version updated")).to_be_visible()
    expect(picker).to_have_text("Live")
    expect(content).to_have_value("Summarise this in one line.")

    expect(_inserted_text(page, command)).to_have_text("Summarise this in one line.")


def test_a_switched_off_prompt_leaves_the_slash_menu_until_switched_on(page_for, librarian):
    account, add = librarian
    prefix = f"memo{uuid.uuid4().hex[:6]}"
    kept, switched = f"{prefix}kept", f"{prefix}off"
    add(kept, "Write a memo.")
    add(switched, "Write a short memo.")
    page = page_for(account)

    def flip(expected: str) -> None:
        page.goto("/workspace/prompts")
        page.get_by_role("textbox", name="Search Prompts").fill(switched)
        row = page.get_by_role("main").get_by_role("button").filter(has_text=f"Prompt {switched}")
        row.get_by_role("switch").click()
        expect(row.get_by_role("switch")).to_have_attribute("aria-checked", expected)

    flip("false")
    menu = _slash_menu(page, prefix)
    expect(_offered(menu, kept)).to_be_visible()
    expect(_offered(menu, switched)).to_have_count(0)

    flip("true")
    menu = _slash_menu(page, prefix)
    expect(_offered(menu, kept)).to_be_visible()
    expect(_offered(menu, switched)).to_be_visible()


def test_a_prompt_shared_with_a_group_reaches_only_its_members(
    page_for, librarian, admin, make_user
):
    account, add = librarian
    member, stranger = make_user(), make_user()
    group_id = make_group(admin, [member], PROMPT_USER)
    prefix = f"team{uuid.uuid4().hex[:6]}"
    shared, public = f"{prefix}group", f"{prefix}all"
    shared_id = add(shared, "Draft the team update.")
    _share(account, shared_id, [grant("group", group_id, "read")])
    _share(account, add(public, "Draft the public update."), [grant("user", "*", "read")])

    member_page = page_for(member)
    expect(_inserted_text(member_page, shared)).to_have_text("Draft the team update.")
    member_page.goto(f"/workspace/prompts/{shared_id}")
    expect(member_page.get_by_text("Read Only", exact=True)).to_be_visible()
    expect(member_page.get_by_role("textbox", name="Prompt Content")).not_to_be_editable()

    stranger_menu = _slash_menu(page_for(stranger), prefix)
    expect(_offered(stranger_menu, public)).to_be_visible()
    expect(_offered(stranger_menu, shared)).to_have_count(0)


def test_a_prompt_shared_with_a_group_in_the_access_dialog_is_offered_until_the_group_is_removed(
    page_for, librarian, admin, make_user
):
    account, add = librarian
    member, stranger = make_user(), make_user()
    group_id = make_group(admin, [member], PROMPT_USER)
    with admin.client() as client:
        group_name = client.get(f"/api/v1/groups/id/{group_id}").json()["name"]
    command = f"crew{uuid.uuid4().hex[:8]}"
    prompt_id = add(command, "Brief the crew.")

    page = page_for(account)
    page.goto(f"/workspace/prompts/{prompt_id}")
    page.get_by_role("main").get_by_role("button", name="Access").click()
    dialog = page.get_by_role("dialog").filter(has_text="Access Control")
    dialog.get_by_role("button", name="Add Access").click()
    picker = page.get_by_role("dialog").filter(has_text="Add Access").last
    picker.get_by_placeholder("Search").fill(group_name)
    picker.get_by_role("button", name=group_name).click()
    picker.get_by_role("button", name="Add", exact=True).click()
    expect(page.get_by_text("Saved").first).to_be_visible()
    expect(dialog.get_by_text(group_name)).to_be_visible()

    member_page, stranger_page = page_for(member), page_for(stranger)
    expect(_inserted_text(member_page, command)).to_have_text("Brief the crew.")
    expect(_offered(_slash_menu(stranger_page, command), command)).to_have_count(0)

    group_row = (
        dialog.locator("div")
        .filter(has_text=group_name)
        .filter(has=page.get_by_label("Access level"))
    )
    group_row.last.get_by_role("button").last.click()  # the row's remove button has no label
    expect(dialog.get_by_text(group_name)).to_have_count(0)
    expect(page.get_by_text("Saved").first).to_be_visible()

    expect(_offered(_slash_menu(member_page, command), command)).to_have_count(0)
