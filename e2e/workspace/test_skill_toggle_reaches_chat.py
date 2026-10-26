"""Regression: turning a skill off or on in the workspace left the chat's skill menu as it was.

Fix `ffa6bb17c` (open-webui/open-webui#30966, issue open-webui/open-webui#30965): the chat reads
the account's skills from a list the app loads once. The workspace's on/off switch saved the
change without reloading that list, so until a page reload the chat still offered a skill that
had just been turned off, and did not offer one just turned on. The switch now reloads the list.

The page stays loaded throughout: from the chat to the workspace and back through the sidebar,
the way a person moves, since a reload would hide the bug.

Discriminates: passes on the efe63bd34 build; on a build with ffa6bb17c reverted both tests fail
(the switched skill is still offered, or still missing). An unrelated chat journey passes on the
reverted build.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Page, expect

from utils.chat_ui import chat_input
from utils.workspace_list import item_row

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def skill_owner(make_user):
    """A fresh admin; `add(name, active)` gives them a skill, deleted again afterwards."""
    account = make_user(role="admin")
    created: list[str] = []

    def add(name: str, active: bool = True) -> None:
        skill_id = f"skill_{uuid.uuid4().hex[:8]}"
        with account.client() as client:
            response = client.post(
                "/api/v1/skills/create",
                json={
                    "id": skill_id,
                    "name": name,
                    "content": "Be brief.",
                    "meta": {},
                    "is_active": active,
                },
            )
        assert response.status_code == 200, response.text
        created.append(skill_id)

    yield account, add
    with account.client() as client:
        for skill_id in created:
            client.delete(f"/api/v1/skills/id/{skill_id}/delete")


def _switch_in_workspace(page: Page, name: str) -> None:
    """From the chat, go to Workspace > Skills through the sidebar and flip the skill's switch."""
    page.get_by_role("link", name="Workspace", exact=True).click()
    page.get_by_role("link", name=re.compile(r"^Skills")).click()
    expect(page).to_have_url(re.compile(r"/workspace/skills$"))
    row = item_row(page, name)
    switch = row.get_by_role("switch")
    was_on = switch.get_attribute("aria-checked") == "true"
    switch.click()
    expect(switch).to_have_attribute("aria-checked", "false" if was_on else "true")
    page.get_by_role("link", name="New Chat").click()
    expect(chat_input(page)).to_be_visible()


def _skills_in_chat_menu(page: Page, anchor: str):
    """Open the chat's Integrations > Skills list; returns it once `anchor` shows in it."""
    expect(chat_input(page)).to_be_visible()
    page.get_by_label("Integrations").click()
    page.get_by_role("menuitem", name=re.compile(r"^Skills")).click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_role("menuitemcheckbox", name=re.compile(re.escape(anchor)))).to_be_visible()
    return menu


def test_a_skill_turned_off_leaves_the_chat_menu_at_once(page_for, skill_owner):
    account, add = skill_owner
    tag = uuid.uuid4().hex[:6]
    anchor, switched = f"Always on {tag}", f"Turned off {tag}"
    add(anchor)
    add(switched)
    page = page_for(account)
    before = _skills_in_chat_menu(page, anchor)
    expect(
        before.get_by_role("menuitemcheckbox", name=re.compile(re.escape(switched)))
    ).to_be_visible()
    page.keyboard.press("Escape")

    _switch_in_workspace(page, switched)
    after = _skills_in_chat_menu(page, anchor)

    expect(
        after.get_by_role("menuitemcheckbox", name=re.compile(re.escape(switched)))
    ).to_have_count(0)


def test_a_skill_turned_on_joins_the_chat_menu_at_once(page_for, skill_owner):
    account, add = skill_owner
    tag = uuid.uuid4().hex[:6]
    anchor, switched = f"Always on {tag}", f"Turned on {tag}"
    add(anchor)
    add(switched, active=False)
    page = page_for(account)
    before = _skills_in_chat_menu(page, anchor)
    expect(
        before.get_by_role("menuitemcheckbox", name=re.compile(re.escape(switched)))
    ).to_have_count(0)
    page.keyboard.press("Escape")

    _switch_in_workspace(page, switched)
    after = _skills_in_chat_menu(page, anchor)

    expect(
        after.get_by_role("menuitemcheckbox", name=re.compile(re.escape(switched)))
    ).to_be_visible()
