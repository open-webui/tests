"""Journey: what the chat input's Integrations menu offers, and to whom.

With web search on for the instance, the menu offers Web Search to every account; the admin's Web
Search switch under Default permissions > Features takes it from users (an admin keeping it is
pinned in e2e/admin/test_search_and_image_switches.py), and a model whose Web Search capability is
off is offered no Web Search either. Tools and Skills each open a list with a search box that
narrows it by name or description and says when nothing matches; a tool picked from the narrowed
list is offered to the model with the next message.

Since dev 834c47ee0 the integrations sit in the chat input's + (More) menu, under its own entries,
and the Integrations button is gone. The Tools and Skills rows count what is picked ("2 selected"),
also after the menu is closed and opened again. The picked tools, skills and switches show beside
the + button only once a person turns on Show Input Selections in Settings > Interface (off by
default); hiding them keeps the picks, so a tool picked before shows at once.

Discriminates: passes on dev ebc6add67. On a build whose Web Search item ignores the permission and
the model's capabilities the withdrawn and capability tests fail; on one that never offers Web
Search the offered test fails; on one whose tool and skill search boxes filter nothing both search
tests fail; on one where a picked tool stays off the picked tool test fails.
The count and Show Input Selections tests pass on dev f659c0d12; in a build of it whose
chips beside + show whatever the setting says the Show Input Selections test fails, and in one
whose Tools and Skills rows show no count the count test fails.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.actors import Actor
from harness.upstream import MOCK_MODEL_ID
from harness.web_retrieval import RETRIEVAL_CONFIG, save_web_settings, serve_search_results
from utils.chat_ui import (
    chat_input,
    close_menu,
    expect_picked,
    expect_reply,
    open_integrations_menu,
    send,
)
from utils.user_menu import open_settings

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

PUFFIN_TOOL = '''class Tools:
    def count_puffins(self, cliff: str) -> str:
        """Count the puffins nesting on a cliff."""
        return "forty"
'''


def ready(page: Page) -> None:
    expect(chat_input(page)).to_be_visible()
    expect(page.get_by_role("button", name=re.compile("^Selected model")).first).to_be_visible()


def open_integrations(page: Page) -> Locator:
    ready(page)
    return open_integrations_menu(page)


def offers_web_search(page: Page) -> bool:
    """Whether the integrations in the More menu hold Web Search."""
    menu = open_integrations(page)
    return menu.get_by_role("menuitemcheckbox", name="Web Search").count() > 0


@pytest.fixture
def web_search_on(admin, preserve, listener):
    """Web search on for the instance, on a local engine nobody needs to reach here."""
    preserve(RETRIEVAL_CONFIG)
    with admin.client() as client:
        save_web_settings(client, **serve_search_results(listener, []))


def save_default_permission(page: Page, switch: str, turn_on: bool) -> None:
    page.goto("/admin/users/groups")
    page.get_by_role("button", name="Default permissions").click()
    dialog = page.get_by_role("dialog")
    target = dialog.get_by_role("switch", name=switch, exact=True)
    if (target.get_attribute("aria-checked") == "true") != turn_on:
        target.click()
    expect(target).to_have_attribute("aria-checked", "true" if turn_on else "false")
    dialog.get_by_role("button", name="Save").click()
    expect(page.get_by_text("Default permissions updated successfully")).to_be_visible()


# --------------------------------------------------------------------------- web search


def test_web_search_is_offered_while_the_instance_has_it_on(page_for, make_user, web_search_on):
    assert offers_web_search(page_for(make_user()))


def test_web_search_is_withdrawn_from_a_user_without_the_permission(
    page_for, admin, make_user, web_search_on, preserve
):
    preserve("permissions")
    save_default_permission(page_for(admin), "Web Search", turn_on=False)

    assert not offers_web_search(page_for(make_user()))


@pytest.fixture
def model_without_web_search(admin, make_user):
    """A fresh user and a preset only they may read, its Web Search capability off."""
    account = make_user()
    model_id = f"no-search-{uuid.uuid4().hex[:8]}"
    form = {
        "id": model_id,
        "base_model_id": MOCK_MODEL_ID,
        "name": model_id,
        "meta": {"capabilities": {"web_search": False}},
        "params": {},
        "access_grants": [
            {"principal_type": "user", "principal_id": account.id, "permission": "read"}
        ],
    }
    with admin.client() as client:
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
    yield account, model_id
    with admin.client() as client:
        client.post("/api/v1/models/model/delete", json={"id": model_id})


def test_a_model_without_the_web_search_capability_is_offered_no_web_search(
    page_for, model_without_web_search, web_search_on
):
    account, model_id = model_without_web_search
    page = page_for(account)
    assert offers_web_search(page)

    page.goto(f"/?model={model_id}")
    expect(page.get_by_role("button", name=f"Selected model: {model_id}")).to_be_visible()

    assert not offers_web_search(page)


# --------------------------------------------------------------------------- tools and skills


@pytest.fixture
def builder(make_user) -> tuple[Actor, str]:
    """A fresh admin with two tools and two skills of their own, deleted again afterwards."""
    account = make_user(role="admin")
    suffix = uuid.uuid4().hex[:8]
    tools = [
        (f"puffins_{suffix}", f"Puffin counter {suffix}", "Counts the seabirds on a cliff"),
        (f"tides_{suffix}", f"Tide table {suffix}", "Reads the harbour tides"),
    ]
    skills = [
        (f"knots-{suffix}", f"Knot tying {suffix}", "Ties a bowline"),
        (f"rigging-{suffix}", f"Rigging {suffix}", "Sets the sails"),
    ]
    with account.client() as client:
        for tool_id, name, description in tools:
            form = {
                "id": tool_id,
                "name": name,
                "content": PUFFIN_TOOL,
                "meta": {"description": description},
            }
            created = client.post("/api/v1/tools/create", json=form)
            assert created.status_code == 200, created.text
        for skill_id, name, description in skills:
            form = {
                "id": skill_id,
                "name": name,
                "description": description,
                "content": "Be brief.",
                "meta": {},
                "is_active": True,
            }
            created = client.post("/api/v1/skills/create", json=form)
            assert created.status_code == 200, created.text
    yield account, suffix
    with account.client() as client:
        for tool_id, _, _ in tools:
            client.delete(f"/api/v1/tools/id/{tool_id}/delete")
        for skill_id, _, _ in skills:
            client.delete(f"/api/v1/skills/id/{skill_id}/delete")


def open_list(page: Page, entry: str) -> Locator:
    menu = open_integrations(page)
    menu.get_by_role("menuitem", name=re.compile(rf"^{entry}")).click()
    expect(menu.get_by_placeholder(f"Search {entry.lower()}")).to_be_visible()
    return menu


def test_the_tool_search_narrows_the_list_by_name_and_by_description(page_for, builder):
    account, suffix = builder
    menu = open_list(page_for(account), "Tools")
    expect(menu.get_by_role("menuitemcheckbox", name=f"Puffin counter {suffix}")).to_be_visible()
    expect(menu.get_by_role("menuitemcheckbox", name=f"Tide table {suffix}")).to_be_visible()

    menu.get_by_placeholder("Search tools").fill(f"tide table {suffix}")
    expect(menu.get_by_role("menuitemcheckbox", name=f"Puffin counter {suffix}")).to_have_count(0)
    expect(menu.get_by_role("menuitemcheckbox", name=f"Tide table {suffix}")).to_be_visible()

    menu.get_by_placeholder("Search tools").fill("seabirds on a cliff")
    expect(menu.get_by_role("menuitemcheckbox", name=f"Puffin counter {suffix}")).to_be_visible()
    expect(menu.get_by_role("menuitemcheckbox", name=f"Tide table {suffix}")).to_have_count(0)

    menu.get_by_placeholder("Search tools").fill(f"lighthouse {suffix}")
    expect(menu.get_by_text("No tools found")).to_be_visible()


def test_a_tool_picked_from_the_search_is_offered_to_the_model(page_for, builder, upstream):
    account, suffix = builder
    page = page_for(account)
    menu = open_list(page, "Tools")
    menu.get_by_placeholder("Search tools").fill("seabirds")
    menu.get_by_role("menuitemcheckbox", name=f"Puffin counter {suffix}").click()
    close_menu(page)
    expect_picked(page, "Tools", 1)

    question = f"how many puffins? {suffix}"
    upstream.queue(reply.text("Plenty.", match=reply.answering(question)))
    send(page, question)
    expect_reply(page, "Plenty.")

    sent = next(filter(reply.answering(question), upstream.chat_requests()))
    assert "count_puffins" in {tool["function"]["name"] for tool in sent.get("tools") or []}


def test_the_skill_search_narrows_the_list(page_for, builder):
    account, suffix = builder
    menu = open_list(page_for(account), "Skills")
    expect(
        menu.get_by_role("menuitemcheckbox", name=re.compile(f"Knot tying {suffix}"))
    ).to_be_visible()
    expect(
        menu.get_by_role("menuitemcheckbox", name=re.compile(f"Rigging {suffix}"))
    ).to_be_visible()

    menu.get_by_placeholder("Search skills").fill(f"rigging {suffix}")
    expect(
        menu.get_by_role("menuitemcheckbox", name=re.compile(f"Knot tying {suffix}"))
    ).to_have_count(0)
    expect(
        menu.get_by_role("menuitemcheckbox", name=re.compile(f"Rigging {suffix}"))
    ).to_be_visible()

    menu.get_by_placeholder("Search skills").fill(f"anchor {suffix}")
    expect(menu.get_by_text("No skills found")).to_be_visible()


def test_the_tools_and_skills_rows_count_what_is_picked(page_for, builder):
    account, suffix = builder
    page = page_for(account)
    menu = open_list(page, "Tools")
    menu.get_by_role("menuitemcheckbox", name=f"Puffin counter {suffix}").click()
    menu.get_by_role("menuitemcheckbox", name=f"Tide table {suffix}").click()
    # the list's heading leads back to the menu's first page
    menu.get_by_role("menuitem", name=re.compile(r"^Tools")).click()
    tools_row = menu.get_by_role("menuitem", name=re.compile(r"^Tools"))
    expect(tools_row).to_contain_text("2 selected")
    skills_row = menu.get_by_role("menuitem", name=re.compile(r"^Skills"))
    expect(skills_row).not_to_contain_text("selected")

    skills_row.click()
    menu.get_by_role("menuitemcheckbox", name=re.compile(f"Rigging {suffix}")).click()
    menu.get_by_role("menuitem", name=re.compile(r"^Skills")).click()
    expect(menu.get_by_role("menuitem", name=re.compile(r"^Skills"))).to_contain_text("1 selected")

    close_menu(page)
    menu = open_integrations(page)
    expect(menu.get_by_role("menuitem", name=re.compile(r"^Tools"))).to_contain_text("2 selected")


def pick_from_list(page: Page, entry: str, name: str) -> None:
    item = open_list(page, entry).get_by_role("menuitemcheckbox", name=name)
    item.click()
    expect(item).to_have_attribute("aria-checked", "true")
    close_menu(page)


def test_picks_show_beside_the_plus_button_once_show_input_selections_is_on(page_for, builder):
    account, suffix = builder
    page = page_for(account)
    pick_from_list(page, "Tools", f"Puffin counter {suffix}")
    expect_picked(page, "Tools", 1)
    tools_chip = page.get_by_role("button", name="Available Tools")
    expect(tools_chip).to_have_count(0)

    open_settings(page)
    settings = page.get_by_role("dialog")
    settings.get_by_label("Personal").get_by_role("tab", name="Interface", exact=True).click()
    switch = settings.get_by_role("switch", name="Show Input Selections")
    expect(switch).to_have_attribute("aria-checked", "false")
    switch.click()
    expect(switch).to_have_attribute("aria-checked", "true")
    page.keyboard.press("Escape")
    expect(settings).to_be_hidden()

    # hiding the chips kept the pick, so it shows at once
    expect(tools_chip).to_have_text("1")

    page.reload()
    expect(chat_input(page)).to_be_visible()
    pick_from_list(page, "Tools", f"Tide table {suffix}")
    expect(tools_chip).to_have_text("1")
