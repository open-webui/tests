"""Journey: the Web Search toggle of the chat input, and who it is offered to.

Turned on in the Integrations menu, Web Search shows as a chip by the input and the model is
offered `search_web` and `fetch_url`; clicking the chip turns it off again, and the next question
goes out without them. With Web Search withdrawn from the default permissions, a group whose
Permissions tab switches Web Search on gives it back to its members, who see the toggle and get
the search, while everyone else still has neither. A chat link carrying `web-search=true` gives a
user without the permission no search either. The engine is a local one nobody needs to reach.

Discriminates: passes on dev ebc6add67; run on a frontend build whose Web Search chip leaves the
toggle on and whose chat sends Web Search without the permission check, against a backend copy
that offers the web tools without the permission check and leaves the user's groups out of their
permissions, every test fails: the chip leaves the search on, the link searches, and the member
gets no toggle.
"""

from __future__ import annotations

import re
import uuid
from typing import Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.actors import Actor
from harness.upstream import MOCK_MODEL_ID
from harness.web_retrieval import RETRIEVAL_CONFIG, save_web_settings, serve_search_results
from utils.chat_ui import chat_input, expect_reply, send
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

WEB_TOOLS = {"search_web", "fetch_url"}
DEFAULT_PERMISSIONS = "/api/v1/users/default/permissions"


@pytest.fixture
def web_search_on(admin, preserve, listener) -> None:
    preserve(RETRIEVAL_CONFIG, "permissions")
    with admin.client() as client:
        save_web_settings(client, **serve_search_results(listener, []))


@pytest.fixture
def withheld_by_default(admin, web_search_on) -> None:
    with admin.client() as client:
        current = client.get(DEFAULT_PERMISSIONS).json()
        withheld = {**current, "features": {**current["features"], "web_search": False}}
        client.post(DEFAULT_PERMISSIONS, json=withheld).raise_for_status()


@pytest.fixture
def crew(admin, make_user) -> Iterator[tuple[str, Actor]]:
    """A group of one member, with no permissions of its own yet; deleted afterwards."""
    member = make_user()
    name = f"Crew {uuid.uuid4().hex[:8]}"
    with admin.client() as client:
        created = client.post("/api/v1/groups/create", json={"name": name, "description": ""})
        assert created.status_code == 200, created.text
        group_id = created.json()["id"]
        added = client.post(
            f"/api/v1/groups/id/{group_id}/users/add", json={"user_ids": [member.id]}
        )
        assert added.status_code == 200, added.text
    yield name, member
    with admin.client() as client:
        client.delete(f"/api/v1/groups/id/{group_id}/delete")


def integrations_menu(page: Page) -> Locator:
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("button", name="Integrations", exact=True).last.click()
    menu = page.get_by_role("menu")
    expect(menu).to_be_visible()
    return menu


def offered_tools(page: Page, upstream, question: str) -> set[str]:
    upstream.queue(reply.text("Calm seas.", match=reply.answering(question)))
    send(page, question)
    expect_reply(page, "Calm seas.")
    sent = next(filter(reply.answering(question), upstream.chat_requests()))
    return {tool["function"]["name"] for tool in sent.get("tools") or []}


def test_the_chip_by_the_input_turns_web_search_off_again(
    page_for, make_user, upstream, web_search_on
):
    page = page_for(make_user())
    toggle = integrations_menu(page).get_by_role("button", name="Web Search")
    toggle.click()
    expect(toggle).to_have_attribute("aria-pressed", "true")
    page.keyboard.press("Escape")

    tooltip_button(page.locator("form"), "Web Search").click()
    expect(integrations_menu(page).get_by_role("button", name="Web Search")).to_have_attribute(
        "aria-pressed", "false"
    )
    page.keyboard.press("Escape")

    assert not WEB_TOOLS & offered_tools(page, upstream, "Is the sea calm at the quay?")


def test_a_group_permission_gives_web_search_back_to_its_members(
    page_for, admin, make_user, upstream, withheld_by_default, crew
):
    name, member = crew
    admin_page = page_for(admin)
    admin_page.goto("/admin/users/groups")
    groups = admin_page.get_by_role("main")
    groups.get_by_role("textbox", name="Search Groups").fill(name)
    groups.get_by_role("button", name=re.compile(rf"^{name} 1 direct members")).click()
    editing = admin_page.get_by_role("dialog").filter(has_text="Edit User Group")
    editing.get_by_role("button", name="Permissions", exact=True).click()
    web_search = editing.get_by_role("switch", name="Web Search", exact=True)
    if web_search.get_attribute("aria-checked") != "true":
        web_search.click()
    expect(web_search).to_have_attribute("aria-checked", "true")
    editing.get_by_role("button", name="Save").click()
    expect(admin_page.get_by_text("Group updated successfully")).to_be_visible()

    outsider = page_for(make_user())
    expect(integrations_menu(outsider).get_by_role("button", name="Web Search")).to_have_count(0)
    member_page = page_for(member)
    integrations_menu(member_page).get_by_role("button", name="Web Search").click()
    member_page.keyboard.press("Escape")

    assert WEB_TOOLS <= offered_tools(member_page, upstream, "Is the sea calm at the slipway?")


def test_a_web_search_link_gives_no_search_without_the_permission(
    page_for, make_user, upstream, withheld_by_default
):
    page = page_for(make_user())
    page.goto(f"/?models={MOCK_MODEL_ID}&web-search=true")

    offered = offered_tools(page, upstream, "Is the sea calm at the harbour mouth?")

    assert not WEB_TOOLS & offered, f"a user without web search was offered {offered}"
