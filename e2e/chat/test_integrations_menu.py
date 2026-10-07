"""Journey: what the chat input's Integrations menu offers, and to whom.

With web search on for the instance, the menu offers Web Search to every account; the admin's
Web Search switch under Default permissions > Features takes it from users while an admin keeps
it, and a model whose Web Search capability is off is offered no Web Search either. Tools and
Skills each open a list with a search box that narrows it by name or description and says when
nothing matches; a tool picked from the narrowed list is offered to the model with the next
message.

Discriminates: passes on dev 30f3f6a8f; on a frontend build whose Integrations menu ignores the
user's web search permission the withdrawn test fails, on one that ignores the model's
capabilities the capability test fails, and on one whose tool and skill search boxes filter
nothing the search tests fail.
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
from utils.chat_ui import chat_input, expect_reply, send

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
    page.get_by_role("button", name="Integrations", exact=True).last.click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_role("button").first).to_be_visible()
    return menu


def offers_web_search(page: Page) -> bool:
    """Whether the Integrations menu holds Web Search; no menu at all offers nothing."""
    ready(page)
    if page.get_by_role("button", name="Integrations", exact=True).count() == 0:
        return False
    menu = open_integrations(page)
    return menu.get_by_role("button", name="Web Search").count() > 0


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


def test_an_admin_keeps_web_search_without_the_permission(page_for, admin, web_search_on, preserve):
    preserve("permissions")
    save_default_permission(page_for(admin), "Web Search", turn_on=False)

    assert offers_web_search(page_for(admin))


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
    menu.get_by_role("button", name=re.compile(rf"^{entry}")).click()
    expect(menu.get_by_placeholder(f"Search {entry.lower()}")).to_be_visible()
    return menu


def test_the_tool_search_narrows_the_list_by_name_and_by_description(page_for, builder):
    account, suffix = builder
    menu = open_list(page_for(account), "Tools")
    expect(menu.get_by_role("button", name=f"Puffin counter {suffix}")).to_be_visible()
    expect(menu.get_by_role("button", name=f"Tide table {suffix}")).to_be_visible()

    menu.get_by_placeholder("Search tools").fill(f"tide table {suffix}")
    expect(menu.get_by_role("button", name=f"Puffin counter {suffix}")).to_have_count(0)
    expect(menu.get_by_role("button", name=f"Tide table {suffix}")).to_be_visible()

    menu.get_by_placeholder("Search tools").fill("seabirds on a cliff")
    expect(menu.get_by_role("button", name=f"Puffin counter {suffix}")).to_be_visible()
    expect(menu.get_by_role("button", name=f"Tide table {suffix}")).to_have_count(0)

    menu.get_by_placeholder("Search tools").fill(f"lighthouse {suffix}")
    expect(menu.get_by_text("No tools found")).to_be_visible()


def test_a_tool_picked_from_the_search_is_offered_to_the_model(page_for, builder, upstream):
    account, suffix = builder
    page = page_for(account)
    menu = open_list(page, "Tools")
    menu.get_by_placeholder("Search tools").fill("seabirds")
    menu.get_by_role("button", name=f"Puffin counter {suffix}").click()
    page.keyboard.press("Escape")
    expect(page.get_by_role("button", name="Available Tools")).to_have_text("1")

    question = f"how many puffins? {suffix}"
    upstream.queue(reply.text("Plenty.", match=reply.answering(question)))
    send(page, question)
    expect_reply(page, "Plenty.")

    sent = next(filter(reply.answering(question), upstream.chat_requests()))
    assert "count_puffins" in {tool["function"]["name"] for tool in sent.get("tools") or []}


def test_the_skill_search_narrows_the_list(page_for, builder):
    account, suffix = builder
    menu = open_list(page_for(account), "Skills")
    expect(menu.get_by_role("button", name=re.compile(f"Knot tying {suffix}"))).to_be_visible()
    expect(menu.get_by_role("button", name=re.compile(f"Rigging {suffix}"))).to_be_visible()

    menu.get_by_placeholder("Search skills").fill(f"rigging {suffix}")
    expect(menu.get_by_role("button", name=re.compile(f"Knot tying {suffix}"))).to_have_count(0)
    expect(menu.get_by_role("button", name=re.compile(f"Rigging {suffix}"))).to_be_visible()

    menu.get_by_placeholder("Search skills").fill(f"anchor {suffix}")
    expect(menu.get_by_text("No skills found")).to_be_visible()
