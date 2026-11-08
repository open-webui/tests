"""Journey: the workspace lists narrow to what the account made or what others share with it.

Each workspace list (models, prompts, tools, skills and knowledge) has a view menu next to its
search: All, Created by you and Shared with you. Two admins each own one item named alike; for
either of them, Created by you lists only their own, Shared with you only the other's and All
both, and the choice is kept when the list is opened again.

An answer to a search asked before the view was changed is dropped. The lists that search on the
server used to add every answer that came back, so one held until after the switch to Created by
you put the other admin's item back under Created by you (open-webui/open-webui#31965), fixed in
dev 4b9d31b39.

Discriminates: passes on dev 1711059db; the late answer tests fail on dev 77febc65d, before
4b9d31b39. In a backend copy whose lists ignore the view asked for every view menu test but the
tools one fails, and the tools one, whose list filters in the page, fails in a frontend build that
ignores the view.

Paging the prompt list from the keyboard keeps the focus in the page buttons, at the last and
the first page too (bc14c3767, PR open-webui/open-webui#33743, and a53089365, PR #33748).
Passes on dev e8c0ad85e and fails on that build with both reverted.
"""

from __future__ import annotations

import uuid
from typing import Callable
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from playwright.sync_api import Page, Route, expect

from harness.actors import Actor
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

STALE_ANSWER = (
    "an answer to the search asked before the view changed was added to the list afterwards, "
    "so the other account's item shows under Created by you (#31965)"
)
TOOL_SOURCE = 'class Tools:\n    def ping(self) -> str:\n        return "pong"\n'


Cleanup = Callable[[httpx.Client], httpx.Response]


def _post(account: Actor, path: str, form: dict) -> dict:
    with account.client() as client:
        created = client.post(path, json=form)
    assert created.status_code == 200, created.text
    return created.json()


def _model(account: Actor, name: str) -> Cleanup:
    model_id = f"view-{uuid.uuid4().hex[:8]}"
    form = {"id": model_id, "name": name, "base_model_id": MOCK_MODEL_ID, "meta": {}, "params": {}}
    _post(account, "/api/v1/models/create", form)
    return lambda client: client.post("/api/v1/models/model/delete", json={"id": model_id})


def _prompt(account: Actor, name: str) -> Cleanup:
    command = f"view{uuid.uuid4().hex[:8]}"
    form = {"command": command, "name": name, "content": "Say hello."}
    created = _post(account, "/api/v1/prompts/create", form)
    return lambda client: client.delete(f"/api/v1/prompts/id/{created['id']}/delete")


def _tool(account: Actor, name: str) -> Cleanup:
    tool_id = f"view_{uuid.uuid4().hex[:8]}"
    form = {"id": tool_id, "name": name, "content": TOOL_SOURCE, "meta": {"description": "pings"}}
    _post(account, "/api/v1/tools/create", form)
    return lambda client: client.delete(f"/api/v1/tools/id/{tool_id}/delete")


def _skill(account: Actor, name: str) -> Cleanup:
    skill_id = f"view-{uuid.uuid4().hex[:8]}"
    form = {"id": skill_id, "name": name, "content": "Be brief.", "meta": {}}
    _post(account, "/api/v1/skills/create", form)
    return lambda client: client.delete(f"/api/v1/skills/id/{skill_id}/delete")


def _knowledge(account: Actor, name: str) -> Cleanup:
    created = _post(account, "/api/v1/knowledge/create", {"name": name, "description": "notes"})
    return lambda client: client.delete(f"/api/v1/knowledge/{created['id']}/delete")


SECTIONS = {
    "models": _model,
    "prompts": _prompt,
    "tools": _tool,
    "skills": _skill,
    "knowledge": _knowledge,
}


def _choose_view(page: Page, current: str, view: str) -> None:
    page.get_by_role("main").get_by_role("button", name=current, exact=True).click()
    page.get_by_role("button", name=view, exact=True).click()


def _listed(page: Page, name: str):
    return page.get_by_role("main").get_by_text(name, exact=True)


def _search(page: Page, section: str, typed: str) -> None:
    """Type the search and wait for its answer, so a later view change cannot race it."""
    search = page.get_by_role("textbox", name=f"Search {section.title()}")
    expect(search).to_be_visible()
    if section == "tools":
        search.fill(typed)
        return
    with page.expect_response(lambda response: "query=" in response.url):
        search.fill(typed)


@pytest.mark.parametrize("section", list(SECTIONS))
def test_the_view_menu_narrows_the_list_to_own_or_shared_items(page_for, make_user, section):
    mine, theirs = make_user(role="admin"), make_user(role="admin")
    tag = uuid.uuid4().hex[:6]
    own_name, other_name = f"Harbour {tag} own", f"Harbour {tag} other"
    cleanups = [
        (mine, SECTIONS[section](mine, own_name)),
        (theirs, SECTIONS[section](theirs, other_name)),
    ]
    try:
        page = page_for(mine, path=f"/workspace/{section}")
        _search(page, section, f"Harbour {tag}")
        expect(_listed(page, own_name)).to_be_visible()
        expect(_listed(page, other_name)).to_be_visible()

        _choose_view(page, "All", "Created by you")
        expect(_listed(page, own_name)).to_be_visible()
        expect(_listed(page, other_name)).to_have_count(0)

        _choose_view(page, "Created by you", "Shared with you")
        expect(_listed(page, other_name)).to_be_visible()
        expect(_listed(page, own_name)).to_have_count(0)

        page.reload()
        _search(page, section, f"Harbour {tag}")
        expect(_listed(page, other_name)).to_be_visible()
        expect(_listed(page, own_name)).to_have_count(0)
        _choose_view(page, "Shared with you", "All")
        expect(_listed(page, own_name)).to_be_visible()
    finally:
        for account, cleanup in cleanups:
            with account.client() as client:
                cleanup(client)


def _hold_unfiltered_searches(page: Page, typed: str) -> list[Route]:
    """Hold the list's answers to `typed` asked before any view was chosen."""
    held: list[Route] = []

    def hold(route: Route) -> None:
        asked = parse_qs(urlparse(route.request.url).query)
        if asked.get("query") == [typed] and "view_option" not in asked:
            held.append(route)
        else:
            route.continue_()

    page.route("**/api/v1/**", hold)
    return held


# the tools list filters in the page, so it has no answer to come back late
@pytest.mark.parametrize("section", ["models", "prompts", "skills", "knowledge"])
def test_a_late_search_answer_leaves_the_chosen_view_alone(page_for, make_user, section):
    mine, theirs = make_user(role="admin"), make_user(role="admin")
    tag = uuid.uuid4().hex[:6]
    typed, own_name, other_name = f"Harbour {tag}", f"Harbour {tag} own", f"Harbour {tag} other"
    cleanups = [
        (mine, SECTIONS[section](mine, own_name)),
        (theirs, SECTIONS[section](theirs, other_name)),
    ]
    try:
        page = page_for(mine, path=f"/workspace/{section}")
        search = page.get_by_role("textbox", name=f"Search {section.title()}")
        expect(search).to_be_visible()
        held = _hold_unfiltered_searches(page, typed)
        search.fill(typed)
        for _ in range(100):
            if held:
                break
            page.wait_for_timeout(100)
        assert held, "the list never asked for the typed search"

        _choose_view(page, "All", "Created by you")
        expect(_listed(page, own_name)).to_be_visible()
        expect(_listed(page, other_name)).to_have_count(0)
        for route in held:
            route.continue_()
        # the late answer would land within a moment; a bounded check that it does not
        page.wait_for_timeout(1500)
        expect(_listed(page, other_name), STALE_ANSWER).to_have_count(0)
    finally:
        for account, cleanup in cleanups:
            with account.client() as client:
                cleanup(client)


def test_paging_the_prompt_list_from_the_keyboard_keeps_the_focus_in_the_page_buttons(
    page_for, make_user
):
    # the list was swapped for a spinner while the page loaded (#33743), and the disabled Next
    # or Previous button at the last or first page dropped the focus to the page (#33748)
    owner = make_user(role="admin")
    tag = uuid.uuid4().hex[:6]
    cleanups = [_prompt(owner, f"Pager {tag} {number:02d}") for number in range(31)]
    try:
        page = page_for(owner, path="/workspace/prompts")
        _search(page, "prompts", f"Pager {tag}")
        main = page.get_by_role("main")
        expect(main.get_by_text(f"Pager {tag}", exact=False).first).to_be_visible()
        next_page = main.get_by_role("button", name="Next page")
        previous_page = main.get_by_role("button", name="Previous page")

        next_page.focus()
        page.keyboard.press("Enter")
        last = main.get_by_role("button", name="Page 2", exact=True)
        expect(last).to_have_attribute("aria-current", "page")
        expect(next_page).to_be_disabled()
        expect(
            last,
            "reaching the last page dropped the keyboard focus",
        ).to_be_focused()

        previous_page.focus()
        page.keyboard.press("Enter")
        first = main.get_by_role("button", name="Page 1", exact=True)
        expect(first).to_have_attribute("aria-current", "page")
        expect(
            first,
            "going back to the first page dropped the keyboard focus",
        ).to_be_focused()
    finally:
        with owner.client() as client:
            for cleanup in cleanups:
                cleanup(client)
