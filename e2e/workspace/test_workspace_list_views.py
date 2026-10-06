"""Journey: the workspace lists narrow to what the account made or what others share with it.

Each workspace list (models, prompts, tools, skills and knowledge) has a view menu next to its
search: All, Created by you and Shared with you. Two admins each own one item named alike; for
either of them, Created by you lists only their own, Shared with you only the other's and All
both, and the choice is kept when the list is opened again.

The late answer tests are red on dev: the lists that search on the server add every answer that
comes back to the list, also one to a search asked before the view was changed. Held until after
the switch to Created by you, that answer puts the other admin's item back under Created by you
(open-webui/open-webui#31965).

Discriminates: passes on dev 30f3f6a8f apart from the late answer tests (the bug above); the
knowledge one passes in a frontend build whose list drops an answer to an older request. In a
backend copy whose lists ignore the view asked for every view menu test but the tools one fails,
and the tools one, whose list filters in the page, fails in a frontend build that ignores the view.
"""

from __future__ import annotations

import uuid
from typing import Callable
from urllib.parse import parse_qs, urlparse

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


def _post(account: Actor, path: str, form: dict) -> dict:
    with account.client() as client:
        created = client.post(path, json=form)
    assert created.status_code == 200, created.text
    return created.json()


def _model(account: Actor, name: str) -> Callable[[], None]:
    model_id = f"view-{uuid.uuid4().hex[:8]}"
    form = {"id": model_id, "name": name, "base_model_id": MOCK_MODEL_ID, "meta": {}, "params": {}}
    _post(account, "/api/v1/models/create", form)
    return lambda client: client.post("/api/v1/models/model/delete", json={"id": model_id})


def _prompt(account: Actor, name: str) -> Callable[[], None]:
    command = f"view{uuid.uuid4().hex[:8]}"
    form = {"command": command, "name": name, "content": "Say hello."}
    created = _post(account, "/api/v1/prompts/create", form)
    return lambda client: client.delete(f"/api/v1/prompts/id/{created['id']}/delete")


def _tool(account: Actor, name: str) -> Callable[[], None]:
    tool_id = f"view_{uuid.uuid4().hex[:8]}"
    form = {"id": tool_id, "name": name, "content": TOOL_SOURCE, "meta": {"description": "pings"}}
    _post(account, "/api/v1/tools/create", form)
    return lambda client: client.delete(f"/api/v1/tools/id/{tool_id}/delete")


def _skill(account: Actor, name: str) -> Callable[[], None]:
    skill_id = f"view-{uuid.uuid4().hex[:8]}"
    form = {"id": skill_id, "name": name, "content": "Be brief.", "meta": {}}
    _post(account, "/api/v1/skills/create", form)
    return lambda client: client.delete(f"/api/v1/skills/id/{skill_id}/delete")


def _knowledge(account: Actor, name: str) -> Callable[[], None]:
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
        page = page_for(mine)
        page.goto(f"/workspace/{section}")
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
        page = page_for(mine)
        page.goto(f"/workspace/{section}")
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
