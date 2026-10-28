"""Journey: the Hidden Models permission decides whether the model editor offers hidden models.

open-webui 0.12.4 (`7f1b61872`) added **Hidden Models** under Models Access. A user with it sees
hidden models in the model editor's base model list next to the visible ones, while the chat's
model selector keeps them out; a user without it sees only the visible ones in the editor. Each
check searches for a tag that a visible and a hidden provider model share, so the visible one
shows the search has run before the hidden one is checked.

Discriminates: passes on dev 948432b43. In a copy of its build, the editor's check of the
permission patched to ignore it (non-admins never get hidden base models) turns the permitted test
red, and patched to grant it to everyone turns the test without the permission red; the preset
journey in e2e/workspace passes on both copies.
"""

from __future__ import annotations

import uuid
from typing import Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.access import grant, make_group
from utils.model_selector import SELECTOR_BUTTON, model_search

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

DEFAULTS = "/api/v1/users/default/permissions"
PUBLIC_READ = grant("user", "*", "read")


@pytest.fixture
def tagged_bases(admin, upstream) -> Iterator[tuple[str, str]]:
    """Two provider models every user may read, one shown and one hidden, named with one tag."""
    tag = uuid.uuid4().hex[:8]
    names = {f"shown-{tag}": f"Shown {tag}", f"hidden-{tag}": f"Hidden {tag}"}
    upstream.models.extend(names)
    with admin.client() as client:
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
        for model_id, name in names.items():
            created = client.post(
                "/api/v1/models/create",
                json={
                    "id": model_id,
                    "base_model_id": None,
                    "name": name,
                    "meta": {"hidden": model_id.startswith("hidden")},
                    "params": {},
                    "access_grants": [PUBLIC_READ],
                },
            )
            assert created.status_code == 200, created.text
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
    yield names[f"shown-{tag}"], names[f"hidden-{tag}"]
    with admin.client() as client:
        for model_id in names:
            client.post("/api/v1/models/model/delete", json={"id": model_id})
            upstream.models.remove(model_id)
        client.get("/api/models", params={"refresh": "true"})


@pytest.fixture
def builder_with(admin, make_user, preserve) -> Iterator:
    """`builder_with(workspace)` is a fresh user in a group of its own with those switches.

    Hidden Models is off in the default permissions meanwhile.
    """
    preserve("permissions")
    with admin.client() as client:
        defaults = client.get(DEFAULTS).json()
        defaults["workspace"]["models_hidden"] = False
        saved = client.post(DEFAULTS, json=defaults)
    assert saved.status_code == 200, saved.text
    groups: list[str] = []

    def build(workspace: dict):
        member = make_user()
        groups.append(make_group(admin, [member], {"workspace": workspace}))
        return member

    yield build
    with admin.client() as client:
        for group_id in groups:
            client.delete(f"/api/v1/groups/id/{group_id}/delete")


def _search(page: Page, opener: Locator, query: str) -> Locator:
    """The options listed after searching for `query` in the model list `opener` opens."""
    opener.click()
    model_search(page).fill(query)
    return page.get_by_role("listbox", name="Available models")


def _base_model_list(page: Page, tag: str) -> Locator:
    editor = page.get_by_role("main")
    return _search(page, editor.get_by_role("button", name="Select a base model", exact=True), tag)


def _chat_model_list(page: Page, tag: str) -> Locator:
    return _search(page, page.get_by_role("button", name=SELECTOR_BUTTON), tag)


def _option(listbox: Locator, name: str) -> Locator:
    return listbox.get_by_role("option", name=f"Select {name} model")


def test_with_the_permission_the_editor_offers_the_hidden_model_and_the_chat_does_not(
    page_for, builder_with, tagged_bases
):
    shown, hidden = tagged_bases
    tag = shown.split()[-1]
    builder = builder_with({"models": True, "models_hidden": True})

    editor_page = page_for(builder, path="/workspace/models/create")
    bases = _base_model_list(editor_page, tag)
    expect(_option(bases, shown)).to_be_visible()
    expect(_option(bases, hidden)).to_be_visible()

    chat_page = page_for(builder)
    models = _chat_model_list(chat_page, tag)
    expect(_option(models, shown)).to_be_visible()
    expect(_option(models, hidden)).to_have_count(0)


def test_without_the_permission_the_editor_leaves_the_hidden_model_out(
    page_for, builder_with, tagged_bases
):
    shown, hidden = tagged_bases
    tag = shown.split()[-1]
    builder = builder_with({"models": True})

    page = page_for(builder, path="/workspace/models/create")
    bases = _base_model_list(page, tag)
    expect(_option(bases, shown)).to_be_visible()
    expect(_option(bases, hidden)).to_have_count(0)
