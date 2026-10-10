"""The menu next to a model in the model selector did nothing: Edit, Keep in Sidebar, Copy Link.

Regression from `8fc416ee7` (open-webui/open-webui#29878), fix open-webui/open-webui#31503. The
selector began closing on any click outside it, in the capture phase, and swallowing that click.
A model's menu opens outside the selector and was not marked as one of its child menus, so a click
on any of its items only closed the selector. The fix marks the menu as a child and closes the
selector when Edit or Delete opens a window, so the first click in that window counts.

Each test opens the selector in a chat, finds its model by name, opens the model's menu and clicks
one item, then checks what that item does in the UI and over the API. The pin and delete routes
themselves always worked, so no integration test can see this bug.

Discriminates: passes on the ef67cc3fa build with #31503 applied, fails on ef67cc3fa (each item
click closes the selector and nothing else happens); with only the child-menu mark applied, the
Edit and Delete tests still fail (the first click in the opened window is swallowed); selecting a
model by its row passes on all three.
"""

from __future__ import annotations

import re
import uuid
from typing import Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.actors import Actor
from harness.ollama_provider import OLLAMA_CONFIG, connect_ollama, serve_ollama
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import chat_input

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

NEW_DESCRIPTION = "Edited from the model selector"


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _refreshed_model_ids(actor: Actor) -> list[str]:
    with actor.client() as client:
        listed = client.get("/api/models", params={"refresh": True})
    listed.raise_for_status()
    return [model["id"] for model in listed.json()["data"]]


def _stored_model(actor: Actor, model_id: str) -> dict:
    with actor.client() as client:
        stored = client.get("/api/v1/models/model", params={"id": model_id})
    assert stored.status_code == 200, stored.text
    return stored.json()


@pytest.fixture
def owner(make_user) -> Actor:
    """An admin of its own, so the pins and models a test adds stay out of other tests."""
    return make_user(role="admin")


@pytest.fixture
def provider_model(upstream, owner) -> Iterator[str]:
    """A model the scripted provider serves, with no settings saved for it yet."""
    model_id = _unique("menu-base")
    upstream.models.append(model_id)
    assert model_id in _refreshed_model_ids(owner)
    yield model_id
    with owner.client() as client:
        client.post("/api/v1/models/model/delete", json={"id": model_id})


@pytest.fixture
def workspace_model(owner) -> Iterator[str]:
    """A workspace model on the scripted model, owned by `owner`."""
    model_id = _unique("menu-workspace")
    with owner.client() as client:
        created = client.post(
            "/api/v1/models/create",
            json={
                "id": model_id,
                "name": model_id,
                "base_model_id": MOCK_MODEL_ID,
                "meta": {"description": "before"},
                "params": {},
            },
        )
        assert created.status_code == 200, created.text
        yield model_id
        client.post("/api/v1/models/model/delete", json={"id": model_id})


@pytest.fixture
def ollama_model(owner, preserve, listener) -> Iterator[tuple[str, object]]:
    """A model on an Ollama stand-in, safe to delete; the menu offers Delete for Ollama models."""
    preserve(OLLAMA_CONFIG)
    model_id = f"{_unique('menu-delete')}:latest"
    server = serve_ollama(listener, model_id)
    with owner.client() as client:
        connect_ollama(client, listener)
    assert model_id in _refreshed_model_ids(owner)
    yield model_id, server


def open_selector(page: Page) -> Locator:
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("button", name=re.compile("^Selected model")).click()
    models = page.get_by_role("listbox", name="Available models")
    expect(models).to_be_visible()
    return models


def model_row(page: Page, name: str) -> Locator:
    models = open_selector(page)
    page.get_by_role("textbox", name="Search In Models").fill(name)
    row = models.get_by_role("option", name=f"Select {name} model")
    expect(row).to_be_visible()
    return row


def click_menu_item(page: Page, name: str, item: str) -> None:
    row = model_row(page, name)
    row.hover()
    row.get_by_label("More Options", exact=True).click()
    menu = page.get_by_role("menu")
    expect(menu).to_be_visible()
    menu.get_by_role("button", name=item, exact=True).click()


def edit_and_save(editor: Locator) -> None:
    # the first click in the editor must count, not just close the selector
    editor.get_by_role("button", name="Custom description enabled").click()
    expect(editor.get_by_role("button", name="Default description enabled")).to_be_visible()
    editor.get_by_role("button", name="Default description enabled").click()
    editor.get_by_placeholder("Add a short description about what this model does").fill(
        NEW_DESCRIPTION
    )
    editor.get_by_role("button", name="Save & Update").click()


@pytest.fixture
def page(page_for, owner) -> Page:
    return page_for(owner)


def test_edit_opens_the_settings_of_a_provider_model(page, owner, provider_model):
    """Red on dev 3dd1db147: the editor of a never-customised provider model stayed blank
    (open-webui/open-webui#32143). Fixed by fdb8cb749: passes on dev 0401b7522 (3 of 3) and
    fails on that build with fdb8cb749 reverted."""
    click_menu_item(page, provider_model, "Edit")

    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("button", name="Save & Update")).to_be_visible()
    edit_and_save(settings)

    expect(settings.get_by_role("button", name="Save & Update")).to_be_hidden()
    assert _stored_model(owner, provider_model)["meta"]["description"] == NEW_DESCRIPTION


def test_edit_opens_the_editor_of_a_workspace_model(page, owner, workspace_model):
    click_menu_item(page, workspace_model, "Edit")

    expect(page).to_have_url(f"{owner.base_url}/workspace/models/edit?id={workspace_model}")
    edit_and_save(page.get_by_role("main"))

    expect(page).to_have_url(f"{owner.base_url}/workspace/models")
    assert _stored_model(owner, workspace_model)["meta"]["description"] == NEW_DESCRIPTION


def test_keep_in_sidebar_pins_the_model(page, owner, provider_model):
    click_menu_item(page, provider_model, "Keep in Sidebar")
    expect(page.get_by_role("menu")).to_be_hidden()
    page.keyboard.press("Escape")
    page.get_by_role("button", name="Open Sidebar", exact=True).click()

    sidebar = page.get_by_role("navigation", name="Chat history")
    expect(sidebar.get_by_role("link", name=provider_model)).to_be_visible()
    with owner.client() as client:
        stored = client.get("/api/v1/users/user/settings").json()
    assert provider_model in stored["ui"]["pinnedModels"]


def test_copy_link_puts_the_model_link_on_the_clipboard(page, owner, provider_model):
    page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    click_menu_item(page, provider_model, "Copy Link")

    expect(page.get_by_text("Copied link to clipboard")).to_be_visible()
    copied = page.evaluate("navigator.clipboard.readText()")
    assert copied == f"{owner.base_url}/?model={provider_model}"


def test_community_reviews_opens_the_model_on_the_community_site(page, provider_model):
    page.context.route("https://openwebui.com/**", lambda route: route.fulfill(body="reviews"))
    with page.expect_popup() as opened:
        click_menu_item(page, provider_model, "Community Reviews")

    popup = opened.value
    popup.wait_for_load_state()
    assert popup.url == f"https://openwebui.com/models?q={provider_model}"


def test_delete_asks_first_and_the_first_click_confirms(page, owner, ollama_model):
    model_id, server = ollama_model
    click_menu_item(page, model_id, "Delete")

    confirm = page.get_by_role("dialog", name="Delete Model")
    expect(confirm).to_be_visible()
    confirm.get_by_role("button", name="Confirm").click()

    expect(confirm).to_be_hidden()
    expect(page.get_by_text(f"Model {model_id} deleted successfully")).to_be_visible()
    assert model_id not in server.models
    assert model_id not in _refreshed_model_ids(owner)


# ---------------------------------------------------------------- nearby


def test_clicking_a_model_row_selects_it_and_closes_the_selector(page, provider_model):
    model_row(page, provider_model).click()

    expect(page.get_by_role("listbox", name="Available models")).to_be_hidden()
    expect(page.get_by_role("button", name=f"Selected model: {provider_model}")).to_be_visible()
