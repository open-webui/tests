"""Journey: the workspace Models list opens a model's editor or chat and switches it on or off.

A builder finds a model of their own in Workspace > Models by searching for its name. The name is a
link to the model's editor, and the arrow beside it, "Try in chat", opens a new chat already on
that model, in a tab beside the list since dev da850248d. The row's Enabled switch, named after
the model, stores the model as off or on, and a model switched off leaves the chat's model selector
straight away, without a reload.

Discriminates: in a frontend build with the name linking to a chat again (the editor link gone),
the editor test goes red; with the arrow, the named switch or the model-list refresh after a
toggle removed, the matching tests go red.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness.upstream import MOCK_MODEL_ID
from utils.chat_surface import open_chat_tab_from
from utils.model_selector import model_options

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def builder(make_user):
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for model in client.get("/api/v1/models/list").json().get("items", []):
            if model["user_id"] == account.id:
                client.post("/api/v1/models/model/delete", json={"id": model["id"]})


@pytest.fixture
def own_model(builder) -> dict:
    model = {"id": f"deck-{uuid.uuid4().hex[:8]}", "name": f"Deck log {uuid.uuid4().hex[:6]}"}
    with builder.client() as client:
        created = client.post(
            "/api/v1/models/create",
            json={**model, "base_model_id": MOCK_MODEL_ID, "meta": {}, "params": {}},
        )
    assert created.status_code == 200, created.text
    return model


def stored_is_active(builder, model_id: str) -> bool:
    with builder.client() as client:
        found = client.get("/api/v1/models/model", params={"id": model_id})
    assert found.status_code == 200, found.text
    return found.json()["is_active"]


def find_in_list(page: Page, model: dict) -> None:
    page.goto("/workspace/models")
    page.get_by_role("textbox", name="Search Models").fill(model["name"])
    expect(page.get_by_role("link", name=model["name"], exact=True)).to_be_visible()


def test_the_name_in_the_list_opens_the_models_editor(page_for, builder, own_model):
    page = page_for(builder)
    find_in_list(page, own_model)

    page.get_by_role("link", name=own_model["name"], exact=True).click()

    expect(page).to_have_url(re.compile(r"/workspace/models/edit\?id=" + own_model["id"]))
    expect(page.get_by_role("button", name="Save & Update")).to_be_visible()


def test_try_in_chat_opens_a_chat_on_that_model(page_for, builder, own_model):
    page = page_for(builder)
    find_in_list(page, own_model)

    try_in_chat = page.get_by_role("link", name=f"Try in chat: {own_model['name']}")
    chat = open_chat_tab_from(page, try_in_chat)

    expect(page).to_have_url(re.compile(r"\?model=" + own_model["id"]))
    expect(chat.button(f"Selected model: {own_model['name']}")).to_be_visible()


def test_the_rows_switch_turns_the_model_off_and_on_and_the_selector_follows(
    page_for, builder, own_model
):
    page = page_for(builder)
    find_in_list(page, own_model)
    switch = page.get_by_role("switch", name=f"Enabled: {own_model['name']}")
    expect(switch).to_be_checked()

    switch.click()

    expect(switch).not_to_be_checked()
    expect(switch).to_be_enabled()
    assert stored_is_active(builder, own_model["id"]) is False
    chat = open_chat_tab_from(
        page, page.get_by_role("link", name=f"Try in chat: {own_model['name']}")
    )
    expect(model_options(chat, MOCK_MODEL_ID)).to_have_count(1)  # the list has loaded
    expect(model_options(chat, own_model["name"])).to_have_count(0)

    find_in_list(page, own_model)
    switch = page.get_by_role("switch", name=f"Enabled: {own_model['name']}")
    switch.click()
    expect(switch).to_be_checked()
    assert stored_is_active(builder, own_model["id"]) is True
