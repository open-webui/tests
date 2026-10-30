"""Journey: a folder's default model is chosen in the folder's settings and left alone by chats.

In a folder's Edit dialog the Default Model is a searchable model picker that reads "Use default"
until a model is chosen; Save stores it as the folder's model, a Reset button beside it clears it
again and the dialog shows the stored choice when it is reopened. A chat started in the folder
opens on that model. Picking another model for one chat in the folder changes that chat only: the
folder keeps its default for the next chat. A folder with no default opens its chats on the
instance's default model, not on the model the last chat used.

Discriminates: in a frontend build with the chat saving its selected models to the folder again
(the change undone), the chat that switched models rewrites the folder's default and the folder
test goes red; with the dialog's picker not storing the choice or the Reset button not clearing
it, the choosing and resetting tests go red; with the folder's chats inheriting the last chat's
model again, the no-default test goes red.
"""

from __future__ import annotations

import json
import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import chat_input, expect_reply, send
from utils.model_selector import model_search
from utils.sidebar_folders import click_folder_menu_button, folder_link, folder_row

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

FOLDER = "Lighthouse"
EVERYONE_READS = {"principal_type": "user", "principal_id": "*", "permission": "read"}


@pytest.fixture
def folder_model(admin):
    """A model every account may use, named apart from the instance's own."""
    model_id = f"lamp-room-{uuid.uuid4().hex[:8]}"
    with admin.client() as client:
        created = client.post(
            "/api/v1/models/create",
            json={
                "id": model_id,
                "base_model_id": MOCK_MODEL_ID,
                "name": model_id,
                "meta": {},
                "params": {},
                "access_grants": [EVERYONE_READS],
            },
        )
        assert created.status_code == 200, created.text
        yield model_id
        client.post("/api/v1/models/model/delete", json={"id": model_id})


def create_folder(owner, data: dict | None = None) -> str:
    with owner.client() as client:
        created = client.post("/api/v1/folders/", json={"name": FOLDER, "data": data})
    assert created.status_code == 200, created.text
    return created.json()["id"]


def stored_default(owner, folder_id: str) -> list[str]:
    with owner.client() as client:
        found = client.get(f"/api/v1/folders/{folder_id}")
    assert found.status_code == 200, found.text
    return (found.json().get("data") or {}).get("model_ids") or []


def open_edit_dialog(page: Page) -> Locator:
    expect(chat_input(page)).to_be_visible()
    opener = page.get_by_role("button", name="Open Sidebar", exact=True)
    if opener.is_visible():
        opener.click()
    sidebar = page.get_by_role("navigation", name="Chat history")
    folders = sidebar.get_by_role("button", name="Folders", exact=True)
    expect(folders).to_be_visible()
    if folders.get_attribute("aria-expanded") != "true":
        folders.click()
    row = folder_row(sidebar, FOLDER)
    click_folder_menu_button(row)
    page.get_by_role("menu").get_by_role("menuitem", name="Edit").click()
    dialog = page.get_by_role("dialog")
    expect(dialog.get_by_placeholder("Enter folder name")).to_have_value(FOLDER)
    return dialog


def choose_default(page: Page, dialog: Locator, model_id: str) -> None:
    dialog.get_by_role("button", name="Use default").click()
    model_search(page).fill(model_id)
    page.get_by_role("option", name=f"Select {model_id} model").click()
    expect(
        dialog.get_by_role("button", name=re.compile(f"Selected model: {model_id}"))
    ).to_be_visible()


def save(page: Page, dialog: Locator) -> None:
    dialog.get_by_role("button", name="Save").click()
    expect(page.get_by_text("Folder updated successfully")).to_be_visible()


def sent_model(page: Page, text: str) -> str:
    with page.expect_request(
        lambda request: request.url.endswith("/api/chat/completions") and request.method == "POST"
    ) as posted:
        send(page, text)
    return json.loads(posted.value.post_data or "{}")["model"]


def test_the_model_chosen_in_the_dialog_is_the_folders_default(
    page_for, make_user, upstream, folder_model
):
    owner = make_user()
    folder_id = create_folder(owner)
    page = page_for(owner)
    dialog = open_edit_dialog(page)

    choose_default(page, dialog, folder_model)
    save(page, dialog)

    assert stored_default(owner, folder_id) == [folder_model]
    reopened = open_edit_dialog(page)
    expect(
        reopened.get_by_role("button", name=re.compile(f"Selected model: {folder_model}"))
    ).to_be_visible()


def test_a_chat_started_in_the_folder_opens_on_its_default_model(
    page_for, make_user, upstream, folder_model
):
    owner = make_user()
    folder_id = create_folder(owner, {"model_ids": [folder_model]})
    page = page_for(owner, path=f"/folders/{folder_id}")
    upstream.queue(reply.text("lamp lit", match=reply.answering("who keeps the light?")))

    assert sent_model(page, "who keeps the light?") == folder_model
    expect_reply(page, "lamp lit")


def test_the_reset_button_clears_the_default(page_for, make_user, folder_model):
    owner = make_user()
    folder_id = create_folder(owner, {"model_ids": [folder_model]})
    page = page_for(owner)
    dialog = open_edit_dialog(page)

    dialog.get_by_role("button", name="Reset").click()
    expect(dialog.get_by_role("button", name="Use default")).to_be_visible()
    expect(dialog.get_by_role("button", name="Reset")).to_have_count(0)
    save(page, dialog)

    assert stored_default(owner, folder_id) == []


def test_switching_models_in_a_folder_chat_leaves_the_folders_default(
    page_for, make_user, upstream, folder_model
):
    owner = make_user()
    folder_id = create_folder(owner, {"model_ids": [folder_model]})
    page = page_for(owner, path=f"/folders/{folder_id}")
    page.get_by_role("button", name=re.compile("Selected model")).click()
    model_search(page).fill(MOCK_MODEL_ID)
    page.get_by_role("option", name=f"Select {MOCK_MODEL_ID} model").click()
    upstream.queue(reply.text("fog horn", match=reply.answering("what is the weather?")))

    assert sent_model(page, "what is the weather?") == MOCK_MODEL_ID
    expect_reply(page, "fog horn")

    assert stored_default(owner, folder_id) == [folder_model], (
        "a chat's model switch overwrote the folder's default"
    )


def test_a_folder_without_a_default_does_not_inherit_the_last_chats_model(
    page_for, make_user, upstream, folder_model
):
    owner = make_user()
    folder_id = create_folder(owner)
    page = page_for(owner)
    page.get_by_role("button", name=re.compile("Selected model")).click()
    model_search(page).fill(folder_model)
    page.get_by_role("option", name=f"Select {folder_model} model").click()
    expect(
        page.get_by_role("button", name=re.compile(f"Selected model: {folder_model}"))
    ).to_be_visible()
    upstream.queue(reply.text("noted", match=reply.answering("log the tide")))
    send(page, "log the tide")
    expect_reply(page, "noted")
    expect(page).to_have_url(re.compile("/c/"))

    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    sidebar = page.get_by_role("navigation", name="Chat history")
    sidebar.get_by_role("button", name="Folders", exact=True).click()
    folder_link(sidebar, FOLDER).click()

    expect(page).to_have_url(re.compile(f"/folders/{folder_id}"))
    expect(
        page.get_by_role("button", name=re.compile(f"Selected model: {MOCK_MODEL_ID}"))
    ).to_be_visible()
