"""Journey: a workspace model's versions in the model editor, as the person editing it meets them.

Saving a changed model with a message beside Save & Update adds a version, and the editor's
version menu lists it as Production under that message. Picking the older version shows its
settings read only, and Set as Production brings them back into the editor, onto the stored
model and into the next chat with it. With unsaved edits in the editor, Set as Production asks
first. An older version can be deleted from its own menu; the Production one offers no delete.
Who may reach the versions over the API is integration/models/test_model_history.py.

Discriminates: passes on the dev 206bf9723 build (3 of 3). In a frontend build whose editor
drops the commit message from the save, whose Set as Production sends the Production version's
id, whose version delete skips its request and whose unsaved-edits check is gone,
`test_a_save_with_a_message_is_listed_as_the_production_version`,
`test_set_as_production_brings_the_old_prompt_back_into_the_editor_and_the_chat`,
`test_unsaved_edits_are_confirmed_before_an_old_version_replaces_them` and
`test_an_older_version_can_be_deleted_but_not_the_production_one` go red (no message, the
stored prompt stays the newer one, no confirmation, the deleted version is still stored) while
the read-only test and an unrelated model editor test pass. In a build where picking a version
always shows Production, `test_an_older_version_opens_read_only_beside_the_production_one` goes
red (no preview opens) and the unrelated test passes.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.actors import Actor
from harness.chat import ask
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

SYSTEM_PROMPT = re.compile("^Write your model system prompt")


@pytest.fixture
def curator(make_user):
    """A fresh admin; its models are deleted afterwards."""
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for model in client.get("/api/v1/models/list").json().get("items", []):
            if model["user_id"] == account.id:
                client.post("/api/v1/models/model/delete", json={"id": model["id"]})


def _model_with_two_versions(account: Actor) -> dict:
    """A model saved twice over the API: "first draft", then "second draft" as "Shorter prompt"."""
    suffix = uuid.uuid4().hex[:8]
    form = {
        "id": f"versions-{suffix}",
        "name": f"Versions {suffix}",
        "base_model_id": MOCK_MODEL_ID,
        "meta": {"description": "version history"},
        "params": {"system": "first draft"},
    }
    with account.client() as client:
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
        saved = client.post(
            "/api/v1/models/model/update",
            json={
                **form,
                "params": {"system": "second draft"},
                "commit_message": "Shorter prompt",
            },
        )
        assert saved.status_code == 200, saved.text
    return {**form, "first_version": created.json()["version_id"]}


def _stored(account: Actor, model_id: str) -> tuple[dict, list[dict]]:
    with account.client() as client:
        model = client.get("/api/v1/models/model", params={"id": model_id})
        versions = client.get("/api/v1/models/model/history", params={"id": model_id})
    assert model.status_code == 200, model.text
    assert versions.status_code == 200, versions.text
    return model.json(), versions.json()


def _open_editor(page: Page, model: dict) -> Locator:
    page.goto(f"/workspace/models/edit?id={model['id']}")
    editor = page.get_by_role("main")
    expect(editor.get_by_placeholder("Model Name")).to_have_value(model["name"])
    return editor


def _version_menu(page: Page) -> Locator:
    page.get_by_label("Select version", exact=True).click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_role("menuitemradio", name="Production")).to_be_visible()
    return menu


def _pick_first_version(page: Page, model: dict) -> Locator:
    """The read-only preview of the model's first version, picked from the version menu."""
    _version_menu(page).get_by_role("menuitemradio", name=model["first_version"][:7]).click()
    preview = page.get_by_role("region", name="Model version preview")
    expect(preview.get_by_role("textbox", name=SYSTEM_PROMPT)).to_have_value("first draft")
    return preview


def test_a_save_with_a_message_is_listed_as_the_production_version(page_for, curator):
    suffix = uuid.uuid4().hex[:8]
    model = {
        "id": f"versions-{suffix}",
        "name": f"Versions {suffix}",
        "base_model_id": MOCK_MODEL_ID,
        "meta": {"description": "version history"},
        "params": {"system": "first draft"},
    }
    with curator.client() as client:
        assert client.post("/api/v1/models/create", json=model).status_code == 200
    page = page_for(curator)
    editor = _open_editor(page, model)

    editor.get_by_role("textbox", name=SYSTEM_PROMPT).fill("second draft")
    editor.get_by_role("textbox", name="Commit message").fill("Shorter prompt")
    editor.get_by_role("button", name="Save & Update").click()
    expect(page).to_have_url(re.compile(r"/workspace/models/?$"))

    _open_editor(page, model)
    menu = _version_menu(page)
    production = menu.get_by_role("menuitemradio", name="Production")
    expect(production).to_contain_text("Shorter prompt")
    expect(menu.get_by_role("menuitemradio")).to_have_count(2)
    stored, versions = _stored(curator, model["id"])
    assert stored["params"]["system"] == "second draft"
    production_version = [v for v in versions if v["id"] == stored["version_id"]]
    assert [v["commit_message"] for v in production_version] == ["Shorter prompt"]


def test_an_older_version_opens_read_only_beside_the_production_one(page_for, curator):
    model = _model_with_two_versions(curator)
    page = page_for(curator)
    _open_editor(page, model)

    preview = _pick_first_version(page, model)

    expect(preview.get_by_role("textbox", name=SYSTEM_PROMPT)).to_be_disabled()
    expect(page.get_by_role("button", name="Set as Production")).to_be_visible()
    expect(page.get_by_role("button", name="Save & Update")).to_have_count(0)


def test_set_as_production_brings_the_old_prompt_back_into_the_editor_and_the_chat(
    page_for, curator, upstream
):
    model = _model_with_two_versions(curator)
    page = page_for(curator)
    editor = _open_editor(page, model)
    _pick_first_version(page, model)

    page.get_by_role("button", name="Set as Production").click()

    expect(page.get_by_text("Production version updated")).to_be_visible()
    expect(editor.get_by_role("textbox", name=SYSTEM_PROMPT)).to_have_value("first draft")
    stored, versions = _stored(curator, model["id"])
    assert stored["params"]["system"] == "first draft"
    assert stored["version_id"] == model["first_version"]
    assert len(versions) == 2

    question = f"which prompt? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("the first one", match=reply.answering(question)))
    with curator.client() as client:
        ask(client, question, model=model["id"])
    sent = [body for body in upstream.chat_requests() if reply.answering(question)(body)]
    assert sent[-1]["messages"][0] == {"role": "system", "content": "first draft"}


def test_unsaved_edits_are_confirmed_before_an_old_version_replaces_them(page_for, curator):
    model = _model_with_two_versions(curator)
    page = page_for(curator)
    editor = _open_editor(page, model)
    editor.get_by_role("textbox", name=SYSTEM_PROMPT).fill("an edit nobody saved")
    _pick_first_version(page, model)

    page.get_by_role("button", name="Set as Production").click()
    confirm = page.get_by_role("dialog").filter(has_text="Discard unsaved changes?")
    expect(confirm).to_be_visible()
    stored, _ = _stored(curator, model["id"])
    assert stored["params"]["system"] == "second draft"

    confirm.get_by_role("button", name="Set as Production").click()
    expect(page.get_by_text("Production version updated")).to_be_visible()
    expect(editor.get_by_role("textbox", name=SYSTEM_PROMPT)).to_have_value("first draft")
    stored, _ = _stored(curator, model["id"])
    assert stored["params"]["system"] == "first draft"


def test_an_older_version_can_be_deleted_but_not_the_production_one(page_for, curator):
    model = _model_with_two_versions(curator)
    page = page_for(curator)
    _open_editor(page, model)
    menu = _version_menu(page)
    older = menu.get_by_role("menuitemradio", name=model["first_version"][:7])
    expect(older).to_be_visible()

    rows = menu.locator("div.group")
    production_row = rows.filter(has_text="Production")
    expect(production_row).to_have_count(1)
    expect(production_row.get_by_label("More Options", exact=True)).to_have_count(0)
    rows.filter(has_text=model["first_version"][:7]).get_by_label(
        "More Options", exact=True
    ).click()
    page.get_by_role("button", name="Delete", exact=True).click()
    page.get_by_role("dialog").filter(has_text="Delete Version").get_by_role(
        "button", name="Delete"
    ).click()

    expect(page.get_by_text("Version deleted")).to_be_visible()
    stored, versions = _stored(curator, model["id"])
    assert [version["id"] for version in versions] == [stored["version_id"]]
    assert stored["params"]["system"] == "second draft"
