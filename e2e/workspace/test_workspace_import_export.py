"""Journey: models, prompts, tools and skills leave the workspace as a JSON file and return.

A fresh admin owns one of each. Exporting from a workspace list downloads a file that holds the
item; after the item is deleted, importing that file from the same list brings it back with its
content intact, with the tool asking for a confirmation first.

Discriminates: passes on dev 176d31d1d. In a frontend copy where the prompts import sends an empty
content, the skills import a changed content and the models import no base model, only the matching
round trip goes red. Retargeted for 9bbb95048, where skills import through a dialog; the skill
round trip passes on dev 178de3666.
"""

from __future__ import annotations

import json
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness.actors import Actor
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

TOOL_SOURCE = "class Tools:\n    def ping(self) -> str:\n        return 'pong'\n"


@pytest.fixture
def keeper(make_user):
    """A fresh admin; what it owns in the workspace is deleted afterwards."""
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for model in client.get("/api/v1/models/list").json().get("items", []):
            if model["user_id"] == account.id:
                client.post("/api/v1/models/model/delete", json={"id": model["id"]})
        for prompt in client.get("/api/v1/prompts/").json():
            if prompt["user_id"] == account.id:
                client.delete(f"/api/v1/prompts/id/{prompt['id']}/delete")
        for tool in client.get("/api/v1/tools/").json():
            if tool["user_id"] == account.id:
                client.delete(f"/api/v1/tools/id/{tool['id']}/delete")
        for skill in client.get("/api/v1/skills/").json():
            if skill["user_id"] == account.id:
                client.delete(f"/api/v1/skills/id/{skill['id']}/delete")


def _create(account: Actor, path: str, form: dict) -> dict:
    with account.client() as client:
        created = client.post(path, json=form)
    assert created.status_code == 200, created.text
    return created.json()


def _remove(account: Actor, path: str, **options) -> None:
    with account.client() as client:
        removed = client.request(options.pop("method", "DELETE"), path, **options)
    assert removed.status_code == 200, removed.text


def _choose_action(page: Page, section: str, action: str) -> None:
    page.goto(f"/workspace/{section}")
    page.get_by_label("Open create menu").click()
    page.get_by_role("button", name=action).click()


def _export(page: Page, section: str) -> list[dict]:
    with page.expect_download() as download:
        _choose_action(page, section, "Export JSON")
    with open(download.value.path()) as saved:
        return json.load(saved)


def _import(page: Page, section: str, content: str) -> None:
    with page.expect_file_chooser() as chooser:
        _choose_action(page, section, "Import JSON")
    chooser.value.set_files(
        files=[{"name": "import.json", "mimeType": "application/json", "buffer": content.encode()}]
    )


def _fetch(account: Actor, path: str) -> dict:
    with account.client() as client:
        fetched = client.get(path)
    assert fetched.status_code == 200, fetched.text
    return fetched.json()


def _find_in_list(page: Page, label: str, name: str) -> None:
    page.get_by_role("textbox", name=label).fill(name)
    expect(page.get_by_text(name, exact=True)).to_be_visible()


def test_an_exported_model_is_restored_by_importing_the_file(page_for, keeper):
    name = f"Exported {uuid.uuid4().hex[:8]}"
    model_id = name.lower().replace(" ", "-")
    form = {"id": model_id, "base_model_id": MOCK_MODEL_ID, "name": name, "meta": {}, "params": {}}
    _create(keeper, "/api/v1/models/create", form)
    page = page_for(keeper)

    saved = [entry for entry in _export(page, "models") if entry["id"] == model_id]
    assert len(saved) == 1, "the exported file does not hold the model"
    _remove(keeper, "/api/v1/models/model/delete", method="POST", json={"id": model_id})
    _import(page, "models", json.dumps(saved))

    _find_in_list(page, "Search Models", name)
    restored = _fetch(keeper, f"/api/v1/models/model?id={model_id}")
    assert restored["base_model_id"] == MOCK_MODEL_ID
    assert restored["name"] == name


def test_an_exported_prompt_is_restored_by_importing_the_file(page_for, keeper):
    suffix = uuid.uuid4().hex[:8]
    name, command = f"Exported prompt {suffix}", f"exported{suffix}"
    prompt = _create(
        keeper,
        "/api/v1/prompts/create",
        {"command": command, "name": name, "content": f"Say hello, take {suffix}."},
    )
    page = page_for(keeper)

    saved = [entry for entry in _export(page, "prompts") if entry["command"] == command]
    assert len(saved) == 1, "the exported file does not hold the prompt"
    _remove(keeper, f"/api/v1/prompts/id/{prompt['id']}/delete")
    _import(page, "prompts", json.dumps(saved))

    _find_in_list(page, "Search Prompts", name)
    with keeper.client() as client:
        listed = client.get("/api/v1/prompts/").json()
    restored = [entry for entry in listed if entry["command"] == command]
    assert [entry["content"] for entry in restored] == [f"Say hello, take {suffix}."]


def test_an_exported_tool_is_restored_by_importing_the_file(page_for, keeper):
    suffix = uuid.uuid4().hex[:8]
    name, tool_id = f"Exported tool {suffix}", f"exported_tool_{suffix}"
    form = {"id": tool_id, "name": name, "content": TOOL_SOURCE, "meta": {"description": "pings"}}
    _create(keeper, "/api/v1/tools/create", form)
    page = page_for(keeper)

    saved = [entry for entry in _export(page, "tools") if entry["id"] == tool_id]
    assert len(saved) == 1, "the exported file does not hold the tool"
    _remove(keeper, f"/api/v1/tools/id/{tool_id}/delete")
    _import(page, "tools", json.dumps(saved))
    page.get_by_role("dialog", name="Confirm your action").get_by_role(
        "button", name="Confirm"
    ).click()

    expect(page.get_by_text("Tool imported successfully")).to_be_visible()
    _find_in_list(page, "Search Tools", name)
    assert _fetch(keeper, f"/api/v1/tools/id/{tool_id}")["content"] == TOOL_SOURCE


def test_an_exported_skill_is_restored_by_importing_the_file(page_for, keeper):
    suffix = uuid.uuid4().hex[:8]
    name, skill_id = f"Exported skill {suffix}", f"exported-skill-{suffix}"
    form = {"id": skill_id, "name": name, "content": "Be brief.", "meta": {}, "is_active": True}
    _create(keeper, "/api/v1/skills/create", form)
    page = page_for(keeper)

    saved = [entry for entry in _export(page, "skills") if entry["id"] == skill_id]
    assert len(saved) == 1, "the exported file does not hold the skill"
    _remove(keeper, f"/api/v1/skills/id/{skill_id}/delete")
    # skills import through a dialog that lists what the file holds before saving it
    page.goto("/workspace/skills")
    page.get_by_label("Open create menu").click()
    with page.expect_file_chooser() as chooser:
        page.get_by_role("button", name="Import", exact=True).click()
    chooser.value.set_files(
        files=[
            {
                "name": "import.json",
                "mimeType": "application/json",
                "buffer": json.dumps(saved).encode(),
            }
        ]
    )
    dialog = page.get_by_role("dialog")
    dialog.get_by_role("button", name="Import selected").click()
    expect(dialog.get_by_text("saved", exact=True)).to_be_visible()
    dialog.get_by_role("button", name="Close").click()
    _find_in_list(page, "Search Skills", name)
    assert _fetch(keeper, f"/api/v1/skills/id/{skill_id}")["content"] == "Be brief."
