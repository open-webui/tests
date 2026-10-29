"""Journey: an admin editing a model sees what its provider reports about it.

The model editor shows admins a Provider details row with what the connection's `/models` lists
for the model: its inputs and outputs, context window, the provider's own context limit and
maximum output, and the parameters it supports. In Admin Settings > Models the row describes the
provider model itself; in the workspace editor a preset shows the details of its base model. A
field the provider leaves out reads "Not provided", an empty list "None reported", and a base
model reporting nothing at all shows the row as "Not provided". A writer of the preset who is no
admin sees no such row. No existing module serves a model with reported details, so these tests
get one of their own.

Discriminates: passes on the dev c5981aaea build (3 of 3). In a build of it whose editor reads the
context window from the provider's limit, drops the supported parameters, opens the row even with
nothing reported and shows it to every editor, every test fails.
"""

from __future__ import annotations

import re
import uuid
from typing import Callable, Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.actors import Actor
from harness.python_tools import EVERYONE_READS
from harness.upstream import MOCK_MODEL_ID
from utils.model_editor import section

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

TITLE = "Provider details"
FULL_DETAILS = {
    "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
    "context_length": 131072,
    "top_provider": {"context_length": 65536, "max_completion_tokens": 8192},
    "supported_parameters": ["temperature", "tools", "reasoning"],
}
FULL_ROWS = {
    "Inputs": "Text, Image",
    "Outputs": "Text",
    "Context window": "131,072 tokens",
    "Provider context": "65,536 tokens",
    "Max output": "8,192 tokens",
    "Supported parameters": "temperature, tools, reasoning",
}


@pytest.fixture
def builder(make_user) -> Actor:
    return make_user(role="admin")


@pytest.fixture
def served(upstream, builder) -> Iterator[Callable[[dict], str]]:
    """`served(details)` makes the scripted provider list a new model reporting `details`."""
    model_ids: list[str] = []

    def serve(details: dict) -> str:
        model_id = f"reported-{uuid.uuid4().hex[:8]}"
        upstream.models.append(model_id)
        upstream.model_details[model_id] = details
        model_ids.append(model_id)
        with builder.client() as client:
            client.get("/api/models", params={"refresh": "true"}).raise_for_status()
        return model_id

    yield serve
    with builder.client() as client:
        for model_id in model_ids:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


@pytest.fixture
def preset_on(builder) -> Iterator[Callable[..., dict]]:
    """`preset_on(base_id)` makes a preset on that base model; deleted afterwards."""
    made: list[dict] = []

    def make(base_id: str, access_grants: list[dict] | None = None) -> dict:
        suffix = uuid.uuid4().hex[:8]
        model = {"id": f"details-{suffix}", "name": f"Tide reader {suffix}"}
        with builder.client() as client:
            created = client.post(
                "/api/v1/models/create",
                json={
                    **model,
                    "base_model_id": base_id,
                    "meta": {},
                    "params": {},
                    "access_grants": access_grants or [EVERYONE_READS],
                },
            )
        assert created.status_code == 200, created.text
        made.append(model)
        return model

    yield make
    with builder.client() as client:
        for model in made:
            client.post("/api/v1/models/model/delete", json={"id": model["id"]})


def workspace_editor(page_for, actor: Actor, model: dict) -> Locator:
    page = page_for(actor, path=f"/workspace/models/edit?id={model['id']}")
    editor = page.get_by_role("main")
    expect(editor.get_by_placeholder("Model Name")).to_have_value(model["name"])
    return editor


def admin_editor(page: Page, model_id: str) -> Locator:
    settings = page.get_by_role("dialog")
    settings.get_by_role("textbox", name="Search Models").fill(model_id)
    row = settings.locator("#model-list > div").filter(has_text=model_id)
    expect(row).to_have_count(1)
    row.get_by_role("button", name=model_id).first.click()
    expect(settings.get_by_placeholder("Model Name")).to_have_value(model_id)
    return settings


def expect_details(editor: Locator, summary: str, rows: dict[str, str]) -> None:
    """The Provider details part reads `summary` closed and lists exactly `rows` opened."""
    title = editor.locator("summary").filter(has_text=TITLE)
    expect(title).to_contain_text(summary)
    part = section(editor, TITLE)
    expect(part.get_by_role("term")).to_have_text(list(rows))
    expect(part.get_by_role("definition")).to_have_text(list(rows.values()))


def test_the_admin_settings_editor_shows_what_the_provider_reports(page_for, builder, served):
    model_id = served(FULL_DETAILS)
    editor = admin_editor(page_for(builder, path="/admin/settings/models"), model_id)

    expect_details(editor, "131,072 tokens", FULL_ROWS)


def test_a_preset_shows_the_details_of_its_base_model(page_for, builder, served, preset_on):
    preset = preset_on(served(FULL_DETAILS))
    editor = workspace_editor(page_for, builder, preset)

    expect_details(editor, "131,072 tokens", FULL_ROWS)


def test_fields_the_provider_leaves_out_read_not_provided(page_for, builder, served, preset_on):
    sparse = {
        "architecture": {"input_modalities": ["text"]},
        "context_length": 32768,
        "supported_parameters": [],
    }
    editor = workspace_editor(page_for, builder, preset_on(served(sparse)))

    expect_details(
        editor,
        "32,768 tokens",
        {
            "Inputs": "Text",
            "Outputs": "Not provided",
            "Context window": "32,768 tokens",
            "Supported parameters": "None reported",
        },
    )


def test_a_base_model_reporting_nothing_shows_the_row_as_not_provided(page_for, builder, preset_on):
    editor = workspace_editor(page_for, builder, preset_on(MOCK_MODEL_ID))

    # the row is the label and its value, with nothing to open
    row = editor.get_by_text(TITLE, exact=True).locator("xpath=ancestor::div[1]")
    expect(row).to_have_text(re.compile(rf"^\s*{TITLE}\s*Not provided\s*$"))
    expect(editor.locator("summary").filter(has_text=TITLE)).to_have_count(0)


def test_a_writer_who_is_no_admin_sees_no_provider_details(
    page_for, builder, make_user, preserve, served, preset_on
):
    preserve("permissions")
    with builder.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["workspace"]["models"] = True
        saved = client.post("/api/v1/users/default/permissions", json=permissions)
    assert saved.status_code == 200, saved.text
    writer = make_user()
    grants = [EVERYONE_READS] + [
        {"principal_type": "user", "principal_id": writer.id, "permission": permission}
        for permission in ("read", "write")
    ]
    preset = preset_on(served(FULL_DETAILS), access_grants=grants)

    # the same preset shows the details to an admin, so their absence below is the role's doing
    expect(workspace_editor(page_for, builder, preset).get_by_text(TITLE)).to_be_visible()

    editor = workspace_editor(page_for, writer, preset)
    expect(editor.get_by_text("System Prompt", exact=True)).to_be_visible()
    expect(editor.get_by_text(TITLE)).to_have_count(0)
