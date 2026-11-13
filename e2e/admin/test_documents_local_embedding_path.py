"""Regression: saving Admin Settings > Documents on a local embedding model path showed an error.

open-webui/open-webui#33977. With `RAG_EMBEDDING_MODEL` set to a folder on disk (a model
downloaded ahead of time and mounted into the container), every Save in the Documents tab raised
"Model filesystem path detected. Model shortname is required for update, cannot continue." The
instance itself loads and uses such a model, so the error flagged a setup that works. The Save
button and the embedding model's own update button now take a model path as they take a name, and
Save stores the embedding settings such as the batch size, which the error used to skip.

Discriminates: passes on a build with the fix, fails on dev 77d3d1c31 (the error toast shows on
Save, and the update button sends nothing to the server).
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.actors import admin_of
from harness.instance import LaunchedInstance
from harness.local_embedding import local_embedding_env, save_keyword_model

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

RETRIEVAL_CONFIG = ("/api/v1/retrieval/config", "/api/v1/retrieval/config/update")
EMBEDDING_SETTINGS = ("/api/v1/retrieval/embedding", "/api/v1/retrieval/embedding/update")
PATH_ERROR = "Model filesystem path detected. Model shortname is required for update"
SAVED = "Settings saved successfully!"


@pytest.fixture(scope="module")
def local_model(tmp_path_factory) -> str:
    return str(
        save_keyword_model(tmp_path_factory.mktemp("local-embedding"), ["orchard", "harbour"])
    )


@pytest.fixture
def local(instance_with, preserve, local_model) -> LaunchedInstance:
    """An instance whose embedding model is a folder on disk, the way a mounted model is set."""
    launched = instance_with(
        {**local_embedding_env(local_model), "RAG_EMBEDDING_MODEL_AUTO_UPDATE": "false"}
    )
    preserve(RETRIEVAL_CONFIG, on=launched)
    preserve(EMBEDDING_SETTINGS, on=launched)
    return launched


@pytest.fixture
def settings(local, page_for) -> tuple[Page, Locator]:
    page = page_for(admin_of(local), path="/admin/settings/documents")
    dialog = page.get_by_role("dialog")
    expect(dialog.get_by_text("Allowed File Extensions", exact=True)).to_be_visible()
    return page, dialog


def test_saving_the_documents_settings_on_a_local_model_path_shows_no_error(
    local, local_model, settings
):
    page, dialog = settings
    dialog.get_by_text("Max Upload Count", exact=True).locator(
        "xpath=following-sibling::div//input"
    ).fill("3")
    dialog.get_by_label("Embedding Batch Size", exact=True).fill("7")

    dialog.get_by_role("button", name="Save", exact=True).click()

    expect(page.get_by_text(SAVED).first).to_be_visible()
    # a toast clears itself after a few seconds, which `to_have_count(0)` would wait for
    shown = page.get_by_text(PATH_ERROR).count()
    assert shown == 0, "Save showed the filesystem path error for a model that is set as a path"
    with admin_of(local).client() as client:
        saved = client.get(RETRIEVAL_CONFIG[0]).json()
        embedding = client.get(EMBEDDING_SETTINGS[0]).json()
    assert saved["FILE_MAX_COUNT"] == 3, (
        f"Max Upload Count was not saved: {saved['FILE_MAX_COUNT']}"
    )
    assert embedding["RAG_EMBEDDING_MODEL"] == local_model, embedding["RAG_EMBEDDING_MODEL"]
    assert embedding["RAG_EMBEDDING_BATCH_SIZE"] == 7, (
        f"the embedding batch size was skipped on Save: {embedding['RAG_EMBEDDING_BATCH_SIZE']}"
    )


def test_the_update_button_sends_a_local_model_path_to_the_server(settings, local_model):
    page, dialog = settings

    with page.expect_response(
        lambda response: response.url.endswith(EMBEDDING_SETTINGS[1])
    ) as sent:
        dialog.get_by_role("button", name="Update embedding model", exact=True).click()

    assert sent.value.status == 200, sent.value.text()
    assert sent.value.json()["RAG_EMBEDDING_MODEL"] == local_model
    expect(page.get_by_text(PATH_ERROR)).to_have_count(0)
