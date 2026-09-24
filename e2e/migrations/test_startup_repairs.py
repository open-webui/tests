"""Regression: a list-shaped default models row left new chats without the default model.

Commit 5c05608e3a (v0.11.1): instances whose `ui.default_models` row was stored as a list
instead of the comma-separated string the app reads had their default models ignored, since the
chat page splits that string to preselect them. The boot's config repair now joins such a row.
The instance boots on a data directory whose legacy `config.json` writes the list-shaped row
(`harness.prepared_data.with_legacy_config`) and lists a model ahead of the default one, so the
page's own fallback to the first model cannot pass for the default.

Twin of unit/migrations/test_startup_repairs.py.
Discriminates: passes on dev bbfa876af; with the default-model pass of `Config.repair_config_rows`
removed from a copy of it the new chat does not open on the default model.
"""

from __future__ import annotations

import json

import pytest
from playwright.sync_api import expect

from harness.prepared_data import with_legacy_config
from harness.upstream import MOCK_MODEL_ID

pytestmark = [
    pytest.mark.regression,
    pytest.mark.slow,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
]

FIRST_LISTED_MODEL = "alpha-model"


@pytest.fixture
def new_chat_page(browser, instance_with, tmp_path_factory):
    repaired = with_legacy_config(instance_with, tmp_path_factory)
    if not repaired.serves_frontend:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    repaired.upstream.models = [FIRST_LISTED_MODEL, MOCK_MODEL_ID]
    with repaired.client() as client:
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
        client.post("/api/v1/users/user/settings/update", json={"ui": {"showChangelog": False}})

    context = browser.new_context(
        viewport={"width": 1920, "height": 1080}, base_url=repaired.base_url
    )
    page = context.new_page()
    token = json.dumps(repaired.admin_token)
    page.add_init_script(f"try {{ localStorage.setItem('token', {token}); }} catch (e) {{}}")
    page.goto("/")
    yield page
    context.close()


def test_a_new_chat_opens_on_the_repaired_default_model(new_chat_page):
    selected = new_chat_page.get_by_role("button", name=f"Selected model: {MOCK_MODEL_ID}")

    expect(selected).to_be_visible(timeout=30_000)
