"""A prompt you can only read offered edit, share, delete and enable controls that looked usable.

Fix commit `ff89756b6`, issue open-webui/open-webui#30219. The workspace prompt list showed the same
controls on every row, so a user with read access saw an active enable switch and an active Edit,
Share and Delete in the row's menu, none of which the server would let them use. The list now
disables them on rows without write access.

Discriminates: passes on dev efe63bd34; with ff89756b6 reverted the reader's switch and its Edit,
Share and Delete are enabled; the writer's row passes on both.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.access import grant, make_group
from harness.actors import Actor
from utils.workspace_list import item_row

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

PROMPT_USER = {"workspace": {"prompts": True}}
WRITE_ONLY_ITEMS = ["Edit", "Share", "Delete"]


@pytest.fixture
def shared_prompt(admin, make_user) -> tuple[str, Actor, Actor]:
    """A prompt the admin shares read-only with one user and writable with another."""
    reader, writer = make_user(), make_user()
    make_group(admin, [reader, writer], PROMPT_USER)
    name = f"Summary {uuid.uuid4().hex[:6]}"
    with admin.client() as client:
        created = client.post(
            "/api/v1/prompts/create",
            json={"command": f"summary-{uuid.uuid4().hex[:8]}", "name": name, "content": "Sum up"},
        )
        assert created.status_code == 200, created.text
        prompt_id = created.json()["id"]
        grants = [
            grant("user", reader.id, "read"),
            grant("user", writer.id, "read"),
            grant("user", writer.id, "write"),
        ]
        shared = client.post(
            f"/api/v1/prompts/id/{prompt_id}/access/update", json={"access_grants": grants}
        )
        assert shared.status_code == 200, shared.text
        yield name, reader, writer
        client.delete(f"/api/v1/prompts/id/{prompt_id}/delete")


def _prompt_row(page: Page, name: str) -> Locator:
    page.goto("/workspace/prompts")
    row = item_row(page, name)
    expect(row).to_be_visible()
    return row


def _open_menu(page: Page, row: Locator) -> None:
    row.hover()
    # the menu's trigger wraps the labelled button
    row.get_by_role("button", name="Prompt Menu").last.click()


def test_a_read_only_prompt_shows_its_write_controls_as_unavailable(page_for, shared_prompt):
    name, reader, _ = shared_prompt
    page = page_for(reader)
    row = _prompt_row(page, name)

    expect(row.get_by_role("switch")).to_be_disabled()
    _open_menu(page, row)
    for item in WRITE_ONLY_ITEMS:
        expect(page.get_by_role("menuitem", name=item, exact=True)).to_be_disabled()


# ---------------------------------------------------------------- nearby


def test_a_read_only_prompt_can_still_be_cloned(page_for, shared_prompt):
    name, reader, _ = shared_prompt
    page = page_for(reader)
    _open_menu(page, _prompt_row(page, name))

    expect(page.get_by_role("menuitem", name="Clone", exact=True)).to_be_enabled()


def test_a_writable_prompt_keeps_its_controls(page_for, shared_prompt):
    name, _, writer = shared_prompt
    page = page_for(writer)
    row = _prompt_row(page, name)

    expect(row.get_by_role("switch")).to_be_enabled()
    _open_menu(page, row)
    for item in WRITE_ONLY_ITEMS:
        expect(page.get_by_role("menuitem", name=item, exact=True)).to_be_enabled()
