"""Regression: a user granted only the Skills workspace section could not reach the workspace.

open-webui discussion #24719, fixed by 359590ca9 (PR #24729): the sidebar's Workspace entry and
the /workspace index redirect each listed the workspace sections a user may open, and Skills was
missing from both. A user whose only workspace permission was Skills saw no Workspace entry, and
opening /workspace sent them home. The fix adds Skills to both lists.

Every workspace section is read from the default-permissions API, so a section added later is
covered as soon as the backend offers it.

Twin of unit/frontend/test_workspace_permissions.py.

Discriminates: passes on the bbfa876af build; with Skills removed from the sidebar check and the
/workspace redirect chain again, the Skills-only user sees no Workspace entry and lands on /.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

import pytest
from playwright.sync_api import Page, expect

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

PERMISSIONS = "/api/v1/users/default/permissions"


@pytest.fixture
def default_permissions(admin, preserve) -> dict:
    preserve("permissions")
    with admin.client() as client:
        current = client.get(PERMISSIONS)
    current.raise_for_status()
    return current.json()


@pytest.fixture
def workspace_sections(default_permissions) -> list[str]:
    # the section toggles, not their `<section>_import` and `<section>_export` companions
    sections = [key for key in default_permissions["workspace"] if "_" not in key]
    assert "skills" in sections, f"the backend no longer offers a skills section: {sections}"
    return sections


@pytest.fixture
def grant_only(admin, default_permissions):
    """Make these workspace sections, and no others, the default for every account."""

    def grant(*sections: str) -> None:
        workspace = {key: key in sections for key in default_permissions["workspace"]}
        with admin.client() as client:
            saved = client.post(PERMISSIONS, json={**default_permissions, "workspace": workspace})
        saved.raise_for_status()

    return grant


@pytest.fixture
def open_as_new_user(page_for, make_user):
    def open_page() -> Page:
        page = page_for(make_user())
        expect(page.get_by_role("link", name="New Chat")).to_be_visible()
        return page

    return open_page


def workspace_entry(page: Page):
    return page.get_by_role("link", name="Workspace", exact=True)


def landing_path(page: Page, path: str) -> str:
    """Where the app settles after opening `path`, once its client-side redirect has run."""
    page.goto(path)
    page.wait_for_url(lambda url: urlparse(url).path != path)
    return urlparse(page.url).path


def test_a_user_granted_only_skills_can_open_the_workspace(grant_only, open_as_new_user):
    grant_only("skills")
    page = open_as_new_user()

    expect(workspace_entry(page)).to_be_visible()
    page.goto("/workspace")
    expect(page).to_have_url(re.compile(r"/workspace/skills$"))


def test_each_section_alone_opens_the_workspace_on_that_section(
    workspace_sections, grant_only, open_as_new_user
):
    seen = {}
    for section in workspace_sections:
        grant_only(section)
        page = open_as_new_user()
        has_entry = workspace_entry(page).is_visible()
        seen[section] = (has_entry, landing_path(page, "/workspace"))

    assert seen == {section: (True, f"/workspace/{section}") for section in workspace_sections}


def test_without_a_section_there_is_no_entry_and_every_workspace_page_sends_you_home(
    workspace_sections, grant_only, open_as_new_user
):
    grant_only()
    page = open_as_new_user()

    expect(workspace_entry(page)).to_have_count(0)
    paths = ["/workspace", *(f"/workspace/{section}" for section in workspace_sections)]
    assert {path: landing_path(page, path) for path in paths} == {path: "/" for path in paths}
