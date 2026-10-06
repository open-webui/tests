"""Journey: what an admin edits in the OAuth / OIDC part of Admin Settings > Authentication.

On an instance signing in through an OpenID provider, with the panel's OAuth settings persisted
(`ENABLE_OAUTH_PERSISTENT_CONFIG`) so they can be edited there, the admin changes the sign-in rules
in the panel and saves, and the next "Continue with" sign-in follows them: Allowed Domains turns
away a person whose email is elsewhere while one from the listed domain gets in, Role Mapping makes
a person whose roles claim names an Admin Role an admin, and Group Mapping with Auto-Create Groups
puts a person in the groups their groups claim names.

Discriminates: passes on dev 30f3f6a8f; in a frontend build whose Authentication form
sends the OAuth settings it loaded instead of the edited ones, every test but the listed domain's
sign-in fails.
"""

from __future__ import annotations

import re
import uuid
from typing import Callable, Iterator

import pytest
from playwright.sync_api import Browser, Locator, Page, expect

from harness.actors import admin_of
from harness.oidc_provider import OAUTH_CONFIG_PATH, session_user, shared_provider, sso_env
from utils.chat_ui import chat_input

pytestmark = [
    pytest.mark.journey,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
    pytest.mark.slow,
]


@pytest.fixture
def idp():
    return shared_provider()


@pytest.fixture
def sso(instance_with, idp, preserve):
    launched = instance_with(
        {**sso_env(idp), "ENABLE_OAUTH_PERSISTENT_CONFIG": "true", "DEFAULT_USER_ROLE": "user"}
    )
    if not launched.serves_frontend:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    preserve((OAUTH_CONFIG_PATH, OAUTH_CONFIG_PATH), on=launched)
    return launched


@pytest.fixture
def signed_out(browser: Browser, sso) -> Iterator[Callable[[], Page]]:
    """`signed_out()` opens the auth page in a browser that has never signed in."""
    contexts = []

    def open_auth() -> Page:
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080}, base_url=sso.base_url
        )
        contexts.append(context)
        page = context.new_page()
        page.goto("/auth")
        return page

    yield open_auth
    for context in contexts:
        context.close()


def oauth_panel(page_for, sso) -> tuple[Page, Locator]:
    page = page_for(admin_of(sso))
    page.goto("/admin/settings/authentication")
    oauth = page.get_by_role("dialog").get_by_role("group")
    expect(oauth.get_by_role("switch", name="OAuth / OIDC")).to_be_checked()
    return page, oauth


def field(panel: Locator, label: str) -> Locator:
    return panel.get_by_text(label, exact=True).locator("xpath=following-sibling::div//input")


def set_switch(panel: Locator, name: str) -> None:
    switch = panel.get_by_role("switch", name=name, exact=True)
    if switch.get_attribute("aria-checked") != "true":
        switch.click()
    expect(switch).to_be_checked()


def save(page: Page) -> None:
    page.get_by_role("dialog").get_by_role("button", name="Save", exact=True).click()
    expect(page.get_by_text("Settings saved successfully!").first).to_be_visible()


def continue_with(page: Page, provider: str = "SSO") -> None:
    page.get_by_role("button", name=f"Continue with {provider}").click()


def test_allowed_domains_turn_away_a_person_from_another_domain(page_for, sso, idp, signed_out):
    page, oauth = oauth_panel(page_for, sso)
    field(oauth, "Allowed Domains").fill("harbour.example")
    save(page)
    idp.sign_in_as(email=f"skipper-{uuid.uuid4().hex[:6]}@elsewhere.example")

    visitor = signed_out()
    continue_with(visitor)

    expect(visitor).to_have_url(re.compile(r"/auth"))
    expect(visitor.get_by_role("button", name="Continue with SSO")).to_be_visible()
    expect(chat_input(visitor)).to_have_count(0)


def test_allowed_domains_let_a_person_from_the_listed_domain_in(page_for, sso, idp, signed_out):
    page, oauth = oauth_panel(page_for, sso)
    field(oauth, "Allowed Domains").fill("harbour.example")
    save(page)
    idp.sign_in_as(email=f"skipper-{uuid.uuid4().hex[:6]}@harbour.example")

    visitor = signed_out()
    continue_with(visitor)

    expect(chat_input(visitor)).to_be_visible()


def test_role_mapping_makes_a_person_with_an_admin_role_an_admin(page_for, sso, idp, signed_out):
    page, oauth = oauth_panel(page_for, sso)
    set_switch(oauth, "Role Mapping")
    field(oauth, "Roles Claim").fill("roles")
    field(oauth, "Admin Roles").fill("harbourmaster")
    field(oauth, "Allowed Roles").fill("harbourmaster,crew")
    save(page)
    idp.sign_in_as(roles=["harbourmaster"])

    visitor = signed_out()
    continue_with(visitor)
    expect(chat_input(visitor)).to_be_visible()

    assert session_user(sso, visitor.evaluate("localStorage.token"))["role"] == "admin"


def test_group_mapping_puts_a_person_in_the_groups_of_their_claim(page_for, sso, idp, signed_out):
    group = f"Harbour crew {uuid.uuid4().hex[:6]}"
    page, oauth = oauth_panel(page_for, sso)
    set_switch(oauth, "Group Mapping")
    set_switch(oauth, "Auto-Create Groups")
    field(oauth, "Group Claim").fill("groups")
    save(page)
    claims = idp.sign_in_as(groups=[group])

    visitor = signed_out()
    continue_with(visitor)
    expect(chat_input(visitor)).to_be_visible()

    with admin_of(sso).client() as client:
        [created] = [
            entry for entry in client.get("/api/v1/groups/").json() if entry["name"] == group
        ]
        members = client.get(f"/api/v1/groups/id/{created['id']}/export").json()["user_ids"]
        found = client.get("/api/v1/users/", params={"query": claims["email"]}).json()["users"]
        client.delete(f"/api/v1/groups/id/{created['id']}/delete")
    assert [person["id"] for person in found] == members
