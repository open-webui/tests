"""Journey: an admin connects LDAP in Admin Settings > Authentication and people sign in with it.

The admin switches LDAP on and fills in the directory's host, port, service account, the
attributes for mail and username and the search base, and saves. A visitor then signs in on the
auth page with their directory username and password and lands in the chat with an account under
their directory mail; a wrong password is refused. With Group Mapping and Auto-Create Groups on,
the person's directory groups become groups they belong to. With LDAP switched off again, the
auth page asks for an email instead of a directory username.

Discriminates: passes on dev 30f3f6a8f; in a frontend build whose Authentication form saves the
LDAP server settings it loaded and the LDAP switch as it was, every test but the wrong password
one fails.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.ldap_server import (
    LDAP_CONFIG,
    PEOPLE_DN,
    SERVICE_DN,
    SERVICE_PASSWORD,
    save_ldap_settings,
    serve_directory,
)
from utils.chat_ui import chat_input

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

ADMIN_CONFIG = "/api/v1/auths/admin/config"
PASSWORD = "directory-pass-1"


@pytest.fixture
def directory(admin, preserve):
    """A directory to connect, with newcomers let in as users."""
    preserve(LDAP_CONFIG, "admin_config")
    with admin.client() as client:
        current = client.get(ADMIN_CONFIG).json()
        client.post(ADMIN_CONFIG, json={**current, "DEFAULT_USER_ROLE": "user"}).raise_for_status()
    with serve_directory() as served:
        yield served


def open_authentication(page: Page) -> Locator:
    # the LDAP switch takes its stored state last, after the rest of the form shows
    with page.expect_response(lambda response: response.url.endswith("/auths/admin/config/ldap")):
        page.goto("/admin/settings/authentication")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("switch", name="LDAP", exact=True)).to_be_visible()
    return settings


def set_switch(settings: Locator, name: str, turn_on: bool) -> None:
    switch = settings.get_by_role("switch", name=name, exact=True)
    if (switch.get_attribute("aria-checked") == "true") != turn_on:
        switch.click()
    expect(switch).to_have_attribute("aria-checked", "true" if turn_on else "false")


def save(page: Page, settings: Locator) -> None:
    settings.get_by_role("button", name="Save", exact=True).click()
    expect(page.get_by_text("Settings saved successfully!").first).to_be_visible()


def connect_directory(page: Page, directory, group_mapping: bool = False) -> None:
    settings = open_authentication(page)
    set_switch(settings, "LDAP", turn_on=True)
    settings.get_by_placeholder("Enter server label").fill("Harbour directory")
    settings.get_by_placeholder("Enter server host").fill(directory.host)
    settings.get_by_placeholder("Enter server port").fill(str(directory.port))
    settings.get_by_placeholder("Enter Application DN", exact=True).fill(SERVICE_DN)
    settings.get_by_placeholder("Enter Application DN Password").fill(SERVICE_PASSWORD)
    settings.get_by_placeholder("Example: mail").fill("mail")
    settings.get_by_placeholder("Example: sAMAccountName or uid or userPrincipalName").fill("uid")
    settings.get_by_placeholder("Example: ou=users,dc=foo,dc=example").fill(PEOPLE_DN)
    set_switch(settings, "TLS", turn_on=False)
    if group_mapping:
        ldap = settings.locator("section").filter(
            has=page.get_by_role("switch", name="LDAP", exact=True)
        )
        set_switch(ldap, "Group Mapping", turn_on=True)
        set_switch(ldap, "Auto-Create Groups", turn_on=True)
    save(page, settings)


def ldap_sign_in(page: Page, username: str, password: str) -> None:
    page.goto("/auth")
    page.get_by_label("Username").fill(username)
    page.get_by_label("Password", exact=True).fill(password)
    page.get_by_role("button", name="Authenticate").click()


def new_uid() -> str:
    return f"keeper-{uuid.uuid4().hex[:8]}"


def test_a_directory_person_signs_in_once_the_admin_connects_ldap(page_for, admin, page, directory):
    connect_directory(page_for(admin), directory)
    uid = new_uid()
    directory.add_person(uid, PASSWORD, cn="Ada Keeper")

    ldap_sign_in(page, uid, PASSWORD)

    expect(chat_input(page)).to_be_visible()
    with admin.client() as client:
        found = client.get("/api/v1/users/", params={"query": f"{uid}@example.org"}).json()
    assert [user["email"] for user in found["users"]] == [f"{uid}@example.org"]


def test_a_wrong_directory_password_is_refused(admin, page, directory):
    with admin.client() as client:
        save_ldap_settings(client, directory)
    uid = new_uid()
    directory.add_person(uid, PASSWORD)

    ldap_sign_in(page, uid, "not-the-password")

    expect(page.get_by_role("button", name="Authenticate")).to_be_visible()
    expect(chat_input(page)).to_be_hidden()


def test_group_mapping_puts_the_person_in_their_directory_group(page_for, admin, page, directory):
    connect_directory(page_for(admin), directory, group_mapping=True)
    uid, group = new_uid(), f"Harbour crew {uuid.uuid4().hex[:6]}"
    directory.add_person(uid, PASSWORD, groups=(group,))

    ldap_sign_in(page, uid, PASSWORD)
    expect(chat_input(page)).to_be_visible()

    with admin.client() as client:
        [person] = client.get("/api/v1/users/", params={"query": f"{uid}@example.org"}).json()[
            "users"
        ]
        [created] = [
            entry for entry in client.get("/api/v1/groups/").json() if entry["name"] == group
        ]
        members = client.get(f"/api/v1/groups/id/{created['id']}/export").json()["user_ids"]
        client.delete(f"/api/v1/groups/id/{created['id']}/delete")
    assert person["id"] in members


def test_with_ldap_switched_off_the_auth_page_asks_for_an_email(page_for, admin, page, directory):
    with admin.client() as client:
        save_ldap_settings(client, directory)
    admin_page = page_for(admin)
    settings = open_authentication(admin_page)
    expect(settings.get_by_placeholder("Enter server host")).to_have_value(directory.host)
    set_switch(settings, "LDAP", turn_on=False)
    save(admin_page, settings)

    page.goto("/auth")

    expect(page.get_by_label("Email")).to_be_visible()
    expect(page.get_by_label("Username")).to_have_count(0)
