"""LDAP sign-in against a real directory served on a local port.

The admin points LDAP at the directory, and a person in it signs in with their uid and password
the way the sign-in form does. Open WebUI binds as the service account, searches for the uid,
binds as the person it found, creates their account and hands out a session; with group
management on, the groups their `memberOf` names are created and joined.

Discriminates: fails with the person's bind in `ldap_auth` given the service account's password
(the directory refuses it, so the sign-in is a 400 and no account appears).
"""

from __future__ import annotations

import uuid
from typing import Iterator

import httpx
import pytest

from harness.ldap_server import (
    LDAP_CONFIG,
    PEOPLE_DN,
    SERVICE_DN,
    SERVICE_PASSWORD,
    Directory,
    save_ldap_settings,
    serve_directory,
)

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]


@pytest.fixture
def directory(admin, preserve) -> Iterator[Directory]:
    preserve(LDAP_CONFIG)
    with serve_directory() as served, admin.client() as client:
        save_ldap_settings(client, served)
        yield served


def sign_in(instance, uid: str, password: str) -> httpx.Response:
    """Press "Sign in" on the LDAP form, without a session."""
    return httpx.post(
        f"{instance.base_url}/api/v1/auths/ldap",
        json={"user": uid, "password": password},
        timeout=60.0,
    )


def new_uid() -> str:
    return f"person-{uuid.uuid4().hex[:8]}"


def test_a_directory_person_signs_in_and_gets_an_account(instance, directory):
    uid = new_uid()
    person = directory.add_person(uid, "directory-pass-1", cn="Directory Person")

    response = sign_in(instance, uid, "directory-pass-1")

    assert response.status_code == 200, response.text
    session = response.json()
    assert session["email"] == f"{uid}@example.org"
    assert session["name"] == "Directory Person"
    assert directory.binds[0].dn == SERVICE_DN and directory.binds[0].password == SERVICE_PASSWORD
    [search] = directory.searches
    assert search.base == PEOPLE_DN and search.filter == f"(&(uid={uid}))"
    [user_bind] = directory.user_binds()
    assert (user_bind.dn, user_bind.password, user_bind.succeeded) == (
        person.dn,
        "directory-pass-1",
        True,
    )
    with instance.client(session["token"]) as client:
        assert client.get("/api/v1/auths/").json()["email"] == f"{uid}@example.org"


def test_a_wrong_password_is_refused_by_the_directory(instance, directory):
    uid = new_uid()
    directory.add_person(uid, "directory-pass-1")

    response = sign_in(instance, uid, "not-the-password")

    assert response.status_code == 400
    [user_bind] = directory.user_binds()
    assert user_bind.succeeded is False


def test_someone_outside_the_directory_is_not_bound(instance, directory):
    response = sign_in(instance, new_uid(), "directory-pass-1")

    assert response.status_code == 400
    assert len(directory.searches) == 1
    assert directory.user_binds() == []


def test_member_of_groups_are_created_and_joined(instance, admin, directory):
    group_name = f"Sales, EMEA {uuid.uuid4().hex[:6]}"
    uid = new_uid()
    directory.add_person(uid, "directory-pass-1", groups=(group_name,))
    with admin.client() as client:
        save_ldap_settings(
            client, directory, enable_group_management=True, enable_group_creation=True
        )

    response = sign_in(instance, uid, "directory-pass-1")

    assert response.status_code == 200, response.text
    assert "memberOf" in directory.searches[0].attributes
    with admin.client() as client:
        groups = client.get("/api/v1/groups/").json()
        [group] = [group for group in groups if group["name"] == group_name]
        try:
            members = client.get(f"/api/v1/groups/id/{group['id']}/export").json()["user_ids"]
            assert response.json()["id"] in members
        finally:
            client.delete(f"/api/v1/groups/id/{group['id']}/delete")
