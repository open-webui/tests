"""LDAP sign-in against a real directory served on a local port.

The admin points LDAP at the directory, and a person in it signs in with their uid and password
the way the sign-in form does. Open WebUI binds as the service account, searches for the uid,
binds as the person it found, creates their account and hands out a session; with group
management on, the groups their `memberOf` names are created and joined. ldap3 does the talking:
its `escape_filter_chars` turns the typed uid into a literal in the search filter, so a uid
holding `*`, `(` or `)` finds exactly that person and a bare `*` finds no one, and with LDAPS on
its `Tls` checks the directory's certificate against the one the admin names, or skips the check
when told to. A person without a mail attribute is refused unless the admin set a default email
domain (v0.12.4); then they sign in as uid@domain, into the account that already holds that
address, while a person with an address keeps it. `LDAP_DEFAULT_EMAIL_DOMAIN` sets the default.

Discriminates: fails with the person's bind in `ldap_auth` given the service account's password
(the directory refuses it, so the sign-in is a 400 and no account appears); without
`escape_filter_chars` the filter-character uids are refused and the bare star finds the person;
building `Tls` with `CERT_NONE` whatever the setting lets the untrusted directory in, and
`use_ssl=False` fails both LDAPS sign-ins. Skipping the default domain fallback fails the two
uid@domain sign-ins; dropping `default_email_domain` from the saved LDAP keys fails those and the
env read-back; ignoring the env var fails the env read-back; applying the domain even when a mail
attribute exists fails the mail-wins test; removing the no-address refusal creates an account and
fails the refusal test.
"""

from __future__ import annotations

import uuid
from typing import Iterator

import httpx
import pytest

from harness.actors import admin_of
from harness.ldap_server import (
    LDAP_CONFIG,
    LDAP_SERVER_CONFIG,
    PEOPLE_DN,
    SERVICE_DN,
    SERVICE_PASSWORD,
    Directory,
    save_ldap_settings,
    serve_directory,
)

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

DEFAULT_DOMAIN = "corp.example"


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


@pytest.mark.parametrize("shape", ["o(brien)", "star*gazer", "(*)"])
def test_a_uid_holding_filter_characters_signs_in_as_that_person(instance, directory, shape):
    uid = f"{shape}-{uuid.uuid4().hex[:8]}"
    email = f"filter-{uuid.uuid4().hex[:8]}@example.org"
    directory.add_person(uid, "directory-pass-1", mail=email)

    response = sign_in(instance, uid, "directory-pass-1")

    assert response.status_code == 200, response.text
    assert response.json()["email"] == email
    [search] = directory.searches
    assert len(search.found) == 1


def test_a_bare_star_finds_no_one(instance, directory):
    directory.add_person(new_uid(), "directory-pass-1")

    response = sign_in(instance, "*", "directory-pass-1")

    assert response.status_code == 400
    [search] = directory.searches
    assert search.found == [], "the star was sent as a wildcard and matched the directory"
    assert directory.user_binds() == []


def test_without_a_mail_attribute_the_person_signs_in_at_the_default_domain(
    instance, admin, directory
):
    uid = new_uid()
    directory.add_person(uid, "directory-pass-1", with_mail=False)
    with admin.client() as client:
        save_ldap_settings(client, directory, default_email_domain=DEFAULT_DOMAIN)
        saved = client.get(LDAP_SERVER_CONFIG[0]).json()
    assert saved["default_email_domain"] == DEFAULT_DOMAIN, f"read back: {saved}"

    response = sign_in(instance, uid, "directory-pass-1")

    assert response.status_code == 200, response.text
    assert response.json()["email"] == f"{uid}@{DEFAULT_DOMAIN}"
    with instance.client(response.json()["token"]) as client:
        assert client.get("/api/v1/auths/").json()["email"] == f"{uid}@{DEFAULT_DOMAIN}"


def test_without_the_default_domain_a_person_without_mail_is_refused(instance, admin, directory):
    uid = new_uid()
    directory.add_person(uid, "directory-pass-1", with_mail=False)

    response = sign_in(instance, uid, "directory-pass-1")

    assert response.status_code == 400, response.text
    assert "valid email" in response.json()["detail"]
    with admin.client() as client:
        listed = client.get("/api/v1/users/", params={"query": uid}).json()["users"]
    assert listed == [], f"an account was created for a person without an address: {listed}"


def test_the_default_domain_signs_into_the_existing_account_with_that_address(
    instance, admin, make_user, directory
):
    uid = new_uid()
    existing = make_user(email=f"{uid}@{DEFAULT_DOMAIN}")
    directory.add_person(uid, "directory-pass-1", with_mail=False)
    with admin.client() as client:
        save_ldap_settings(client, directory, default_email_domain=DEFAULT_DOMAIN)

    # the uid as typed in other case still finds the same account
    for typed in (uid, uid.upper()):
        response = sign_in(instance, typed, "directory-pass-1")
        assert response.status_code == 200, response.text
        assert response.json()["id"] == existing.id, (
            f"signing in as {typed!r} landed in {response.json()['email']}, not {existing.email}"
        )


def test_a_mail_attribute_wins_over_the_default_domain(instance, admin, make_user, directory):
    uid = new_uid()
    mail = f"own-{uuid.uuid4().hex[:8]}@example.org"
    other = make_user(email=f"{uid}@{DEFAULT_DOMAIN}")
    directory.add_person(uid, "directory-pass-1", mail=mail)
    with admin.client() as client:
        save_ldap_settings(client, directory, default_email_domain=DEFAULT_DOMAIN)

    response = sign_in(instance, uid, "directory-pass-1")

    assert response.status_code == 200, response.text
    assert response.json()["email"] == mail
    assert response.json()["id"] != other.id, (
        "a person with an address landed in the domain account"
    )


@pytest.mark.slow
def test_the_default_domain_env_var_sets_the_saved_setting(instance_with):
    configured = instance_with({"LDAP_DEFAULT_EMAIL_DOMAIN": "env.example"})
    with admin_of(configured).client() as client:
        settings = client.get(LDAP_SERVER_CONFIG[0]).json()
    assert settings["default_email_domain"] == "env.example", f"read back: {settings}"


@pytest.fixture
def ldaps_directory(admin, preserve) -> Iterator[Directory]:
    preserve(LDAP_CONFIG)
    with serve_directory(tls=True) as served:
        yield served


def test_a_person_signs_in_over_ldaps_with_the_certificate_checked(
    instance, admin, ldaps_directory
):
    uid = new_uid()
    ldaps_directory.add_person(uid, "directory-pass-1")
    with admin.client() as client:
        save_ldap_settings(
            client,
            ldaps_directory,
            use_tls=True,
            validate_cert=True,
            certificate_path=ldaps_directory.certificate_path,
        )

    response = sign_in(instance, uid, "directory-pass-1")

    assert response.status_code == 200, response.text
    # the service account's connection and the person's
    assert ldaps_directory.tls_handshakes == 2
    [user_bind] = ldaps_directory.user_binds()
    assert user_bind.succeeded


def test_a_directory_whose_certificate_is_not_trusted_is_refused(instance, admin, ldaps_directory):
    uid = new_uid()
    ldaps_directory.add_person(uid, "directory-pass-1")
    with admin.client() as client:
        save_ldap_settings(client, ldaps_directory, use_tls=True, validate_cert=True)

    response = sign_in(instance, uid, "directory-pass-1")

    assert response.status_code == 400
    assert ldaps_directory.binds == [], "the service account's password went to an untrusted host"


def test_with_the_check_off_an_ldaps_directory_is_used_as_it_is(instance, admin, ldaps_directory):
    uid = new_uid()
    ldaps_directory.add_person(uid, "directory-pass-1")
    with admin.client() as client:
        save_ldap_settings(client, ldaps_directory, use_tls=True, validate_cert=False)

    response = sign_in(instance, uid, "directory-pass-1")

    assert response.status_code == 200, response.text
    assert ldaps_directory.tls_handshakes == 2
