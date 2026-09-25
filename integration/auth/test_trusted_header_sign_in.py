"""Journey: signing in behind an authenticating reverse proxy that names the person in headers.

With `WEBUI_AUTH_TRUSTED_EMAIL_HEADER` set, the sign-in reads the email from that header instead
of the form and creates the account on the first request, named from the name header (URL
decoded) or else after the email. The groups header replaces the account's memberships with the
existing groups it lists, never creating one, and the role header sets the role when it names a
valid one. Without the header the sign-in is refused even with a correct password, a session
presented next to another person's header is refused, and the password change is off because the
proxy owns the credentials.

Discriminates: in a backend copy, dropping the trusted-groups sync from the sign-in turns the
groups test red, skipping the trusted-role update turns the role test red, and dropping the
trusted-header check from the password change turns the password test red.
"""

from __future__ import annotations

import secrets
import urllib.parse

import httpx
import pytest

from harness.actors import create_user
from harness.instance import ADMIN_EMAIL, ADMIN_PASSWORD
from harness.oidc_provider import group_member_ids, group_named

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source, pytest.mark.slow]

EMAIL_HEADER = "X-Forwarded-Email"
NAME_HEADER = "X-Forwarded-Name"
GROUPS_HEADER = "X-Forwarded-Groups"
ROLE_HEADER = "X-Forwarded-Role"
TRUSTED_HEADERS = {
    "WEBUI_AUTH_TRUSTED_EMAIL_HEADER": EMAIL_HEADER,
    "WEBUI_AUTH_TRUSTED_NAME_HEADER": NAME_HEADER,
    "WEBUI_AUTH_TRUSTED_GROUPS_HEADER": GROUPS_HEADER,
    "WEBUI_AUTH_TRUSTED_ROLE_HEADER": ROLE_HEADER,
}


@pytest.fixture
def proxied(instance_with):
    return instance_with(TRUSTED_HEADERS)


def new_email() -> str:
    return f"proxied-{secrets.token_hex(4)}@example.com"


def sign_in(instance, headers: dict[str, str], **form) -> httpx.Response:
    """The web client's sign-in, which sends an empty form when a proxy signs people in."""
    return httpx.post(
        f"{instance.base_url}/api/v1/auths/signin",
        json={"email": "", "password": "", **form},
        headers=headers,
        timeout=60.0,
    )


def signed_in(instance, headers: dict[str, str]) -> dict:
    answer = sign_in(instance, headers)
    assert answer.status_code == 200, answer.text
    return answer.json()


def test_the_first_request_creates_the_account_from_the_headers(proxied):
    email = new_email()
    name = "Jörg Østergaard"

    session = signed_in(
        proxied, {EMAIL_HEADER: email.upper(), NAME_HEADER: urllib.parse.quote(name)}
    )

    assert (session["email"], session["name"], session["role"]) == (email, name, "pending")
    with proxied.client(session["token"]) as client:
        assert client.get("/api/v1/auths/").json()["id"] == session["id"]


def test_a_later_request_is_the_same_account_and_keeps_its_name(proxied):
    email = new_email()
    first = signed_in(proxied, {EMAIL_HEADER: email, NAME_HEADER: "First Name"})

    again = signed_in(proxied, {EMAIL_HEADER: email, NAME_HEADER: "Other Name"})

    assert again["id"] == first["id"]
    assert again["name"] == "First Name"


def test_without_a_name_header_the_account_is_named_after_the_email(proxied):
    email = new_email()
    assert signed_in(proxied, {EMAIL_HEADER: email})["name"] == email


def test_without_the_email_header_a_correct_password_is_refused(proxied):
    refused = sign_in(proxied, {}, email=ADMIN_EMAIL, password=ADMIN_PASSWORD)
    assert refused.status_code == 400, refused.text
    assert "token" not in refused.json()


def test_the_role_header_sets_a_valid_role_and_ignores_the_rest(proxied):
    email = new_email()

    promoted = signed_in(proxied, {EMAIL_HEADER: email, ROLE_HEADER: "ADMIN"})
    unknown = signed_in(proxied, {EMAIL_HEADER: email, ROLE_HEADER: "superuser"})
    absent = signed_in(proxied, {EMAIL_HEADER: email})
    demoted = signed_in(proxied, {EMAIL_HEADER: email, ROLE_HEADER: "user"})

    assert [promoted["role"], unknown["role"], absent["role"], demoted["role"]] == [
        "admin",
        "admin",
        "admin",
        "user",
    ]
    with proxied.client(demoted["token"]) as client:
        assert client.get("/api/v1/users/").status_code == 401, "the demoted account is still admin"


def test_the_groups_header_replaces_memberships_with_the_listed_groups(proxied):
    email = new_email()
    with (
        group_named(proxied, f"red-{secrets.token_hex(3)}") as red,
        group_named(proxied, f"blue-{secrets.token_hex(3)}") as blue,
    ):
        both = signed_in(
            proxied, {EMAIL_HEADER: email, GROUPS_HEADER: f"{red['name']} , {blue['name']}"}
        )
        in_both = (
            both["id"] in group_member_ids(proxied, red),
            both["id"] in group_member_ids(proxied, blue),
        )

        signed_in(proxied, {EMAIL_HEADER: email, GROUPS_HEADER: blue["name"]})
        after_blue_only = (
            both["id"] in group_member_ids(proxied, red),
            both["id"] in group_member_ids(proxied, blue),
        )

        signed_in(proxied, {EMAIL_HEADER: email, GROUPS_HEADER: ""})
        after_empty = both["id"] in group_member_ids(proxied, blue)

    assert in_both == (True, True)
    assert after_blue_only == (False, True), "a group left out of the header kept the account"
    assert after_empty is True, "an empty groups header changed the memberships"


def test_the_groups_header_never_creates_a_group(proxied):
    missing = f"nowhere-{secrets.token_hex(3)}"

    signed_in(proxied, {EMAIL_HEADER: new_email(), GROUPS_HEADER: missing})

    with proxied.client() as admin:
        names = [group["name"] for group in admin.get("/api/v1/groups/").json()]
    assert missing not in names


def test_a_session_next_to_another_persons_header_is_refused(proxied):
    first = signed_in(proxied, {EMAIL_HEADER: new_email(), ROLE_HEADER: "user"})
    other_email = new_email()

    with proxied.client(first["token"]) as client:
        own = client.get("/api/v1/auths/", headers={EMAIL_HEADER: first["email"]})
        crossed = client.get("/api/v1/auths/", headers={EMAIL_HEADER: other_email})

    assert own.status_code == 200, own.text
    assert crossed.status_code == 401, crossed.text


def test_the_password_change_is_off_behind_the_proxy(proxied):
    account = create_user(proxied)

    with account.client() as client:
        changed = client.post(
            "/api/v1/auths/update/password",
            json={"password": account.password, "new_password": "new-password-123"},
        )

    assert changed.status_code == 400, changed.text


def test_the_client_is_told_a_proxy_signs_people_in(proxied):
    features = httpx.get(f"{proxied.base_url}/api/config", timeout=60.0).json()["features"]
    assert features["auth_trusted_header"] is True
