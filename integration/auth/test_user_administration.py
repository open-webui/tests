"""Journey: what admins may and may not do to accounts in the admin panel.

The primary admin, the first account on the instance, cannot be demoted, edited by another admin
or deleted, and no admin can delete themselves, so an instance always keeps an admin. An email
change onto an address another account already holds is refused. Deleting an account ends its
sessions and takes its chats with it, which a later read shows.

Discriminates: fails with the other-admin half of the primary-admin guard removed from the
update route (another admin renames the primary admin).
"""

from __future__ import annotations

import uuid

import pytest

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]


def update(actor, user_id: str, **changes):
    with actor.client() as client:
        return client.post(f"/api/v1/users/{user_id}/update", json=changes)


def delete(actor, user_id: str):
    with actor.client() as client:
        return client.delete(f"/api/v1/users/{user_id}")


def stored_user(admin, user_id: str) -> dict:
    with admin.client() as client:
        fetched = client.get(f"/api/v1/users/{user_id}")
    assert fetched.status_code == 200, fetched.text
    return fetched.json()


@pytest.mark.parametrize(
    "changes", [{"role": "user"}, {"name": "Renamed by another admin"}], ids=["demote", "rename"]
)
def test_another_admin_cannot_change_the_primary_admin(admin, make_user, changes):
    other_admin = make_user(role="admin")

    refused = update(other_admin, admin.id, **changes)

    assert refused.status_code == 403, refused.text
    primary = stored_user(admin, admin.id)
    assert (primary["role"], primary["name"]) == ("admin", admin.name)


def test_the_primary_admin_cannot_demote_themselves(admin):
    refused = update(admin, admin.id, role="user")

    assert refused.status_code == 403, refused.text
    assert stored_user(admin, admin.id)["role"] == "admin"


def test_another_admin_cannot_delete_the_primary_admin(admin, make_user):
    other_admin = make_user(role="admin")

    assert delete(other_admin, admin.id).status_code == 403
    assert stored_user(admin, admin.id)["role"] == "admin"


def test_an_admin_cannot_delete_themselves(admin, make_user):
    other_admin = make_user(role="admin")

    assert delete(other_admin, other_admin.id).status_code == 403
    assert delete(admin, admin.id).status_code == 403
    assert stored_user(admin, other_admin.id)["role"] == "admin"


def test_another_admin_still_manages_regular_accounts(admin, make_user):
    other_admin, account = make_user(role="admin"), make_user()

    promoted = update(other_admin, account.id, role="admin", name="Promoted")

    assert promoted.status_code == 200, promoted.text
    stored = stored_user(admin, account.id)
    assert (stored["role"], stored["name"]) == ("admin", "Promoted")


@pytest.mark.parametrize("spelling", [str, str.upper], ids=["same-case", "upper-case"])
def test_an_email_change_onto_a_taken_address_is_refused(admin, make_user, spelling):
    account, holder = make_user(), make_user()

    refused = update(admin, account.id, email=spelling(holder.email))

    assert refused.status_code == 400, refused.text
    assert stored_user(admin, account.id)["email"] == account.email


def test_an_email_change_onto_a_free_address_lands(admin, make_user):
    account = make_user()
    new_email = f"moved-{uuid.uuid4().hex[:8]}@example.com"

    changed = update(admin, account.id, email=new_email.upper())

    assert changed.status_code == 200, changed.text
    assert stored_user(admin, account.id)["email"] == new_email


def test_deleting_an_account_ends_its_sessions_and_takes_its_chats(admin, make_user):
    account = make_user()
    with account.client() as client:
        created = client.post("/api/v1/chats/new", json={"chat": {"title": "to be deleted"}})
    assert created.status_code == 200, created.text
    chat_id = created.json()["id"]
    with admin.client() as client:
        assert client.get(f"/api/v1/chats/{chat_id}").status_code == 200

    assert delete(admin, account.id).json() is True

    with admin.client() as client:
        assert client.get(f"/api/v1/users/{account.id}").status_code == 400
        assert client.get(f"/api/v1/chats/{chat_id}").status_code == 401
        assert client.get(f"/api/v1/chats/list/user/{account.id}").json() == []
    with account.client() as client:
        assert client.get("/api/v1/auths/").status_code == 401
