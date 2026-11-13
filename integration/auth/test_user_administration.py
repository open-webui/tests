"""Journey: what admins may and may not do to accounts in the admin panel.

The primary admin, the first account on the instance, cannot demote themselves, and no admin can
delete themselves, so an instance always keeps an admin. An email
change onto an address another account already holds is refused. Deleting an account ends its
sessions and takes its chats with it, which a later read shows. Since #34223 (44d2dd3d5) it also
takes the account off the access list of everything shared with it, where it used to stay as a row
showing only its id, while the others on the list stay.

Discriminates: each refusal is asserted as a 403 with the stored account unchanged, so a route
that lets the primary admin demote themselves or an admin delete themselves turns its test red.
The access list test passes on dev 485103cde and fails with 44d2dd3d5 reverted (the deleted
account's grants are still listed).
"""

from __future__ import annotations

import uuid

import pytest

from harness.upstream import MOCK_MODEL_ID

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


def test_the_primary_admin_cannot_demote_themselves(admin):
    refused = update(admin, admin.id, role="user")

    assert refused.status_code == 403, refused.text
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


def _user_grants(owner, kind: str, resource_id: str) -> set[tuple[str, str]]:
    path = (
        f"/api/v1/notes/{resource_id}"
        if kind == "note"
        else f"/api/v1/models/model?id={resource_id}"
    )
    with owner.client() as client:
        fetched = client.get(path)
    assert fetched.status_code == 200, fetched.text
    return {
        (grant["principal_id"], grant["permission"])
        for grant in fetched.json().get("access_grants") or []
        if grant["principal_type"] == "user"
    }


def _shared_note(owner, grants: list[dict]) -> str:
    with owner.client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={"title": f"rota {uuid.uuid4().hex[:8]}", "data": {"content": {"md": ""}}},
        )
        assert created.status_code == 200, created.text
        note_id = created.json()["id"]
        shared = client.post(
            f"/api/v1/notes/{note_id}/access/update", json={"access_grants": grants}
        )
    assert shared.status_code == 200, shared.text
    return note_id


def _shared_model(owner, grants: list[dict]) -> str:
    model_id = f"rota-{uuid.uuid4().hex[:8]}"
    form = {
        "id": model_id,
        "name": model_id,
        "base_model_id": MOCK_MODEL_ID,
        "meta": {},
        "params": {},
        "access_grants": grants,
    }
    with owner.client() as client:
        created = client.post("/api/v1/models/create", json=form)
    assert created.status_code == 200, created.text
    return model_id


@pytest.mark.parametrize("kind", ["note", "model"])
def test_deleting_an_account_takes_it_off_the_access_lists_it_was_on(admin, make_user, kind):
    """#34223: the deleted account stayed on each access list as a row showing only its id."""
    owner, leaving, staying = make_user(role="admin"), make_user(), make_user()
    grants = [
        {"principal_type": "user", "principal_id": account.id, "permission": permission}
        for account in (leaving, staying)
        for permission in ("read", "write")
    ]
    share = _shared_note if kind == "note" else _shared_model
    resource_id = share(owner, grants)
    before = _user_grants(owner, kind, resource_id)
    assert {grant_holder for grant_holder, _ in before} == {leaving.id, staying.id}, before

    assert delete(admin, leaving.id).json() is True

    after = _user_grants(owner, kind, resource_id)
    assert after == {(staying.id, "read"), (staying.id, "write")}, (
        f"the {kind}'s access list still names the deleted account: {after}"
    )
