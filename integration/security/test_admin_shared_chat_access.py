"""Regression: turning off blanket admin chat access revoked shares an admin was given.

open-webui 0.11.0, fix `a35b37adc` (#27127): reading a chat branched on the caller's role. An
admin got a chat only through `ENABLE_ADMIN_CHAT_ACCESS` and never reached the access grant and
shared folder checks every other account got, so with the setting off an admin who was explicitly
given a chat, directly or through a shared folder, was refused it. The fix lets every role fall
through to those checks; the setting still keeps admins out of chats nobody shared with them.
0.11.1 moved the resolution into `Chats.get_chat_by_id_for_user` unchanged.

Twin of unit/security/test_admin_shared_chat_access.py.

Discriminates: passes on bbfa876af, fails when the admin branch of `get_chat_by_id_for_user`
stops falling through (the a35b37adc~1 shape: the direct and folder shares answer 401 to the
admin); the other tests pass on both.
"""

from __future__ import annotations

import pytest

from harness.actors import admin_of, create_user

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

# shares one boot with test_tool_source_exposure
RESTRICTED_ADMIN_ENV = {"ENABLE_ADMIN_CHAT_ACCESS": "false", "BYPASS_ADMIN_ACCESS_CONTROL": "false"}


def _read_grant(principal_id: str) -> list[dict]:
    return [{"principal_type": "user", "principal_id": principal_id, "permission": "read"}]


def _create_chat(owner, folder_id: str | None = None) -> str:
    with owner.client() as client:
        created = client.post(
            "/api/v1/chats/new",
            json={"chat": {"title": "Quarterly numbers"}, "folder_id": folder_id},
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _share_chat(owner, chat_id: str, recipient) -> None:
    with owner.client() as client:
        shared = client.post(
            f"/api/v1/chats/shared/{chat_id}/access/update",
            json={"access_grants": _read_grant(recipient.id)},
        )
    assert shared.status_code == 200, shared.text


def _chat_in_shared_folder(owner, recipient) -> str:
    with owner.client() as client:
        folder = client.post("/api/v1/folders/", json={"name": "Finance"})
        assert folder.status_code == 200, folder.text
        folder_id = folder.json()["id"]
        shared = client.post(
            f"/api/v1/folders/{folder_id}/access/update",
            json={"access_grants": _read_grant(recipient.id)},
        )
    assert shared.status_code == 200, shared.text
    return _create_chat(owner, folder_id)


def _read_chat(reader, chat_id: str) -> int:
    with reader.client() as client:
        return client.get(f"/api/v1/chats/{chat_id}").status_code


@pytest.fixture
def restricted(instance_with):
    """An instance with ENABLE_ADMIN_CHAT_ACCESS off, its admin and a chat owner on it."""
    launched = instance_with(RESTRICTED_ADMIN_ENV)
    return admin_of(launched), create_user(launched), launched


def _recipient(restricted, role: str):
    admin, _, launched = restricted
    return admin if role == "admin" else create_user(launched)


# narrow: a share reaches an admin while blanket access is off


def test_admin_reads_a_chat_shared_with_them(restricted):
    admin, owner, _ = restricted
    chat_id = _create_chat(owner)
    _share_chat(owner, chat_id, admin)

    assert _read_chat(admin, chat_id) == 200, (
        "an admin explicitly given a chat was refused it because ENABLE_ADMIN_CHAT_ACCESS is off "
        "(#27127)"
    )


def test_admin_reads_a_chat_in_a_folder_shared_with_them(restricted):
    admin, owner, _ = restricted
    chat_id = _chat_in_shared_folder(owner, admin)

    assert _read_chat(admin, chat_id) == 200, (
        "an admin lost a chat reachable through a folder shared with them (#27127)"
    )


def test_admin_without_a_share_stays_refused(restricted):
    admin, owner, _ = restricted
    chat_id = _create_chat(owner)

    assert _read_chat(admin, chat_id) == 401, (
        "ENABLE_ADMIN_CHAT_ACCESS=false must still keep admins out of chats nobody shared"
    )


# broad: an explicit share beats the setting for every role and every read path


@pytest.mark.parametrize("role", ["admin", "user"])
@pytest.mark.parametrize("share", ["direct", "folder"])
def test_an_explicit_share_is_honoured_for_every_role(restricted, role, share):
    _, owner, _ = restricted
    recipient = _recipient(restricted, role)
    if share == "direct":
        chat_id = _create_chat(owner)
        _share_chat(owner, chat_id, recipient)
    else:
        chat_id = _chat_in_shared_folder(owner, recipient)

    assert _read_chat(recipient, chat_id) == 200, f"a {role} lost a {share} share (#27127)"


@pytest.mark.parametrize("role", ["admin", "user"])
def test_a_share_link_is_honoured_for_every_role(restricted, role):
    _, owner, _ = restricted
    recipient = _recipient(restricted, role)
    chat_id = _create_chat(owner)
    with owner.client() as client:
        link = client.post(f"/api/v1/chats/{chat_id}/share")
    assert link.status_code == 200, link.text
    share_id = link.json()["share_id"]
    _share_chat(owner, chat_id, recipient)

    with recipient.client() as client:
        opened = client.get(f"/api/v1/chats/share/{share_id}")

    assert opened.status_code == 200, f"a {role} could not open the link shared with them"
    assert opened.json()["title"] == "Quarterly numbers"


# nearby: the setting on, owners and strangers


def test_admin_reads_any_chat_when_blanket_access_is_on(admin, make_user):
    owner = make_user()

    assert _read_chat(admin, _create_chat(owner)) == 200


def test_user_without_a_share_is_refused(restricted):
    _, owner, launched = restricted

    assert _read_chat(create_user(launched), _create_chat(owner)) == 401


def test_owner_reads_their_own_chat(restricted):
    _, owner, _ = restricted

    assert _read_chat(owner, _create_chat(owner)) == 200


def test_missing_chat_is_refused_not_crashed(restricted):
    admin, _, _ = restricted

    assert _read_chat(admin, "no-such-chat") == 401
