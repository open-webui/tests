"""Journey: personal API keys, from generation to the switches that take them away.

With API keys switched on and the `features.api_keys` permission granted, an account generates a
key, reads it back and signs requests with it as itself; generating again replaces the key and
deleting it ends it. Switching API keys off, or withdrawing the permission, refuses generation
and stops existing keys working, though an admin keeps the permission regardless. With endpoint
restrictions on, a key only reaches the paths on the allowed list, while a session token of the
same account goes everywhere as before. An entry covers the paths under it by whole segments
only, a key restricted to the routes a chat client needs still runs that client and nothing
else, and a key sent in the `x-api-key` header (for a proxy that keeps `Authorization` for
itself) acts as its owner under the same restrictions. A key acts on the account's status route:
since 24e30d1cb the session route `/api/v1/auths/` answers a signed-in session only.

Since 360d81dfe and fdaa240d4 (v0.12.5) an admin rotates or deletes an account's key from the
Edit User dialog. A rotation hands back a new working key once, uncached, and ends the old one; it
is refused while API keys are off or the account lacks the permission (admins exempt) and stores
nothing then. Deleting works even with API keys off. Only the first admin manages the first
admin's key, and an unknown account is not found.

Discriminates: fails with the endpoint-restriction check in `get_current_user_by_api_key`
removed (the key reaches a path outside the allowed list); matching an entry as a plain string
prefix fails the segment test, and dropping the custom header from the ASGI middleware fails the
`x-api-key` test. In a backend copy on dev 05e115366, dropping the `no-store` header, the
first-user guard, the switch and permission gate on rotation or the unknown-user check each turns
its admin test red, as does gating the admin delete on the API keys switch.
"""

from __future__ import annotations

import uuid

import httpx
import openai
import pytest

from harness import upstream as reply
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

ADMIN_CONFIG = "/api/v1/auths/admin/config"
DEFAULT_PERMISSIONS = "/api/v1/users/default/permissions"
OWN_ACCOUNT = "/api/v1/users/user/status"


def save_admin_config(admin, **changes) -> None:
    with admin.client() as client:
        current = client.get(ADMIN_CONFIG)
        current.raise_for_status()
        saved = client.post(ADMIN_CONFIG, json={**current.json(), **changes})
    assert saved.status_code == 200, saved.text


def allow_api_keys_for_users(admin, allowed: bool) -> None:
    with admin.client() as client:
        permissions = client.get(DEFAULT_PERMISSIONS).json()
        permissions["features"]["api_keys"] = allowed
        saved = client.post(DEFAULT_PERMISSIONS, json=permissions)
    assert saved.status_code == 200, saved.text


def generate_key(account) -> str:
    with account.client() as client:
        generated = client.post("/api/v1/auths/api_key")
    assert generated.status_code == 200, generated.text
    return generated.json()["api_key"]


def status_with_key(instance, api_key: str, path: str = OWN_ACCOUNT) -> int:
    with instance.client(api_key) as client:
        return client.get(path).status_code


@pytest.fixture
def api_keys_on(admin, preserve):
    preserve("admin_config", "permissions")
    save_admin_config(admin, ENABLE_API_KEYS=True, ENABLE_API_KEYS_ENDPOINT_RESTRICTIONS=False)
    allow_api_keys_for_users(admin, True)


def test_a_key_is_generated_read_back_and_acts_as_its_owner(instance, make_user, api_keys_on):
    account = make_user()

    api_key = generate_key(account)
    with account.client() as client:
        read_back = client.get("/api/v1/auths/api_key")
    with instance.client(api_key) as client:
        session = client.get(OWN_ACCOUNT)

    assert api_key.startswith("sk-")
    assert read_back.json()["api_key"] == api_key
    assert session.status_code == 200, session.text
    assert session.json()["id"] == account.id


def test_a_deleted_key_stops_working(instance, make_user, api_keys_on):
    account = make_user()
    api_key = generate_key(account)

    with account.client() as client:
        deleted = client.delete("/api/v1/auths/api_key")
        read_back = client.get("/api/v1/auths/api_key")

    assert deleted.status_code == 200 and deleted.json() is True
    assert read_back.status_code == 404
    assert status_with_key(instance, api_key) == 401


def test_generating_again_replaces_the_old_key(instance, make_user, api_keys_on):
    account = make_user()
    old_key = generate_key(account)

    new_key = generate_key(account)

    assert new_key != old_key
    assert status_with_key(instance, new_key) == 200
    assert status_with_key(instance, old_key) == 401


def test_api_keys_switched_off_refuse_generation_and_existing_keys(
    instance, admin, make_user, api_keys_on
):
    account = make_user()
    api_key = generate_key(account)

    save_admin_config(admin, ENABLE_API_KEYS=False)

    with account.client() as client:
        assert client.post("/api/v1/auths/api_key").status_code == 403
    with admin.client() as client:
        assert client.post("/api/v1/auths/api_key").status_code == 403
    assert status_with_key(instance, api_key) == 403


def test_without_the_permission_a_user_gets_no_key_and_an_admin_still_does(
    instance, admin, make_user, api_keys_on
):
    account, other_admin = make_user(), make_user(role="admin")
    api_key = generate_key(account)

    allow_api_keys_for_users(admin, False)

    with account.client() as client:
        assert client.post("/api/v1/auths/api_key").status_code == 403
    assert status_with_key(instance, api_key) == 403
    assert status_with_key(instance, generate_key(other_admin)) == 200


def test_a_restricted_key_reaches_only_the_allowed_paths(instance, admin, make_user, api_keys_on):
    account = make_user()
    api_key = generate_key(account)

    save_admin_config(
        admin,
        ENABLE_API_KEYS_ENDPOINT_RESTRICTIONS=True,
        API_KEYS_ALLOWED_ENDPOINTS="/api/v1/chats, /api/models",
    )

    assert status_with_key(instance, api_key, "/api/v1/chats/") == 200
    assert status_with_key(instance, api_key, "/api/models") == 200
    assert status_with_key(instance, api_key, OWN_ACCOUNT) == 403
    assert status_with_key(instance, api_key, "/api/v1/users/user/settings") == 403
    with account.client() as client:
        assert client.get(OWN_ACCOUNT).status_code == 200
        assert client.get("/api/v1/users/user/settings").status_code == 200


def test_an_allowed_entry_covers_whole_path_segments_only(instance, admin, make_user, api_keys_on):
    api_key = generate_key(make_user())

    save_admin_config(
        admin,
        ENABLE_API_KEYS_ENDPOINT_RESTRICTIONS=True,
        API_KEYS_ALLOWED_ENDPOINTS="/api/v1/chat, /api/v1/users/user",
    )

    assert status_with_key(instance, api_key, OWN_ACCOUNT) == 200
    assert status_with_key(instance, api_key, "/api/v1/chats/") == 403


def test_a_key_restricted_to_the_chat_routes_runs_a_client_and_nothing_more(
    instance, admin, make_user, api_keys_on, upstream
):
    api_key = generate_key(make_user())
    upstream.queue(reply.text("still answering"))
    save_admin_config(
        admin,
        ENABLE_API_KEYS_ENDPOINT_RESTRICTIONS=True,
        API_KEYS_ALLOWED_ENDPOINTS="/api/models, /api/chat/completions",
    )
    client = openai.OpenAI(api_key=api_key, base_url=f"{instance.base_url}/api", max_retries=0)

    listed = [model.id for model in client.models.list().data]
    completion = client.chat.completions.create(
        model=MOCK_MODEL_ID, messages=[{"role": "user", "content": "are you there?"}]
    )
    with instance.client(api_key) as raw:
        upload = raw.post("/api/v1/files/", files={"file": ("a.txt", b"text", "text/plain")})

    assert MOCK_MODEL_ID in listed
    assert completion.choices[0].message.content == "still answering"
    assert upload.status_code == 403, upload.text


def test_a_key_in_the_x_api_key_header_acts_as_its_owner_under_the_restrictions(
    instance, admin, make_user, api_keys_on
):
    account = make_user()
    api_key = generate_key(account)

    with httpx.Client(base_url=instance.base_url, headers={"x-api-key": api_key}) as client:
        own_account = client.get(OWN_ACCOUNT)
        save_admin_config(
            admin,
            ENABLE_API_KEYS_ENDPOINT_RESTRICTIONS=True,
            API_KEYS_ALLOWED_ENDPOINTS="/api/models",
        )
        allowed = client.get("/api/models")
        outside = client.get(OWN_ACCOUNT)

    assert own_account.status_code == 200, own_account.text
    assert own_account.json()["id"] == account.id
    assert allowed.status_code == 200, allowed.text
    assert outside.status_code == 403, outside.text


def rotate_for(admin, account_id: str) -> httpx.Response:
    with admin.client() as client:
        return client.post(f"/api/v1/users/{account_id}/api_key/rotate")


def delete_for(admin, account_id: str) -> httpx.Response:
    with admin.client() as client:
        return client.delete(f"/api/v1/users/{account_id}/api_key")


def own_key_status(account) -> int:
    with account.client() as client:
        return client.get("/api/v1/auths/api_key").status_code


def test_an_admin_rotates_an_accounts_key_and_only_the_new_one_works(
    instance, admin, make_user, api_keys_on
):
    account = make_user()
    old_key = generate_key(account)

    rotated = rotate_for(admin, account.id)

    assert rotated.status_code == 200, rotated.text
    assert rotated.headers.get("cache-control") == "no-store", rotated.headers
    new_key = rotated.json()["api_key"]
    assert new_key.startswith("sk-") and new_key != old_key
    with instance.client(new_key) as client:
        session = client.get(OWN_ACCOUNT)
    assert session.status_code == 200, session.text
    assert session.json()["id"] == account.id
    assert status_with_key(instance, old_key) == 401
    with account.client() as client:
        assert client.get("/api/v1/auths/api_key").json()["api_key"] == new_key
    with admin.client() as client:
        listed = client.get(f"/api/v1/users/{account.id}")
    assert listed.status_code == 200, listed.text
    assert new_key not in listed.text, "the account's record shows the admin its key"


def test_an_admin_deletes_an_accounts_key_even_with_api_keys_switched_off(
    instance, admin, make_user, api_keys_on
):
    account = make_user()
    api_key = generate_key(account)
    save_admin_config(admin, ENABLE_API_KEYS=False)

    deleted = delete_for(admin, account.id)

    assert deleted.status_code == 200 and deleted.json() is True, deleted.text
    # with keys back on, a key that survived would answer 200 again
    save_admin_config(admin, ENABLE_API_KEYS=True)
    assert own_key_status(account) == 404
    assert status_with_key(instance, api_key) == 401


def test_an_admin_cannot_rotate_a_key_the_account_could_not_generate(admin, make_user, api_keys_on):
    account, other_admin = make_user(), make_user(role="admin")

    allow_api_keys_for_users(admin, False)
    refused_without_permission = rotate_for(admin, account.id)
    allowed_for_an_admin = rotate_for(admin, other_admin.id)
    save_admin_config(admin, ENABLE_API_KEYS=False)
    refused_while_off = rotate_for(admin, other_admin.id)
    # the account reads its own key only with keys on and the permission granted
    save_admin_config(admin, ENABLE_API_KEYS=True)
    allow_api_keys_for_users(admin, True)

    assert refused_without_permission.status_code == 403, refused_without_permission.text
    assert "api_key" not in refused_without_permission.json()
    assert own_key_status(account) == 404, "a refused rotation still stored a key"
    assert allowed_for_an_admin.status_code == 200, allowed_for_an_admin.text
    assert refused_while_off.status_code == 403, refused_while_off.text
    with other_admin.client() as client:
        stored = client.get("/api/v1/auths/api_key").json()["api_key"]
    assert stored == allowed_for_an_admin.json()["api_key"], "a refused rotation replaced the key"


def test_only_the_first_admin_manages_the_first_admins_key(instance, admin, make_user, api_keys_on):
    other_admin = make_user(role="admin")

    own_rotation = rotate_for(admin, admin.id)
    rotated_by_another = rotate_for(other_admin, admin.id)
    deleted_by_another = delete_for(other_admin, admin.id)
    first_admin_key_status = status_with_key(instance, own_rotation.json().get("api_key", ""))
    own_deletion = delete_for(admin, admin.id)

    assert own_rotation.status_code == 200, own_rotation.text
    assert rotated_by_another.status_code == 403, rotated_by_another.text
    assert "api_key" not in rotated_by_another.json()
    assert deleted_by_another.status_code == 403, deleted_by_another.text
    assert first_admin_key_status == 200, "another admin replaced or revoked the first admin's key"
    assert own_deletion.status_code == 200 and own_deletion.json() is True, own_deletion.text
    assert own_key_status(admin) == 404


def test_rotating_or_deleting_the_key_of_an_unknown_account_is_not_found(admin, api_keys_on):
    unknown = f"no-such-user-{uuid.uuid4().hex}"

    assert rotate_for(admin, unknown).status_code == 404
    assert delete_for(admin, unknown).status_code == 404
