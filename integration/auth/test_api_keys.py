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

Discriminates: fails with the endpoint-restriction check in `get_current_user_by_api_key`
removed (the key reaches a path outside the allowed list); matching an entry as a plain string
prefix fails the segment test, and dropping the custom header from the ASGI middleware fails the
`x-api-key` test.
"""

from __future__ import annotations

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
