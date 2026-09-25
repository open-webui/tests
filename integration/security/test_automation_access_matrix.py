"""Journey: an automation is its owner's alone, and it only files into places the owner may write.

Automations have no sharing: every per-item route looks the automation up for its owner and
answers 404 to anyone else, the admin included. A stranger's or the admin's read, update, toggle,
run, delete or run history of someone else's automation is refused, and the owner still sees the
same name, schedule, state and runs afterwards. Saving an automation into another user's folder
answers 404 and saving it to post into a channel the user may only read, or not even see,
answers 403; the automation is then neither created nor moved.

Discriminates: in a backend copy, reducing `check_automation_access` to a None check turns the
stranger and admin rows red (200, and the update, toggle and delete change the owner's
automation), and dropping the write check from `check_automation_channel_access` turns the
read-only channel rows red (200).
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from harness.actors import Actor
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

OK, NOT_FOUND, REFUSED = 200, 404, 403
DAILY = "RRULE:FREQ=DAILY"


def _form(**changes) -> dict:
    return {
        "name": f"digest {uuid.uuid4().hex[:8]}",
        "is_active": False,
        "data": {"prompt": "summarise the day", "model_id": MOCK_MODEL_ID, "rrule": DAILY},
        **changes,
    }


def _create(actor: Actor, **changes) -> httpx.Response:
    with actor.client() as client:
        return client.post("/api/v1/automations/create", json=_form(**changes))


def _owner_view(owner: Actor, automation_id: str) -> dict | None:
    with owner.client() as client:
        automation = client.get(f"/api/v1/automations/{automation_id}")
        if automation.status_code != 200:
            return None
        runs = client.get(f"/api/v1/automations/{automation_id}/runs")
    assert runs.status_code == 200, runs.text
    body = automation.json()
    return {
        "name": body["name"],
        "is_active": body["is_active"],
        "folder_id": body["folder_id"],
        "data": body["data"],
        "runs": len(runs.json()),
    }


def _automation_ids(actor: Actor) -> list[str]:
    with actor.client() as client:
        listed = client.get("/api/v1/automations/list")
    assert listed.status_code == 200, listed.text
    return [item["id"] for item in listed.json()["items"]]


@pytest.fixture
def automations_allowed(admin, preserve):
    preserve("permissions")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["features"]["automations"] = True
        granted = client.post("/api/v1/users/default/permissions", json=permissions)
    assert granted.status_code == 200, granted.text


ROUTES = [
    ("GET", "/api/v1/automations/{id}", None),
    ("POST", "/api/v1/automations/{id}/update", _form(name="taken over", is_active=True)),
    ("POST", "/api/v1/automations/{id}/toggle", None),
    ("POST", "/api/v1/automations/{id}/run", None),
    ("DELETE", "/api/v1/automations/{id}/delete", None),
    ("GET", "/api/v1/automations/{id}/runs", None),
]


@pytest.mark.parametrize(
    "method, path, body",
    ROUTES,
    ids=[f"{row[0]} {row[1].split('{id}')[1] or '/'}" for row in ROUTES],
)
def test_only_the_owner_reaches_an_automation(
    method, path, body, automations_allowed, admin, make_user, upstream
):
    owner, stranger = make_user(), make_user()

    answered, unchanged = {}, {}
    for role, actor in (("owner", owner), ("stranger", stranger), ("admin", admin)):
        created = _create(owner)
        assert created.status_code == 200, created.text
        automation_id = created.json()["id"]
        before = _owner_view(owner, automation_id)
        with actor.client() as client:
            answered[role] = client.request(
                method, path.format(id=automation_id), json=body
            ).status_code
        unchanged[role] = _owner_view(owner, automation_id) == before

    assert answered == {"owner": OK, "stranger": NOT_FOUND, "admin": NOT_FOUND}
    assert unchanged["stranger"] and unchanged["admin"], "a refused request changed the automation"


def _folder_of(actor: Actor) -> str:
    with actor.client() as client:
        created = client.post("/api/v1/folders/", json={"name": f"folder {uuid.uuid4().hex[:8]}"})
    assert created.status_code == 200, created.text
    return created.json()["id"]


def test_an_automation_cannot_be_created_in_another_users_folder(automations_allowed, make_user):
    account, other = make_user(), make_user()

    refused = _create(account, folder_id=_folder_of(other))

    assert refused.status_code == NOT_FOUND, refused.text
    assert _automation_ids(account) == []


def test_an_automation_cannot_be_moved_into_another_users_folder(automations_allowed, make_user):
    account, other = make_user(), make_user()
    automation_id = _create(account).json()["id"]
    before = _owner_view(account, automation_id)

    with account.client() as client:
        refused = client.post(
            f"/api/v1/automations/{automation_id}/update",
            json=_form(folder_id=_folder_of(other)),
        )

    assert refused.status_code == NOT_FOUND, refused.text
    assert _owner_view(account, automation_id) == before


def test_an_automation_is_filed_into_the_users_own_folder(automations_allowed, make_user):
    account = make_user()
    folder_id = _folder_of(account)

    created = _create(account, folder_id=folder_id)

    assert created.status_code == OK, created.text
    assert created.json()["folder_id"] == folder_id


@pytest.fixture
def channels_on(admin, preserve):
    preserve("admin_config")
    with admin.client() as client:
        config = client.get("/api/v1/auths/admin/config").json()
        client.post(
            "/api/v1/auths/admin/config", json={**config, "ENABLE_CHANNELS": True}
        ).raise_for_status()


def _channel_for(admin: Actor, account: Actor, permissions: tuple[str, ...]) -> str:
    grants = [
        {"principal_type": "user", "principal_id": account.id, "permission": permission}
        for permission in permissions
    ]
    with admin.client() as client:
        created = client.post(
            "/api/v1/channels/create",
            json={"name": f"room-{uuid.uuid4().hex[:8]}", "access_grants": grants},
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _posting_to(channel_id: str) -> dict:
    form = _form()
    form["data"]["target"] = {"type": "channel", "channel_id": channel_id}
    return form


@pytest.mark.parametrize("permissions", [(), ("read",)], ids=["no grant", "read grant"])
def test_an_automation_cannot_post_into_a_channel_the_user_cannot_write(
    permissions, automations_allowed, channels_on, admin, make_user
):
    account = make_user()
    channel_id = _channel_for(admin, account, permissions)
    automation_id = _create(account).json()["id"]
    before = _owner_view(account, automation_id)

    with account.client() as client:
        created = client.post("/api/v1/automations/create", json=_posting_to(channel_id))
        updated = client.post(
            f"/api/v1/automations/{automation_id}/update", json=_posting_to(channel_id)
        )

    assert (created.status_code, updated.status_code) == (REFUSED, REFUSED)
    assert _automation_ids(account) == [automation_id]
    assert _owner_view(account, automation_id) == before


def test_an_automation_posts_into_a_channel_the_user_may_write(
    automations_allowed, channels_on, admin, make_user
):
    account = make_user()
    channel_id = _channel_for(admin, account, ("read", "write"))

    with account.client() as client:
        created = client.post("/api/v1/automations/create", json=_posting_to(channel_id))

    assert created.status_code == OK, created.text
    assert created.json()["data"]["target"]["channel_id"] == channel_id
