"""Journey: a memory is its owner's alone, and Memories switched off closes every memory route.

Memories are never shared: the update, delete and bulk routes look a memory up for the caller,
so another user, the admin included, sending the owner's memory id changes nothing. The single
update and the bulk edit answer 404 and the delete answers `false`; the owner's memories read the
same afterwards. With Memories switched off every route answers 404 to everyone, and without
`features.memories` every route answers 403 to a user while an admin still gets through. The
chat-side gates are pinned by integration/memory/test_memory_switch_gating.py.

Discriminates: in a backend copy, dropping the `user_id` filter from
`Memories.delete_memory_by_id_and_user_id` turns the delete rows red (the stranger's delete
answers `true` and the memory is gone), and dropping the switch check from
`check_memories_permission` turns the switched-off rows red (200).
"""

from __future__ import annotations

from typing import Callable

import pytest

from harness.actors import Actor

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

ADMIN_CONFIG = "/api/v1/auths/admin/config"
DEFAULT_PERMISSIONS = "/api/v1/users/default/permissions"
STORED = "Prefers answers in metric units."


def _add_memory(actor: Actor, content: str = STORED) -> str:
    with actor.client() as client:
        added = client.post("/api/v1/memories/add", json={"content": content, "type": "user"})
    assert added.status_code == 200, added.text
    return added.json()["id"]


def _memories(actor: Actor) -> list[tuple[str, str]]:
    with actor.client() as client:
        listed = client.get("/api/v1/memories/")
    assert listed.status_code == 200, listed.text
    return sorted((memory["id"], memory["content"]) for memory in listed.json())


# method, path, body for a memory id; a foreign caller's answer
FOREIGN_ROUTES = [
    ("POST", "/api/v1/memories/{id}/update", lambda memory_id: {"content": "rewritten"}, 404),
    ("DELETE", "/api/v1/memories/{id}", lambda memory_id: None, 200),
    (
        "POST",
        "/api/v1/memories/update",
        lambda memory_id: {"operations": [{"action": "replace", "id": memory_id, "content": "x"}]},
        404,
    ),
    (
        "POST",
        "/api/v1/memories/update",
        lambda memory_id: {"operations": [{"action": "remove", "id": memory_id}]},
        404,
    ),
]


@pytest.mark.parametrize("caller", ["stranger", "admin"])
@pytest.mark.parametrize(
    "method, path, body, answer",
    FOREIGN_ROUTES,
    ids=["update", "delete", "bulk replace", "bulk remove"],
)
def test_another_users_memory_id_changes_nothing(method, path, body, answer, caller, make_user):
    owner = make_user()
    other = make_user(role="admin" if caller == "admin" else "user")
    memory_id = _add_memory(owner)
    before = _memories(owner)

    with other.client() as client:
        response = client.request(method, path.format(id=memory_id), json=body(memory_id))

    assert response.status_code == answer, response.text
    if answer == 200:
        assert response.json() is False
    assert _memories(owner) == before
    assert _memories(other) == []


@pytest.mark.parametrize(
    "method, path, body, answer",
    FOREIGN_ROUTES,
    ids=["update", "delete", "bulk replace", "bulk remove"],
)
def test_the_owner_changes_their_own_memory(method, path, body, answer, make_user):
    owner = make_user()
    memory_id = _add_memory(owner)

    with owner.client() as client:
        response = client.request(method, path.format(id=memory_id), json=body(memory_id))

    assert response.status_code == 200, response.text
    assert _memories(owner) != [(memory_id, STORED)]


# every memory route, with a body built from the caller's own memory id
GATED_ROUTES: list[tuple[str, str, Callable[[str], dict | None]]] = [
    ("GET", "/api/v1/memories/", lambda memory_id: None),
    ("POST", "/api/v1/memories/add", lambda memory_id: {"content": "added while barred"}),
    ("POST", "/api/v1/memories/{id}/update", lambda memory_id: {"content": "changed"}),
    ("DELETE", "/api/v1/memories/{id}", lambda memory_id: None),
    (
        "POST",
        "/api/v1/memories/update",
        lambda memory_id: {"operations": [{"action": "remove", "id": memory_id}]},
    ),
    ("POST", "/api/v1/memories/search", lambda memory_id: {"query": "units"}),
    ("POST", "/api/v1/memories/query", lambda memory_id: {"content": "units"}),
]


def _switch(admin: Actor, switch: str, enabled: bool) -> None:
    with admin.client() as client:
        if switch == "feature":
            config = client.get(ADMIN_CONFIG).json()
            saved = client.post(ADMIN_CONFIG, json={**config, "ENABLE_MEMORIES": enabled})
        else:
            permissions = client.get(DEFAULT_PERMISSIONS).json()
            permissions["features"]["memories"] = enabled
            saved = client.post(DEFAULT_PERMISSIONS, json=permissions)
    assert saved.status_code == 200, saved.text


@pytest.mark.parametrize("switch, refusal", [("feature", 404), ("permission", 403)])
@pytest.mark.parametrize(
    "method, path, body", GATED_ROUTES, ids=[f"{row[0]} {row[1]}" for row in GATED_ROUTES]
)
def test_switched_off_memories_refuse_every_route(
    method, path, body, switch, refusal, admin, preserve, make_user
):
    account, other_admin = make_user(), make_user(role="admin")
    own_memory, admin_memory = _add_memory(account), _add_memory(other_admin)
    before = _memories(account)
    preserve("admin_config", "permissions")
    _switch(admin, switch, enabled=False)

    answered = {}
    for role, actor, memory_id in (
        ("user", account, own_memory),
        ("admin", other_admin, admin_memory),
    ):
        with actor.client() as client:
            response = client.request(method, path.format(id=memory_id), json=body(memory_id))
        answered[role] = response.status_code

    admin_answer = refusal if switch == "feature" else 200
    assert answered == {"user": refusal, "admin": admin_answer}
    _switch(admin, switch, enabled=True)
    assert _memories(account) == before
