"""Owner, stranger, reader, writer and admin against one shared resource.

`Shareable` says how one kind of resource is created and shared over the API. `cast(...)` adds
the accounts, sharing with the reader and writer directly or through a group of their own, and
`statuses(...)` sends one route as each of them, each on a fresh resource shared the same way,
so a route that deletes or toggles never changes what the next account meets.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Callable

from harness.actors import Actor

ROLES = ("owner", "stranger", "reader", "writer", "admin")


@dataclass(frozen=True)
class Shareable:
    """A kind of resource: where it is created and where its access grants are replaced."""

    create_path: str
    create_body: Callable[[], dict]
    access_path: str  # with `{id}`
    # added to the access body, for kinds that also name the resource there
    access_fields: Callable[[str], dict] = field(default=lambda resource_id: {})


@dataclass
class Cast:
    kind: Shareable
    owner: Actor
    stranger: Actor
    reader: Actor
    writer: Actor
    admin: Actor
    grants: list[dict]

    def actor(self, role: str) -> Actor:
        return getattr(self, role)


def grant(principal_type: str, principal_id: str, permission: str) -> dict:
    return {
        "principal_type": principal_type,
        "principal_id": principal_id,
        "permission": permission,
    }


def make_group(admin: Actor, members: list[Actor]) -> str:
    """A group holding `members`, added the way the admin panel adds one."""
    with admin.client() as client:
        created = client.post(
            "/api/v1/groups/create",
            json={"name": f"group {uuid.uuid4().hex[:8]}", "description": "access matrix"},
        )
        assert created.status_code == 200, created.text
        group_id = created.json()["id"]
        added = client.post(
            f"/api/v1/groups/id/{group_id}/users/add",
            json={"user_ids": [member.id for member in members]},
        )
    assert added.status_code == 200, added.text
    return group_id


def cast(kind: Shareable, admin: Actor, make_user: Callable[..., Actor], via: str = "user") -> Cast:
    """Fresh accounts, with read for the reader and read plus write for the writer, as the UI sets.

    `via="group"` grants to a one-member group per account instead of to the account.
    """
    owner, stranger, reader, writer = make_user(), make_user(), make_user(), make_user()
    if via == "group":
        reader_principal = ("group", make_group(admin, [reader]))
        writer_principal = ("group", make_group(admin, [writer]))
    else:
        reader_principal, writer_principal = ("user", reader.id), ("user", writer.id)
    grants = [
        grant(*reader_principal, "read"),
        grant(*writer_principal, "read"),
        grant(*writer_principal, "write"),
    ]
    return Cast(kind, owner, stranger, reader, writer, admin, grants)


def create_shared(cast: Cast) -> str:
    """A new resource of the owner's, shared with the cast's reader and writer."""
    kind = cast.kind
    with cast.owner.client() as client:
        created = client.post(kind.create_path, json=kind.create_body())
        assert created.status_code == 200, created.text
        resource_id = created.json()["id"]
        shared = client.post(
            kind.access_path.format(id=resource_id),
            json={**kind.access_fields(resource_id), "access_grants": cast.grants},
        )
    assert shared.status_code == 200, shared.text
    return resource_id


def statuses(cast: Cast, method: str, path: str, body: dict | None = None) -> dict[str, int]:
    """`{role: status}` for one route with `{id}` in its path, sent by every role in the cast."""
    answered = {}
    for role in ROLES:
        resource_id = create_shared(cast)
        with cast.actor(role).client() as client:
            response = client.request(method, path.format(id=resource_id), json=body)
        answered[role] = response.status_code
    return answered
