"""Owner, stranger, reader, writer and admin against one shared resource.

`Shareable` says how one kind of resource is created and shared over the API. `cast(...)` adds
the accounts, sharing with the reader and writer directly or through a group of their own, and
`statuses(...)` sends one route as each of them, each on a fresh resource shared the same way,
so a route that deletes or toggles never changes what the next account meets. `attempts(...)`
does the same and reads the owner's view of the resource around each request, so a refused
write can be shown to have changed nothing; `reads(...)` builds that view from the owner's GET
routes, compared byte for byte.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Callable

import httpx

from harness.actors import Actor

ROLES = ("owner", "stranger", "reader", "writer", "admin")


@dataclass(frozen=True)
class Shareable:
    """A kind of resource: where it is created and where its access grants are replaced."""

    create_path: str = ""
    create_body: Callable[[], dict] = dict
    access_path: str = ""  # with `{id}`
    # added to the access body, for kinds that also name the resource there
    access_fields: Callable[[str], dict] = field(default=lambda resource_id: {})
    # workspace permissions the owner needs to create one, e.g. {"workspace": {"tools": True}}
    owner_permissions: dict | None = None
    # for kinds shared through another resource: creates, shares with the grants, returns the id
    create_shared: Callable[[httpx.Client, list[dict]], str] | None = None


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


def make_group(admin: Actor, members: list[Actor], permissions: dict | None = None) -> str:
    """A group holding `members`, added the way the admin panel adds one."""
    form = {"name": f"group {uuid.uuid4().hex[:8]}", "description": "access matrix"}
    if permissions:
        form["permissions"] = permissions
    with admin.client() as client:
        created = client.post("/api/v1/groups/create", json=form)
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
    if kind.owner_permissions:
        make_group(admin, [owner], kind.owner_permissions)
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
        if kind.create_shared:
            return kind.create_shared(client, cast.grants)
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


@dataclass(frozen=True)
class Attempt:
    """One account's request on a fresh resource, and the owner's view of it before and after."""

    status: int
    before: object
    after: object


def attempts(
    cast: Cast,
    method: str,
    path: str,
    body: dict | Callable[[Actor, dict], dict] | None = None,
    setup: Callable[[Actor, str], dict] | None = None,
    look: Callable[[httpx.Client, dict], object] | None = None,
) -> dict[str, Attempt]:
    """`{role: Attempt}` for one route, sent by every role in the cast.

    `setup(owner, resource_id)` adds fields to each fresh resource (a file, an event) that the
    path, `body(actor, fields)` and `look(owner_client, fields)` read next to `id`.
    """
    answered = {}
    for role in ROLES:
        resource_id = create_shared(cast)
        fields = {"id": resource_id, **(setup(cast.owner, resource_id) if setup else {})}
        actor = cast.actor(role)
        payload = body(actor, fields) if callable(body) else body
        with cast.owner.client() as owner_client:
            before = look(owner_client, fields) if look else None
        with actor.client() as client:
            response = client.request(method, path.format(**fields), json=payload)
        with cast.owner.client() as owner_client:
            after = look(owner_client, fields) if look else None
        answered[role] = Attempt(response.status_code, before, after)
    return answered


def reads(*paths: str) -> Callable[[httpx.Client, dict], list[tuple[int, bytes]]]:
    """A `look` that fetches `paths`, formatted with the fields, and keeps each answer whole."""

    def look(client: httpx.Client, fields: dict) -> list[tuple[int, bytes]]:
        responses = [client.get(path.format(**fields)) for path in paths]
        return [(response.status_code, response.content) for response in responses]

    return look
