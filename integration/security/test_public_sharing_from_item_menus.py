"""Journey: making a skill, tool or note public takes the public sharing permission, whoever asks.

The Access dialog opened from a skill, tool or note's menu saves every change straight to the
item's access route, and offers Public only to an account with the public sharing permission for
that kind. The server holds the same line on its own: a public grant sent by an owner without the
permission is dropped while the rest of the save stands, so an unrelated account still cannot
open the item; with the permission the grant is kept and any account can open it. The browser
side is e2e/workspace/test_access_from_item_menus.py.

Discriminates: passes on dev 206bf9723 (3 of 3). In a backend copy whose grant filter no longer
drops public grants every "without" case goes red (the stranger opens the item), and in one
whose grant filter asks for a permission nobody holds every "with" case goes red (the public
grant is dropped).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Callable

import httpx
import pytest

from harness.access import grant, make_group
from harness.actors import Actor

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

TOOL_SOURCE = 'class Tools:\n    def ping(self) -> str:\n        return "pong"\n'
EVERYONE_READS = grant("user", "*", "read")


@dataclass(frozen=True)
class Kind:
    workspace: dict
    public_flag: str
    create: Callable[[httpx.Client], str]
    item_path: str
    access_path: str


def _create_skill(client: httpx.Client) -> str:
    skill_id = f"public-skill-{uuid.uuid4().hex[:8]}"
    created = client.post(
        "/api/v1/skills/create",
        json={
            "id": skill_id,
            "name": f"Knots {skill_id}",
            "description": "knots",
            "content": "Tie a bowline.",
            "meta": {},
        },
    )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _create_tool(client: httpx.Client) -> str:
    tool_id = f"public_tool_{uuid.uuid4().hex[:8]}"
    created = client.post(
        "/api/v1/tools/create",
        json={
            "id": tool_id,
            "name": "Ping",
            "content": TOOL_SOURCE,
            "meta": {"description": "pings"},
        },
    )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _create_note(client: httpx.Client) -> str:
    created = client.post(
        "/api/v1/notes/create",
        json={"title": "Agenda", "data": {"content": {"md": "agenda"}}, "access_grants": []},
    )
    assert created.status_code == 200, created.text
    return created.json()["id"]


KINDS = {
    "skill": Kind(
        {"workspace": {"skills": True}},
        "public_skills",
        _create_skill,
        "/api/v1/skills/id/{id}",
        "/api/v1/skills/id/{id}/access/update",
    ),
    "tool": Kind(
        {"workspace": {"tools": True}},
        "public_tools",
        _create_tool,
        "/api/v1/tools/id/{id}",
        "/api/v1/tools/id/{id}/access/update",
    ),
    "note": Kind(
        {},
        "public_notes",
        _create_note,
        "/api/v1/notes/{id}",
        "/api/v1/notes/{id}/access/update",
    ),
}


def _owner(admin: Actor, make_user, kind: Kind, may_publish: bool) -> Actor:
    account = make_user()
    sharing = {kind.public_flag.removeprefix("public_"): True, kind.public_flag: may_publish}
    make_group(admin, [account], {**kind.workspace, "sharing": sharing})
    return account


def _made_public_by(owner: Actor, kind: Kind, reader: Actor) -> tuple[str, list[tuple]]:
    """The owner saves the grants the dialog sends for Public plus a reader; returns what stuck."""
    with owner.client() as client:
        item_id = kind.create(client)
        saved = client.post(
            kind.access_path.format(id=item_id),
            json={"access_grants": [EVERYONE_READS, grant("user", reader.id, "read")]},
        )
        assert saved.status_code == 200, saved.text
        stored = client.get(kind.item_path.format(id=item_id))
    assert stored.status_code == 200, stored.text
    grants = [
        (entry["principal_type"], entry["principal_id"], entry["permission"])
        for entry in stored.json()["access_grants"]
    ]
    return item_id, grants


@pytest.mark.parametrize("kind_name", list(KINDS))
def test_without_the_public_permission_the_public_grant_is_dropped_and_the_rest_kept(
    admin, make_user, kind_name
):
    kind = KINDS[kind_name]
    owner = _owner(admin, make_user, kind, may_publish=False)
    reader, stranger = make_user(), make_user()

    item_id, grants = _made_public_by(owner, kind, reader)
    with stranger.client() as client:
        opened_by_stranger = client.get(kind.item_path.format(id=item_id))
    with reader.client() as client:
        opened_by_reader = client.get(kind.item_path.format(id=item_id))

    assert ("user", "*", "read") not in grants
    assert ("user", reader.id, "read") in grants
    assert opened_by_stranger.status_code != 200, opened_by_stranger.text
    assert opened_by_reader.status_code == 200, opened_by_reader.text


@pytest.mark.parametrize("kind_name", list(KINDS))
def test_with_the_public_permission_any_account_opens_the_item(admin, make_user, kind_name):
    kind = KINDS[kind_name]
    owner = _owner(admin, make_user, kind, may_publish=True)
    reader, stranger = make_user(), make_user()

    item_id, grants = _made_public_by(owner, kind, reader)
    with stranger.client() as client:
        opened_by_stranger = client.get(kind.item_path.format(id=item_id))

    assert ("user", "*", "read") in grants
    assert opened_by_stranger.status_code == 200, opened_by_stranger.text
