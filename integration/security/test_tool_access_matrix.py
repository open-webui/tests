"""Journey: who may open, edit, share and delete a shared tool.

A user with the workspace tools permission creates a tool and shares it with a reader (read)
and a writer (read and write), directly or through a group. A stranger is refused everything;
the reader may open the tool but not change it; the writer may rename, share and delete it,
but changing its Python source takes the workspace tools permission too, since saving the source
runs it. The admin may do all of it. Every refused write leaves the owner's tool as it was.

Discriminates: in a backend copy, removing the workspace tools check on changed source from the
update handler turns the new-source rows red (the writer gets 200 and the owner's source is
replaced), and asking for `read` instead of `write` in the delete handler turns the delete rows
red (the reader gets 200).
"""

from __future__ import annotations

import uuid

import pytest

from harness.access import Shareable, attempts, cast, make_group, reads
from harness.actors import Actor

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

TOOL_BUILDER = {"workspace": {"tools": True}}
SOURCE = '''class Tools:
    def echo(self, text: str) -> str:
        """Say the text back.

        :param text: what to say
        """
        return text
'''
NEW_SOURCE = SOURCE.replace("return text", "return text.upper()")


def _tool() -> dict:
    return {
        "id": f"tool_{uuid.uuid4().hex[:8]}",
        "name": "Echo",
        "content": SOURCE,
        "meta": {"description": "access matrix"},
    }


TOOL = Shareable(
    create_path="/api/v1/tools/create",
    create_body=_tool,
    access_path="/api/v1/tools/id/{id}/access/update",
    owner_permissions=TOOL_BUILDER,
)
OWNERS_TOOL = reads("/api/v1/tools/id/{id}")


def _edit(content: str):
    def body(actor: Actor, fields: dict) -> dict:
        return {**_tool(), "id": fields["id"], "name": "Renamed", "content": content}

    return body


REFUSED, ALLOWED = 401, 200
READ = {"owner": ALLOWED, "stranger": REFUSED, "reader": ALLOWED, "writer": ALLOWED}
WRITE = {"owner": ALLOWED, "stranger": REFUSED, "reader": REFUSED, "writer": ALLOWED}
NEEDS_TOOLS_PERMISSION = {**WRITE, "writer": REFUSED}

# method, path, body, what each account gets
MATRIX = [
    ("GET", "/api/v1/tools/id/{id}", None, READ),
    ("POST", "/api/v1/tools/id/{id}/update", _edit(SOURCE), WRITE),
    ("POST", "/api/v1/tools/id/{id}/update", _edit(NEW_SOURCE), NEEDS_TOOLS_PERMISSION),
    ("POST", "/api/v1/tools/id/{id}/access/update", {"access_grants": []}, WRITE),
    ("DELETE", "/api/v1/tools/id/{id}/delete", None, WRITE),
]
ROW_IDS = ["get", "rename", "new source", "share", "delete"]


@pytest.mark.parametrize("via", ["user", "group"])
@pytest.mark.parametrize("method, path, body, expected", MATRIX, ids=ROW_IDS)
def test_each_account_gets_what_its_grant_allows(
    method, path, body, expected, via, admin, make_user
):
    accounts = cast(TOOL, admin, make_user, via=via)

    answered = attempts(accounts, method, path, body, look=OWNERS_TOOL)

    assert {role: attempt.status for role, attempt in answered.items()} == {
        **expected,
        "admin": ALLOWED,
    }, f"{method} {path} shared via {via}"
    for role, attempt in answered.items():
        if attempt.status != ALLOWED:
            assert attempt.after == attempt.before, f"a refused {role} changed the owner's tool"


def test_a_writer_with_the_tools_permission_may_change_the_source(admin, make_user):
    accounts = cast(TOOL, admin, make_user)
    make_group(admin, [accounts.writer], TOOL_BUILDER)

    answered = attempts(
        accounts, "POST", "/api/v1/tools/id/{id}/update", _edit(NEW_SOURCE), look=OWNERS_TOOL
    )

    assert answered["writer"].status == ALLOWED
    assert answered["writer"].after != answered["writer"].before
