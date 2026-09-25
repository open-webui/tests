"""Journey: who may open, edit, switch off, share and delete a shared skill.

A user with the workspace skills permission creates a skill and shares it with a reader (read)
and a writer (read and write), directly or through a group. A stranger is refused everything;
the reader may open the skill but not change it; the writer and the admin may do all of it.
Every refused write leaves the owner's skill as it was.

Discriminates: in a backend copy, asking for `read` instead of `write` in the toggle handler's
grant check turns the toggle rows red (the reader gets 200 and the owner's skill switches off),
and dropping the grant check from the update handler turns its rows red (the stranger and the
reader rewrite the owner's skill).
"""

from __future__ import annotations

import uuid

import pytest

from harness.access import Shareable, attempts, cast, reads

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]


def _skill() -> dict:
    suffix = uuid.uuid4().hex[:8]
    return {
        "id": f"skill-{suffix}",
        "name": f"Release notes {suffix}",  # names are unique
        "content": "Write release notes from the merged pull requests.",
    }


SKILL = Shareable(
    create_path="/api/v1/skills/create",
    create_body=_skill,
    access_path="/api/v1/skills/id/{id}/access/update",
    owner_permissions={"workspace": {"skills": True}},
)
OWNERS_SKILL = reads("/api/v1/skills/id/{id}")

REFUSED, ALLOWED = 401, 200
READ = {"owner": ALLOWED, "stranger": REFUSED, "reader": ALLOWED, "writer": ALLOWED}
WRITE = {"owner": ALLOWED, "stranger": REFUSED, "reader": REFUSED, "writer": ALLOWED}

# method, path, body, what each account gets
MATRIX = [
    ("GET", "/api/v1/skills/id/{id}", None, READ),
    (
        "POST",
        "/api/v1/skills/id/{id}/update",
        lambda actor, fields: {**_skill(), "id": fields["id"], "content": "rewritten"},
        WRITE,
    ),
    ("POST", "/api/v1/skills/id/{id}/toggle", None, WRITE),
    ("POST", "/api/v1/skills/id/{id}/access/update", {"access_grants": []}, WRITE),
    ("DELETE", "/api/v1/skills/id/{id}/delete", None, WRITE),
]


@pytest.mark.parametrize("via", ["user", "group"])
@pytest.mark.parametrize(
    "method, path, body, expected", MATRIX, ids=[f"{row[0]} {row[1]}" for row in MATRIX]
)
def test_each_account_gets_what_its_grant_allows(
    method, path, body, expected, via, admin, make_user
):
    accounts = cast(SKILL, admin, make_user, via=via)

    answered = attempts(accounts, method, path, body, look=OWNERS_SKILL)

    assert {role: attempt.status for role, attempt in answered.items()} == {
        **expected,
        "admin": ALLOWED,
    }, f"{method} {path} shared via {via}"
    for role, attempt in answered.items():
        if attempt.status != ALLOWED:
            assert attempt.after == attempt.before, f"a refused {role} changed the owner's skill"
