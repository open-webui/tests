"""Journey: who may open, edit, version, share and delete a shared prompt.

A user with the workspace prompts permission creates a prompt, saves a second version of it and
shares it with a reader (read) and a writer (read and write), directly or through a group. A
stranger is refused everything; the reader may open the prompt and its history but not change,
re-version, share or delete it, nor delete an old version; the writer and the admin may do all
of it. Every refused write leaves the owner's prompt as it was.

Discriminates: in a backend copy, asking for `read` instead of `write` in the version handler's
grant check turns the `/update/version` rows red (the reader gets 200 and the owner's live
version changes), and dropping the grant check from the history entry delete handler turns its
rows red (the stranger and the reader delete the owner's old version).
"""

from __future__ import annotations

import uuid

import pytest

from harness.access import Shareable, attempts, cast, reads
from harness.actors import Actor

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

PROMPT_WRITER = {"workspace": {"prompts": True}}


def _command() -> str:
    return f"matrix-{uuid.uuid4().hex[:8]}"


PROMPT = Shareable(
    create_path="/api/v1/prompts/create",
    create_body=lambda: {"command": _command(), "name": "Summary", "content": "first version"},
    access_path="/api/v1/prompts/id/{id}/access/update",
    owner_permissions=PROMPT_WRITER,
)
OWNERS_PROMPT = reads("/api/v1/prompts/id/{id}", "/api/v1/prompts/id/{id}/history")


def _second_version(owner: Actor, prompt_id: str) -> dict:
    """Saves a second version as the owner; `first_version` is the one no longer live."""
    with owner.client() as client:
        prompt = client.get(f"/api/v1/prompts/id/{prompt_id}").json()
        saved = client.post(
            f"/api/v1/prompts/id/{prompt_id}/update",
            json={
                "command": prompt["command"],
                "name": prompt["name"],
                "content": "second version",
                "access_grants": prompt["access_grants"],
            },
        )
        assert saved.status_code == 200, saved.text
        history = client.get(f"/api/v1/prompts/id/{prompt_id}/history").json()
    live_version = saved.json()["version_id"]
    first_version = next(entry["id"] for entry in history if entry["id"] != live_version)
    return {"first_version": first_version}


REFUSED, ALLOWED = 401, 200
READ = {"owner": ALLOWED, "stranger": REFUSED, "reader": ALLOWED, "writer": ALLOWED}
WRITE = {"owner": ALLOWED, "stranger": REFUSED, "reader": REFUSED, "writer": ALLOWED}
HISTORY = "/api/v1/prompts/id/{id}/history"

# method, path, body, setup, what each account gets
MATRIX = [
    # the prompt itself answers 404 to a stranger, so its id cannot be probed
    ("GET", "/api/v1/prompts/id/{id}", None, None, {**READ, "stranger": 404}),
    (
        "POST",
        "/api/v1/prompts/id/{id}/update",
        lambda actor, fields: {"command": _command(), "name": "Renamed", "content": "rewritten"},
        None,
        WRITE,
    ),
    (
        "POST",
        "/api/v1/prompts/id/{id}/update/meta",
        lambda actor, fields: {"command": _command(), "name": "Renamed"},
        None,
        WRITE,
    ),
    (
        "POST",
        "/api/v1/prompts/id/{id}/update/version",
        lambda actor, fields: {"version_id": fields["first_version"]},
        _second_version,
        WRITE,
    ),
    ("POST", "/api/v1/prompts/id/{id}/access/update", {"access_grants": []}, None, WRITE),
    ("POST", "/api/v1/prompts/id/{id}/toggle", None, None, WRITE),
    ("DELETE", "/api/v1/prompts/id/{id}/delete", None, None, WRITE),
    ("GET", HISTORY, None, None, READ),
    ("GET", HISTORY + "/{first_version}", None, _second_version, READ),
    ("DELETE", HISTORY + "/{first_version}", None, _second_version, WRITE),
]


@pytest.mark.parametrize("via", ["user", "group"])
@pytest.mark.parametrize(
    "method, path, body, setup, expected",
    MATRIX,
    ids=[f"{row[0]} {row[1]}" for row in MATRIX],
)
def test_each_account_gets_what_its_grant_allows(
    method, path, body, setup, expected, via, admin, make_user
):
    accounts = cast(PROMPT, admin, make_user, via=via)

    answered = attempts(accounts, method, path, body, setup=setup, look=OWNERS_PROMPT)

    assert {role: attempt.status for role, attempt in answered.items()} == {
        **expected,
        "admin": ALLOWED,
    }, f"{method} {path} shared via {via}"
    for role, attempt in answered.items():
        if attempt.status != ALLOWED:
            assert attempt.after == attempt.before, f"a refused {role} changed the owner's prompt"
