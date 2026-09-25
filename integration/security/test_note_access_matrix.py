"""Journey: who may read, change, share, pin and delete a shared note.

The owner shares a note with a reader (read) and a writer (read and write), directly or through
a group, the way the share dialog does. A stranger is refused everything; the reader may open
and pin the note but not change, share or delete it; the writer and the admin may do all of it.

Discriminates: in a backend copy, dropping the owner-or-write check from the note update
handler turns the `/update` rows red (the stranger and the reader get 200), and asking for
`read` instead of `write` in the delete handler turns the `/delete` rows red (the reader gets
200).
"""

from __future__ import annotations

import uuid

import pytest

from harness.access import Shareable, cast, statuses

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

NOTE = Shareable(
    create_path="/api/v1/notes/create",
    create_body=lambda: {"title": f"shared {uuid.uuid4().hex[:8]}", "data": {"content": {}}},
    access_path="/api/v1/notes/{id}/access/update",
)

REFUSED, ALLOWED = 403, 200
READ = {"owner": ALLOWED, "stranger": REFUSED, "reader": ALLOWED, "writer": ALLOWED}
WRITE = {"owner": ALLOWED, "stranger": REFUSED, "reader": REFUSED, "writer": ALLOWED}

# method, path, body, what each account gets
MATRIX = [
    ("GET", "/api/v1/notes/{id}", None, READ),
    ("POST", "/api/v1/notes/{id}/pin", None, READ),
    ("POST", "/api/v1/notes/{id}/update", {"title": "renamed"}, WRITE),
    ("POST", "/api/v1/notes/{id}/access/update", {"access_grants": []}, WRITE),
    ("DELETE", "/api/v1/notes/{id}/delete", None, WRITE),
]


@pytest.mark.parametrize("via", ["user", "group"])
@pytest.mark.parametrize(
    "method, path, body, expected", MATRIX, ids=[f"{row[0]} {row[1]}" for row in MATRIX]
)
def test_each_account_gets_what_its_grant_allows(
    method, path, body, expected, via, admin, make_user
):
    accounts = cast(NOTE, admin, make_user, via=via)

    answered = statuses(accounts, method, path, body)

    assert answered == {**expected, "admin": ALLOWED}, f"{method} {path} shared via {via}"
