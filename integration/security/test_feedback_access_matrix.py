"""Journey: a rating is its author's to read, change and delete, and the admin's to review.

Feedback is never shared. The per-item routes look a rating up for its author unless the caller
is an admin, so another user reading, rewriting or deleting it is answered 404 and the author
still reads the same rating afterwards, while the admin reaches it. Clearing one's own ratings
and listing them stay within the caller's own.

Discriminates: in a backend copy, making the update route take the admin branch for every
caller turns the stranger's update row red (200 and the rating is rewritten), and dropping the
`user_id` filter from `Feedbacks.delete_feedbacks_by_user_id` turns the clear-all test red (the
author's rating is gone).
"""

from __future__ import annotations

import pytest

from harness.actors import Actor
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

OK, NOT_FOUND = 200, 404
REWRITTEN = {"type": "rating", "data": {"rating": -1, "comment": "rewritten by someone else"}}


def _rate(actor: Actor) -> str:
    with actor.client() as client:
        created = client.post(
            "/api/v1/evaluations/feedback",
            json={
                "type": "rating",
                "data": {"rating": 1, "model_id": MOCK_MODEL_ID, "comment": "clear answer"},
            },
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _author_view(author: Actor, feedback_id: str) -> dict | None:
    with author.client() as client:
        feedback = client.get(f"/api/v1/evaluations/feedback/{feedback_id}")
    if feedback.status_code != 200:
        return None
    return feedback.json()["data"]


ROUTES = [
    ("GET", "/api/v1/evaluations/feedback/{id}", None),
    ("POST", "/api/v1/evaluations/feedback/{id}", REWRITTEN),
    ("DELETE", "/api/v1/evaluations/feedback/{id}", None),
]


@pytest.mark.parametrize("method, path, body", ROUTES, ids=[row[0] for row in ROUTES])
def test_the_author_and_the_admin_reach_a_rating_and_nobody_else(
    method, path, body, admin, make_user
):
    author, stranger = make_user(), make_user()

    answered, unchanged = {}, {}
    for role, actor in (("author", author), ("stranger", stranger), ("admin", admin)):
        feedback_id = _rate(author)
        before = _author_view(author, feedback_id)
        with actor.client() as client:
            response = client.request(method, path.format(id=feedback_id), json=body)
        answered[role] = response.status_code
        unchanged[role] = _author_view(author, feedback_id) == before

    assert answered == {"author": OK, "stranger": NOT_FOUND, "admin": OK}
    assert unchanged["stranger"], "a refused stranger changed the rating"


def test_clearing_ones_ratings_leaves_other_users_ratings_alone(make_user):
    author, stranger = make_user(), make_user()
    feedback_id = _rate(author)
    before = _author_view(author, feedback_id)
    _rate(stranger)

    with stranger.client() as client:
        cleared = client.delete("/api/v1/evaluations/feedbacks")
        remaining = client.get("/api/v1/evaluations/feedbacks/user")

    assert cleared.status_code == OK, cleared.text
    assert remaining.json()["items"] == []
    assert _author_view(author, feedback_id) == before


def test_a_users_rating_list_holds_only_their_own(make_user):
    author, stranger = make_user(), make_user()
    feedback_id = _rate(author)

    with stranger.client() as client:
        listed = client.get("/api/v1/evaluations/feedbacks/user")
    with author.client() as client:
        own = client.get("/api/v1/evaluations/feedbacks/user")

    assert listed.json()["items"] == []
    assert [item["id"] for item in own.json()["items"]] == [feedback_id]
