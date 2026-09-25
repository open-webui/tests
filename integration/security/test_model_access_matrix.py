"""Journey: who may open, edit, switch off, share and delete a shared workspace model.

A user with the workspace model permission builds a preset on the scripted model and shares it
with a reader (read) and a writer (read and write), directly or through a group. A stranger is
refused everything; the reader may open the preset but not change it; the writer and the admin
may do all of it. Every refused write leaves the owner's preset as it was. A preset shared with
nobody stays out of other users' model picker, and a chat naming it is refused.

Discriminates: in a backend copy, asking for `read` instead of `write` in the update handler's
grant check turns the `/model/update` rows red (the reader gets 200 and the owner's preset
changes), dropping the grant check from `get_filtered_models` turns the picker test red, and
dropping it from both `check_model_access` copies turns the chat test red (the stranger's chat
is answered).
"""

from __future__ import annotations

import uuid

import pytest

from harness.access import Shareable, attempts, cast, reads
from harness.actors import Actor
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

MODEL_CURATOR = {"workspace": {"models": True}}


def _preset() -> dict:
    return {
        "id": f"preset-{uuid.uuid4().hex[:8]}",
        "base_model_id": MOCK_MODEL_ID,
        "name": "Shared preset",
        "meta": {"description": "access matrix"},
        "params": {"system": "the owner's curated prompt"},
    }


MODEL = Shareable(
    create_path="/api/v1/models/create",
    create_body=_preset,
    access_path="/api/v1/models/model/access/update",
    access_fields=lambda model_id: {"id": model_id},
    owner_permissions=MODEL_CURATOR,
)
OWNERS_PRESET = reads("/api/v1/models/model?id={id}")

ALLOWED = 200
READ = {"owner": ALLOWED, "stranger": 401, "reader": ALLOWED, "writer": ALLOWED}
# the update and share handlers refuse with 400, toggle and delete with 401
WRITE_400 = {"owner": ALLOWED, "stranger": 400, "reader": 400, "writer": ALLOWED}
WRITE_401 = {"owner": ALLOWED, "stranger": 401, "reader": 401, "writer": ALLOWED}


def _renamed(actor: Actor, fields: dict) -> dict:
    return {**_preset(), "id": fields["id"], "name": "renamed", "params": {"system": "replaced"}}


# method, path, body, what each account gets
MATRIX = [
    ("GET", "/api/v1/models/model?id={id}", None, READ),
    ("POST", "/api/v1/models/model/toggle?id={id}", None, WRITE_401),
    ("POST", "/api/v1/models/model/update", _renamed, WRITE_400),
    (
        "POST",
        "/api/v1/models/model/access/update",
        lambda actor, fields: {"id": fields["id"], "access_grants": []},
        WRITE_400,
    ),
    ("POST", "/api/v1/models/model/delete", lambda actor, fields: {"id": fields["id"]}, WRITE_401),
]


@pytest.mark.parametrize("via", ["user", "group"])
@pytest.mark.parametrize(
    "method, path, body, expected", MATRIX, ids=[f"{row[0]} {row[1]}" for row in MATRIX]
)
def test_each_account_gets_what_its_grant_allows(
    method, path, body, expected, via, admin, make_user
):
    accounts = cast(MODEL, admin, make_user, via=via)

    answered = attempts(accounts, method, path, body, look=OWNERS_PRESET)

    assert {role: attempt.status for role, attempt in answered.items()} == {
        **expected,
        "admin": ALLOWED,
    }, f"{method} {path} shared via {via}"
    for role, attempt in answered.items():
        if attempt.status != ALLOWED:
            assert attempt.after == attempt.before, f"a refused {role} changed the owner's preset"


@pytest.fixture
def private_preset(admin, make_user):
    """A preset shared with nobody, its owner and a reader it was shared with instead."""
    accounts = cast(MODEL, admin, make_user)
    with accounts.owner.client() as client:
        created = client.post("/api/v1/models/create", json=_preset())
        assert created.status_code == 200, created.text
        model_id = created.json()["id"]
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
    return model_id, accounts


def _listed(actor) -> set[str]:
    with actor.client() as client:
        response = client.get("/api/models")
    assert response.status_code == 200, response.text
    return {model["id"] for model in response.json()["data"]}


def _chat_with(actor, model_id: str):
    body = {"model": model_id, "messages": [{"role": "user", "content": "hi"}], "stream": False}
    with actor.client() as client:
        return client.post("/api/chat/completions", json=body)


def test_a_private_preset_stays_out_of_another_users_picker(private_preset):
    model_id, accounts = private_preset

    assert model_id in _listed(accounts.owner)
    assert model_id not in _listed(accounts.stranger), "a stranger's picker lists a private preset"


def test_chatting_with_someone_elses_private_preset_is_refused(private_preset, upstream):
    model_id, accounts = private_preset

    response = _chat_with(accounts.stranger, model_id)

    assert response.status_code == 400 and "Model not found" in response.text, (
        f"a stranger chatted with a private preset: HTTP {response.status_code} {response.text}"
    )
    assert upstream.chat_requests() == []


def test_the_owner_chats_with_their_private_preset(private_preset, upstream):
    model_id, accounts = private_preset

    assert _chat_with(accounts.owner, model_id).status_code == 200
