"""Journey: a workspace model's version history, as the model editor keeps and restores it.

Creating a model records its first version. Every save that changes the configuration adds a
version holding a snapshot of it, with the commit message typed beside Save, and makes it the
Production version; a save that changes nothing adds none. Opening an older version shows its
snapshot, and setting it as Production copies that snapshot back onto the model without adding
a version. The Production version cannot be deleted; an older one can. Only accounts that may
change the model see or touch its versions: the owner, a writer and the admin, never a reader
or a stranger, and a refused request leaves the versions and the model as they were. Restoring
a version re-checks what it points at, so a writer cannot bring back the owner's private
knowledge base through an old version.

Discriminates: passes on dev 206bf9723 (3 of 3). In backend copies: creating a model without
storing its first version turns `test_creating_a_model_records_its_first_version` red; a save
that changes the model adding no version turns `test_a_save_adds_a_version_and_makes_it_production`
red; every save adding a version turns `test_a_save_that_changes_nothing_adds_no_version` red;
opening a version returning the live configuration and versions being found through any model
turn `test_an_old_version_shows_the_configuration_it_was_saved_with` and
`test_a_version_of_one_model_cannot_be_reached_through_another` red; Set as Production keeping
the live configuration and the delete no longer guarding Production turn
`test_setting_an_old_version_as_production_brings_it_back` and
`test_the_production_version_cannot_be_deleted` red; the history routes asking for `read`
instead of `write` and restoring without re-checking knowledge access turn every reader row of
`test_only_accounts_that_may_change_the_model_reach_its_versions` and
`test_a_writer_cannot_restore_a_version_pointing_at_a_knowledge_base_they_cannot_read` red.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from harness.access import Shareable, attempts, cast, make_group
from harness.actors import Actor
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

MODEL_CURATOR = {"workspace": {"models": True}}
KNOWLEDGE_CURATOR = {"workspace": {"models": True, "knowledge": True}}


def _preset_form(system: str, model_id: str | None = None, **meta) -> dict:
    return {
        "id": model_id or f"history-{uuid.uuid4().hex[:8]}",
        "base_model_id": MOCK_MODEL_ID,
        "name": f"Preset {system}",
        "meta": {"description": "version history", **meta},
        "params": {"system": system},
    }


MODEL = Shareable(
    create_path="/api/v1/models/create",
    create_body=lambda: _preset_form("first draft"),
    access_path="/api/v1/models/model/access/update",
    access_fields=lambda model_id: {"id": model_id},
    owner_permissions=MODEL_CURATOR,
)


@pytest.fixture
def curator(admin, make_user) -> Actor:
    account = make_user()
    make_group(admin, [account], MODEL_CURATOR)
    return account


def _create(client: httpx.Client, system: str, **meta) -> dict:
    created = client.post("/api/v1/models/create", json=_preset_form(system, **meta))
    assert created.status_code == 200, created.text
    return created.json()


def _save(client: httpx.Client, model_id: str, system: str, **fields) -> dict:
    form = {**_preset_form(system, model_id), **fields}
    saved = client.post("/api/v1/models/model/update", json=form)
    assert saved.status_code == 200, saved.text
    return saved.json()


def _versions(client: httpx.Client, model_id: str) -> list[dict]:
    listed = client.get("/api/v1/models/model/history", params={"id": model_id})
    assert listed.status_code == 200, listed.text
    return listed.json()


def _version(client: httpx.Client, model_id: str, version_id: str) -> dict:
    opened = client.get(f"/api/v1/models/model/history/{version_id}", params={"id": model_id})
    assert opened.status_code == 200, opened.text
    return opened.json()


def _model(client: httpx.Client, model_id: str) -> dict:
    read = client.get("/api/v1/models/model", params={"id": model_id})
    assert read.status_code == 200, read.text
    return read.json()


def test_creating_a_model_records_its_first_version(curator):
    with curator.client() as client:
        model = _create(client, "first draft")
        versions = _versions(client, model["id"])

    assert [version["id"] for version in versions] == [model["version_id"]]
    assert versions[0]["user_id"] == curator.id
    assert versions[0]["user"] == {"name": curator.name}


def test_a_save_adds_a_version_and_makes_it_production(curator):
    with curator.client() as client:
        model = _create(client, "first draft")
        saved = _save(client, model["id"], "second draft", commit_message="tighten the prompt")
        versions = _versions(client, model["id"])
        latest = _version(client, model["id"], saved["version_id"])

    assert saved["version_id"] != model["version_id"]
    assert {version["id"] for version in versions} == {model["version_id"], saved["version_id"]}
    assert latest["commit_message"] == "tighten the prompt"
    assert latest["parent_id"] == model["version_id"]
    assert latest["snapshot"]["params"]["system"] == "second draft"
    assert latest["snapshot"]["name"] == "Preset second draft"


def test_a_save_that_changes_nothing_adds_no_version(curator):
    with curator.client() as client:
        model = _create(client, "first draft")
        saved = _save(client, model["id"], "first draft", commit_message="nothing changed")
        versions = _versions(client, model["id"])

    assert [version["id"] for version in versions] == [model["version_id"]]
    assert saved["version_id"] == model["version_id"]


def test_an_old_version_shows_the_configuration_it_was_saved_with(curator):
    with curator.client() as client:
        model = _create(client, "first draft")
        _save(client, model["id"], "second draft")
        first = _version(client, model["id"], model["version_id"])
        current = _model(client, model["id"])

    assert first["snapshot"]["params"]["system"] == "first draft"
    assert first["snapshot"]["name"] == "Preset first draft"
    assert current["params"]["system"] == "second draft"


def test_setting_an_old_version_as_production_brings_it_back(curator):
    with curator.client() as client:
        model = _create(client, "first draft")
        _save(client, model["id"], "second draft")
        restored = client.post(
            "/api/v1/models/model/update/version",
            params={"id": model["id"]},
            json={"version_id": model["version_id"]},
        )
        assert restored.status_code == 200, restored.text
        current = _model(client, model["id"])
        versions = _versions(client, model["id"])

    assert (current["name"], current["params"]["system"]) == ("Preset first draft", "first draft")
    assert current["version_id"] == model["version_id"]
    assert len(versions) == 2


def test_the_production_version_cannot_be_deleted(curator):
    with curator.client() as client:
        model = _create(client, "first draft")
        saved = _save(client, model["id"], "second draft")
        refused = client.delete(
            f"/api/v1/models/model/history/{saved['version_id']}", params={"id": model["id"]}
        )
        deleted = client.delete(
            f"/api/v1/models/model/history/{model['version_id']}", params={"id": model["id"]}
        )
        remaining = [version["id"] for version in _versions(client, model["id"])]
        current = _model(client, model["id"])

    assert refused.status_code == 400, refused.text
    assert deleted.status_code == 200, deleted.text
    assert remaining == [saved["version_id"]]
    assert current["params"]["system"] == "second draft"


def test_a_version_of_one_model_cannot_be_reached_through_another(curator, make_user, admin):
    other_curator = make_user()
    make_group(admin, [other_curator], MODEL_CURATOR)
    with curator.client() as client:
        mine = _create(client, "my prompt")
    with other_curator.client() as client:
        theirs = _create(client, "their prompt")
        opened = client.get(
            f"/api/v1/models/model/history/{mine['version_id']}", params={"id": theirs["id"]}
        )
        deleted = client.delete(
            f"/api/v1/models/model/history/{mine['version_id']}", params={"id": theirs["id"]}
        )
    with curator.client() as client:
        still_there = [version["id"] for version in _versions(client, mine["id"])]

    assert opened.status_code == 404, opened.text
    assert deleted.status_code == 404, deleted.text
    assert still_there == [mine["version_id"]]


def _second_version(owner: Actor, model_id: str) -> dict:
    """A second save, so the first version is an older one the route can act on."""
    with owner.client() as client:
        first = _model(client, model_id)["version_id"]
        _save(client, model_id, "second draft")
    return {"history_id": first}


def _owners_view(owner_client: httpx.Client, fields: dict) -> tuple:
    model = _model(owner_client, fields["id"])
    versions = sorted(version["id"] for version in _versions(owner_client, fields["id"]))
    return model["params"]["system"], model["version_id"], versions


REFUSED, ALLOWED = 403, 200
MAY_CHANGE = {
    "owner": ALLOWED,
    "stranger": REFUSED,
    "reader": REFUSED,
    "writer": ALLOWED,
    "admin": ALLOWED,
}

# method, path, body, whether an allowed request changes the owner's view
ROUTES = [
    ("GET", "/api/v1/models/model/history?id={id}", None, False),
    ("GET", "/api/v1/models/model/history/{history_id}?id={id}", None, False),
    ("DELETE", "/api/v1/models/model/history/{history_id}?id={id}", None, True),
    (
        "POST",
        "/api/v1/models/model/update/version?id={id}",
        lambda actor, fields: {"version_id": fields["history_id"]},
        True,
    ),
]


@pytest.mark.parametrize("via", ["user", "group"])
@pytest.mark.parametrize(
    "method, path, body, changes", ROUTES, ids=["list", "open", "delete", "set as production"]
)
def test_only_accounts_that_may_change_the_model_reach_its_versions(
    admin, make_user, via, method, path, body, changes
):
    accounts = cast(MODEL, admin, make_user, via=via)

    answered = attempts(accounts, method, path, body=body, setup=_second_version, look=_owners_view)

    assert {role: attempt.status for role, attempt in answered.items()} == MAY_CHANGE
    for role, attempt in answered.items():
        if MAY_CHANGE[role] == REFUSED or not changes:
            assert attempt.after == attempt.before, role
        else:
            assert attempt.after != attempt.before, role


def test_a_writer_cannot_restore_a_version_pointing_at_a_knowledge_base_they_cannot_read(
    admin, make_user
):
    owner, writer = make_user(), make_user()
    make_group(admin, [owner], KNOWLEDGE_CURATOR)
    with owner.client() as client:
        created = client.post(
            "/api/v1/knowledge/create",
            json={"name": "Salary reviews", "description": "private", "access_grants": []},
        )
        assert created.status_code == 200, created.text
        knowledge = [{"type": "collection", "id": created.json()["id"], "name": "Salary reviews"}]
        model = _create(client, "with the reviews", knowledge=knowledge)
        _save(client, model["id"], "without the reviews")
        shared = client.post(
            "/api/v1/models/model/access/update",
            json={
                "id": model["id"],
                "access_grants": [
                    {"principal_type": "user", "principal_id": writer.id, "permission": "read"},
                    {"principal_type": "user", "principal_id": writer.id, "permission": "write"},
                ],
            },
        )
        assert shared.status_code == 200, shared.text

    with writer.client() as client:
        refused = client.post(
            "/api/v1/models/model/update/version",
            params={"id": model["id"]},
            json={"version_id": model["version_id"]},
        )
    with owner.client() as client:
        current = _model(client, model["id"])
        restored_by_owner = client.post(
            "/api/v1/models/model/update/version",
            params={"id": model["id"]},
            json={"version_id": model["version_id"]},
        )

    assert refused.status_code == 403, refused.text
    assert current["params"]["system"] == "without the reviews"
    assert not current["meta"].get("knowledge")
    assert restored_by_owner.status_code == 200, restored_by_owner.text
    assert restored_by_owner.json()["meta"]["knowledge"][0]["id"] == knowledge[0]["id"]
