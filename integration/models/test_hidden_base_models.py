"""Journey: the Hidden Models permission, which lets a user build a workspace model on a hidden one.

open-webui 0.12.4 (`7f1b61872`) added **Hidden Models** under Models Access, stored as
`workspace.models_hidden`: off by default, switched on per group or in the default permissions,
and for everyone by `USER_PERMISSIONS_WORKSPACE_MODELS_HIDDEN`. It lets the model editor offer
hidden models as the base model. As the docs put it, it changes only that list, not who can use
a hidden model: a user who can read a hidden model may still use it as a base over the API without
the permission, and the workspace model then chats on it like any other.

Discriminates: passes on dev 948432b43. In a backend copy, leaving `models_hidden` out of the
default permissions form turns the default-permissions test red (the saved switch reads back off),
leaving it out of `DEFAULT_USER_PERMISSIONS` turns the environment test red, `get_permissions`
ignoring a group's `models_hidden` turns the group test red, and dropping hidden models from a
user's model list turns both chat tests red (the hidden base is not listed for them).
"""

from __future__ import annotations

import uuid
from typing import Iterator

import pytest

from harness.access import grant, make_group
from harness.actors import Actor, create_user
from harness.chat import ask
from harness.upstream import answering, text

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

DEFAULTS = "/api/v1/users/default/permissions"
ENV_ON = {"USER_PERMISSIONS_WORKSPACE_MODELS_HIDDEN": "true"}
PUBLIC_READ = grant("user", "*", "read")


def _hidden_models(actor: Actor) -> bool:
    """The Hidden Models switch as the web client reads it, from both places it reads it."""
    with actor.client() as client:
        listed = client.get("/api/v1/users/permissions")
        session = client.get("/api/v1/auths/")
    assert listed.status_code == 200, listed.text
    assert session.status_code == 200, session.text
    assert listed.json() == session.json()["permissions"]
    return listed.json()["workspace"]["models_hidden"]


def _set_default(admin: Actor, value: bool) -> dict:
    with admin.client() as client:
        current = client.get(DEFAULTS)
        assert current.status_code == 200, current.text
        changed = current.json()
        changed["workspace"]["models_hidden"] = value
        saved = client.post(DEFAULTS, json=changed)
        assert saved.status_code == 200, saved.text
        stored = client.get(DEFAULTS)
    assert stored.status_code == 200, stored.text
    return stored.json()


@pytest.fixture
def groups(admin) -> Iterator[list[str]]:
    made: list[str] = []
    yield made
    with admin.client() as client:
        for group_id in made:
            client.delete(f"/api/v1/groups/id/{group_id}/delete")


def _member_of(admin: Actor, groups: list[str], make_user, workspace: dict) -> Actor:
    member = make_user()
    groups.append(make_group(admin, [member], {"workspace": workspace}))
    return member


@pytest.fixture
def hidden_base(admin, upstream) -> Iterator[str]:
    """A provider model every user may read, hidden by the admin; models built on it go after."""
    model_id = f"hidden-base-{uuid.uuid4().hex[:8]}"
    upstream.models.append(model_id)
    with admin.client() as client:
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
        hidden = client.post(
            "/api/v1/models/create",
            json={
                "id": model_id,
                "base_model_id": None,
                "name": f"Hidden {model_id}",
                "meta": {"hidden": True},
                "params": {},
                "access_grants": [PUBLIC_READ],
            },
        )
        assert hidden.status_code == 200, hidden.text
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
    yield model_id
    with admin.client() as client:
        for model in client.get("/api/v1/models/list").json().get("items", []):
            if model.get("base_model_id") == model_id:
                client.post("/api/v1/models/model/delete", json={"id": model["id"]})
        client.post("/api/v1/models/model/delete", json={"id": model_id})
        upstream.models.remove(model_id)
        client.get("/api/models", params={"refresh": "true"})


def _listed_hidden(actor: Actor, model_id: str) -> bool:
    """Whether the account's model list, which the model editor reads, has the model as hidden."""
    with actor.client() as client:
        listed = client.get("/api/models")
    assert listed.status_code == 200, listed.text
    entry = next((model for model in listed.json()["data"] if model["id"] == model_id), None)
    assert entry is not None, f"{model_id} is missing from the account's model list"
    return bool(((entry.get("info") or {}).get("meta") or {}).get("hidden"))


def _build_on(actor: Actor, base_model_id: str):
    model_id = f"on-hidden-{uuid.uuid4().hex[:8]}"
    body = {
        "id": model_id,
        "base_model_id": base_model_id,
        "name": f"Built on {base_model_id}",
        "meta": {},
        "params": {},
        "access_grants": [],
    }
    with actor.client() as client:
        return model_id, client.post("/api/v1/models/create", json=body)


def _chats_on(actor: Actor, model_id: str, base_model_id: str, upstream) -> None:
    question = f"which model answers {uuid.uuid4().hex[:6]}?"
    upstream.queue(text("the hidden base answers", match=answering(question)))
    with actor.client() as client:
        # the web client reloads its model list after a save, which registers the new model
        listed = client.get("/api/models")
        assert listed.status_code == 200, listed.text
        assert model_id in [model["id"] for model in listed.json()["data"]], listed.text
        _, answer = ask(client, question, model=model_id)
    assert answer["content"] == "the hidden base answers", answer
    asked = [request["model"] for request in filter(answering(question), upstream.chat_requests())]
    assert asked == [base_model_id], f"the provider was asked for {asked}, not the hidden base"


# the switch


def test_the_switch_is_off_by_default_and_saves_in_the_default_permissions(
    admin, make_user, preserve
):
    preserve("permissions")
    assert _set_default(admin, False)["workspace"]["models_hidden"] is False
    assert _hidden_models(make_user()) is False

    stored = _set_default(admin, True)

    assert stored["workspace"]["models_hidden"] is True, "the saved switch read back off"
    assert _hidden_models(make_user()) is True, "a user did not get the switched-on default"
    assert _set_default(admin, False)["workspace"]["models_hidden"] is False
    assert _hidden_models(make_user()) is False


def test_a_groups_switch_saves_and_reaches_only_its_members(admin, make_user, preserve, groups):
    preserve("permissions")
    _set_default(admin, False)
    member = _member_of(admin, groups, make_user, {"models_hidden": True})
    outsider = make_user()

    with admin.client() as client:
        stored = client.get(f"/api/v1/groups/id/{groups[0]}")
    assert stored.status_code == 200, stored.text
    assert stored.json()["permissions"]["workspace"]["models_hidden"] is True, stored.json()
    assert _hidden_models(member) is True, "the group's Hidden Models did not reach its member"
    assert _hidden_models(outsider) is False, "Hidden Models reached a user outside the group"


@pytest.mark.slow
def test_the_environment_switches_it_on_for_every_user(instance_with):
    instance = instance_with(ENV_ON)
    with instance.client() as client:
        defaults = client.get(DEFAULTS)
    assert defaults.status_code == 200, defaults.text

    assert defaults.json()["workspace"]["models_hidden"] is True, defaults.json()["workspace"]
    assert _hidden_models(create_user(instance)) is True


# building on a hidden model


def test_a_workspace_model_on_a_hidden_base_chats_for_a_permitted_user(
    admin, make_user, preserve, groups, hidden_base, upstream
):
    preserve("permissions")
    _set_default(admin, False)
    builder = _member_of(admin, groups, make_user, {"models": True, "models_hidden": True})
    assert _listed_hidden(builder, hidden_base), "the base is not marked hidden for the user"

    model_id, created = _build_on(builder, hidden_base)

    assert created.status_code == 200, created.text
    assert created.json()["base_model_id"] == hidden_base
    _chats_on(builder, model_id, hidden_base, upstream)


def test_without_the_permission_the_server_still_accepts_a_hidden_base(
    admin, make_user, preserve, groups, hidden_base, upstream
):
    # the permission only widens the editor's list; a readable hidden model stays usable
    preserve("permissions")
    _set_default(admin, False)
    builder = _member_of(admin, groups, make_user, {"models": True})
    assert _hidden_models(builder) is False
    assert _listed_hidden(builder, hidden_base), "the base is not marked hidden for the user"

    model_id, created = _build_on(builder, hidden_base)

    assert created.status_code == 200, (
        f"the server refused a hidden base without the permission: HTTP {created.status_code} "
        f"{created.text}"
    )
    _chats_on(builder, model_id, hidden_base, upstream)
