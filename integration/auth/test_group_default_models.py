"""Journey: the models a group's members start a new chat with, resolved through the hierarchy.

An admin sets default models on a group; its members, and the members of its subgroups, get them
as `default_models` in the config the web client loads, in place of the instance's Selected
Models. When a member's groups set different defaults the deepest group wins, a subgroup that sets
none passes on the nearest parent's, and between groups at the same depth the one created first
wins. Models that are hidden or that the member cannot read are dropped from the winner's list,
and when none are left the member gets the instance's own default again. Moving a group changes
what its members resolve at once, and the admin's group preview says where a group's effective
defaults come from. A list that is not made of model ids is refused.

The tests that start on a model created through the API are red on dev 708e6642a: the model is
not used as a group's default until someone loads the model list, and a model hidden later stays
the default (open-webui/open-webui#34987).

Discriminates: in a backend copy, sorting `resolve_group_default_models` by shallowest instead of
deepest turns the deepest-wins test red (the top group's models win), sorting by newest turns
the oldest-wins test red, and ignoring the ancestors in the member's group list (no inherited
memberships) turns the inheritance and move tests red.
"""

from __future__ import annotations

import time
import uuid

import pytest

from harness.access import grant, make_group
from harness.group_tree import group_of, move_group
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

PUBLIC = grant("user", "*", "read")


@pytest.fixture
def global_default(admin):
    """`set(value)` changes the instance's Selected Models; the original is put back."""
    with admin.client() as client:
        original = client.get("/api/v1/configs/models").json()

        def set_default(value: str) -> None:
            saved = client.post(
                "/api/v1/configs/models", json={**original, "DEFAULT_MODELS": value}
            )
            assert saved.status_code == 200, saved.text

        yield set_default
        client.post("/api/v1/configs/models", json=original)


@pytest.fixture
def groups(admin):
    """Groups made through `make_group` are deleted again, deepest first."""
    made: list[str] = []
    yield made
    with admin.client() as client:
        for group_id in reversed(made):
            client.delete(f"/api/v1/groups/id/{group_id}/delete")


def _group(admin, groups: list[str], members, default_models=None, parent_id=None) -> str:
    group_id = make_group(admin, members, parent_id=parent_id, default_models=default_models)
    groups.append(group_id)
    return group_id


def _preset(admin, name: str, grants: list[dict] | None = None, hidden: bool = False) -> str:
    model_id = f"preset-{name}-{uuid.uuid4().hex[:6]}"
    body = {
        "id": model_id,
        "base_model_id": MOCK_MODEL_ID,
        "name": name,
        "meta": {"hidden": hidden},
        "params": {},
        "access_grants": [PUBLIC] if grants is None else grants,
    }
    with admin.client() as client:
        created = client.post("/api/v1/models/create", json=body)
        assert created.status_code == 200, created.text
    return model_id


def _starts_with(actor) -> list[str]:
    """The `default_models` the web client loads for this account, as a list."""
    with actor.client() as client:
        config = client.get("/api/config")
    assert config.status_code == 200, config.text
    value = config.json().get("default_models") or ""
    return [model_id for model_id in value.split(",") if model_id]


def _set_defaults(admin, group_id: str, default_models: list[str]):
    stored = group_of(admin, group_id)
    with admin.client() as client:
        return client.post(
            f"/api/v1/groups/id/{group_id}/update",
            json={
                "name": stored["name"],
                "description": stored["description"],
                "data": {"config": {"default_models": default_models}},
            },
        )


def test_a_groups_default_models_replace_the_instances_for_its_members(
    admin, make_user, groups, global_default
):
    member, outsider = make_user(), make_user()
    first, second = _preset(admin, "first"), _preset(admin, "second")
    global_default(MOCK_MODEL_ID)
    _group(admin, groups, [member], default_models=[first, second])

    assert _starts_with(member) == [first, second]
    assert _starts_with(outsider) == [MOCK_MODEL_ID]


def test_a_subgroup_without_defaults_passes_on_the_nearest_parents(admin, make_user, groups):
    leaf_member = make_user()
    wanted = _preset(admin, "wanted")
    top = _group(admin, groups, [], default_models=[wanted])
    middle = _group(admin, groups, [], parent_id=top)
    _group(admin, groups, [leaf_member], parent_id=middle)

    assert _starts_with(leaf_member) == [wanted]


def test_the_deepest_group_wins_over_its_parent(admin, make_user, groups):
    top_member, leaf_member = make_user(), make_user()
    broad, narrow = _preset(admin, "broad"), _preset(admin, "narrow")
    top = _group(admin, groups, [top_member], default_models=[broad])
    _group(admin, groups, [leaf_member], default_models=[narrow], parent_id=top)

    assert _starts_with(leaf_member) == [narrow], "the parent's defaults beat the subgroup's"
    assert _starts_with(top_member) == [broad]


def test_a_deeper_group_wins_even_when_it_was_created_later(admin, make_user, groups):
    member = make_user()
    shallow, deep = _preset(admin, "shallow"), _preset(admin, "deep")
    _group(admin, groups, [member], default_models=[shallow])
    parent = _group(admin, groups, [])
    _group(admin, groups, [member], default_models=[deep], parent_id=parent)

    assert _starts_with(member) == [deep]


def test_between_groups_at_the_same_depth_the_older_one_wins(admin, make_user, groups):
    member = make_user()
    older, newer = _preset(admin, "older"), _preset(admin, "newer")
    _group(admin, groups, [member], default_models=[older])
    time.sleep(1.1)  # groups are stamped to the second
    _group(admin, groups, [member], default_models=[newer])

    assert _starts_with(member) == [older]


def test_hidden_and_unreadable_models_are_dropped_from_the_winners_list(
    admin, make_user, groups, global_default
):
    member = make_user()
    visible = _preset(admin, "visible")
    hidden = _preset(admin, "hidden", hidden=True)
    private = _preset(admin, "private", grants=[])
    global_default(MOCK_MODEL_ID)
    _group(admin, groups, [member], default_models=[hidden, private, visible])

    assert _starts_with(member) == [visible]


def test_the_instances_default_returns_when_no_group_model_is_left(
    admin, make_user, groups, global_default
):
    member = make_user()
    hidden = _preset(admin, "hidden", hidden=True)
    private = _preset(admin, "private", grants=[])
    global_default(MOCK_MODEL_ID)
    _group(admin, groups, [member], default_models=[hidden, private])

    assert _starts_with(member) == [MOCK_MODEL_ID]


def test_a_default_model_hidden_later_is_dropped_at_once(admin, make_user, groups, global_default):
    """A model hidden after the model list loaded stops being a default (#34987)."""
    member = make_user()
    later_hidden = _preset(admin, "later-hidden")
    global_default(MOCK_MODEL_ID)
    _group(admin, groups, [member], default_models=[later_hidden])
    with admin.client() as client:
        assert client.get("/api/models").status_code == 200
    assert _starts_with(member) == [later_hidden]

    time.sleep(1.1)  # models are stamped to the second
    with admin.client() as client:
        stored = client.get(f"/api/v1/models/model?id={later_hidden}").json()
        hidden = client.post(
            f"/api/v1/models/model/update?id={later_hidden}",
            json={**stored, "meta": {**stored["meta"], "hidden": True}, "access_grants": [PUBLIC]},
        )
    assert hidden.status_code == 200, hidden.text

    assert _starts_with(member) == [MOCK_MODEL_ID]


def test_clearing_a_groups_defaults_gives_its_members_the_parents_again(admin, make_user, groups):
    member = make_user()
    parents, own = _preset(admin, "parents"), _preset(admin, "own")
    top = _group(admin, groups, [], default_models=[parents])
    leaf = _group(admin, groups, [member], default_models=[own], parent_id=top)
    assert _starts_with(member) == [own]

    cleared = _set_defaults(admin, leaf, [])

    assert cleared.status_code == 200, cleared.text
    assert _starts_with(member) == [parents]


def test_moving_a_group_changes_what_its_members_start_with(
    admin, make_user, groups, global_default
):
    member = make_user()
    inherited = _preset(admin, "inherited")
    global_default(MOCK_MODEL_ID)
    top = _group(admin, groups, [], default_models=[inherited])
    leaf = _group(admin, groups, [member])
    assert _starts_with(member) == [MOCK_MODEL_ID]

    assert move_group(admin, leaf, top).status_code == 200
    assert _starts_with(member) == [inherited]

    assert move_group(admin, leaf, None).status_code == 200
    assert _starts_with(member) == [MOCK_MODEL_ID]


def test_the_preview_says_where_the_effective_defaults_come_from(admin, groups):
    wanted = _preset(admin, "wanted")
    top = _group(admin, groups, [], default_models=[wanted])
    leaf = _group(admin, groups, [], parent_id=top)

    with admin.client() as client:
        leaf_preview = client.get(f"/api/v1/groups/id/{leaf}/preview").json()
        top_preview = client.get(f"/api/v1/groups/id/{top}/preview").json()

    assert leaf_preview["default_models"] == {
        "local": None,
        "effective": [wanted],
        "source_group_id": top,
    }
    assert top_preview["default_models"]["local"] == [wanted]
    assert top_preview["default_models"]["source_group_id"] == top


@pytest.mark.parametrize("bad", [[""], ["  "], "model-a", [1], {"a": 1}])
def test_default_models_that_are_not_model_ids_are_refused(admin, bad):
    form = {
        "name": f"bad {uuid.uuid4().hex[:6]}",
        "description": "",
        "data": {"config": {"default_models": bad}},
    }
    with admin.client() as client:
        refused = client.post("/api/v1/groups/create", json=form)

    assert refused.status_code == 422, refused.text


def test_default_models_are_trimmed_and_listed_once(admin, groups):
    group_id = make_group(admin, [])
    groups.append(group_id)

    saved = _set_defaults(admin, group_id, [" a-model ", "a-model", "b-model"])

    assert saved.status_code == 200, saved.text
    assert saved.json()["data"]["config"]["default_models"] == ["a-model", "b-model"]
