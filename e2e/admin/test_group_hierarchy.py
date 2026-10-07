"""Journey: an admin builds and reads a group hierarchy in Admin Panel > Users > Groups.

The list shows a subgroup under its parent as a tree: an arrow collapses and expands a parent's
subgroups, and a search for a path such as "Parent / Child" or for a child alone keeps the parents
of the match in view. In the group editor's General tab the Parent group dropdown sets a parent (or
No parent) and offers no group from the group's own branch; the New Group dialog has the same
dropdown. Dragging a group onto another makes it a subgroup and dragging it onto Move to top level
takes it out. The Users tab's Direct / Inherited menu splits the group's own members from inherited
ones with the subgroup each comes through, and the Permissions tab notes that parent permissions are
inherited and marks the ones that remain enabled. Deleting a parent names where its subgroups go and
they show at that level afterwards. The Default models field names where inherited defaults come
from and sets a group's own. In their own browser, a subgroup member finds a model shared only with
the parent group in the model selector, starts a new chat on the parent's default model, and an open
tab gains the shared model when the member joins the subgroup, without a reload. Each test works as
a fresh admin on groups of its own.

Discriminates: passes on dev b859124f9; in a frontend copy, the list ignoring parents (every
group top-level), the Parent group dropdown offering the group's own branch, the permissions
note and its parent-based marking gone, "Inherited via" gone, the inherited default models
naming the global defaults and the delete confirmation always saying "top-level" turn the tree,
search, dropdown, creation, drag, Users tab, Permissions tab, delete and Default models tests
red; in a backend copy, the update and create routes ignoring the parent turn the dropdown,
creation and drag tests red, subgroup deletion leaving subgroups top-level turns the middle
delete test red, group memberships ignoring parents turns the Users tab, shared model, starting
model and open tab tests red, and the access refresh message not being sent turns the open tab
test red.
"""

from __future__ import annotations

import re
import uuid
from typing import Callable, Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.access import grant
from harness.actors import Actor
from harness.python_tools import EVERYONE_READS
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import chat_input
from utils.model_selector import model_options, select_model

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def admin_page(make_user, page_for) -> Page:
    """A fresh admin's page, so nothing here touches the shared admin's settings."""
    return page_for(make_user(role="admin"))


@pytest.fixture
def group_ids(admin) -> Iterator[list[str]]:
    """Ids of the groups a test made, deleted afterwards (the ones already gone are skipped)."""
    ids: list[str] = []
    yield ids
    with admin.client() as client:
        for group_id in reversed(ids):
            client.delete(f"/api/v1/groups/id/{group_id}/delete")


@pytest.fixture
def make_group(admin, group_ids) -> Iterator[Callable[..., dict]]:
    """`make_group(name, parent=None, **fields)` creates a group over the API and returns it."""
    with admin.client() as client:

        def create(name: str, parent: dict | None = None, **fields) -> dict:
            body = {"name": name, "description": "", **fields}
            if parent:
                body["parent_group_id"] = parent["id"]
            response = client.post("/api/v1/groups/create", json=body)
            assert response.status_code == 200, response.text
            group_ids.append(response.json()["id"])
            return response.json()

        yield create


@pytest.fixture
def suffix() -> str:
    return uuid.uuid4().hex[:6]


@pytest.fixture
def lineage(make_group, suffix) -> dict[str, dict]:
    """Top > Middle > Leaf, three groups of one family, named apart from other tests' groups."""
    top = make_group(f"Top {suffix}")
    middle = make_group(f"Middle {suffix}", parent=top)
    leaf = make_group(f"Leaf {suffix}", parent=middle)
    return {"top": top, "middle": middle, "leaf": leaf}


def _group_rows(page: Page) -> Locator:
    return page.get_by_label("Group hierarchy").get_by_role("group")


def _row(page: Page, name: str) -> Locator:
    return page.get_by_label("Group hierarchy").get_by_role("group", name=name, exact=True)


def _open_groups(page: Page) -> None:
    page.goto("/admin/users/groups")
    expect(page.get_by_label("Group hierarchy")).to_be_visible()


def _search(page: Page, text: str) -> None:
    page.get_by_role("main").get_by_role("textbox", name="Search Groups").fill(text)


def _edit(page: Page, name: str) -> Locator:
    _row(page, name).get_by_role("button", name=re.compile(rf"^{name} \d+ direct members")).click()
    return page.get_by_role("dialog").filter(has_text="Edit User Group")


def _parent_dropdown(editing: Locator) -> Locator:
    return editing.get_by_label("Parent group")


def _parent_choices(page: Page) -> Locator:
    return page.get_by_role("menuitemradio")


def _stored_parent(admin: Actor, group: dict) -> str | None:
    with admin.client() as client:
        stored = client.get(f"/api/v1/groups/id/{group['id']}")
    stored.raise_for_status()
    return stored.json().get("parent_group_id")


def test_a_subgroup_appears_under_its_parent_and_the_arrow_folds_it(admin_page, lineage, suffix):
    _open_groups(admin_page)
    _search(admin_page, suffix)
    expect(_group_rows(admin_page)).to_have_text(
        [
            re.compile(rf"^\s*Top {suffix}"),
            re.compile(rf"^\s*Middle {suffix}"),
            re.compile(rf"^\s*Leaf {suffix}"),
        ]
    )

    admin_page.get_by_role("button", name="Clear search").click()
    admin_page.get_by_role("button", name=f"Collapse Middle {suffix}").click()
    expect(_row(admin_page, f"Leaf {suffix}")).to_have_count(0)
    expect(_row(admin_page, f"Middle {suffix}")).to_be_visible()

    admin_page.get_by_role("button", name=f"Collapse Top {suffix}").click()
    expect(_row(admin_page, f"Middle {suffix}")).to_have_count(0)

    admin_page.get_by_role("button", name=f"Expand Top {suffix}").click()
    expect(_row(admin_page, f"Middle {suffix}")).to_be_visible()
    expect(_row(admin_page, f"Leaf {suffix}")).to_have_count(0)
    admin_page.get_by_role("button", name=f"Expand Middle {suffix}").click()
    expect(_row(admin_page, f"Leaf {suffix}")).to_be_visible()


def test_searching_by_path_or_by_child_keeps_the_parents_of_a_match(
    admin_page, make_group, lineage, suffix
):
    make_group(f"Other {suffix}")
    _open_groups(admin_page)
    branch = [
        re.compile(rf"^\s*Top {suffix}"),
        re.compile(rf"^\s*Middle {suffix}"),
        re.compile(rf"^\s*Leaf {suffix}"),
    ]

    _search(admin_page, f"Middle {suffix} / Leaf {suffix}")
    expect(_group_rows(admin_page)).to_have_text(branch)

    _search(admin_page, f"Leaf {suffix}")
    expect(_group_rows(admin_page)).to_have_text(branch)
    expect(_row(admin_page, f"Other {suffix}")).to_have_count(0)


def test_the_parent_dropdown_sets_and_clears_a_parent(admin, admin_page, make_group, suffix):
    parent = make_group(f"Parent {suffix}")
    child = make_group(f"Child {suffix}")
    _open_groups(admin_page)
    _search(admin_page, suffix)

    editing = _edit(admin_page, f"Child {suffix}")
    expect(_parent_dropdown(editing)).to_have_text("No parent")
    _parent_dropdown(editing).click()
    admin_page.get_by_role("menuitemradio", name=f"Parent {suffix}").click()
    expect(_parent_dropdown(editing)).to_have_text(f"Parent {suffix}")
    editing.get_by_role("button", name="Save").click()
    expect(admin_page.get_by_text("Group updated successfully")).to_be_visible()
    assert _stored_parent(admin, child) == parent["id"]
    expect(_group_rows(admin_page)).to_have_text(
        [re.compile(rf"^\s*Parent {suffix}"), re.compile(rf"^\s*Child {suffix}")]
    )

    editing = _edit(admin_page, f"Child {suffix}")
    _parent_dropdown(editing).click()
    admin_page.get_by_role("menuitemradio", name="No parent").click()
    with admin_page.expect_response(lambda response: response.url.endswith("/update")):
        editing.get_by_role("button", name="Save").click()
    assert _stored_parent(admin, child) is None


def test_the_parent_dropdown_offers_no_group_from_the_groups_own_branch(
    admin_page, make_group, lineage, suffix
):
    make_group(f"Other {suffix}")
    _open_groups(admin_page)
    _search(admin_page, suffix)

    editing = _edit(admin_page, f"Middle {suffix}")
    _parent_dropdown(editing).click()
    admin_page.get_by_role("textbox", name="Search parent groups").fill(suffix)
    expect(_parent_choices(admin_page)).to_have_text(
        ["No parent", f"Other {suffix}", f"Top {suffix}"]
    )


def test_a_group_created_with_a_parent_lands_under_it(
    admin, admin_page, make_group, group_ids, suffix
):
    parent = make_group(f"Parent {suffix}")
    _open_groups(admin_page)
    admin_page.get_by_role("button", name="New Group").click()
    creating = admin_page.get_by_role("dialog").filter(has_text="Add User Group")
    creating.get_by_placeholder("Group Name").fill(f"Born {suffix}")
    _parent_dropdown(creating).click()
    admin_page.get_by_role("textbox", name="Search parent groups").fill(suffix)
    admin_page.get_by_role("menuitemradio", name=f"Parent {suffix}").click()
    creating.get_by_role("button", name="Save").click()
    expect(admin_page.get_by_text("Group created successfully")).to_be_visible()

    _search(admin_page, suffix)
    expect(_group_rows(admin_page)).to_have_text(
        [re.compile(rf"^\s*Parent {suffix}"), re.compile(rf"^\s*Born {suffix}")]
    )
    with admin.client() as client:
        groups = client.get("/api/v1/groups/").json()
    born = next(group for group in groups if group["name"] == f"Born {suffix}")
    group_ids.append(born["id"])
    assert born["parent_group_id"] == parent["id"]


def _drag_to_top_level(page: Page, name: str) -> None:
    """Drags the group's row onto Move to top level, which only shows once a drag has begun."""
    handle = _row(page, name).get_by_role("button", name=re.compile(rf"^{name} \d+ direct"))
    box = handle.bounding_box()
    page.mouse.move(box["x"] + 20, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.mouse.move(box["x"] + 40, box["y"] + box["height"] / 2 + 10, steps=5)
    target = page.get_by_role("region", name="Move to top level")
    expect(target).to_be_visible()
    target_box = target.bounding_box()
    page.mouse.move(
        target_box["x"] + target_box["width"] / 2,
        target_box["y"] + target_box["height"] / 2,
        steps=10,
    )
    page.mouse.up()


def test_dragging_a_group_onto_another_makes_it_a_subgroup(admin, admin_page, make_group, suffix):
    parent = make_group(f"Parent {suffix}")
    child = make_group(f"Child {suffix}")
    _open_groups(admin_page)
    _search(admin_page, suffix)

    _row(admin_page, f"Child {suffix}").get_by_role("button", name=re.compile("^Child")).drag_to(
        _row(admin_page, f"Parent {suffix}")
    )

    expect(admin_page.get_by_text("Group moved successfully")).to_be_visible()
    assert _stored_parent(admin, child) == parent["id"]
    expect(_group_rows(admin_page)).to_have_text(
        [re.compile(rf"^\s*Parent {suffix}"), re.compile(rf"^\s*Child {suffix}")]
    )


def test_dragging_a_subgroup_onto_move_to_top_level_takes_it_out(
    admin, admin_page, lineage, suffix
):
    _open_groups(admin_page)
    _search(admin_page, suffix)

    _drag_to_top_level(admin_page, f"Middle {suffix}")

    expect(admin_page.get_by_text("Group moved successfully")).to_be_visible()
    assert _stored_parent(admin, lineage["middle"]) is None
    expect(_group_rows(admin_page)).to_have_text(
        [
            re.compile(rf"^\s*Middle {suffix}"),
            re.compile(rf"^\s*Leaf {suffix}"),
            re.compile(rf"^\s*Top {suffix}"),
        ]
    )


def _add_member(admin: Actor, group: dict, member: Actor) -> None:
    with admin.client() as client:
        added = client.post(
            f"/api/v1/groups/id/{group['id']}/users/add", json={"user_ids": [member.id]}
        )
    assert added.status_code == 200, added.text


def test_the_users_tab_splits_direct_members_from_inherited_ones_with_their_subgroup(
    admin, admin_page, make_user, lineage, suffix
):
    direct = make_user(name=f"Direct {suffix}")
    from_middle = make_user(name=f"FromMiddle {suffix}")
    from_leaf = make_user(name=f"FromLeaf {suffix}")
    _add_member(admin, lineage["top"], direct)
    _add_member(admin, lineage["middle"], from_middle)
    _add_member(admin, lineage["leaf"], from_leaf)
    _open_groups(admin_page)
    _search(admin_page, suffix)

    editing = _edit(admin_page, f"Top {suffix}")
    editing.get_by_role("button", name="Users", exact=True).click()
    editing.get_by_role("textbox", name="Search", exact=True).fill(f"{suffix}")
    expect(editing.get_by_role("checkbox", name=direct.name)).to_have_attribute(
        "aria-checked", "true"
    )
    expect(editing.get_by_role("checkbox", name=from_middle.name)).to_have_attribute(
        "aria-checked", "false"
    )

    # since 71167131d the two lists are picked from a Direct / Inherited menu
    editing.get_by_role("button", name="Direct", exact=True).click()
    admin_page.get_by_role("button", name="Inherited", exact=True).click()
    inherited = editing.get_by_role("table")
    middle_row = inherited.get_by_role("row").filter(has_text=from_middle.name)
    expect(middle_row).to_contain_text(f"Top {suffix} / Middle {suffix}")
    leaf_row = inherited.get_by_role("row").filter(has_text=from_leaf.name)
    expect(leaf_row).to_contain_text(f"Top {suffix} / Middle {suffix} / Leaf {suffix}")
    expect(inherited.get_by_text(direct.name)).to_have_count(0)


def test_the_permissions_tab_notes_inheritance_and_marks_what_a_parent_keeps_enabled(
    admin_page, make_group, suffix
):
    top = make_group(f"Top {suffix}", permissions={"workspace": {"models": True}})
    middle = make_group(f"Middle {suffix}", parent=top)
    make_group(f"Leaf {suffix}", parent=middle)
    _open_groups(admin_page)
    _search(admin_page, suffix)
    note = "Parent permissions are inherited."
    kept = "This is a default user permission and will remain enabled."

    editing = _edit(admin_page, f"Leaf {suffix}")
    editing.get_by_role("button", name="Permissions", exact=True).click()
    expect(editing.get_by_text(note)).to_be_visible()
    models_access = editing.get_by_role("switch", name="Models Access")
    expect(models_access).to_have_attribute("aria-checked", "false")
    expect(
        models_access.locator("xpath=ancestor::div[contains(@class, 'flex-col')][1]")
    ).to_contain_text(kept)
    knowledge_access = editing.get_by_role("switch", name="Knowledge Access")
    expect(
        knowledge_access.locator("xpath=ancestor::div[contains(@class, 'flex-col')][1]")
    ).not_to_contain_text(kept)
    admin_page.keyboard.press("Escape")
    expect(editing).to_be_hidden()

    editing = _edit(admin_page, f"Top {suffix}")
    editing.get_by_role("button", name="Permissions", exact=True).click()
    expect(editing.get_by_role("switch", name="Models Access")).to_have_attribute(
        "aria-checked", "true"
    )
    expect(editing.get_by_text(note)).to_have_count(0)


def _confirm_deleting(page: Page, name: str) -> Locator:
    editing = _edit(page, name)
    editing.get_by_role("button", name="Delete", exact=True).click()
    return page.get_by_role("dialog").filter(has_text="Delete group")


def test_deleting_a_middle_group_moves_its_subgroups_up_a_level(admin, admin_page, lineage, suffix):
    _open_groups(admin_page)
    _search(admin_page, suffix)

    confirming = _confirm_deleting(admin_page, f"Middle {suffix}")
    expect(confirming).to_contain_text(f"Its child groups will move to Top {suffix}.")
    confirming.get_by_role("button", name="Confirm").click()

    expect(admin_page.get_by_text("Group deleted successfully")).to_be_visible()
    expect(_group_rows(admin_page)).to_have_text(
        [re.compile(rf"^\s*Top {suffix}"), re.compile(rf"^\s*Leaf {suffix}")]
    )
    assert _stored_parent(admin, lineage["leaf"]) == lineage["top"]["id"]


def test_deleting_a_top_level_group_makes_its_subgroups_top_level(
    admin, admin_page, lineage, suffix
):
    _open_groups(admin_page)
    _search(admin_page, suffix)

    confirming = _confirm_deleting(admin_page, f"Top {suffix}")
    expect(confirming).to_contain_text("Its child groups will become top-level groups.")
    confirming.get_by_role("button", name="Confirm").click()

    expect(admin_page.get_by_text("Group deleted successfully")).to_be_visible()
    expect(_group_rows(admin_page)).to_have_text(
        [re.compile(rf"^\s*Middle {suffix}"), re.compile(rf"^\s*Leaf {suffix}")]
    )
    assert _stored_parent(admin, lineage["middle"]) is None


@pytest.fixture
def make_preset(admin, suffix) -> Iterator[Callable[..., dict]]:
    """`make_preset(label, grants)` adds a preset of the scripted model; gives its id and name."""
    ids: list[str] = []
    with admin.client() as client:

        def create(label: str, grants: list[dict]) -> dict:
            model_id = f"{label.lower()}-{suffix}"
            name = f"{label} {suffix}"
            form = {
                "id": model_id,
                "base_model_id": MOCK_MODEL_ID,
                "name": name,
                "meta": {},
                "params": {},
                "access_grants": grants,
            }
            created = client.post("/api/v1/models/create", json=form)
            assert created.status_code == 200, created.text
            ids.append(model_id)
            return {"id": model_id, "name": name}

        yield create
        for model_id in ids:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


def test_a_subgroup_member_sees_a_model_shared_only_with_the_parent_group(
    admin, page_for, make_user, make_group, make_preset, lineage
):
    shared = make_preset("Shared", [grant("group", lineage["top"]["id"], "read")])
    member = make_user()
    outsider = make_user()
    _add_member(admin, lineage["leaf"], member)

    member_page = page_for(member)
    expect(chat_input(member_page)).to_be_visible()
    expect(model_options(member_page, shared["name"])).to_have_count(1)

    outsider_page = page_for(outsider)
    expect(chat_input(outsider_page)).to_be_visible()
    expect(model_options(outsider_page, MOCK_MODEL_ID)).to_have_count(1)
    expect(model_options(outsider_page, shared["name"])).to_have_count(0)


def _default_models(admin: Actor, group: dict) -> list[str]:
    with admin.client() as client:
        stored = client.get(f"/api/v1/groups/id/{group['id']}")
    stored.raise_for_status()
    return ((stored.json().get("data") or {}).get("config") or {}).get("default_models") or []


def test_a_new_chat_starts_on_the_default_model_of_a_parent_group(
    admin, page_for, make_user, make_group, make_preset, suffix
):
    preset = make_preset("Starter", [EVERYONE_READS])
    top = make_group(f"Top {suffix}", data={"config": {"default_models": [preset["id"]]}})
    leaf = make_group(f"Leaf {suffix}", parent=top)
    member = make_user()
    outsider = make_user()
    _add_member(admin, leaf, member)

    member_page = page_for(member)
    expect(
        member_page.get_by_role("button", name=f"Selected model: {preset['name']}")
    ).to_be_visible()

    outsider_page = page_for(outsider)
    expect(outsider_page.get_by_role("button", name=re.compile("^Selected model"))).to_be_visible()
    expect(
        outsider_page.get_by_role("button", name=f"Selected model: {preset['name']}")
    ).to_have_count(0)


def test_the_default_models_field_shows_where_the_defaults_come_from_and_sets_its_own(
    admin, admin_page, make_group, make_preset, suffix
):
    inherited = make_preset("Inherited", [EVERYONE_READS])
    own = make_preset("Own", [EVERYONE_READS])
    top = make_group(f"Top {suffix}", data={"config": {"default_models": [inherited["id"]]}})
    middle = make_group(f"Middle {suffix}", parent=top)
    _open_groups(admin_page)
    _search(admin_page, suffix)

    editing = _edit(admin_page, f"Middle {suffix}")
    expect(editing.get_by_text(f"Top {suffix}: {inherited['name']}")).to_be_visible()

    editing.get_by_role("button", name="Inherit", exact=True).click()
    select_model(admin_page, own["name"])
    editing.get_by_text("Edit User Group").click()
    expect(editing.get_by_role("button", name="Inherit", exact=True)).to_be_visible()
    expect(editing.get_by_text(f"Top {suffix}: {inherited['name']}")).to_have_count(0)
    editing.get_by_role("button", name="Save").click()
    expect(admin_page.get_by_text("Group updated successfully")).to_be_visible()
    assert _default_models(admin, middle) == [own["id"]]

    editing = _edit(admin_page, f"Middle {suffix}")
    editing.get_by_role("button", name="Inherit", exact=True).click()
    expect(editing.get_by_text(f"Top {suffix}: {inherited['name']}")).to_be_visible()
    with admin_page.expect_response(lambda response: response.url.endswith("/update")):
        editing.get_by_role("button", name="Save").click()
    assert _default_models(admin, middle) == []


def test_an_open_tab_picks_up_a_group_change_without_a_reload(
    admin, page_for, make_user, make_preset, lineage
):
    shared = make_preset("Shared", [grant("group", lineage["top"]["id"], "read")])
    member = make_user()
    member_page = page_for(member)
    expect(chat_input(member_page)).to_be_visible()
    expect(model_options(member_page, shared["name"])).to_have_count(0)

    _add_member(admin, lineage["leaf"], member)

    expect(model_options(member_page, shared["name"])).to_have_count(1)
