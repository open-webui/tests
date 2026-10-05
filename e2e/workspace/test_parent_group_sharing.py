"""Journey: what a subgroup member meets in the browser when something is shared with the parent.

A parent group holds a subgroup with one member, and an outsider belongs to neither. A note, a
prompt, a knowledge base, a tool and a channel shared only with the parent group reach the
subgroup's member where a person uses them: the note opens from the Notes page, the prompt is
offered by `/`, the knowledge base by `#`, the tool under Integrations and the channel, shared to
write, opens and takes a message. The note is gone once the member leaves the subgroup and the
prompt once the subgroup is taken out of the parent. A knowledge base an admin shares with the
parent group in the Access dialog reaches the subgroup's member, and a workspace permission
switched on for the parent lets them create a prompt there. A sharing permission held through
the parent lets the member share their own note with a parent whose sharing is set to Members,
and a member of another subgroup then reads it. Moving the member's group under the parent in the
admin's group editor gives them what the parent was shared.

Discriminates: passes on dev 30f3f6a8f; in a backend copy, making `user_group_memberships`
ignore `include_inherited` turns every test here red (only direct groups count, so nothing the
parent was given or allowed reaches the subgroup's member); in another, an edit with a null
parent keeping the old one and removing a member from a group leaving them in it turns the
prompt and note tests red (the prompt is still offered and the note still listed afterwards).
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.access import grant, make_group
from harness.actors import Actor
from harness.channel_quotes import enable_channels
from harness.group_tree import move_group
from harness.python_tools import EVERYONE_READS
from utils.chat_ui import chat_input

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

TOOL_SOURCE = '''class Tools:
    def ping(self) -> str:
        """Answer pong."""
        return "pong"
'''


@dataclass
class Family:
    parent: str
    child: str
    member: Actor
    outsider: Actor


def _unique(prefix: str) -> str:
    return f"{prefix} {uuid.uuid4().hex[:6]}"


@pytest.fixture
def group_ids(admin) -> Iterator[list[str]]:
    """Ids of the groups a test made, deleted afterwards."""
    ids: list[str] = []
    yield ids
    with admin.client() as client:
        for group_id in reversed(ids):
            client.delete(f"/api/v1/groups/id/{group_id}/delete")


@pytest.fixture
def family(admin, make_user, group_ids) -> Family:
    member, outsider = make_user(), make_user()
    parent = make_group(admin, [])
    child = make_group(admin, [member], parent_id=parent)
    group_ids.extend([parent, child])
    return Family(parent, child, member, outsider)


def _created(response) -> str:
    assert response.status_code == 200, response.text
    return response.json()["id"]


def _parent_reads(family: Family) -> list[dict]:
    return [grant("group", family.parent, "read")]


def _leave_child_group(admin: Actor, family: Family) -> None:
    with admin.client() as client:
        removed = client.post(
            f"/api/v1/groups/id/{family.child}/users/remove",
            json={"user_ids": [family.member.id]},
        )
    assert removed.status_code == 200, removed.text


def _slash_menu(page: Page, typed: str) -> Locator:
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    chat_input(page).click()
    # the composer keeps a draft across reloads
    page.keyboard.press("ControlOrMeta+a")
    page.keyboard.press("Backspace")
    page.keyboard.type(typed)
    expect(chat_input(page)).to_have_text(typed)
    return page.get_by_role("tooltip")


def _read_only_note(page: Page, title: str) -> Locator:
    """The row of a note the account can only read, on the Notes page's Read Only list."""
    page.goto("/notes")
    main = page.get_by_role("main")
    main.get_by_role("button", name="Write", exact=True).click()
    page.get_by_role("button", name="Read Only", exact=True).click()
    expect(main.get_by_role("button", name="Read Only", exact=True)).to_be_visible()
    return main.get_by_role("button", name="Open note").filter(has_text=title)


def test_a_note_shared_with_the_parent_opens_for_a_subgroup_member_until_they_leave(
    admin, family, page_for
):
    title = _unique("Harbour rota")
    with admin.client() as client:
        _created(
            client.post(
                "/api/v1/notes/create",
                json={
                    "title": title,
                    "data": {"content": {"md": "Ada takes the night watch."}},
                    "access_grants": _parent_reads(family),
                },
            )
        )

    page = page_for(family.member)
    _read_only_note(page, title).click()
    expect(page.get_by_text("Ada takes the night watch.")).to_be_visible()
    expect(page.get_by_role("main").get_by_text("Read-Only Access")).to_be_visible()

    outsider_page = page_for(family.outsider)
    expect(_read_only_note(outsider_page, title)).to_have_count(0)

    _leave_child_group(admin, family)
    expect(_read_only_note(page, title)).to_have_count(0)


def _create_prompt(admin: Actor, command: str, grants: list[dict]) -> None:
    with admin.client() as client:
        _created(
            client.post(
                "/api/v1/prompts/create",
                json={
                    "command": command,
                    "name": command,
                    "content": "Who takes the night watch?",
                    "access_grants": grants,
                },
            )
        )


def _offered_prompt(page: Page, token: str, command: str) -> Locator:
    """`command` in the `/` list for `token`, once the prompt everyone reads under it is listed."""
    menu = _slash_menu(page, f"/{token}")
    expect(menu.get_by_role("button", name=f"open-{token}")).to_be_visible()
    return menu.get_by_role("button", name=command)


def test_a_prompt_shared_with_the_parent_is_offered_until_the_subgroup_is_detached(
    admin, family, page_for
):
    token = uuid.uuid4().hex[:8]
    _create_prompt(admin, f"rota-{token}", _parent_reads(family))
    _create_prompt(admin, f"open-{token}", [EVERYONE_READS])

    page = page_for(family.member)
    expect(_offered_prompt(page, token, f"rota-{token}")).to_be_visible()
    outsider_page = page_for(family.outsider)
    expect(_offered_prompt(outsider_page, token, f"rota-{token}")).to_have_count(0)

    moved = move_group(admin, family.child, None)
    assert moved.status_code == 200, moved.text
    expect(_offered_prompt(page, token, f"rota-{token}")).to_have_count(0)


@pytest.fixture
def curator(make_user) -> Iterator[Actor]:
    """A fresh admin, whose knowledge bases are deleted afterwards."""
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for knowledge in client.get("/api/v1/knowledge/").json().get("items", []):
            if knowledge["user_id"] == account.id:
                client.delete(f"/api/v1/knowledge/{knowledge['id']}/delete")


def _group_name(admin: Actor, group_id: str) -> str:
    with admin.client() as client:
        group = client.get(f"/api/v1/groups/id/{group_id}")
    assert group.status_code == 200, group.text
    return group.json()["name"]


def test_a_knowledge_base_shared_with_the_parent_in_the_access_dialog_reaches_a_subgroup_member(
    admin, curator, family, page_for
):
    name = _unique("Lighthouse logs")
    with curator.client() as client:
        knowledge_id = _created(
            client.post("/api/v1/knowledge/create", json={"name": name, "description": ""})
        )
    parent_name = _group_name(admin, family.parent)

    page = page_for(curator)
    page.goto(f"/workspace/knowledge/{knowledge_id}")
    page.get_by_role("main").get_by_role("button", name="Access").click()
    dialog = page.get_by_role("dialog").filter(has_text="Access Control")
    dialog.get_by_role("button", name="Add Access").click()
    picker = page.get_by_role("dialog").filter(has_text="Add Access").last
    picker.get_by_placeholder("Search").fill(parent_name)
    picker.get_by_role("button", name=parent_name).click()
    picker.get_by_role("button", name="Add", exact=True).click()
    expect(page.get_by_text("Saved").first).to_be_visible()
    expect(dialog.get_by_text(parent_name)).to_be_visible()

    member_page = page_for(family.member)
    offered = _slash_menu(member_page, "#Lighthouse").get_by_role("button", name=name)
    expect(offered).to_be_visible()
    outsider_page = page_for(family.outsider)
    expect(
        _slash_menu(outsider_page, "#Lighthouse").get_by_role("button", name=name)
    ).to_have_count(0)


def test_a_tool_shared_with_the_parent_is_listed_under_integrations_for_a_subgroup_member(
    admin, family, page_for
):
    name = _unique("Tide table")
    tool_id = f"tide_{uuid.uuid4().hex[:6]}"
    with admin.client() as client:
        _created(
            client.post(
                "/api/v1/tools/create",
                json={
                    "id": tool_id,
                    "name": name,
                    "content": TOOL_SOURCE,
                    "meta": {"description": "tides"},
                    "access_grants": _parent_reads(family),
                },
            )
        )
    try:
        page = page_for(family.member)
        expect(chat_input(page)).to_be_visible()
        page.get_by_label("Integrations").click()
        page.get_by_role("button", name=re.compile(r"^Tools")).click()
        expect(page.get_by_role("button", name=name)).to_be_visible()
    finally:
        with admin.client() as client:
            client.delete(f"/api/v1/tools/id/{tool_id}/delete")


@pytest.fixture
def channels_on(admin, preserve):
    preserve("admin_config")
    enable_channels(admin)


def test_a_channel_shared_with_the_parent_takes_a_subgroup_members_message(
    admin, channels_on, family, page_for
):
    name = f"harbour-{uuid.uuid4().hex[:6]}"
    parent_writes = [grant("group", family.parent, "read"), grant("group", family.parent, "write")]
    with admin.client() as client:
        channel_id = _created(
            client.post(
                "/api/v1/channels/create",
                json={"name": name, "type": None, "access_grants": parent_writes},
            )
        )

    page = page_for(family.member)
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    sidebar = page.get_by_role("navigation", name="Chat history")
    sidebar.get_by_role("button", name="Channels", exact=True).click()
    sidebar.get_by_role("link", name=name).click()
    expect(page).to_have_url(re.compile(f"/channels/{channel_id}$"))
    chat_input(page).click()
    page.keyboard.type("the tide turns at six")
    with page.expect_response(lambda response: response.url.endswith("/messages/post")) as posted:
        page.keyboard.press("Enter")
    assert posted.value.status == 200
    expect(page.get_by_role("main").get_by_text("the tide turns at six")).to_be_visible()

    with admin.client() as client:
        messages = client.get(f"/api/v1/channels/{channel_id}/messages").json()
    assert [message["content"] for message in messages] == ["the tide turns at six"]

    outsider_page = page_for(family.outsider)
    outsider_page.goto(f"/channels/{channel_id}")
    expect(chat_input(outsider_page)).to_be_visible()
    expect(outsider_page).not_to_have_url(re.compile("/channels/"))


def test_a_workspace_permission_on_the_parent_lets_a_subgroup_member_create_a_prompt(
    admin, make_user, group_ids, page_for
):
    member, outsider = make_user(), make_user()
    parent = make_group(admin, [], {"workspace": {"prompts": True}})
    child = make_group(admin, [member], parent_id=parent)
    group_ids.extend([parent, child])
    command = f"tide{uuid.uuid4().hex[:6]}"

    page = page_for(member)
    page.goto("/workspace/prompts")
    page.get_by_role("main").get_by_role("button", name="Create", exact=True).click()
    creating = page.get_by_role("dialog").filter(has_text="Create Prompt")
    creating.get_by_role("textbox", name="Name", exact=True).fill("Tide check")
    creating.get_by_role("textbox", name="Command").fill(command)
    creating.get_by_role("textbox", name=re.compile("^Write a summary in 50 words")).fill(
        "When is high tide?"
    )
    creating.get_by_role("button", name="Save & Create").click()
    expect(page.get_by_role("main").get_by_text(f"/{command}")).to_be_visible()
    expect(page).to_have_url(re.compile("/workspace/prompts"))

    outsider_page = page_for(outsider)
    outsider_page.goto("/workspace/prompts")
    expect(chat_input(outsider_page)).to_be_visible()
    expect(outsider_page).not_to_have_url(re.compile("/workspace"))


def test_a_subgroup_member_shares_a_note_with_a_members_only_parent(
    admin, make_user, group_ids, page_for
):
    author, sibling, outsider = make_user(), make_user(), make_user()
    parent = make_group(admin, [], {"sharing": {"notes": True}})
    with admin.client() as client:
        stored = client.get(f"/api/v1/groups/id/{parent}").json()
        restricted = client.post(
            f"/api/v1/groups/id/{parent}/update",
            json={
                "name": stored["name"],
                "description": stored["description"],
                "data": {"config": {"share": "members"}},
            },
        )
    assert restricted.status_code == 200, restricted.text
    first_child = make_group(admin, [author], parent_id=parent)
    second_child = make_group(admin, [sibling], parent_id=parent)
    group_ids.extend([parent, first_child, second_child])
    parent_name = stored["name"]
    title = _unique("Mooring plan")
    with author.client() as client:
        note_id = _created(
            client.post(
                "/api/v1/notes/create",
                json={"title": title, "data": {"content": {"md": "Berth 4 is free."}}},
            )
        )

    page = page_for(author)
    page.goto(f"/notes/{note_id}")
    page.get_by_role("main").get_by_role("button", name="Access").click()
    panel = page.get_by_role("dialog").filter(has_text="Access Control")
    panel.get_by_role("button", name="Add Access").click()
    picker = page.get_by_role("dialog").filter(has=page.get_by_role("button", name="Add"))
    picker.get_by_placeholder("Search").fill(parent_name)
    picker.get_by_role("button", name=parent_name).click()
    picker.get_by_role("button", name="Add", exact=True).click()
    expect(page.get_by_text("Saved", exact=True)).to_be_visible()

    sibling_page = page_for(sibling)
    _read_only_note(sibling_page, title).click()
    expect(sibling_page.get_by_text("Berth 4 is free.")).to_be_visible()

    with outsider.client() as client:
        offered = client.get("/api/v1/groups/", params={"share": "true"}).json()
    assert parent not in [group["id"] for group in offered]


def test_moving_a_group_under_the_parent_in_the_editor_gives_its_member_the_parents_prompt(
    admin, make_user, group_ids, page_for
):
    member = make_user()
    parent = make_group(admin, [])
    loose = make_group(admin, [member])
    group_ids.extend([parent, loose])
    parent_name, loose_name = _group_name(admin, parent), _group_name(admin, loose)
    token = uuid.uuid4().hex[:8]
    _create_prompt(admin, f"watch-{token}", [grant("group", parent, "read")])
    _create_prompt(admin, f"open-{token}", [EVERYONE_READS])
    member_page = page_for(member)
    expect(_offered_prompt(member_page, token, f"watch-{token}")).to_have_count(0)

    admin_page = page_for(make_user(role="admin"))
    admin_page.goto("/admin/users/groups")
    hierarchy = admin_page.get_by_label("Group hierarchy")
    admin_page.get_by_role("main").get_by_role("textbox", name="Search Groups").fill(loose_name)
    hierarchy.get_by_role("group", name=loose_name, exact=True).get_by_role(
        "button", name=re.compile(rf"^{loose_name} \d+ direct members")
    ).click()
    editing = admin_page.get_by_role("dialog").filter(has_text="Edit User Group")
    editing.get_by_label("Parent group").click()
    admin_page.get_by_role("textbox", name="Search parent groups").fill(parent_name)
    admin_page.get_by_role("menuitemradio", name=parent_name).click()
    editing.get_by_role("button", name="Save").click()
    expect(admin_page.get_by_text("Group updated successfully")).to_be_visible()
    expect(_offered_prompt(member_page, token, f"watch-{token}")).to_be_visible()
