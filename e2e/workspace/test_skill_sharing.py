"""Journey: a skill shared in its Access dialog, as the accounts it reaches and misses see it.

An admin writes a skill and shares it from the skill editor's Access dialog. Shared with a group
for reading, it is offered by the `$` menu to the group's member, whose mention puts its
instructions in the system prompt the model is sent, and an outsider is never offered it.
Taking the group off the list again withdraws it from the member. A user given write access in
the same dialog changes its instructions in the editor and the owner's next mention sends the
new ones, while a reader opens the editor read only, without a Save. Who the API lets do what
with a shared skill is integration/security/test_skill_access_matrix.py.

Discriminates: passes on the dev ebc6add67 build; in a backend copy whose access update stores
no grants every test here goes red (the member is never offered the skill, the writer meets a
read-only editor); in one whose access update only adds grants, the withdraw test goes red (the
member is still offered the skill) and the other two pass. Retargeted for 9bbb95048, where the
instructions became SKILL.md in the skill's file editor; passes on dev 178de3666.
"""

from __future__ import annotations

import uuid
from typing import Callable, Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.access import make_group
from harness.actors import Actor
from utils.chat_ui import chat_input, expect_reply
from utils.skill_editor import code_editor, replace_text

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

SKILLS_WORKSPACE = {"workspace": {"skills": True}}


def _unique(prefix: str) -> str:
    return f"{prefix} {uuid.uuid4().hex[:6]}"


@pytest.fixture
def owner(make_user) -> Iterator[Actor]:
    """A fresh admin; the skills it made are deleted afterwards."""
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for skill in client.get("/api/v1/skills/").json():
            if skill["user_id"] == account.id:
                client.delete(f"/api/v1/skills/id/{skill['id']}/delete")


@pytest.fixture
def group_of(admin) -> Iterator[Callable[..., tuple[str, str]]]:
    """`group_of(*members)` is a new group's (id, name); the groups are deleted afterwards."""
    made: list[str] = []

    def create(*members: Actor) -> tuple[str, str]:
        group_id = make_group(admin, list(members), permissions=SKILLS_WORKSPACE)
        made.append(group_id)
        with admin.client() as client:
            name = client.get(f"/api/v1/groups/id/{group_id}").json()["name"]
        return group_id, name

    yield create
    with admin.client() as client:
        for group_id in made:
            client.delete(f"/api/v1/groups/id/{group_id}/delete")


def _skill(owner: Actor, instructions: str) -> dict:
    skill = {"id": f"knots-{uuid.uuid4().hex[:8]}", "name": _unique("Knots")}
    with owner.client() as client:
        created = client.post(
            "/api/v1/skills/create",
            json={**skill, "description": "tying knots", "content": instructions, "meta": {}},
        )
    assert created.status_code == 200, created.text
    return skill


def _suffix(skill: dict) -> str:
    return skill["name"].split()[1]


def _access_dialog(page: Page, skill: dict) -> Locator:
    page.goto(f"/workspace/skills/edit?id={skill['id']}")
    page.get_by_role("main").get_by_role("button", name="Access").click()
    dialog = page.get_by_role("dialog").filter(has_text="Access Control")
    expect(dialog).to_be_visible()
    return dialog


def _add_access(page: Page, dialog: Locator, name: str) -> None:
    dialog.get_by_role("button", name="Add Access").click()
    picker = page.get_by_role("dialog").filter(has_text="Add Access").last
    picker.get_by_placeholder("Search").fill(name)
    picker.get_by_role("button", name=name).click()
    picker.get_by_role("button", name="Add", exact=True).click()
    expect(page.get_by_text("Saved").first).to_be_visible()
    expect(dialog.get_by_text(name)).to_be_visible()


def _principal_row(dialog: Locator, name: str) -> Locator:
    return dialog.get_by_role("combobox", name="Access level").locator(
        f"xpath=ancestor::div[contains(@class, 'justify-between')][.//*[contains(., '{name}')]][1]"
    )


def _dollar_menu(page: Page, typed: str) -> Locator:
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    chat_input(page).click()
    # the composer keeps a draft across reloads
    page.keyboard.press("ControlOrMeta+a")
    page.keyboard.press("Backspace")
    with page.expect_response(lambda response: f"/skills/list?query={typed}" in response.url):
        page.keyboard.type(f"${typed}")
    return page.get_by_role("button")


def _mention_and_ask(page: Page, upstream, skill: dict) -> str:
    """Mention the skill from the `$` menu, ask, and return the system prompt the model got."""
    question = f"which knot? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("A bowline.", match=reply.answering(question)))
    _dollar_menu(page, _suffix(skill)).filter(has_text=skill["name"]).click()
    page.keyboard.type(question)
    page.keyboard.press("Enter")
    expect_reply(page, "A bowline.")
    request = next(filter(reply.answering(question), upstream.chat_requests()))
    return "\n".join(
        str(entry["content"]) for entry in request["messages"] if entry["role"] == "system"
    )


def test_a_skill_shared_with_a_group_reaches_its_member_and_not_an_outsider(
    page_for, owner, make_user, group_of, upstream
):
    member, outsider = make_user(), make_user()
    _, group_name = group_of(member)
    skill = _skill(owner, "Tie a bowline for a fixed loop.")
    page = page_for(owner)

    _add_access(page, _access_dialog(page, skill), group_name)

    member_page = page_for(member)
    assert "Tie a bowline for a fixed loop." in _mention_and_ask(member_page, upstream, skill)
    outsider_menu = _dollar_menu(page_for(outsider), _suffix(skill))
    expect(outsider_menu.filter(has_text=skill["name"])).to_have_count(0)


def test_taking_the_group_off_the_access_list_withdraws_the_skill(
    page_for, owner, make_user, group_of
):
    member = make_user()
    _, group_name = group_of(member)
    skill = _skill(owner, "Tie a clove hitch to a post.")
    page = page_for(owner)
    dialog = _access_dialog(page, skill)
    _add_access(page, dialog, group_name)
    member_page = page_for(member)
    expect(_dollar_menu(member_page, _suffix(skill)).filter(has_text=skill["name"])).to_be_visible()

    _principal_row(dialog, group_name).get_by_role("button").last.click()
    expect(page.get_by_text("Saved").last).to_be_visible()
    expect(dialog.get_by_text("No access grants. Private to you.")).to_be_visible()

    expect(_dollar_menu(member_page, _suffix(skill)).filter(has_text=skill["name"])).to_have_count(
        0
    )


def test_a_writer_changes_the_instructions_and_a_reader_meets_a_read_only_editor(
    page_for, owner, make_user, group_of, upstream
):
    writer, reader = make_user(), make_user()
    group_of(writer, reader)
    skill = _skill(owner, "Tie a reef knot.")
    page = page_for(owner)
    dialog = _access_dialog(page, skill)
    _add_access(page, dialog, writer.name)
    _principal_row(dialog, writer.name).get_by_role("combobox").select_option("write")
    expect(page.get_by_text("Saved").last).to_be_visible()
    _add_access(page, dialog, reader.name)

    writer_page = page_for(writer)
    writer_page.goto(f"/workspace/skills/edit?id={skill['id']}")
    expect(code_editor(writer_page)).to_contain_text("Tie a reef knot.")
    replace_text(writer_page, "Tie a sheet bend.")
    writer_page.get_by_role("main").get_by_role("button", name="Save", exact=True).click()
    expect(writer_page.get_by_text("Skill updated successfully")).to_be_visible()

    reader_page = page_for(reader)
    reader_page.goto(f"/workspace/skills/edit?id={skill['id']}")
    expect(reader_page.get_by_text("Read Only", exact=True)).to_be_visible()
    expect(reader_page.get_by_text("Tie a sheet bend.")).to_be_visible()
    expect(code_editor(reader_page)).to_have_attribute("contenteditable", "false")
    expect(reader_page.get_by_role("main").get_by_role("button", name="Save")).to_have_count(0)

    system = _mention_and_ask(page, upstream, skill)
    assert "Tie a sheet bend." in system
    assert "Tie a reef knot." not in system
