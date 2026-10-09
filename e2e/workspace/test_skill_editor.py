"""Journey: a skill written, changed and deleted in the workspace, as a chat mention sees it.

A fresh admin writes a skill in Workspace > Skills. Mentioning it with `$` in a chat puts its
instructions in the system prompt the model is sent. After its instructions are changed in the
editor, the next mention sends the new ones and none of the old. Deleted from the list's menu,
it leaves the list and the `$` picker, while a skill next to it stays.

Discriminates: passes on dev 176d31d1d. In a backend copy where `/api/v1/skills/create` drops
the content, where `/api/v1/skills/id/{id}/update` leaves the content as it was and where the
delete answers without deleting, the matching test goes red. Retargeted for 9bbb95048, where the
instructions became the skill's SKILL.md in the file editor and Save & Create opens the saved
skill; passes on dev 178de3666.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.chat_ui import chat_input, expect_reply
from utils.skill_editor import code_editor, replace_text

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def skill_writer(make_user):
    """A fresh admin; the skills it made are deleted afterwards."""
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for skill in client.get("/api/v1/skills/").json():
            if skill["user_id"] == account.id:
                client.delete(f"/api/v1/skills/id/{skill['id']}/delete")


def _write_skill(page: Page, name: str, instructions: str) -> None:
    page.goto("/workspace/skills")
    page.get_by_role("main").get_by_role("button", name="Create", exact=True).click()
    expect(page).to_have_url(re.compile(r"/workspace/skills/create$"))
    editor = page.get_by_role("main")
    editor.get_by_placeholder("Skill Name").fill(name)
    editor.get_by_placeholder("Skill Description").fill("how to answer")
    replace_text(page, instructions)
    editor.get_by_role("button", name="Save & Create").click()
    expect(page).to_have_url(re.compile(r"/workspace/skills/edit\?id="))


def _skill_row(page: Page, name: str) -> Locator:
    page.goto("/workspace/skills")
    page.get_by_role("textbox", name="Search Skills").fill(name)
    row = page.get_by_role("main").get_by_role("button").filter(has_text=name)
    expect(row).to_be_visible()
    return row


def _skill_menu_item(page: Page, name: str, item: str) -> None:
    row = _skill_row(page, name)
    row.hover()
    row.get_by_role("button", name="Skill Menu").last.click()
    page.get_by_role("button", name=item, exact=True).click()


def _dollar_picker(page: Page, query: str) -> Locator:
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    chat_input(page).click()
    page.keyboard.type(f"${query}")
    return page.get_by_role("button")


def _system_prompt_after_mention(page: Page, upstream, name: str) -> str:
    question = f"what now? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("On it.", match=reply.answering(question)))
    _dollar_picker(page, name.split()[0]).filter(has_text=name).click()
    page.keyboard.type(question)
    page.keyboard.press("Enter")
    expect_reply(page, "On it.")
    request = next(filter(reply.answering(question), upstream.chat_requests()))
    return "\n".join(
        str(entry["content"]) for entry in request["messages"] if entry["role"] == "system"
    )


def test_a_skill_written_in_the_editor_reaches_the_model_when_mentioned(
    page_for, skill_writer, upstream
):
    name = f"Bullet {uuid.uuid4().hex[:6]}"
    page = page_for(skill_writer)
    _write_skill(page, name, "Answer in exactly three bullet points.")

    system = _system_prompt_after_mention(page, upstream, name)

    assert "Answer in exactly three bullet points." in system


def test_changed_instructions_replace_the_old_ones_in_the_next_mention(
    page_for, skill_writer, upstream
):
    name = f"Tone {uuid.uuid4().hex[:6]}"
    page = page_for(skill_writer)
    _write_skill(page, name, "Answer formally.")

    _skill_menu_item(page, name, "Edit")
    expect(page).to_have_url(re.compile(r"/workspace/skills/edit\?id="))
    expect(code_editor(page)).to_contain_text("Answer formally.")
    replace_text(page, "Answer like a sailor.")
    page.get_by_role("main").get_by_role("button", name="Save", exact=True).click()
    expect(page.get_by_text("Skill updated successfully")).to_be_visible()

    system = _system_prompt_after_mention(page, upstream, name)

    assert "Answer like a sailor." in system
    assert "Answer formally." not in system


def test_a_deleted_skill_leaves_the_list_and_the_dollar_picker(page_for, skill_writer):
    tag = uuid.uuid4().hex[:6]
    kept, deleted = f"Kept {tag}", f"Gone {tag}"
    page = page_for(skill_writer)
    _write_skill(page, kept, "Stay.")
    _write_skill(page, deleted, "Leave.")

    _skill_menu_item(page, deleted, "Delete")
    page.get_by_role("dialog", name="Delete skill?").get_by_role("button", name="Confirm").click()
    expect(page.get_by_role("main").get_by_role("button").filter(has_text=deleted)).to_have_count(0)

    picker = _dollar_picker(page, tag)
    expect(picker.filter(has_text=kept)).to_be_visible()
    expect(picker.filter(has_text=deleted)).to_have_count(0)
