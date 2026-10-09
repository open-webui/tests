"""Journey: skills brought in as files, sent out one at a time, and named in another language.

A SKILL.md picked through the Skills page's Import is listed in the import dialog under the name
and id from its front matter; Import selected saves it with that description and the whole file
as the instructions, which a `$` mention sends to the model. A JSON file holding several skills
imports every one of them. A row's own Export JSON downloads that skill alone, with its
instructions. A German name and description given in the editor's Editing language show in place
of the default ones once the interface is German, in the list and in the `$` menu, and the
English interface keeps the default name; the German name also finds the skill in the list's
search. The round trip through Export JSON and Import JSON is
e2e/workspace/test_workspace_import_export.py.

The German-name search test is red on purpose: the Translations docs page says Workspace search
matches the translated name as well as the original, but the skill list's search goes to the
server, which matches only the original name, description and id, so typing the name a German
user is shown finds nothing (open-webui/open-webui#32018).

Discriminates: passes on the dev ebc6add67 build apart from the search test; in a frontend copy
that never resolves a translated name the German name test goes red. Retargeted for 9bbb95048,
where import moved into a dialog backed by the server and Export became Export JSON with the
skill's files: passes on dev 178de3666 apart from the search test; in a backend copy whose import
ignores the front matter the SKILL.md test goes red (the skill is named "Imported skill"), and in
one whose export leaves out the files the export test goes red.
"""

from __future__ import annotations

import json
import uuid
from typing import Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.actors import Actor
from utils.chat_ui import chat_input

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

GERMAN_SEARCH = "Skills durchsuchen"


@pytest.fixture
def keeper(make_user) -> Iterator[Actor]:
    """A fresh admin; the skills it made are deleted afterwards."""
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for skill in client.get("/api/v1/skills/").json():
            if skill["user_id"] == account.id:
                client.delete(f"/api/v1/skills/id/{skill['id']}/delete")


def _create(keeper: Actor, name: str, instructions: str, **fields) -> str:
    skill_id = f"skill-{uuid.uuid4().hex[:8]}"
    form = {"id": skill_id, "name": name, "content": instructions, "meta": {}, **fields}
    with keeper.client() as client:
        created = client.post("/api/v1/skills/create", json=form)
    assert created.status_code == 200, created.text
    return skill_id


def _stored(keeper: Actor, skill_id: str) -> dict:
    with keeper.client() as client:
        fetched = client.get(f"/api/v1/skills/id/{skill_id}")
    assert fetched.status_code == 200, fetched.text
    return fetched.json()


def _import(page: Page, file_name: str, content: str, mime_type: str) -> None:
    page.goto("/workspace/skills")
    page.get_by_label("Open create menu").click()
    with page.expect_file_chooser() as chooser:
        page.get_by_role("button", name="Import", exact=True).click()
    chooser.value.set_files(
        files=[{"name": file_name, "mimeType": mime_type, "buffer": content.encode()}]
    )


def _import_selected(page: Page, count: int) -> None:
    dialog = page.get_by_role("dialog")
    expect(dialog.get_by_role("textbox", name="Skill ID")).to_have_count(count)
    dialog.get_by_role("button", name="Import selected").click()
    expect(dialog.get_by_text("saved", exact=True)).to_have_count(count)
    dialog.get_by_role("button", name="Close").click()


def _list_rows(page: Page, query: str, search_label: str = "Search Skills") -> Locator:
    """The list's rows once its search for `query` has answered."""
    search = page.get_by_role("textbox", name=search_label)
    expect(search).to_be_visible()
    with page.expect_response(
        lambda response: "query=" in response.url and "/skills/" in response.url
    ):
        search.fill(query)
    return page.get_by_role("main").get_by_role("button")


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


def _system_prompt_after_mention(page: Page, upstream, typed: str, shown: str) -> str:
    question = f"what now? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("On it.", match=reply.answering(question)))
    _dollar_menu(page, typed).filter(has_text=shown).click()
    page.keyboard.type(question)
    page.keyboard.press("Enter")
    # the conversation's label is translated, so the reply is found by its class
    expect(page.locator(".chat-assistant").last).to_contain_text("On it.")
    request = next(filter(reply.answering(question), upstream.chat_requests()))
    return "\n".join(
        str(entry["content"]) for entry in request["messages"] if entry["role"] == "system"
    )


def test_a_skill_markdown_file_is_imported_named_from_its_front_matter(page_for, keeper, upstream):
    suffix = uuid.uuid4().hex[:6]
    markdown = (
        f"---\nname: tide-tables-{suffix}\ndescription: Reading tide tables\n---\n"
        "Read the high tide column first.\n"
    )
    page = page_for(keeper)

    _import(page, "SKILL.md", markdown, "text/markdown")

    dialog = page.get_by_role("dialog")
    expect(dialog.get_by_role("textbox", name="Skill name")).to_have_value(f"tide-tables-{suffix}")
    expect(dialog.get_by_role("textbox", name="Skill ID")).to_have_value(f"tide-tables-{suffix}")
    _import_selected(page, 1)
    stored = _stored(keeper, f"tide-tables-{suffix}")
    assert stored["description"] == "Reading tide tables"
    assert stored["content"] == markdown

    system = _system_prompt_after_mention(page, upstream, suffix, f"tide-tables-{suffix}")
    assert "Read the high tide column first." in system


def test_a_json_file_of_several_skills_imports_each_of_them(page_for, keeper):
    suffix = uuid.uuid4().hex[:6]
    skills = [
        {"id": f"{kind}-{suffix}", "name": f"{kind.title()} {suffix}", "content": f"{kind} rules"}
        for kind in ("anchoring", "docking")
    ]
    page = page_for(keeper)

    _import(page, "skills.json", json.dumps(skills), "application/json")

    _import_selected(page, 2)
    rows = _list_rows(page, suffix)
    expect(rows.filter(has_text=f"Anchoring {suffix}")).to_be_visible()
    expect(rows.filter(has_text=f"Docking {suffix}")).to_be_visible()
    assert _stored(keeper, f"docking-{suffix}")["content"] == "docking rules"


def test_a_rows_own_export_downloads_that_skill_alone_with_its_instructions(page_for, keeper):
    suffix = uuid.uuid4().hex[:6]
    exported_id = _create(keeper, f"Exported {suffix}", "Keep the log tidy.")
    _create(keeper, f"Neighbour {suffix}", "Stay out of it.")
    page = page_for(keeper)
    page.goto("/workspace/skills")
    row = _list_rows(page, f"Exported {suffix}").filter(has_text=f"Exported {suffix}")
    row.hover()
    row.get_by_role("button", name="Skill Menu").last.click()

    with page.expect_download() as download:
        page.get_by_role("button", name="Export JSON", exact=True).click()
    with open(download.value.path()) as saved:
        exported = json.load(saved)

    assert exported["id"] == exported_id, exported
    assert exported["name"] == f"Exported {suffix}"
    assert exported["files"] == [{"path": "SKILL.md", "content": "Keep the log tidy."}]


def _name_in_german(page: Page, skill_id: str, name: str, description: str) -> None:
    page.goto(f"/workspace/skills/edit?id={skill_id}")
    editor = page.get_by_role("main")
    editor.get_by_role("combobox", name="Editing language").select_option("de-DE")
    # with a language picked the fields show the default text as their placeholder
    editor.get_by_role("textbox", name="Skill Name").fill(name)
    editor.get_by_role("textbox", name="Skill Description").fill(description)
    editor.get_by_role("button", name="Save", exact=True).click()
    expect(page.get_by_text("Skill updated successfully")).to_be_visible()


def _switch_to_german(page: Page) -> None:
    page.goto("/?settings=general")
    settings = page.get_by_role("dialog")
    settings.get_by_role("combobox", name="Language").select_option("de-DE")
    expect(page.locator("html")).to_have_attribute("lang", "de-DE")


@pytest.fixture
def german_named(page_for, keeper) -> tuple[Page, str, str, str]:
    """(page, skill id, default name, German name) for a skill named in German in the editor."""
    suffix = uuid.uuid4().hex[:6]
    default_name, german_name = f"Knots {suffix}", f"Knoten {suffix}"
    skill_id = _create(keeper, default_name, "Tie a bowline.", description="tying knots")
    page = page_for(keeper)
    _name_in_german(page, skill_id, german_name, "Knoten knüpfen")
    return page, skill_id, default_name, german_name


def test_a_german_name_given_in_the_editor_shows_in_a_german_interface(
    german_named, keeper, upstream
):
    page, skill_id, default_name, german_name = german_named
    assert _stored(keeper, skill_id)["meta"]["i18n"]["de-DE"]["name"] == german_name
    page.goto("/workspace/skills")
    expect(_list_rows(page, default_name).filter(has_text=default_name)).to_be_visible()

    _switch_to_german(page)
    page.goto("/workspace/skills")
    rows = _list_rows(page, default_name, GERMAN_SEARCH)
    expect(rows.filter(has_text=german_name)).to_be_visible()
    expect(rows.filter(has_text=default_name)).to_have_count(0)

    system = _system_prompt_after_mention(page, upstream, default_name.split()[1], german_name)
    assert "Tie a bowline." in system


def test_the_german_name_finds_the_skill_in_a_german_interface(german_named):
    page, _, _, german_name = german_named
    _switch_to_german(page)
    page.goto("/workspace/skills")

    rows = _list_rows(page, german_name, GERMAN_SEARCH)

    expect(
        rows.filter(has_text=german_name),
        "the list shows the skill under its German name, but searching that name finds nothing "
        "(the search matches only the original name, description and id, "
        "open-webui/open-webui#32018)",
    ).to_be_visible()
