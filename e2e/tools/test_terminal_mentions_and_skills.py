"""Journey: a terminal's files and skills in the lists of the message box.

A real Open Terminal runs with a folder, a file and a skill of the test's own in its home. A
person with the terminal picked types `@` in the message box: the list offers the terminal's
matching folders and files under Filesystem, and nothing of it while no terminal is picked. A
picked file is attached to the message, is named to the model with its path in the terminal and
is stored with the message; clicking it in the sent message opens the file in a file tab.

Typing `$` or `/` lists the terminal's skills beside the workspace's, each terminal skill
labelled Terminal where a workspace skill shows its id. A picked terminal skill is a mention in
the message, the model reads it from the terminal, and the sent message shows the mention. The
Skills list of the message box's More menu marks a terminal skill the same way, and a skill
switched on there is listed, as Terminal, under Available Skills.

The page learns a terminal's skills only when one of these lists opens. Until then it knows
none: after a reload the sent mention reads as its raw markup, and with no workspace skill the
More menu has no Skills entry (finding `open-webui-terminal-skills-unknown-until-listed`, both
tests red on dev c15f28649).

`/` also offers Create skill in a chat with a terminal; that command is driven in
e2e/chat/test_prompt_prefix_caching.py.

Discriminates: the other tests pass on the dev c15f28649 build. In frontend copies, `@`
listing nothing of the terminal turns the `@` test red; terminal skills listed by their id
turn the list tests and the mention test red; Available Skills or the Skills menu not marking
a terminal skill turns the Available Skills test red. In a copy that reads the terminal's
skills as soon as the chat's terminal is known, the reload test passes and the More menu
shows the Skills entry.
"""

from __future__ import annotations

import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, read_grant
from harness.upstream import MOCK_MODEL_ID
from utils.cached_chat import ask, choose_terminal
from utils.chat_ui import (
    REPLY_TIMEOUT_MS,
    chat_input,
    close_menu,
    conversation,
    expect_reply,
    open_integrations_menu,
    show_input_selections,
)

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

CHAT_URL = re.compile(r"/c/([0-9a-f-]{36})$")
RACKING = "rack the perry in march\n"
PRESSING = "Press the apples slowly."


@dataclass
class Cellar:
    """What the test keeps on the terminal and in the workspace, each named with `tag`."""

    tag: str
    folder: Path
    notes: Path
    terminal_skill: str
    workspace_skill: str
    workspace_skill_id: str


@pytest.fixture
def person(make_user):
    return make_user()


@pytest.fixture
def terminal_name(admin, preserve, open_terminal, person) -> str:
    preserve(TERMINAL_SERVERS_CONFIG)
    connection = open_terminal.connection(
        name=f"Cider house {uuid.uuid4().hex[:6]}",
        config={"access_grants": [read_grant(person.id)]},
    )
    with admin.client() as client:
        configure_terminals(client, connection)
    return connection["name"]


@pytest.fixture
def cellar(open_terminal):
    """A folder with a file and a skill in the terminal's home, removed afterwards."""
    tag = uuid.uuid4().hex[:6]
    # the model's own name starts the same way, so `@mock` lists it beside these
    folder = open_terminal.home / f"mock-cellar-{tag}"
    folder.mkdir()
    notes = folder / f"mock-racking-{tag}.txt"
    notes.write_text(RACKING)
    skill = open_terminal.home / ".agents" / "skills" / f"pressing-{tag}"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: pressing-{tag}\ndescription: Pressing apples {tag}\n---\n{PRESSING}\n"
    )
    yield Cellar(tag, folder, notes, f"pressing-{tag}", f"Bottling {tag}", f"bottling-{tag}")
    shutil.rmtree(folder, ignore_errors=True)
    shutil.rmtree(skill, ignore_errors=True)


@pytest.fixture
def workspace_skill(admin, person, cellar):
    """A workspace skill shared with the person, named with the same tag as the terminal's."""
    with admin.client() as client:
        created = client.post(
            "/api/v1/skills/create",
            json={
                "id": cellar.workspace_skill_id,
                "name": cellar.workspace_skill,
                "description": f"Bottling cider {cellar.tag}",
                "content": "Bottle the cider cold.",
                "meta": {},
                "access_grants": [read_grant(person.id)],
            },
        )
        assert created.status_code == 200, created.text
        yield
        client.delete(f"/api/v1/skills/id/{cellar.workspace_skill_id}/delete")


def _type(page: Page, text: str) -> None:
    chat_input(page).click()
    page.keyboard.type(text)


def _clear_input(page: Page) -> None:
    chat_input(page).click()
    page.keyboard.press("Control+a")
    page.keyboard.press("Backspace")
    expect(chat_input(page)).to_have_text("")


def _offered(page: Page, name: str | re.Pattern) -> Locator:
    """An entry of the list that opens under the message box while `@`, `$` or `/` is typed."""
    return page.get_by_role("tooltip").get_by_role("button", name=name, exact=True)


def _sent_message(page: Page) -> Locator:
    return conversation(page).get_by_role("listitem").first


def _terminal_skill_row(cellar: Cellar) -> str:
    return f"{cellar.terminal_skill} Terminal"


def _send_a_terminal_skill_mention(page: Page, upstream, cellar: Cellar) -> str:
    """Mention the terminal's skill in a first message; returns the question once answered."""
    question = f"how hard do I press? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Gently.", match=reply.answering(question)))
    _type(page, f"${cellar.tag}")
    _offered(page, _terminal_skill_row(cellar)).click()
    page.keyboard.type(question)
    page.keyboard.press("Enter")
    expect_reply(page, "Gently.")
    return question


def test_at_lists_the_picked_terminals_files_and_a_picked_one_is_sent_and_opens_from_the_message(
    page_for, person, terminal_name, cellar, upstream
):
    page = page_for(person)

    _type(page, "@mock")

    expect(_offered(page, re.compile(MOCK_MODEL_ID))).to_be_visible()
    expect(page.get_by_role("tooltip").get_by_text("Filesystem")).to_have_count(0)
    expect(_offered(page, cellar.notes.name)).to_have_count(0)

    _clear_input(page)
    choose_terminal(page, terminal_name)
    _type(page, f"@{cellar.tag}")

    expect(page.get_by_role("tooltip").get_by_text("Filesystem", exact=True)).to_be_visible()
    expect(_offered(page, cellar.folder.name)).to_be_visible()
    _offered(page, cellar.notes.name).click()

    composer = page.locator("form").filter(has=chat_input(page))
    expect(
        composer.get_by_role("button", name=re.compile(re.escape(cellar.notes.name)))
    ).to_be_visible()
    expect(chat_input(page)).to_have_text("")
    prompt = f"what do the notes say? {uuid.uuid4().hex[:6]}"
    ask(page, upstream, prompt, reply.text("Rack in March."))

    asked = next(filter(reply.answering(prompt), upstream.chat_requests()))
    told = [entry["content"] for entry in asked["messages"] if entry["role"] == "user"][-1]
    assert f'<file type="filesystem" id="{cellar.notes}"' in told, (
        f"the model was not told the attached terminal file's path: {told!r}"
    )
    expect(page).to_have_url(CHAT_URL, timeout=REPLY_TIMEOUT_MS)
    with person.client() as client:
        stored = client.get(f"/api/v1/chats/{CHAT_URL.search(page.url).group(1)}")
    assert stored.status_code == 200, stored.text
    messages = stored.json()["chat"]["history"]["messages"].values()
    files = [file for message in messages if message["role"] == "user" for file in message["files"]]
    assert [(file["type"], file["path"]) for file in files] == [
        ("filesystem", str(cellar.notes))
    ], f"the message should carry the picked terminal file with its path: {files}"

    _sent_message(page).get_by_role("button", name=re.compile(re.escape(cellar.notes.name))).click()

    expect(page.get_by_role("tab", name=cellar.notes.name, exact=True)).to_have_attribute(
        "aria-selected", "true"
    )
    files_tab = page.get_by_role("region", name="File browser")
    expect(files_tab.get_by_text(RACKING.strip())).to_be_visible()


@pytest.mark.parametrize("trigger", ["$", "/"])
def test_typing_lists_the_terminals_skills_as_terminal_beside_the_workspaces(
    page_for, person, terminal_name, cellar, workspace_skill, trigger
):
    page = page_for(person)
    workspace_row = f"{cellar.workspace_skill} {cellar.workspace_skill_id}"

    _type(page, f"{trigger}{cellar.tag}")

    expect(_offered(page, workspace_row)).to_be_visible()
    expect(_offered(page, _terminal_skill_row(cellar))).to_have_count(0)

    _clear_input(page)
    choose_terminal(page, terminal_name)
    _type(page, f"{trigger}{cellar.tag}")

    expect(_offered(page, _terminal_skill_row(cellar))).to_be_visible()
    expect(_offered(page, workspace_row)).to_be_visible()


def test_a_picked_terminal_skill_is_a_mention_the_model_reads_from_the_terminal(
    page_for, person, terminal_name, cellar, upstream
):
    page = page_for(person)
    choose_terminal(page, terminal_name)

    question = _send_a_terminal_skill_mention(page, upstream, cellar)

    mention = _sent_message(page).get_by_role("button", name=f"$terminal:{cellar.terminal_skill}")
    expect(mention).to_be_visible()
    asked = next(filter(reply.answering(question), upstream.chat_requests()))
    read = "\n".join(reply.context_results(asked, "view_skill"))
    assert PRESSING in read, (
        f"the model was not given the terminal skill it was pointed at: {read!r}"
    )


def test_a_terminal_skill_switched_on_in_the_skills_menu_is_listed_under_available_skills(
    page_for, person, terminal_name, cellar, workspace_skill
):
    show_input_selections(person)
    page = page_for(person)
    choose_terminal(page, terminal_name)

    menu = open_integrations_menu(page)
    menu.get_by_role("menuitem", name=re.compile(r"^Skills")).click()
    workspace_row = menu.get_by_role("menuitemcheckbox", name=cellar.workspace_skill, exact=True)
    terminal_row = menu.get_by_role(
        "menuitemcheckbox", name=_terminal_skill_row(cellar), exact=True
    )
    expect(workspace_row).to_be_visible()
    terminal_row.click()
    expect(terminal_row).to_have_attribute("aria-checked", "true")
    workspace_row.click()
    expect(workspace_row).to_have_attribute("aria-checked", "true")
    close_menu(page)

    available = page.get_by_role("main").get_by_role("button", name="Available Skills", exact=True)
    expect(available).to_have_text("2")
    available.click()
    dialog = page.get_by_role("dialog", name="Available Skills")
    expect(dialog.get_by_text(cellar.workspace_skill, exact=True)).to_be_visible()
    labelled = re.compile(
        rf"{re.escape(cellar.terminal_skill)}\s*Pressing apples {cellar.tag}\s*Terminal"
    )
    expect(dialog).to_contain_text(labelled)
    # the workspace skill is listed without the label
    expect(dialog.get_by_text("Terminal", exact=True)).to_have_count(1)


def test_a_sent_terminal_skill_mention_still_reads_as_a_mention_after_a_reload(
    page_for, person, terminal_name, cellar, upstream
):
    page = page_for(person)
    choose_terminal(page, terminal_name)
    _send_a_terminal_skill_mention(page, upstream, cellar)
    mention = _sent_message(page).get_by_role("button", name=f"$terminal:{cellar.terminal_skill}")
    expect(mention).to_be_visible()

    page.reload()
    expect(chat_input(page)).to_be_visible()
    expect_reply(page, "Gently.")

    # red on dev c15f28649: the message reads "<$terminal:pressing-…|pressing-…>" until a skill
    # list is opened (finding open-webui-terminal-skills-unknown-until-listed)
    expect(mention).to_be_visible()
    expect(_sent_message(page)).not_to_contain_text("<$terminal:")


def test_the_more_menu_offers_a_terminals_skills_before_anything_is_typed(
    page_for, person, terminal_name, cellar
):
    page = page_for(person)
    choose_terminal(page, terminal_name)

    menu = open_integrations_menu(page)

    # red on dev c15f28649: with no workspace skill the entry only shows once `$` or `/` listed
    # the terminal's skills (finding open-webui-terminal-skills-unknown-until-listed)
    menu.get_by_role("menuitem", name=re.compile(r"^Skills")).click(timeout=10_000)
    expect(
        menu.get_by_role("menuitemcheckbox", name=_terminal_skill_row(cellar), exact=True)
    ).to_be_visible()
