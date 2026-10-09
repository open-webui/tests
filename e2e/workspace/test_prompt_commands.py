"""Journey: a prompt's slash command is unique, normalised by the dialog and offered under its name.

Saving a prompt in the Create Prompt dialog with a command that is already taken, by the same
account or by another account's private prompt, is refused with the "already registered" toast and
the first prompt keeps its text in the chat's `/` menu. Renaming a prompt's command to a taken one
in the editor is refused the same way and the prompt keeps its old command. The command
follows the name as a lower-case slug until it is typed by hand; a hand-typed command with a
leading slash or other odd characters is refused, and one made of letters, digits, hyphens and
underscores is kept as typed and offered under that name. A prompt whose command is the name of a
built-in command (`model`) is listed next to that command in the `/` menu, and each row does its
own thing.

Discriminates: passes on dev ebc6add67. In a backend copy where the command lookup never finds
a prompt, the three refusal tests go red (the second prompt is not refused, the rename is not
refused); in a frontend build whose dialog keeps the raw name as the command, accepts any command
text and lower-cases what is typed, the slug, typed-case and odd-character tests go red; where the
`/` menu drops a prompt that shares a name with a built-in command, the built-in test goes red.
Retargeted for 37138282f, where the rewritten editor names its text box Prompt Content, refuses an
odd command on creation with its own message and leaves a refused rename in the field: the tests
pass on that build.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.access import make_group
from harness.actors import Actor
from utils.chat_ui import chat_input

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

TAKEN = "Uh-oh! This command is already registered. Please choose another command string."
ODD_COMMAND = "Enter a name, content, and a valid command."


@pytest.fixture
def author(make_user):
    """A fresh admin; `add(command, content)` saves a prompt of theirs and returns its id."""
    account = make_user(role="admin")
    created: list[str] = []

    def add(command: str, content: str, owner: Actor = account) -> str:
        form = {"command": command, "name": f"Prompt {command}", "content": content}
        with owner.client() as client:
            response = client.post("/api/v1/prompts/create", json=form)
        assert response.status_code == 200, response.text
        created.append(response.json()["id"])
        return created[-1]

    yield account, add
    with account.client() as client:
        for prompt_id in created:
            client.delete(f"/api/v1/prompts/id/{prompt_id}/delete")


def _slash_menu(page: Page, typed: str) -> Locator:
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    chat_input(page).click()
    page.keyboard.type(f"/{typed}")
    return page.get_by_role("tooltip")


def _create_dialog(page: Page, name: str, command: str | None, content: str) -> Locator:
    """Fills the Create Prompt dialog (the command only when given) without saving it."""
    page.goto("/workspace/prompts/create")
    dialog = page.get_by_role("dialog").filter(has_text="Create Prompt")
    dialog.get_by_role("textbox", name="Prompt Name").fill(name)
    if command is not None:
        dialog.get_by_role("textbox", name="Command").fill(command)
    dialog.get_by_role("textbox", name="Prompt Content").fill(content)
    return dialog


def test_creating_a_prompt_with_a_command_the_same_account_holds_is_refused(page_for, author):
    account, add = author
    command = f"dup-{uuid.uuid4().hex[:8]}"
    add(command, "The first text.")
    page = page_for(account)

    dialog = _create_dialog(page, f"Second {command}", command, "The second text.")
    dialog.get_by_role("button", name="Save & Create").click()

    expect(page.get_by_text(TAKEN).first).to_be_visible()
    expect(dialog).to_be_visible()
    menu = _slash_menu(page, command)
    expect(menu.get_by_role("button", name=command)).to_have_count(1)
    menu.get_by_role("button", name=command).click()
    expect(chat_input(page)).to_have_text("The first text.")


def test_creating_a_prompt_with_a_command_another_accounts_private_prompt_holds_is_refused(
    page_for, author, admin, make_user
):
    owner, add = author
    other = make_user()
    make_group(admin, [other], {"workspace": {"prompts": True}})
    command = f"mine-{uuid.uuid4().hex[:8]}"
    add(command, "Only for the owner.")
    page = page_for(other)

    dialog = _create_dialog(page, f"Taking {command}", command, "Taking it over.")
    dialog.get_by_role("button", name="Save & Create").click()

    expect(page.get_by_text(TAKEN).first).to_be_visible()
    expect(dialog).to_be_visible()
    expect(_slash_menu(page_for(owner), command).get_by_role("button", name=command)).to_have_count(
        1
    )
    expect(_slash_menu(page, command).get_by_role("button", name=command)).to_have_count(0)


def test_renaming_a_command_to_a_taken_one_in_the_editor_is_refused(page_for, author):
    account, add = author
    suffix = uuid.uuid4().hex[:8]
    held, moving = f"held-{suffix}", f"moving-{suffix}"
    add(held, "Held text.")
    moving_id = add(moving, "Moving text.")
    page = page_for(account)

    page.goto(f"/workspace/prompts/{moving_id}")
    field = page.get_by_role("textbox", name="command")
    field.fill(held)

    expect(page.get_by_text(TAKEN).first).to_be_visible()
    with account.client() as client:
        assert client.get(f"/api/v1/prompts/id/{moving_id}").json()["command"] == moving
    expect(_slash_menu(page, moving).get_by_role("button", name=moving)).to_have_count(1)
    menu = _slash_menu(page, held)
    expect(menu.get_by_role("button", name=held)).to_have_count(1)
    menu.get_by_role("button", name=held).click()
    expect(chat_input(page)).to_have_text("Held text.")


def test_the_command_follows_the_name_as_a_slug_and_is_offered_under_that_name(page_for, author):
    account, _ = author
    suffix = uuid.uuid4().hex[:6]
    page = page_for(account)

    dialog = _create_dialog(page, f"Café   Menu! {suffix}", None, "Today's specials.")
    expect(dialog.get_by_role("textbox", name="Command")).to_have_value(f"cafe-menu-{suffix}")
    dialog.get_by_role("button", name="Save & Create").click()
    expect(page.get_by_role("main").get_by_text(f"Café   Menu! {suffix}")).to_be_visible()

    menu = _slash_menu(page, f"cafe-menu-{suffix}")
    menu.get_by_role("button", name=f"cafe-menu-{suffix}").click()
    expect(chat_input(page)).to_have_text("Today's specials.")


def test_a_typed_command_keeps_its_letters_case_and_underscores(page_for, author):
    account, _ = author
    command = f"Daily_Brief-{uuid.uuid4().hex[:6]}"
    page = page_for(account)

    dialog = _create_dialog(page, f"Brief {command}", command, "Brief the team.")
    dialog.get_by_role("button", name="Save & Create").click()
    expect(page.get_by_role("main").get_by_text(f"Brief {command}")).to_be_visible()

    with account.client() as client:
        listed = client.get("/api/v1/prompts/").json()
    stored = [prompt for prompt in listed if prompt["name"] == f"Brief {command}"]
    assert [prompt["command"] for prompt in stored] == [command]
    menu = _slash_menu(page, command.lower())
    menu.get_by_role("button", name=command).click()
    expect(chat_input(page)).to_have_text("Brief the team.")
    with account.client() as client:
        client.delete(f"/api/v1/prompts/id/{stored[0]['id']}/delete")


@pytest.mark.parametrize("typed", ["/slashed", "odd chars!", "café"])
def test_a_typed_command_with_a_leading_slash_or_odd_characters_is_refused(page_for, author, typed):
    account, _ = author
    name = f"Odd {uuid.uuid4().hex[:8]}"
    page = page_for(account)

    dialog = _create_dialog(page, name, typed, "Never saved.")
    dialog.get_by_role("button", name="Save & Create").click()

    expect(page.get_by_text(ODD_COMMAND).first).to_be_visible()
    expect(dialog).to_be_visible()
    with account.client() as client:
        names = [prompt["name"] for prompt in client.get("/api/v1/prompts/").json()]
    assert name not in names


def test_a_prompt_named_like_a_built_in_command_is_listed_next_to_it(page_for, author):
    account, add = author
    prompt_id = add("model", "Pick a vessel.")
    with account.client() as client:
        client.post(
            f"/api/v1/prompts/id/{prompt_id}/update/meta",
            json={"name": "Vessel picker", "command": "model"},
        )
    page = page_for(account)

    menu = _slash_menu(page, "model")
    built_in = menu.get_by_role("button", name=re.compile(r"^Model:"))
    prompt = menu.get_by_role("button").filter(has_text="Vessel picker")
    expect(built_in).to_have_count(1)
    expect(prompt).to_have_count(1)

    prompt.click()
    expect(chat_input(page)).to_have_text("Pick a vessel.")
    expect(page.get_by_role("textbox", name="Search In Models")).to_have_count(0)

    # the chat keeps the inserted text as a draft, so clear it rather than reload
    page.keyboard.press("ControlOrMeta+A")
    page.keyboard.press("Delete")
    page.keyboard.type("/model")
    page.get_by_role("tooltip").get_by_role("button", name=re.compile(r"^Model:")).click()
    expect(page.get_by_role("textbox", name="Search In Models")).to_be_visible()
