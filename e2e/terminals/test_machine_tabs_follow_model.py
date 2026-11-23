"""Journey: the open shell and Files tabs of a chat follow the chat's model.

An admin shares a Terminals terminal with one machine per model. A person picks it, opens a
shell and Files, and switches the chat to another model. Both tabs move to the new model's
machine without a reload: the shell of the old model's machine is stopped, the same tab connects
to a new shell in the new model's machine, and Files lists that machine's home. In a saved chat
the moved tabs are what a reload brings back: the shell tab connects again, nothing reads
"Session no longer available", and Files still lists the new model's machine (#34815). Picking
a model and at once another, while the shell of the first is still starting, leaves the shell in
the last model's machine and none running in the others. A shell on a terminal with one machine
per chat or per user keeps running through a model switch, in the same machine, and a shell left
open in a chat keeps running while the person visits a chat on another model and comes back.

Since 6f2fc7bc7 (dev after 0.13.3, titled `refac`) every machine tab kept the model it was opened
with: after a switch the open shell and Files stayed in the old model's machine, and after a
reload the shell read "Session no longer available" while Files listed the new model's machine.

Discriminates: the three per model switch tests are red on the dev b393504d9 build, where Files
still lists the old model's machine after the switch and the shell still writes into it; all
six pass with the fix. In a frontend copy of the fix that moves a shell without waiting for one
still starting, the twice in a row test fails with "Session no longer available". The per chat,
per user and visit tests pass on both.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.actors import Actor
from harness.python_tools import EVERYONE_READS
from harness.upstream import MOCK_MODEL_ID
from utils.cached_chat import choose_terminal, open_files_tab
from utils.chat_surface import main_chat, open_sidebar
from utils.chat_ui import expect_reply, send
from utils.file_browser import expect_listed, file_browser, refresh
from utils.model_selector import model_search, select_model
from utils.shell_tab import (
    READABLE_SHELL,
    expect_on_screen,
    open_shell,
    screen_of,
    shell_notice,
    shell_panel_by_tab,
    shell_tab_of,
    type_into,
)

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def make_model(admin: Actor):
    """`make_model()`: a model everyone may use, on the mock model; all go after the test."""
    made = []

    def make() -> dict:
        model = {
            "id": f"ferry-{uuid.uuid4().hex[:8]}",
            "name": f"Ferry desk {uuid.uuid4().hex[:6]}",
        }
        with admin.client() as client:
            created = client.post(
                "/api/v1/models/create",
                json={
                    **model,
                    "base_model_id": MOCK_MODEL_ID,
                    "meta": {},
                    "params": {},
                    "access_grants": [EVERYONE_READS],
                },
            )
        assert created.status_code == 200, created.text
        made.append(model)
        return model

    yield make
    with admin.client() as client:
        for model in made:
            client.post("/api/v1/models/model/delete", json={"id": model["id"]})


@pytest.fixture
def second_model(make_model) -> dict:
    return make_model()


def _write(shell: Locator, note: str) -> None:
    """Write a file into the shell's home and wait until the shell confirms it."""
    expect(shell.locator(".xterm-rows")).not_to_have_text("")
    expect_on_screen(shell, "$")
    marker = f"written-{uuid.uuid4().hex[:6]}"
    type_into(shell, f"echo moored > ~/{note}; echo {marker}")
    shell.get_by_role("textbox", name="Terminal input").press("Enter")
    expect_on_screen(shell, marker)


def _shells(person: Actor, terminal_id: str, model_id: str, chat_id: str = "") -> list[str]:
    """The shells running in the machine of `model_id`, as the page would list them."""
    headers = {"X-Terminal-Model-Id": model_id, **({"X-Session-Id": chat_id} if chat_id else {})}
    with person.client() as client:
        listed = client.get(f"/api/v1/machines/{terminal_id}/api/terminals", headers=headers)
    assert listed.status_code == 200, listed.text
    return [shell["id"] for shell in listed.json()]


def _switch_with_shell_and_files_open(
    page: Page, person: Actor, terminal_id: str, terminals, model: dict, chat_id: str = ""
) -> tuple[str, str]:
    """Open a shell and Files, switch the chat to `model` and check both moved to its machine;
    returns the shell tab's id and the file the shell wrote after the switch."""
    shell = open_shell(page)
    shell_tab_id = shell.get_attribute("aria-labelledby")
    before = f"before-{uuid.uuid4().hex[:6]}.txt"
    _write(shell, before)
    assert terminals.holding(before) == {f"model:{MOCK_MODEL_ID}": "moored\n"}
    assert len(_shells(person, terminal_id, MOCK_MODEL_ID, chat_id)) == 1
    files = open_files_tab(page)
    expect_listed(files, before)

    main_chat(page).show()
    select_model(main_chat(page), model["name"])
    expect(
        main_chat(page).panel.get_by_role("button", name=f"Selected model: {model['name']}")
    ).to_be_visible()
    # the new model's machine is fresh, so Files shows its empty home
    page.get_by_role("tab", name=re.compile(r"^(Files|Home)$")).click()
    expect_listed(file_browser(page))

    shell_tab_of(shell).click()
    after = f"after-{uuid.uuid4().hex[:6]}.txt"
    _write(shell, after)
    assert terminals.holding(after) == {f"model:{model['id']}": "moored\n"}, (
        f"the open shell wrote into {terminals.holding(after)} after the switch to {model['id']}"
    )
    assert _shells(person, terminal_id, MOCK_MODEL_ID, chat_id) == [], (
        "the shell of the old model's machine still runs after the switch"
    )

    page.get_by_role("tab", name=re.compile(r"^(Files|Home)$")).click()
    files = file_browser(page)
    refresh(files)
    expect_listed(files, after)
    return shell_tab_id, after


def test_open_shell_and_files_move_to_the_machine_of_a_new_chats_model(
    make_user, shared_terminal, terminals, page_for, second_model
):
    person = make_user()
    name = f"Harbour {uuid.uuid4().hex[:6]}"
    terminal_id = shared_terminal(person, name=name, scope="model")
    page = page_for(person, init_script=READABLE_SHELL)
    choose_terminal(page, name)
    _switch_with_shell_and_files_open(page, person, terminal_id, terminals, second_model)


def test_open_shell_and_files_follow_a_saved_chats_model_and_come_back_after_a_reload(
    make_user, shared_terminal, terminals, page_for, upstream, second_model
):
    person = make_user()
    name = f"Harbour {uuid.uuid4().hex[:6]}"
    terminal_id = shared_terminal(person, name=name, scope="model")
    page = page_for(person, init_script=READABLE_SHELL)
    choose_terminal(page, name)
    question = f"plan the crossing {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Board at noon.", match=reply.answering(question)))
    send(page, question)
    expect_reply(page, "Board at noon.")
    expect(page).to_have_url(re.compile(r"/c/"))
    chat_id = page.url.rsplit("/", 1)[-1]
    shell_tab_id, after = _switch_with_shell_and_files_open(
        page, person, terminal_id, terminals, second_model, chat_id
    )

    page.reload()
    page.locator(f'[id="{shell_tab_id}"]').click()
    shell = shell_panel_by_tab(page, shell_tab_id)
    expect(shell.locator(".xterm-rows")).to_be_visible()
    expect(page.get_by_text("Session no longer available")).to_have_count(0)
    expect(shell_notice(shell)).to_have_count(0)
    again = f"again-{uuid.uuid4().hex[:6]}.txt"
    _write(shell, again)
    assert terminals.holding(again) == {f"model:{second_model['id']}": "moored\n"}, (
        f"after the reload the shell wrote into {terminals.holding(again)}; it reads "
        f"{screen_of(shell)!r}"
    )

    page.get_by_role("tab", name=re.compile(r"^(Files|Home)$")).click()
    expect_listed(file_browser(page), after, again)


@pytest.mark.parametrize("scope", ["chat", "user"])
def test_a_shell_on_a_machine_per_chat_or_per_user_keeps_running_through_a_model_switch(
    scope, make_user, shared_terminal, terminals, page_for, upstream, second_model
):
    person = make_user()
    name = f"Harbour {uuid.uuid4().hex[:6]}"
    shared_terminal(person, name=name, scope=scope)
    page = page_for(person, init_script=READABLE_SHELL)
    choose_terminal(page, name)
    question = f"plan the crossing {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Board at noon.", match=reply.answering(question)))
    send(page, question)
    expect_reply(page, "Board at noon.")
    expect(page).to_have_url(re.compile(r"/c/"))
    machine = f"chat:{page.url.rsplit('/', 1)[-1]}" if scope == "chat" else f"user:{person.id}"
    shell = open_shell(page)
    tag = uuid.uuid4().hex[:6]
    expect_on_screen(shell, "$")
    type_into(shell, f"berth=north-{tag}")
    shell.get_by_role("textbox", name="Terminal input").press("Enter")

    main_chat(page).show()
    select_model(main_chat(page), second_model["name"])
    expect(
        main_chat(page).panel.get_by_role("button", name=f"Selected model: {second_model['name']}")
    ).to_be_visible()

    shell_tab_of(shell).click()
    note = f"after-{uuid.uuid4().hex[:6]}.txt"
    type_into(shell, f"echo $berth > ~/{note}; echo written-$berth")
    shell.get_by_role("textbox", name="Terminal input").press("Enter")
    expect_on_screen(shell, f"written-north-{tag}")
    assert terminals.holding(note) == {machine: f"north-{tag}\n"}


def test_an_open_shell_switched_twice_in_a_row_ends_up_in_the_last_models_machine(
    make_user, shared_terminal, terminals, page_for, make_model
):
    first, last = make_model(), make_model()
    person = make_user()
    name = f"Harbour {uuid.uuid4().hex[:6]}"
    terminal_id = shared_terminal(person, name=name, scope="model")
    page = page_for(person, init_script=READABLE_SHELL)
    choose_terminal(page, name)
    shell = open_shell(page)
    _write(shell, f"before-{uuid.uuid4().hex[:6]}.txt")

    main_chat(page).show()
    # the second pick lands while the shell of the first one is still starting
    select_model(main_chat(page), first["name"])
    expect(model_search(page)).to_be_hidden()
    select_model(main_chat(page), last["name"])
    expect(
        main_chat(page).panel.get_by_role("button", name=f"Selected model: {last['name']}")
    ).to_be_visible()

    shell_tab_of(shell).click()
    after = f"after-{uuid.uuid4().hex[:6]}.txt"
    _write(shell, after)
    assert terminals.holding(after) == {f"model:{last['id']}": "moored\n"}, (
        f"the open shell wrote into {terminals.holding(after)} after switching to {last['id']}"
    )
    assert _shells(person, terminal_id, MOCK_MODEL_ID) == []
    assert _shells(person, terminal_id, first["id"]) == []


def test_a_shell_keeps_running_when_the_person_goes_to_a_chat_on_another_model_and_back(
    make_user, shared_terminal, terminals, page_for, upstream, second_model
):
    person = make_user()
    name = f"Harbour {uuid.uuid4().hex[:6]}"
    shared_terminal(person, name=name, scope="model")
    page = page_for(person, init_script=READABLE_SHELL)
    choose_terminal(page, name)
    first = f"plan the crossing {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Board at noon.", match=reply.answering(first)))
    send(page, first)
    expect_reply(page, "Board at noon.")
    expect(page).to_have_url(re.compile(r"/c/"))
    first_url = page.url
    shell = open_shell(page)
    shell_tab_id = shell.get_attribute("aria-labelledby")
    tag = uuid.uuid4().hex[:6]
    expect_on_screen(shell, "$")
    type_into(shell, f"berth=north-{tag}")
    shell.get_by_role("textbox", name="Terminal input").press("Enter")

    page.goto(re.sub(r"/c/.*$", "/", first_url))
    select_model(main_chat(page), second_model["name"])
    second = f"plan the return {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Leave at dusk.", match=reply.answering(second)))
    send(page, second)
    expect_reply(page, "Leave at dusk.")
    expect(page).not_to_have_url(first_url)

    chat_path = first_url[first_url.index("/c/") :]
    open_sidebar(page).locator(f'a[href="{chat_path}"]').first.click()
    expect(page).to_have_url(first_url)
    page.locator(f'[id="{shell_tab_id}"]').click()
    shell = shell_panel_by_tab(page, shell_tab_id)
    note = f"back-{uuid.uuid4().hex[:6]}.txt"
    expect(shell.locator(".xterm-rows")).not_to_have_text("")
    type_into(shell, f"echo $berth > ~/{note}; echo written-$berth")
    shell.get_by_role("textbox", name="Terminal input").press("Enter")
    expect_on_screen(shell, f"written-north-{tag}")
    assert terminals.holding(note) == {f"model:{MOCK_MODEL_ID}": f"north-{tag}\n"}
