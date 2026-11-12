"""Journey: the chat input's Terminal menu lists a person's terminals and keeps the chat's pick.

Rebuilt on dev after 0.13.3 (3d98e86e2, f4cf2ca0c, c647f3f42, 41179ba85, titled `refac`). The
menu lists the terminals of the admin's connections under Terminals and, for a person who may
use Direct Integrations, the terminals from their own settings under Direct. The + beside
Direct opens the person's Settings on Integrations; the cog beside Terminals, for admins only,
opens the admin's Integrations settings. The picked terminal carries a check and names the menu's
button; picking it again unpicks it, and picking a terminal of the other kind unpicks the first.
In a saved chat the pick is stored on the chat and is back when the chat is opened again. Since
d42988f0c a pick shows at once and is saved in the background, so one the server refuses stays
picked on the page, says why and is not stored: the chat opens again without it (before, a
refused pick was not made). A picked terminal that is deleted reads
Unavailable terminal for a person with Direct Integrations and is unpicked for everyone else.
The menu moves into the chat input's More menu when the input is too narrow for it.

A local fake terminal stands for the admin's connection and for the personal terminal; the
deleted terminal is one on the real Terminals service on Docker (`harness.terminals`), and its
test skips without the images or Docker. New terminal, the third header entry, is in
test_new_terminal_dialog.py and test_terminal_settings_dialog.py (only for a person who may
create terminals), and the menu missing before the first terminal exists is the finding
`open-webui-no-terminal-menu-before-the-first-terminal`.

Discriminates: passes on the dev c15f28649 build with Terminals 0.4.1 and Open Terminal 0.17.7.
On builds of a frontend copy with one change each: the Settings cog shown to everyone, personal
terminals listed without the permission, and Add Terminal opening the General settings (the
listed apart test); no terminal carrying the check, the picked terminal kept when it is picked
again, and the personal terminal left on when one of the admin's is picked (the picked test:
the model is still offered the personal terminal's tool); the pick not stored on a saved chat,
a saved chat opened without its terminal, and a refused pick that says nothing (the saved
chat test); a lost terminal reading Terminal, and one that stays picked for everyone (the
deleted test); the controls never moved into the More menu (the narrow test). The cog's target
has no mutation of its own.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.chat_history import seed_chat
from harness.listener import json_answer
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    configure_terminals,
    read_grant,
    serving_terminal,
)
from harness.terminals import create_template, share_template
from utils.cached_chat import ask, chat_requests, choose_terminal, offered_tools
from utils.chat_surface import MAIN_CHAT, ChatSurface
from utils.chat_ui import close_menu, expect_reply, open_input_menu, send
from utils.tooltips import tooltip_button
from utils.user_terminals import TEMPLATES, listed_terminal, open_terminal_menu

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

HOME = "/home/user"


def _spec(operation: str) -> dict:
    """An Open Terminal that offers the model one tool, `operation`."""
    described = {"operationId": operation, "responses": {"200": {"description": "ok"}}}
    return {
        "openapi": "3.0.0",
        "info": {"title": "Open Terminal", "version": "1"},
        "paths": {f"/{operation}": {"post": described}},
    }


def _serve(server, operation: str) -> None:
    server.cors = True
    server.route("GET", "/openapi.json", json_answer(_spec(operation)))
    server.route("GET", "/api/config", json_answer({"features": {}}))
    server.route("GET", "/files/cwd", json_answer({"cwd": HOME, "home": HOME}))
    server.route("GET", "/files/list", json_answer({"dir": HOME, "entries": []}))
    server.route("GET", "/files/read", (404, {"Content-Type": "text/plain"}, b"no such file"))
    server.route("GET", "/skills", json_answer([]))
    server.route("GET", "/files/search", json_answer({"results": []}))
    server.route("POST", f"/{operation}", json_answer({"ran": operation}))


@pytest.fixture
def fake_terminal():
    """A terminal behind the admin's connections; the model's tool on it is `run_on_server`."""
    with serving_terminal() as server:
        _serve(server, "run_on_server")
        yield server


@pytest.fixture
def personal_terminal():
    """A terminal the browser reaches itself; the model's tool on it is `run_on_laptop`."""
    with serving_terminal() as server:
        _serve(server, "run_on_laptop")
        yield server


@pytest.fixture
def connect(admin, preserve, fake_terminal):
    """`connect(names, *people)`: the admin connects the fake terminal once per name, for those
    people to use; returns the connections."""
    preserve(TERMINAL_SERVERS_CONFIG)

    def save(names: list[str], *people) -> list[dict]:
        grants = [read_grant(person.id) for person in people]
        connections = [
            fake_terminal.connection(name=name, config={"access_grants": grants}) for name in names
        ]
        with admin.client() as client:
            configure_terminals(client, *connections)
        return connections

    return save


def _add_personal_terminal(person, server, name: str) -> None:
    """The terminal as the person's Settings > Integrations stores it."""
    terminal = {
        "url": server.base_url,
        "key": "",
        "name": name,
        "enabled": False,
        "auth_type": "bearer",
        "path": "/openapi.json",
        "config": {"enable": True},
    }
    with person.client() as client:
        saved = client.post(
            "/api/v1/users/user/settings/update",
            json={"ui": {"showChangelog": False, "terminalServers": [terminal]}},
        )
    assert saved.status_code == 200, saved.text


def _names() -> tuple[str, str, str]:
    tag = uuid.uuid4().hex[:6]
    return f"Harbour {tag}", f"Quarry {tag}", f"Laptop {tag}"


def _menu_lines(menu: Locator) -> list[str]:
    """The menu from top to bottom: each heading and each terminal."""
    return [line.strip() for line in menu.inner_text().splitlines() if line.strip()]


def _settings_tab(settings: Locator, group: str) -> Locator:
    """The Integrations tab of the Settings window's Personal or Admin group."""
    return settings.get_by_label(group, exact=True).get_by_role("tab", name="Integrations")


def _terminal_button(page: Page, label: str) -> Locator:
    """The chat input's Terminal button, which reads the picked terminal."""
    return page.get_by_role("main").get_by_role("button", name=label, exact=True)


def _in_more_menu(page: Page, label: str) -> Locator:
    """The Terminal entry of the chat input's open More menu, which reads the picked terminal."""
    return page.get_by_role("menu").get_by_role("menuitem", name=label, exact=True)


def _checked(page: Page) -> dict[str, str]:
    """Which terminals of the open Terminal menu carry the check."""
    rows = page.get_by_role("menu").get_by_role("menuitemradio")
    return {row.inner_text().strip(): row.get_attribute("aria-checked") for row in rows.all()}


def _pick(page: Page, name: str, shown: str) -> None:
    """Click the terminal `name` in the open Terminal menu; the button then reads `shown`."""
    listed_terminal(page, name).click()
    expect(_terminal_button(page, shown)).to_be_visible()
    expect(page.get_by_role("menu")).to_have_count(0)


def _terminal_tools_sent_for(upstream, prompt: str) -> set[str]:
    """Which of the two terminals' tools the model was offered with `prompt`."""
    asked = [request for request in chat_requests(upstream) if reply.answering(prompt)(request)]
    assert asked, f"the provider never got {prompt!r}"
    return offered_tools(asked[-1]) & {"run_on_server", "run_on_laptop"}


def _saved_chat(person, title: str) -> str:
    with person.client() as client:
        chat_id, _ = seed_chat(
            client,
            [{"role": "user", "content": title}, {"role": "assistant", "content": "Noted."}],
        )
    return chat_id


def _stored_terminal(person, chat_id: str) -> str | None:
    with person.client() as client:
        stored = client.get(f"/api/v1/chats/{chat_id}")
    assert stored.status_code == 200, stored.text
    return stored.json()["chat"].get("terminal_id")


def test_terminals_and_personal_terminals_are_listed_apart_and_each_heading_opens_its_settings(
    page_for, make_user, connect, personal_terminal, direct_integrations
):
    direct_integrations(True)
    owner, person = make_user(role="admin"), make_user()
    shared, _, laptop = _names()
    connect([shared], person)
    for account in (owner, person):
        _add_personal_terminal(account, personal_terminal, laptop)
    page = page_for(owner)

    menu = open_terminal_menu(page)
    assert [line.lower() for line in _menu_lines(menu)] == [
        "direct",
        laptop.lower(),
        "terminals",
        shared.lower(),
    ], f"the admin's Terminal menu reads {_menu_lines(menu)}"

    tooltip_button(menu, "Add Terminal", role="menuitem").click()
    settings = page.get_by_role("dialog").filter(has=page.get_by_role("tab", name="Integrations"))
    expect(_settings_tab(settings, "Personal")).to_have_attribute("aria-selected", "true")
    expect(settings.get_by_text(laptop)).to_be_visible()
    page.keyboard.press("Escape")
    expect(settings).to_be_hidden()

    menu = open_terminal_menu(page)
    tooltip_button(menu, "Settings", role="menuitem").click()
    expect(_settings_tab(settings, "Admin")).to_have_attribute("aria-selected", "true")
    expect(settings.get_by_text(shared)).to_be_visible()

    # without the Direct Tool Servers permission a person's own terminals are not listed
    person_page = page_for(person)
    menu = open_terminal_menu(person_page)
    assert [line.lower() for line in _menu_lines(menu)] == ["terminals", shared.lower()], (
        f"a regular person's Terminal menu reads {_menu_lines(menu)}"
    )
    expect(menu.get_by_role("menuitem")).to_have_text([""])


def test_the_picked_terminal_is_checked_and_named_and_picking_it_again_unpicks_it(
    page_for, make_user, connect, personal_terminal, direct_integrations, upstream
):
    direct_integrations(True)
    person = make_user(role="admin")
    harbour, quarry, laptop = _names()
    connect([harbour, quarry])
    _add_personal_terminal(person, personal_terminal, laptop)
    page = page_for(person)
    tag = uuid.uuid4().hex[:6]

    open_terminal_menu(page)
    assert _checked(page) == {laptop: "false", harbour: "false", quarry: "false"}
    _pick(page, harbour, shown=harbour)
    open_terminal_menu(page, picked=harbour)
    assert _checked(page) == {laptop: "false", harbour: "true", quarry: "false"}, (
        f"with {harbour} picked the menu checks {_checked(page)}"
    )
    close_menu(page)
    ask(page, upstream, f"on the server {tag}", reply.text("On the server."))
    assert _terminal_tools_sent_for(upstream, f"on the server {tag}") == {"run_on_server"}

    # a personal terminal takes the pick from the admin's
    open_terminal_menu(page, picked=harbour)
    _pick(page, laptop, shown=laptop)
    open_terminal_menu(page, picked=laptop)
    assert _checked(page) == {laptop: "true", harbour: "false", quarry: "false"}, (
        f"with {laptop} picked the menu checks {_checked(page)}"
    )
    close_menu(page)
    ask(page, upstream, f"on the laptop {tag}", reply.text("On the laptop."))
    assert _terminal_tools_sent_for(upstream, f"on the laptop {tag}") == {"run_on_laptop"}

    # and one of the admin's takes it back
    open_terminal_menu(page, picked=laptop)
    _pick(page, quarry, shown=quarry)
    ask(page, upstream, f"back on the server {tag}", reply.text("Back on the server."))
    assert _terminal_tools_sent_for(upstream, f"back on the server {tag}") == {"run_on_server"}, (
        "the personal terminal still reached the model after another terminal was picked"
    )

    open_terminal_menu(page, picked=quarry)
    _pick(page, quarry, shown="Terminal")
    open_terminal_menu(page)
    assert _checked(page) == {laptop: "false", harbour: "false", quarry: "false"}, (
        f"with the pick taken back the menu checks {_checked(page)}"
    )
    close_menu(page)
    ask(page, upstream, f"without a terminal {tag}", reply.text("Without one."))
    assert _terminal_tools_sent_for(upstream, f"without a terminal {tag}") == set(), (
        "a terminal that was unpicked still reached the model"
    )


def test_a_pick_in_a_saved_chat_is_stored_with_the_chat_and_a_refused_one_is_not_stored(
    page_for, make_user, connect
):
    person = make_user()
    harbour, quarry, _ = _names()
    connections = connect([harbour, quarry], person)
    tag = uuid.uuid4().hex[:6]
    kept, other = _saved_chat(person, f"kept {tag}"), _saved_chat(person, f"other {tag}")
    page = page_for(person, path=f"/c/{kept}")
    expect_reply(page, "Noted.")

    open_terminal_menu(page)
    _pick(page, harbour, shown=harbour)
    assert _stored_terminal(person, kept) == connections[0]["id"], (
        f"the chat is stored with the terminal {_stored_terminal(person, kept)!r}"
    )

    # another chat has no pick of its own; coming back, the first chat has its terminal again
    page.goto(f"/c/{other}")
    expect_reply(page, "Noted.")
    expect(_terminal_button(page, "Terminal")).to_be_visible()
    expect(_terminal_button(page, harbour)).to_have_count(0)
    page.goto(f"/c/{kept}")
    expect(_terminal_button(page, harbour)).to_be_visible()

    open_terminal_menu(page, picked=harbour)
    _pick(page, harbour, shown="Terminal")
    assert _stored_terminal(person, kept) is None, (
        f"the unpicked chat is still stored with {_stored_terminal(person, kept)!r}"
    )

    # the pick shows at once; the server refusing to store it says why and the chat keeps none
    refusal = f"The chat could not be saved {tag}"
    page.route(
        f"**/api/v1/chats/{kept}",
        lambda route: (
            route.fulfill(status=500, json={"detail": refusal})
            if route.request.method == "POST"
            else route.continue_()
        ),
    )
    open_terminal_menu(page)
    listed_terminal(page, quarry).click()
    expect(page.get_by_text(refusal)).to_be_visible()
    expect(_terminal_button(page, quarry)).to_be_visible()
    assert _stored_terminal(person, kept) is None, (
        f"the refused pick is stored with the chat as {_stored_terminal(person, kept)!r}"
    )
    page.unroute(f"**/api/v1/chats/{kept}")
    page.goto(f"/c/{kept}")
    expect_reply(page, "Noted.")
    expect(_terminal_button(page, "Terminal")).to_be_visible()
    expect(_terminal_button(page, quarry)).to_have_count(0)


def test_a_picked_terminal_that_is_deleted_reads_unavailable_terminal_until_another_is_picked(
    page_for, make_user, admin, connect_terminals, direct_integrations
):
    direct_integrations(True)
    owner, person = make_user(role="admin"), make_user()
    tag = uuid.uuid4().hex[:6]
    deleted, left = f"Harbour {tag}", f"Quarry {tag}"
    connection = connect_terminals(read_grant(person.id))
    with admin.client() as client:
        ids = {
            name: create_template(client, connection, name=name)["terminal_id"]
            for name in (deleted, left)
        }
        for terminal_id in ids.values():
            share_template(client, terminal_id, read_grant(person.id))
    page, person_page = page_for(owner), page_for(person)
    for picking in (page, person_page):
        choose_terminal(picking, deleted)

    with admin.client() as client:
        gone = client.delete(f"{TEMPLATES}/{ids[deleted]}")
    assert gone.status_code in (200, 204), gone.text

    # for a person without Direct Integrations the pick is dropped with the terminal
    expect(_terminal_button(person_page, "Terminal")).to_be_visible()
    expect(_terminal_button(person_page, deleted)).to_have_count(0)
    # only the people the terminal is shared with are told at once; an admin's page keeps
    # listing it until it loads again
    page.reload()
    expect(_terminal_button(page, "Unavailable terminal")).to_be_visible()
    open_terminal_menu(page, picked="Unavailable terminal")
    assert _checked(page) == {left: "false"}, f"the menu lists {_checked(page)}"
    _pick(page, left, shown=left)
    expect(_terminal_button(page, "Unavailable terminal")).to_have_count(0)


def test_a_chat_input_too_narrow_for_the_terminal_offers_it_in_its_more_menu(
    page_for, make_user, connect
):
    person = make_user()
    harbour, quarry, _ = _names()
    connect([harbour, quarry], person)
    page = page_for(person)
    choose_terminal(page, harbour)
    open_input_menu(page)
    expect(_in_more_menu(page, harbour)).to_have_count(0)
    close_menu(page)

    page.set_viewport_size({"width": 400, "height": 800})

    expect(_terminal_button(page, harbour)).to_have_count(0)
    open_input_menu(page)
    _in_more_menu(page, harbour).click()
    listed_terminal(page, quarry).click()
    expect(
        page.get_by_role("menu").get_by_role("menuitem", name=quarry, exact=True)
    ).to_be_visible()
    close_menu(page)
    page.set_viewport_size({"width": 1920, "height": 1080})
    expect(_terminal_button(page, quarry)).to_be_visible()


def _chat_terminal_button(chat: ChatSurface, label: str) -> Locator:
    """The Terminal button of this chat's input, which reads the chat's picked terminal."""
    return chat.panel.get_by_role("button", name=label, exact=True)


def _pick_in(chat: ChatSurface, name: str, picked: str | None = None) -> None:
    """Open the chat's Terminal menu, whose button reads `picked`, and click the terminal `name`."""
    # a new chat page shows the button only once it has loaded
    expect(_chat_terminal_button(chat, picked or "Terminal")).to_be_visible()
    tooltip_button(chat.panel, picked or "Terminal").click()
    expect(chat.page.get_by_role("menu")).to_be_visible()
    listed_terminal(chat.page, name).click()
    expect(chat.page.get_by_role("menu")).to_have_count(0)


def _chat_asking(person, question: str) -> str:
    """The id of the person's saved chat whose messages hold `question`."""
    with person.client() as client:
        listed = client.get("/api/v1/chats/", params={"page": 1})
        assert listed.status_code == 200, listed.text
        for listed_chat in listed.json():
            stored = client.get(f"/api/v1/chats/{listed_chat['id']}")
            assert stored.status_code == 200, stored.text
            messages = stored.json()["chat"].get("history", {}).get("messages", {}).values()
            if any(question in str(message.get("content")) for message in messages):
                return listed_chat["id"]
    raise AssertionError(f"no saved chat holds {question!r}")


@pytest.mark.chat_surfaces(MAIN_CHAT)
def test_a_pick_in_a_saved_chat_is_stored_with_it_and_back_after_a_reload(
    chat_for, make_user, connect
):
    person = make_user()
    harbour, _, _ = _names()
    connections = connect([harbour], person)
    chat_id = _saved_chat(person, f"kept {uuid.uuid4().hex[:6]}")
    chat = chat_for(person, chat_id=chat_id)
    expect(chat.last_reply).to_contain_text("Noted.")

    _pick_in(chat, harbour)
    expect(_chat_terminal_button(chat, harbour)).to_be_visible()
    assert _stored_terminal(person, chat_id) == connections[0]["id"], (
        f"the chat is stored with the terminal {_stored_terminal(person, chat_id)!r}"
    )

    chat = chat.reload()
    expect(_chat_terminal_button(chat, harbour)).to_be_visible()


@pytest.mark.chat_surfaces(MAIN_CHAT)
def test_a_terminal_picked_before_the_first_message_is_saved_with_the_new_chat(
    chat_for, make_user, connect, upstream
):
    person = make_user()
    harbour, _, _ = _names()
    connections = connect([harbour], person)
    chat = chat_for(person)

    _pick_in(chat, harbour)
    expect(_chat_terminal_button(chat, harbour)).to_be_visible()
    question = f"first question {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("First answer.", match=reply.answering(question)))
    send(chat, question)
    expect_reply(chat, "First answer.")

    chat_id = _chat_asking(person, question)
    assert _stored_terminal(person, chat_id) == connections[0]["id"], (
        f"the new chat is stored with the terminal {_stored_terminal(person, chat_id)!r}"
    )
