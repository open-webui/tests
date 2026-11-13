"""Journey: the chat input's Terminal menu lists a person's terminals and keeps the chat's pick.

Rebuilt on dev after 0.13.3 (3d98e86e2, f4cf2ca0c, c647f3f42, 41179ba85, titled `refac`). The
menu lists the terminals of the admin's connections under Terminals; the cog beside
Terminals, for admins only, opens the admin's Integrations settings. The picked terminal carries
a check and names the menu's button; picking it again unpicks it, and picking another terminal
unpicks the first.
In a saved chat the pick is stored on the chat and is back when the chat is opened again. Since
d42988f0c a pick shows at once and is saved in the background, so one the server refuses stays
picked on the page, says why and is not stored: the chat opens again without it (before, a
refused pick was not made). A picked terminal that is deleted is unpicked at once for a
person it was shared with, and reads Unavailable terminal on an admin's page after a reload.
When the input is too narrow for the menu's button to show the terminal's name, the button keeps
only its icon and the chat input's More menu offers the menu as well (e4b969f60; before, the
button left the input).
While a reply is being written, the Terminal button and the working directory button beside it
cannot be used; once it is written both can be changed again, also in a chat that already has
messages (a518ff163, 0.13.3; before, both stayed locked after the first message).

A local fake terminal stands for the admin's connection; the deleted terminal is one on the
real Terminals service on Docker (`harness.terminals`), and its test skips without the images or
Docker. New terminal, the third header entry, is in
test_new_terminal_dialog.py and test_terminal_settings_dialog.py (only for a person who may
create terminals), and the menu missing before the first terminal exists is the finding
`open-webui-no-terminal-menu-before-the-first-terminal`.

Discriminates: passes on the dev c15f28649 build with Terminals 0.4.1 and Open Terminal 0.17.7. On
builds of a frontend copy with one change each: the Settings cog shown to everyone (the cog test);
no terminal carrying the check, the picked terminal kept when it is picked again (the picked test);
the pick not stored on a saved chat, a saved chat opened without its terminal, and a refused pick
that says nothing (the saved chat test); a lost terminal reading Terminal, and one that stays picked
for everyone (the deleted test); the controls never moved into the More menu (the narrow test). The
reply-written test is red on dev 485103cde and on 0.13.3: both buttons stay usable while the reply
is written (#34383); the lock it expects is in the 0.13.3 release notes, and only the MoA merge
sets the flag the input reads. It passes with the fix on branch
fix/terminal-folder-unlocked-during-reply (bde3943d3). The cog's target has no mutation of its
own. A person's own terminals, listed under Direct, were removed from Open WebUI in 44d55bd0f
and 4720ff19e and their test with them.
"""

from __future__ import annotations

import threading
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
from utils.chat_ui import close_menu, expect_reply, last_reply, open_input_menu, send
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


def _names() -> tuple[str, str]:
    tag = uuid.uuid4().hex[:6]
    return f"Harbour {tag}", f"Quarry {tag}"


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
    """Whether the terminal's tool was offered to the model with `prompt`."""
    asked = [request for request in chat_requests(upstream) if reply.answering(prompt)(request)]
    assert asked, f"the provider never got {prompt!r}"
    return offered_tools(asked[-1]) & {"run_on_server"}


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


def test_the_terminals_heading_offers_admins_alone_a_cog_to_their_integrations_settings(
    page_for, make_user, connect
):
    owner, person = make_user(role="admin"), make_user()
    shared, _ = _names()
    connect([shared], person)
    page = page_for(owner)

    menu = open_terminal_menu(page)
    assert [line.lower() for line in _menu_lines(menu)] == ["terminals", shared.lower()], (
        f"the admin's Terminal menu reads {_menu_lines(menu)}"
    )
    tooltip_button(menu, "Settings", role="menuitem").click()
    settings = page.get_by_role("dialog").filter(has=page.get_by_role("tab", name="Integrations"))
    expect(_settings_tab(settings, "Admin")).to_have_attribute("aria-selected", "true")
    expect(settings.get_by_text(shared)).to_be_visible()

    person_page = page_for(person)
    menu = open_terminal_menu(person_page)
    assert [line.lower() for line in _menu_lines(menu)] == ["terminals", shared.lower()], (
        f"a regular person's Terminal menu reads {_menu_lines(menu)}"
    )
    expect(menu.get_by_role("menuitem", name="Settings")).to_have_count(0)


def test_the_picked_terminal_is_checked_and_named_and_picking_it_again_unpicks_it(
    page_for, make_user, connect, upstream
):
    person = make_user(role="admin")
    harbour, quarry = _names()
    connect([harbour, quarry])
    page = page_for(person)
    tag = uuid.uuid4().hex[:6]

    open_terminal_menu(page)
    assert _checked(page) == {harbour: "false", quarry: "false"}
    _pick(page, harbour, shown=harbour)
    open_terminal_menu(page, picked=harbour)
    assert _checked(page) == {harbour: "true", quarry: "false"}, (
        f"with {harbour} picked the menu checks {_checked(page)}"
    )
    close_menu(page)
    ask(page, upstream, f"on the server {tag}", reply.text("On the server."))
    assert _terminal_tools_sent_for(upstream, f"on the server {tag}") == {"run_on_server"}

    # another terminal takes the pick
    open_terminal_menu(page, picked=harbour)
    _pick(page, quarry, shown=quarry)
    open_terminal_menu(page, picked=quarry)
    assert _checked(page) == {harbour: "false", quarry: "true"}, (
        f"with {quarry} picked the menu checks {_checked(page)}"
    )

    _pick(page, quarry, shown="Terminal")
    open_terminal_menu(page)
    assert _checked(page) == {harbour: "false", quarry: "false"}, (
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
    harbour, quarry = _names()
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


def test_the_terminal_and_its_folder_cannot_be_changed_while_a_reply_is_written(
    page_for, make_user, connect, upstream
):
    person = make_user()
    harbour, _ = _names()
    connect([harbour], person)
    page = page_for(person)
    open_terminal_menu(page)
    _pick(page, harbour, shown=harbour)
    folder = page.get_by_role("main").get_by_role("button", name="Choose working directory")
    expect(folder).to_be_enabled()

    question = f"take your time {uuid.uuid4().hex[:6]}"
    released = threading.Event()
    upstream.queue(
        reply.text(
            ["Still writing ", "and done now."],
            hold=released,
            hold_after=2,  # the role chunk, then the first piece
            match=reply.answering(question),
        )
    )
    try:
        send(page, question)
        expect(last_reply(page)).to_contain_text("Still writing")
        expect(_terminal_button(page, harbour)).to_be_disabled()
        expect(folder).to_be_disabled()
    finally:
        released.set()

    expect_reply(page, "and done now.")
    # a518ff163: once the reply is written both can be changed again
    expect(_terminal_button(page, harbour)).to_be_enabled()
    expect(folder).to_be_enabled()


def test_a_picked_terminal_that_is_deleted_reads_unavailable_terminal_until_another_is_picked(
    page_for, make_user, admin, connect_terminals
):
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

    # for the person it was shared with the pick is dropped with the terminal
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
    harbour, quarry = _names()
    connect([harbour, quarry], person)
    page = page_for(person)
    choose_terminal(page, harbour)
    shown_name = _terminal_button(page, harbour).get_by_text(harbour, exact=True)
    expect(shown_name).to_be_visible()
    open_input_menu(page)
    expect(_in_more_menu(page, harbour)).to_have_count(0)
    close_menu(page)

    page.set_viewport_size({"width": 400, "height": 800})

    # the button stays as an icon without the name (e4b969f60)
    expect(_terminal_button(page, harbour)).to_be_visible()
    expect(shown_name).to_be_hidden()
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
    harbour, _ = _names()
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
    harbour, _ = _names()
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
