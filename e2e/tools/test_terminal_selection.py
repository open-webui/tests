"""Journey: which terminal a chat is on, and where that choice is kept.

Since dev 6f2fc7bc7 a chat stores the terminal it was started with, and opening the chat picks
that terminal again, or none when the chat has none or the person can no longer use it. The
browser also remembers the last pick, which only decides what the next new chat starts with. A
layout saved for a chat names a terminal too; since 41179ba85 it no longer changes the chat's
own. A model can name a default terminal, which a new chat takes when all its models share it
and the person may use it; the model editor shows one that is gone as Unavailable, takes it off
with None, and offers the Terminal capability and the default only where a terminal can be had.
Changing the terminal drops the skills picked from the one before, and a file of another
terminal cannot be attached beside one already attached. The terminal list is read again when
the page reconnects and when the person's access changes (a group they were added to).

Two fake terminals stand in for two connections of the admin; each names itself in the answer of
its `run_command`, which the test reads from what the model was sent. The skills and attachments
tests use real Open Terminals.

The reload test is red on dev c15f28649 for a person who cannot add terminals of their own, an
admin with Direct Integrations off and a user without the Direct Tool Servers permission: finding
"The terminal picked for a new chat is dropped when the page is reloaded" (since 537e223ed). It
needs six reloads to keep the pick, as a user keeps it in one reload of three or four.

Discriminates: passes on the dev c15f28649 build except those two cases. A backend copy that
stores no terminal on a chat the server creates turns the each-chat test red. On a build with
these changes: a saved chat no longer picking its own terminal (each chat, started without; also
taken away and layout); the first model's default taken although the models disagree (models
share); no Unavailable entry (gone default); the Terminal capability offered without any
terminal (editor offers); no terminal list read on reconnect (granted while open, while the group
test still passes on the access change alone); the skills of the terminal before kept (skills);
a file of a second terminal let through on drop (attachments). On a second build: the layout's
terminal picked as before 41179ba85 (layout; also each chat, started without); None keeping the
model's default, so nothing is left to save (gone default); a default picked without the person
having the terminal (may use it); no terminal list read on reconnect or on an access change
(group, granted while open); a chat opening on a terminal the person lost (taken away, the admin
case; for a user the page's own check still unpicks it).
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from typing import Callable, Iterator
from urllib.parse import urlparse

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import skill_files
from harness import upstream as reply
from harness.access import grant, make_group
from harness.fixtures import SETTINGS
from harness.listener import json_answer
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    FakeTerminalServer,
    configure_terminals,
    read_grant,
    serving_terminal,
)
from harness.upstream import MOCK_MODEL_ID, conversation
from utils.cached_chat import (
    ask,
    calling,
    chat_requests,
    choose_terminal,
    offered_tools,
    pick_terminal,
)
from utils.chat_ui import (
    REPLY_TIMEOUT_MS,
    chat_input,
    close_menu,
    open_integrations_menu,
    show_input_selections,
    split_screen,
)
from utils.model_editor import section
from utils.user_terminals import listed_terminal, open_terminal_menu

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

QUAY = "Quay shell"
SLIP = "Slipway shell"
BOX = "Dry dock"
WORKBENCH = "Workbench"
MIXED_ATTACHMENTS = "Attached files must belong to this chat and the same terminal."
EVERYONE_READS = grant("user", "*", "read")
CONNECTION_LOST = "Connection lost. Reconnecting..."
RECONNECTED = "Reconnected"
# the socket retries up to 5 s apart
RECONNECT_TIMEOUT_MS = 15_000
RUN_COMMAND_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Terminal", "version": "1"},
    "paths": {
        "/execute": {
            "post": {"operationId": "run_command", "responses": {"200": {"description": "ok"}}}
        }
    },
}


def _serve(server, name: str) -> None:
    server.route("GET", "/openapi.json", json_answer(RUN_COMMAND_SPEC))
    server.route("POST", "/execute", json_answer({"ran_on": name}))
    # a new chat starts in the folder the page names, as on Open Terminal
    server.route(
        "POST",
        "/files/cwd",
        lambda request: json_answer({"cwd": json.loads(request.body)["path"]}),
    )


@pytest.fixture
def servers(preserve) -> Iterator[dict[str, FakeTerminalServer]]:
    """Two fake terminals by name; the test connects them."""
    preserve(TERMINAL_SERVERS_CONFIG)
    with serving_terminal() as quay, serving_terminal() as slip:
        _serve(quay, QUAY)
        _serve(slip, SLIP)
        yield {QUAY: quay, SLIP: slip}


def connect(admin, servers, **grants: list[dict]) -> dict[str, dict]:
    """The admin saves a connection to each terminal named in `grants` (`quay=[...]`,
    `slip=[...]`) with those grants, replacing the current ones; the connections by name. A
    terminal keeps its id from one call to the next."""
    names = {"quay": QUAY, "slip": SLIP}
    connections = {}
    for key, granted in grants.items():
        server = servers[names[key]]
        connections[names[key]] = server.connection(
            id=f"{key}-{server.base_url.rsplit(':', 1)[1]}",
            name=names[key],
            config={"access_grants": granted},
        )
    with admin.client() as client:
        configure_terminals(client, *connections.values())
    return connections


@pytest.fixture
def terminals(admin, servers) -> dict[str, dict]:
    """Both terminals connected for every account; the connection of each by its name."""
    return connect(admin, servers, quay=[EVERYONE_READS], slip=[EVERYONE_READS])


def terminal_button(page: Page, shown: str = "Terminal") -> Locator:
    """The chat input's Terminal button; it reads the picked terminal's name, else Terminal."""
    return page.get_by_role("main").get_by_role("button", name=shown, exact=True)


def expect_terminal(page: Page, name: str | None) -> None:
    """The chat input shows `name` as the chat's terminal, or none picked."""
    expect(chat_input(page)).to_be_visible()
    expect(terminal_button(page, name or "Terminal")).to_be_visible()
    for other in {QUAY, SLIP, "Unavailable terminal"} - {name}:
        expect(terminal_button(page, other), f"{other} shows as picked").to_have_count(0)


def switch_terminal(page: Page, picked: str, name: str) -> None:
    """With `picked` as the chat's terminal, pick `name` in the Terminal menu."""
    open_terminal_menu(page, picked)
    listed_terminal(page, name).click()
    expect(terminal_button(page, name)).to_be_visible()
    expect(page.get_by_role("menu")).to_have_count(0)


def where_the_command_ran(page: Page, upstream) -> str:
    """The model runs a command in the open chat; the name of the terminal that answered."""
    prompt = f"which berth? {uuid.uuid4().hex[:8]}"
    ask(page, upstream, prompt, calling("run_command", {"command": "hostname"}), reply.text("ok"))
    answered = [request for request in upstream.chat_requests() if reply.answering(prompt)(request)]
    results = [entry["content"] for entry in conversation(answered[-1]) if entry["role"] == "tool"]
    assert results, f"run_command gave the model nothing: {conversation(answered[-1])}"
    assert '"ran_on"' in results[-1], f"no terminal ran the command: {results[-1]}"
    return json.loads(results[-1])["ran_on"]


def chat_id_of(page: Page) -> str:
    expect(page).to_have_url(re.compile(r"/c/[0-9a-f-]+"))
    return urlparse(page.url).path.removeprefix("/c/")


def stored_chat(person, chat_id: str) -> dict:
    with person.client() as client:
        found = client.get(f"/api/v1/chats/{chat_id}")
    assert found.status_code == 200, found.text
    return found.json()["chat"]


def load_chat(page: Page, chat_id: str) -> None:
    """Load the chat's page anew in the same browser."""
    page.goto(f"/c/{chat_id}")
    # a cold load, which `send` allows the same time for
    expect(chat_input(page)).to_be_visible(timeout=REPLY_TIMEOUT_MS)


def open_chat(page: Page, chat_id: str) -> None:
    """Open the chat from the sidebar, without loading the page again."""
    opener = page.get_by_role("button", name="Open Sidebar", exact=True)
    if opener.is_visible():
        opener.click()
    # every chat of the test is titled New Chat, so its row is found by where it leads
    sidebar = page.get_by_role("navigation", name="Chat history")
    sidebar.locator(f'a[href="/c/{chat_id}"]').first.click()
    expect(page).to_have_url(re.compile(rf"/c/{chat_id}$"))
    expect(chat_input(page)).to_be_visible()


def start_new_chat(page: Page) -> None:
    page.get_by_role("link", name="New Chat", exact=True).first.click()
    expect(page).to_have_url(re.compile(r"/$"))
    expect(chat_input(page)).to_be_visible()


def test_each_chat_opens_on_the_terminal_it_was_started_with(
    page_for, make_user, terminals, upstream
):
    person = make_user()
    page = page_for(person)
    choose_terminal(page, QUAY)
    assert where_the_command_ran(page, upstream) == QUAY
    on_quay = chat_id_of(page)

    start_new_chat(page)
    expect_terminal(page, QUAY)
    switch_terminal(page, QUAY, SLIP)
    assert where_the_command_ran(page, upstream) == SLIP
    on_slip = chat_id_of(page)

    assert stored_chat(person, on_quay).get("terminal_id") == terminals[QUAY]["id"]
    assert stored_chat(person, on_slip).get("terminal_id") == terminals[SLIP]["id"]
    # Slipway is the page's last pick; the first chat still opens on its own
    open_chat(page, on_quay)
    expect_terminal(page, QUAY)
    assert where_the_command_ran(page, upstream) == QUAY, "the reopened chat ran elsewhere"
    open_chat(page, on_slip)
    expect_terminal(page, SLIP)
    # and so it does in a browser that remembers no pick
    expect_terminal(page_for(person, path=f"/c/{on_quay}"), QUAY)


def offered_to_the_model(page: Page, upstream) -> set[str]:
    """The tools the model is offered for a message sent in the open chat."""
    prompt = f"anything to do? {uuid.uuid4().hex[:8]}"
    ask(page, upstream, prompt, reply.text("nothing"))
    answered = [request for request in chat_requests(upstream) if reply.answering(prompt)(request)]
    assert answered, "the model was never asked"
    return offered_tools(answered[-1])


def test_a_chat_started_without_a_terminal_opens_without_one_whatever_is_picked_now(
    page_for, make_user, terminals, upstream
):
    person = make_user()
    page = page_for(person)
    assert "run_command" not in offered_to_the_model(page, upstream)
    plain = chat_id_of(page)
    start_new_chat(page)
    choose_terminal(page, QUAY)
    assert where_the_command_ran(page, upstream) == QUAY

    open_chat(page, plain)

    expect_terminal(page, None)
    offered = offered_to_the_model(page, upstream)
    assert "run_command" not in offered, f"the chat without a terminal was handed one: {offered}"
    assert stored_chat(person, plain).get("terminal_id") is None, stored_chat(person, plain)


@pytest.mark.parametrize(
    "role, own_terminals",
    [("user", False), ("admin", True)],
    ids=["user-loses-the-grant", "admin-with-direct-integrations-loses-the-connection"],
)
def test_a_chat_whose_terminal_was_taken_away_opens_with_none_picked(
    page_for, make_user, admin, servers, upstream, direct_integrations, role, own_terminals
):
    """A person who may add terminals of their own keeps a pick the admin's list does not hold,
    so for them the chat alone decides that its terminal is gone."""
    direct_integrations(own_terminals)
    person = make_user(role=role)
    mine = [read_grant(person.id)]
    connect(admin, servers, quay=mine, slip=mine)
    page = page_for(person)
    choose_terminal(page, QUAY)
    assert where_the_command_ran(page, upstream) == QUAY
    chat_id = chat_id_of(page)
    load_chat(page, chat_id)
    expect_terminal(page, QUAY)

    if role == "admin":
        connect(admin, servers, slip=mine)
    else:
        connect(admin, servers, quay=[], slip=mine)
    load_chat(page, chat_id)

    expect_terminal(page, None)
    open_terminal_menu(page)
    expect(listed_terminal(page, SLIP)).to_be_visible()
    expect(listed_terminal(page, QUAY), "the terminal taken away is still listed").to_have_count(0)


def reload_and_expect_still_picked(page: Page, name: str) -> None:
    page.reload()
    # a cold load, which `send` allows the same time for
    expect(chat_input(page)).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    composer = page.get_by_role("main")
    composer.get_by_role("button", name=re.compile(f"^({name}|Terminal)$")).click()
    # the terminal is listed once the page has its terminals; only then is the pick settled
    expect(listed_terminal(page, name)).to_be_visible()
    expect(listed_terminal(page, name), f"the reload dropped the pick of {name}").to_have_attribute(
        "aria-checked", "true"
    )
    page.keyboard.press("Escape")
    expect(page.get_by_role("menu")).to_have_count(0)
    expect_terminal(page, name)


def allow_direct_tool_servers(admin, allowed: bool) -> None:
    """The default Direct Tool Servers permission: whether a user may add terminals of their own
    while Direct Integrations is on."""
    read_path, write_path = SETTINGS["permissions"]
    with admin.client() as client:
        current = client.get(read_path).json()
        current["features"]["direct_tool_servers"] = allowed
        saved = client.post(write_path, json=current)
    assert saved.status_code == 200, saved.text


@pytest.mark.parametrize(
    "role, direct_integrations_on, direct_tool_servers",
    [("admin", True, False), ("user", True, True), ("admin", False, False), ("user", True, False)],
    ids=[
        "admin-with-direct-integrations",
        "user-with-direct-tool-servers",
        "admin",
        "user",
    ],
)
def test_the_terminal_picked_for_a_new_chat_is_still_picked_after_a_reload(
    page_for,
    make_user,
    admin,
    terminals,
    direct_integrations,
    preserve,
    role,
    direct_integrations_on,
    direct_tool_servers,
):
    """Red on dev c15f28649 for a person who cannot add terminals of their own, an admin with
    Direct Integrations off and a user without the Direct Tool Servers permission: the pick is
    dropped while the page loads (finding "The terminal picked for a new chat is dropped when the
    page is reloaded")."""
    preserve("permissions")
    direct_integrations(direct_integrations_on)
    allow_direct_tool_servers(admin, direct_tool_servers)
    page = page_for(make_user(role=role))
    choose_terminal(page, QUAY)

    # the pick is lost in most reloads, not in all of them
    for _ in range(6):
        reload_and_expect_still_picked(page, QUAY)


def test_a_terminal_granted_while_the_page_is_open_shows_once_the_page_reconnects(
    page_for, make_user, admin, servers
):
    person = make_user()
    connect(admin, servers, quay=[], slip=[read_grant(person.id)])
    page = page_for(person)
    open_terminal_menu(page)
    expect(listed_terminal(page, SLIP)).to_be_visible()
    expect(listed_terminal(page, QUAY)).to_have_count(0)
    page.keyboard.press("Escape")
    expect(page.get_by_role("menu")).to_have_count(0)

    connect(admin, servers, quay=[read_grant(person.id)], slip=[read_grant(person.id)])
    page.context.set_offline(True)
    expect(page.get_by_text(CONNECTION_LOST)).to_be_visible()
    page.context.set_offline(False)
    expect(page.get_by_text(RECONNECTED)).to_be_visible(timeout=RECONNECT_TIMEOUT_MS)

    open_terminal_menu(page)
    expect(listed_terminal(page, QUAY)).to_be_visible()
    expect(listed_terminal(page, SLIP)).to_be_visible()


def test_a_person_added_to_a_group_gets_the_groups_terminal_without_a_reload(
    page_for, make_user, admin, servers
):
    person = make_user()
    group_id = make_group(admin, [])
    connect(admin, servers, quay=[grant("group", group_id, "read")], slip=[read_grant(person.id)])
    page = page_for(person)
    open_terminal_menu(page)
    expect(listed_terminal(page, SLIP)).to_be_visible()
    expect(listed_terminal(page, QUAY)).to_have_count(0)
    page.keyboard.press("Escape")
    expect(page.get_by_role("menu")).to_have_count(0)

    with admin.client() as client:
        added = client.post(
            f"/api/v1/groups/id/{group_id}/users/add", json={"user_ids": [person.id]}
        )
    assert added.status_code == 200, added.text

    open_terminal_menu(page)
    expect(listed_terminal(page, QUAY)).to_be_visible(timeout=RECONNECT_TIMEOUT_MS)
    listed_terminal(page, QUAY).click()
    expect(terminal_button(page, QUAY)).to_be_visible()


@pytest.fixture
def make_model(admin) -> Iterator[Callable[..., dict]]:
    """`make_model(terminal_id)` saves a model every account may use with that default terminal
    (none without); the models are deleted afterwards."""
    made: list[str] = []

    def create(terminal_id: str | None = None) -> dict:
        model_id = f"berth-{uuid.uuid4().hex[:8]}"
        meta: dict = {"capabilities": {"terminal": True}}
        if terminal_id:
            meta["terminalId"] = terminal_id
        model = {
            "id": model_id,
            "base_model_id": MOCK_MODEL_ID,
            "name": model_id,
            "meta": meta,
            "params": {},
            "access_grants": [EVERYONE_READS],
        }
        with admin.client() as client:
            created = client.post("/api/v1/models/create", json=model)
        assert created.status_code == 200, created.text
        made.append(model_id)
        return model

    yield create
    with admin.client() as client:
        for model_id in made:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


def open_chat_on(person, page_for, *models: dict) -> Page:
    """A new chat of `person` on `models`, in a browser that remembers no pick."""
    page = page_for(person, path=f"/?models={','.join(model['id'] for model in models)}")
    # with several models the button reads the first and how many more
    selected = re.compile(rf"^Selected model: {models[0]['name']}( \+{len(models) - 1})?$")
    expect(page.get_by_role("button", name=selected)).to_be_visible()
    return page


def test_a_new_chat_starts_on_the_default_terminal_all_its_models_share(
    page_for, make_user, terminals, make_model, upstream
):
    on_quay, also_on_quay = make_model(terminals[QUAY]["id"]), make_model(terminals[QUAY]["id"])
    on_slip = make_model(terminals[SLIP]["id"])
    person = make_user()

    alone = open_chat_on(person, page_for, on_quay)
    expect_terminal(alone, QUAY)
    assert where_the_command_ran(alone, upstream) == QUAY

    expect_terminal(open_chat_on(person, page_for, on_quay, also_on_quay), QUAY)
    expect_terminal(open_chat_on(person, page_for, on_slip), SLIP)
    # models that disagree leave the chat as it was, here without a terminal
    expect_terminal(open_chat_on(person, page_for, on_quay, on_slip), None)


def test_a_models_default_terminal_is_picked_only_for_a_person_who_may_use_it(
    page_for, make_user, admin, servers, make_model, direct_integrations, preserve
):
    """The people may add terminals of their own: the page then keeps a pick the admin's list
    does not hold, so the model's default alone decides whether it is picked."""
    preserve("permissions")
    direct_integrations(True)
    allow_direct_tool_servers(admin, True)
    allowed, other = make_user(), make_user()
    connections = connect(
        admin,
        servers,
        quay=[read_grant(allowed.id)],
        slip=[read_grant(allowed.id), read_grant(other.id)],
    )
    model = make_model(connections[QUAY]["id"])

    expect_terminal(open_chat_on(allowed, page_for, model), QUAY)

    page = open_chat_on(other, page_for, model)
    expect_terminal(page, None)
    open_terminal_menu(page)
    expect(listed_terminal(page, SLIP)).to_be_visible()
    expect(listed_terminal(page, QUAY)).to_have_count(0)


def default_terminal(editor: Locator) -> Locator:
    return editor.get_by_role("button", name=re.compile(r"^Default terminal: "))


def open_model_editor(person, page_for, model: dict) -> Locator:
    page = page_for(person, path=f"/workspace/models/edit?id={model['id']}")
    editor = page.get_by_role("main")
    # the editor fills in after the app loads the model list, part of the same cold load
    expect(editor.get_by_placeholder("Model Name")).to_have_value(
        model["name"], timeout=REPLY_TIMEOUT_MS
    )
    return editor


def stored_model(admin, model: dict) -> dict:
    with admin.client() as client:
        found = client.get(f"/api/v1/models/model?id={model['id']}")
    assert found.status_code == 200, found.text
    return found.json()


def test_a_default_terminal_that_is_gone_reads_unavailable_and_none_takes_it_off_the_model(
    page_for, admin, terminals, make_model
):
    gone = f"gone-{uuid.uuid4().hex[:8]}"
    model = make_model(gone)
    editor = open_model_editor(admin, page_for, model)
    page = editor.page
    expect(default_terminal(editor)).to_have_accessible_name("Default terminal: Unavailable")

    default_terminal(editor).click()
    for offered in ("None", "Unavailable", QUAY, SLIP):
        expect(page.get_by_role("button", name=offered, exact=True)).to_be_visible()
    page.get_by_role("button", name="None", exact=True).click()
    expect(default_terminal(editor)).to_have_accessible_name("Default terminal: None")

    def updates_the_model(response) -> bool:
        return response.request.method == "POST" and "/api/v1/models/model/update" in response.url

    save = editor.get_by_role("button", name="Save & Update")
    expect(save, "taking the default off left nothing to save").to_be_enabled()
    with page.expect_response(updates_the_model) as saved:
        save.click()
    assert saved.value.ok, saved.value.text()
    meta = stored_model(admin, model)["meta"]
    assert "terminalId" not in meta, f"the model still names a default terminal: {meta}"
    assert meta.get("capabilities", {}).get("terminal") is True, meta


def test_the_model_editor_offers_a_terminal_only_where_one_can_be_had(
    page_for, make_user, admin, servers, make_model, direct_integrations
):
    builder = make_user(role="admin")
    model = make_model()
    connect(admin, servers, quay=[EVERYONE_READS])
    direct_integrations(False)
    editor = open_model_editor(builder, page_for, model)
    capability = section(editor, "Capabilities").get_by_role("switch", name="Terminal", exact=True)
    expect(capability).to_be_checked()
    expect(default_terminal(editor)).to_have_accessible_name("Default terminal: None")

    # no connection, but people may add terminals of their own
    connect(admin, servers)
    direct_integrations(True)
    editor = open_model_editor(builder, page_for, model)
    capability = section(editor, "Capabilities").get_by_role("switch", name="Terminal", exact=True)
    expect(capability).to_be_checked()
    expect(editor.get_by_text("No terminal servers available")).to_be_visible()
    expect(default_terminal(editor)).to_have_count(0)

    direct_integrations(False)
    editor = open_model_editor(builder, page_for, model)
    capabilities = section(editor, "Capabilities")
    expect(capabilities.get_by_role("switch", name="Vision", exact=True)).to_be_visible()
    expect(capabilities.get_by_role("switch", name="Terminal", exact=True)).to_have_count(0)
    expect(editor.get_by_text("No terminal servers available")).to_have_count(0)
    expect(default_terminal(editor)).to_have_count(0)


def save_layout(person, chat_id: str, terminal_id: str) -> None:
    """Store a layout for the chat that names `terminal_id`, as the page saves one."""
    conversation_only = {
        "tabs": [{"id": "main-chat", "kind": "chat", "label": "Chat"}],
        "activeId": "main-chat",
        "history": [],
        "groups": [{"id": "main", "tabIds": ["main-chat"], "activeId": "main-chat"}],
        "activeGroupId": "main",
        "layout": {"type": "group", "groupId": "main"},
    }
    state = {"schemaVersion": 1, "selectedTerminalId": terminal_id, "panels": conversation_only}
    with person.client() as client:
        saved = client.post(f"/api/v1/chats/{chat_id}/view", json={"state": state})
    assert saved.status_code == 200, saved.text


def test_a_layout_saved_with_another_terminal_leaves_the_chat_on_its_own(
    page_for, make_user, terminals, upstream
):
    """Until dev 41179ba85 the terminal saved with a chat's layout was picked when the chat
    opened, whatever the chat itself was on."""
    person = make_user()
    page = page_for(person)
    choose_terminal(page, QUAY)
    assert where_the_command_ran(page, upstream) == QUAY
    chat_id = chat_id_of(page)
    # the open page saves its own layout as it changes
    page.close()
    save_layout(person, chat_id, terminals[SLIP]["id"])

    reopened = page_for(person, path=f"/c/{chat_id}")

    expect_terminal(reopened, QUAY)
    assert where_the_command_ran(reopened, upstream) == QUAY, (
        "the chat ran on the layout's terminal"
    )


def test_changing_the_terminal_drops_the_skills_picked_from_the_one_before(
    page_for, make_user, admin, servers, open_terminal
):
    person = make_user(role="admin")
    show_input_selections(person)
    # the menu lists skills once the person has one of their own
    skill_files.create(person, {"SKILL.md": skill_files.skill_md("Tide tables", "Read them.")})
    skill = f"knots-{uuid.uuid4().hex[:6]}"
    skill_folder = open_terminal.home / ".agents" / "skills" / skill
    skill_folder.mkdir(parents=True)
    (skill_folder / "SKILL.md").write_text(skill_files.skill_md(skill, "Use a bowline."))
    with admin.client() as client:
        configure_terminals(
            client, open_terminal.connection(name=WORKBENCH), servers[SLIP].connection(name=SLIP)
        )
    try:
        page = page_for(person)
        choose_terminal(page, WORKBENCH)
        menu = open_integrations_menu(page)
        menu.get_by_role("menuitem", name=re.compile(r"^Skills")).click()
        entry = menu.get_by_role("menuitemcheckbox", name=re.compile(skill))
        expect(entry).to_contain_text("Terminal")
        entry.click()
        expect(entry).to_have_attribute("aria-checked", "true")
        close_menu(page)
        # beside the + the chat input counts the skills picked for the next message
        picked_skills = page.get_by_role("main").get_by_role("button", name="Available Skills")
        expect(picked_skills).to_have_text("1")

        switch_terminal(page, WORKBENCH, SLIP)

        expect(picked_skills, "the skill of the terminal before is still picked").to_have_count(0)
    finally:
        shutil.rmtree(skill_folder)
        skill_files.delete_skills_of(person)


def test_files_of_two_terminals_cannot_be_attached_to_one_message(
    page_for, make_user, admin, preserve, open_terminal, agent_terminal
):
    preserve(TERMINAL_SERVERS_CONFIG)
    person = make_user()
    tag = uuid.uuid4().hex[:6]
    tides = open_terminal.home / f"tides-{tag}.txt"
    tides.write_text("high water 06:12\n")
    # the agent terminal is a second real Open Terminal here
    knots = agent_terminal.terminal.home / f"knots-{tag}.txt"
    knots.write_text("bowline\n")
    mine = {"access_grants": [read_grant(person.id)]}
    with admin.client() as client:
        configure_terminals(
            client,
            open_terminal.connection(name=WORKBENCH, config=mine),
            agent_terminal.connection(name=BOX, config=mine),
        )
    page = page_for(person)
    pick_terminal(page, WORKBENCH)
    split_screen(page)
    files = page.get_by_role("region", name="File browser")
    attached = page.locator("form")

    files.get_by_text(tides.name, exact=True).drag_to(chat_input(page))
    expect(attached.get_by_text(tides.name)).to_be_visible()

    switch_terminal(page, WORKBENCH, BOX)
    files.get_by_text(knots.name, exact=True).drag_to(chat_input(page))

    notifications = page.get_by_role("region", name=re.compile(r"^Notifications"))
    expect(notifications.get_by_text(MIXED_ATTACHMENTS)).to_be_visible()
    expect(
        attached.get_by_text(knots.name), "the other terminal's file was attached"
    ).to_have_count(0)
    expect(attached.get_by_text(tides.name)).to_be_visible()
