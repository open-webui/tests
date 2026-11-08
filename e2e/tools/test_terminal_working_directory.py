"""Journey: a person picks the folder a chat's terminal works in, for one chat or for a folder.

New in 0.13.1. With a terminal picked in a new chat, the chat input shows a working directory
button beside the Terminal menu. Its menu lists recently used folders and "No folder", and "Open
folder" opens a picker where the person walks to a folder and presses "Use this directory"; the
button then shows the folder's name. Every command the model runs in the chat runs there. Once
the chat exists the button keeps showing its folder, also when the chat is opened again, and can
no longer be changed. A folder's Edit dialog holds, under Advanced, a Default terminal and a
Working directory on it (typed or picked with Browse directories); a new chat in the folder
starts on that terminal and in that directory, and a directory that does not exist refuses the
first message with a "Working directory is unavailable" toast before the model is asked (since
21da7627b the page opens the directory itself; the reply carried the error before). A model's
editor also offers a Default terminal, which a new chat on the model starts with. The recently
used folders are kept per person on the terminal, so another person on it sees none of them,
and the menu's search narrows them by name. Since a9478213a the picker clones a Git repository
from an HTTP(S) or SSH address into the folder it shows and uses the clone as the directory; an
address of another kind and a repository that cannot be cloned show the reason. A personal
terminal, one the person added in their own settings, is reached by the browser only, so the
browser has to move it to the chosen folder.

The personal terminal test was red on dev f659c0d12: the picked folder was set on the terminal
without the chat's session and the server leaves personal terminals alone since 5fae57b03, so the
commands ran in the terminal's home (open-webui/open-webui#33876). Since 21da7627b the page sets
the folder for the new chat's session itself.

A real Open Terminal serves the files and runs the commands; its `run_command` answer names the
folder the command ran in, which the test reads from what the model was sent.

Discriminates: passes on the dev f659c0d12 build, except the personal terminal test. With the
backend's directory call skipped in a backend copy, the picked-folder, recent-folder, folder
default and personal terminal tests fail. On a build of f659c0d12 whose working directory menu
leaves the chat's folder unset when one is picked, whose folder dialog saves no working directory
and whose model editor does not take a default terminal, the picked-folder, recent-folder, folder
dialog and model tests fail. The missing directory test passes on the dev a072aaf31 build and
fails on a build of it that sends the message although the terminal refused the directory.
On a copy of the dev e8c0ad85e build: recent folders kept for all people alike, a clone run with a
bad option and the clone error replaced by a fixed text turn the per-person, clone and both clone
error tests red; on a second copy, the menu search matching every folder and file:// addresses let
through turn the per-person and not-HTTP tests red.

Gap (strict xfail, not a bug in what exists): a folder's Default terminal lists only admin
terminals, so a person cannot make their own terminal a folder's default.

Switching a new chat to a model without a default terminal keeps the terminal picked in it
(b433aab65, a01b2ed55; the switch turned it off). The test passes on dev e8c0ad85e and fails on a
build of it whose model switch takes the model's default terminal, none when it has none.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import threading
import uuid
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.access import grant
from harness.backends import wait_until
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals
from harness.upstream import MOCK_MODEL_ID, conversation
from utils.cached_chat import ask, calling
from utils.chat_ui import REPLY_TIMEOUT_MS, chat_input
from utils.model_editor import save
from utils.model_selector import SELECTOR_BUTTON, select_model
from utils.sidebar_folders import click_folder_menu_button, folder_row, open_advanced
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

TERMINAL = "Workbench"
UNAVAILABLE = "Working directory is unavailable"
PERSONAL = "My laptop"
EVERYONE_READS = grant("user", "*", "read")


@pytest.fixture
def terminal(admin, open_terminal, preserve) -> dict:
    """The Open Terminal, connected by the admin for every account; returns the connection."""
    preserve(TERMINAL_SERVERS_CONFIG)
    connection = open_terminal.connection(name=TERMINAL, config={"access_grants": [EVERYONE_READS]})
    with admin.client() as client:
        configure_terminals(client, connection)
    return connection


def _directory(open_terminal, name: str = "harbour") -> Path:
    """A fresh folder in the terminal's home with one folder inside, so the picker lists it."""
    folder = open_terminal.home / f"{name}-{uuid.uuid4().hex[:8]}"
    (folder / "charts").mkdir(parents=True)
    return folder


def notifications(page: Page) -> Locator:
    return page.get_by_role("region", name=re.compile(r"^Notifications"))


def directory_button(page: Page) -> Locator:
    return page.get_by_role("main").get_by_role("button", name="Choose working directory")


def pick_terminal(page: Page, name: str) -> None:
    """Pick `name` in the chat input's Terminal menu; the working directory button then shows."""
    expect(chat_input(page)).to_be_visible()
    tooltip_button(page.get_by_role("main"), "Terminal").click()
    page.get_by_role("menu").get_by_role("menuitemradio", name=name).click()
    expect(directory_button(page)).to_be_visible()


def pick_in_picker(dialog: Locator, folder: Path) -> None:
    """Walk from the terminal's home into `folder` and use it."""
    dialog.get_by_text(folder.name, exact=True).click()
    expect(dialog.get_by_text("charts", exact=True)).to_be_visible()
    dialog.get_by_role("button", name="Use this directory").click()
    expect(dialog).to_have_count(0)


def choose_directory(page: Page, folder: Path) -> None:
    directory_button(page).click()
    page.get_by_role("menuitem", name="Open folder").click()
    pick_in_picker(page.get_by_role("dialog", name="Choose working directory"), folder)
    expect(directory_button(page)).to_contain_text(folder.name)


def where_pwd_ran(page: Page, upstream) -> str:
    """The model runs `pwd` in the open chat; the folder Open Terminal says it ran in."""
    prompt = f"where are we? {uuid.uuid4().hex[:8]}"
    ask(page, upstream, prompt, calling("run_command", {"command": "pwd"}), reply.text("found"))
    answered = [request for request in upstream.chat_requests() if reply.answering(prompt)(request)]
    results = [entry["content"] for entry in conversation(answered[-1]) if entry["role"] == "tool"]
    assert results, f"run_command never ran; the model was sent {conversation(answered[-1])}"
    return json.loads(results[-1])["cwd"]


def open_folder_page(person, page_for, folder_id: str) -> Page:
    page = page_for(person, path=f"/folders/{folder_id}")
    # the folder page reads the folder and its terminal's models before it shows the chat input,
    # the same cold load `send` allows for
    expect(chat_input(page)).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    return page


def create_folder(owner, data: dict) -> tuple[str, str]:
    name = f"Dock {uuid.uuid4().hex[:6]}"
    with owner.client() as client:
        created = client.post("/api/v1/folders/", json={"name": name, "data": data})
    assert created.status_code == 200, created.text
    return created.json()["id"], name


def stored_folder_data(owner, folder_id: str) -> dict:
    with owner.client() as client:
        found = client.get(f"/api/v1/folders/{folder_id}")
    assert found.status_code == 200, found.text
    return found.json().get("data") or {}


def open_folder_dialog(page: Page, name: str) -> Locator:
    expect(chat_input(page)).to_be_visible()
    opener = page.get_by_role("button", name="Open Sidebar", exact=True)
    if opener.is_visible():
        opener.click()
    sidebar = page.get_by_role("navigation", name="Chat history")
    folders = sidebar.get_by_role("button", name="Folders", exact=True)
    expect(folders).to_be_visible()
    if folders.get_attribute("aria-expanded") != "true":
        folders.click()
    click_folder_menu_button(folder_row(sidebar, name))
    page.get_by_role("menu").get_by_role("menuitem", name="Edit").click()
    dialog = page.get_by_role("dialog", name="Edit Folder")
    expect(dialog.get_by_placeholder("Enter folder name")).to_have_value(name)
    open_advanced(dialog)
    return dialog


def default_terminal(scope: Locator) -> Locator:
    return scope.get_by_role("button", name=re.compile(r"^Default terminal: "))


def choose_default_terminal(page: Page, scope: Locator, name: str) -> None:
    default_terminal(scope).click()
    page.get_by_role("button", name=name, exact=True).click()
    expect(default_terminal(scope)).to_have_accessible_name(f"Default terminal: {name}")


def test_commands_run_in_the_folder_picked_for_the_chat_and_it_stays_after_a_reload(
    page_for, make_user, terminal, open_terminal, upstream
):
    person = make_user()
    page = page_for(person)
    pick_terminal(page, TERMINAL)
    folder = _directory(open_terminal)

    choose_directory(page, folder)

    assert where_pwd_ran(page, upstream) == str(folder), "the command ran outside the folder"
    expect(page).to_have_url(re.compile("/c/"))
    reopened = page_for(person, path=urlparse(page.url).path)
    expect(directory_button(reopened)).to_contain_text(folder.name)
    # a518ff163: the folder can be changed again once the chat has messages
    expect(directory_button(reopened)).to_be_enabled()


def test_a_recently_used_folder_and_no_folder_can_be_picked_from_the_menu(
    page_for, make_user, terminal, open_terminal, upstream
):
    person = make_user()
    folder = _directory(open_terminal)
    first = page_for(person)
    pick_terminal(first, TERMINAL)
    choose_directory(first, folder)

    page = page_for(person)
    pick_terminal(page, TERMINAL)
    directory_button(page).click()
    page.get_by_role("menuitemradio", name=folder.name).click()
    expect(directory_button(page)).to_contain_text(folder.name)
    assert where_pwd_ran(page, upstream) == str(folder)

    again = page_for(person)
    pick_terminal(again, TERMINAL)
    directory_button(again).click()
    again.get_by_role("menuitemradio", name=folder.name).click()
    expect(directory_button(again)).to_contain_text(folder.name)
    directory_button(again).click()
    again.get_by_role("menuitemradio", name="No folder").click()
    expect(directory_button(again)).not_to_contain_text(folder.name)
    assert where_pwd_ran(again, upstream) == str(open_terminal.home)


def test_a_new_chat_in_a_folder_starts_on_its_terminal_and_directory(
    page_for, make_user, terminal, open_terminal, upstream
):
    person = make_user()
    folder = _directory(open_terminal)
    folder_id, _ = create_folder(
        person, {"terminal_id": terminal["id"], "working_directory": str(folder)}
    )
    page = open_folder_page(person, page_for, folder_id)

    expect(page.get_by_role("main").get_by_role("button", name=TERMINAL)).to_be_visible()
    expect(directory_button(page)).to_contain_text(folder.name)
    assert where_pwd_ran(page, upstream) == str(folder), "the folder's directory was not used"


def test_a_folders_missing_directory_refuses_the_first_message(
    page_for, make_user, terminal, open_terminal, upstream
):
    person = make_user()
    folder_id, _ = create_folder(
        person,
        {"terminal_id": terminal["id"], "working_directory": str(open_terminal.home / "gone")},
    )
    page = open_folder_page(person, page_for, folder_id)
    expect(directory_button(page)).to_contain_text("gone")
    before = len(upstream.chat_requests())

    chat_input(page).click()
    page.keyboard.type("anything there?")
    page.keyboard.press("Enter")

    expect(notifications(page).get_by_text(UNAVAILABLE)).to_be_visible()
    expect(chat_input(page)).to_contain_text("anything there?")
    assert len(upstream.chat_requests()) == before, "the model was asked anyway"


def test_the_folder_dialog_saves_a_default_terminal_and_directory(
    page_for, make_user, terminal, open_terminal
):
    person = make_user()
    folder = _directory(open_terminal)
    folder_id, name = create_folder(person, {})
    page = page_for(person)
    dialog = open_folder_dialog(page, name)
    expect(dialog.get_by_label("Working directory")).to_have_count(0)

    choose_default_terminal(page, dialog, TERMINAL)
    dialog.get_by_role("button", name="Browse directories").click()
    pick_in_picker(page.get_by_role("dialog", name="Choose working directory"), folder)
    expect(dialog.get_by_label("Working directory")).to_have_value(f"{folder}/")
    dialog.get_by_role("button", name="Save").click()
    expect(page.get_by_text("Folder updated successfully")).to_be_visible()

    data = stored_folder_data(person, folder_id)
    assert (data.get("terminal_id"), data.get("working_directory")) == (
        terminal["id"],
        f"{folder}/",
    ), data
    reopened = open_folder_dialog(page, name)
    expect(default_terminal(reopened)).to_have_accessible_name(f"Default terminal: {TERMINAL}")
    expect(reopened.get_by_label("Working directory")).to_have_value(f"{folder}/")


def test_clearing_the_folders_terminal_clears_its_directory(
    page_for, make_user, terminal, open_terminal
):
    person = make_user()
    folder_id, name = create_folder(
        person,
        {"terminal_id": terminal["id"], "working_directory": str(_directory(open_terminal))},
    )
    page = page_for(person)
    dialog = open_folder_dialog(page, name)

    choose_default_terminal(page, dialog, "None")
    expect(dialog.get_by_label("Working directory")).to_have_count(0)
    dialog.get_by_role("button", name="Save").click()
    expect(page.get_by_text("Folder updated successfully")).to_be_visible()

    data = stored_folder_data(person, folder_id)
    assert (data.get("terminal_id"), data.get("working_directory")) == (None, None), data


@pytest.fixture
def workbench_model(admin):
    model_id = f"bench-{uuid.uuid4().hex[:8]}"
    model = {
        "id": model_id,
        "base_model_id": MOCK_MODEL_ID,
        "name": model_id,
        "meta": {"capabilities": {"terminal": True}},
        "params": {},
        "access_grants": [EVERYONE_READS],
    }
    with admin.client() as client:
        created = client.post("/api/v1/models/create", json=model)
        assert created.status_code == 200, created.text
        yield model
        client.post("/api/v1/models/model/delete", json={"id": model_id})


def test_a_models_default_terminal_is_picked_for_its_new_chats(
    page_for, admin, terminal, workbench_model
):
    page = page_for(admin, path=f"/workspace/models/edit?id={workbench_model['id']}")
    editor = page.get_by_role("main")
    # the editor fills in after the app loads the model list, part of the same cold load
    expect(editor.get_by_placeholder("Model Name")).to_have_value(
        workbench_model["name"], timeout=REPLY_TIMEOUT_MS
    )

    choose_default_terminal(page, editor, TERMINAL)
    save(editor)

    with admin.client() as client:
        stored = client.get(f"/api/v1/models/model?id={workbench_model['id']}").json()
    assert stored["meta"].get("terminalId") == terminal["id"], stored["meta"]
    page = page_for(admin, path=f"/?model={workbench_model['id']}")
    expect(page.get_by_role("main").get_by_role("button", name=TERMINAL)).to_be_visible()
    expect(directory_button(page)).to_be_visible()


def test_switching_to_a_model_without_a_default_terminal_keeps_the_picked_one(
    page_for, make_user, terminal, open_terminal, workbench_model, upstream
):
    page = page_for(make_user())
    pick_terminal(page, TERMINAL)

    select_model(page, workbench_model["name"])

    expect(page.get_by_role("button", name=SELECTOR_BUTTON)).to_contain_text(
        workbench_model["name"]
    )
    expect(page.get_by_role("main").get_by_role("button", name=TERMINAL)).to_be_visible()
    assert where_pwd_ran(page, upstream) == str(open_terminal.home), (
        "the command did not run on the terminal picked before the model switch"
    )


def save_personal_terminal(person, open_terminal) -> None:
    """Add the Open Terminal as `person`'s own terminal, as Settings > Integrations does."""
    personal = {
        "url": open_terminal.base_url,
        "key": open_terminal.api_key,
        "name": PERSONAL,
        "enabled": False,
        "auth_type": "bearer",
        "path": "/openapi.json",
        "config": {"enable": True},
    }
    with person.client() as client:
        saved = client.post(
            "/api/v1/users/user/settings/update",
            json={"ui": {"showChangelog": False, "terminalServers": [personal]}},
        )
    assert saved.status_code == 200, saved.text


def test_a_personal_terminal_runs_in_the_folder_picked_for_the_chat(
    page_for, make_user, open_terminal, upstream, direct_integrations
):
    direct_integrations(True)
    person = make_user(role="admin")
    save_personal_terminal(person, open_terminal)
    page = page_for(person)
    pick_terminal(page, PERSONAL)
    folder = _directory(open_terminal)

    choose_directory(page, folder)

    assert where_pwd_ran(page, upstream) == str(folder), (
        "the personal terminal ran the command outside the picked folder"
    )


@pytest.mark.xfail(
    strict=True,
    reason="gap: a folder's Default terminal lists only admin terminals, never the user's own",
)
def test_a_folder_can_default_to_the_users_personal_terminal(
    page_for, admin, make_user, open_terminal, direct_integrations, preserve
):
    preserve(TERMINAL_SERVERS_CONFIG)
    with admin.client() as client:
        configure_terminals(client)
    direct_integrations(True)
    person = make_user(role="admin")
    save_personal_terminal(person, open_terminal)
    _, name = create_folder(person, {})
    page = page_for(person)
    dialog = open_folder_dialog(page, name)

    expect(dialog.get_by_text("No terminal servers available")).to_have_count(0)
    choose_default_terminal(page, dialog, PERSONAL)


def test_recent_folders_are_kept_for_each_person_and_the_menu_search_narrows_them(
    page_for, make_user, terminal, open_terminal
):
    person = make_user()
    north = _directory(open_terminal, "north-quay")
    south = _directory(open_terminal, "south-quay")
    for folder in (north, south):
        # the picker opens on the folder picked last, so each pick starts on a page of its own
        picking = page_for(person)
        pick_terminal(picking, TERMINAL)
        choose_directory(picking, folder)
    records = open_terminal.home / ".open-webui" / "users" / person.id / "folders"
    assert wait_until(lambda: len(list(records.glob("*.json"))) == 2, 10), (
        f"the two picked folders were not kept for the person on the terminal: {records}"
    )

    page = page_for(person)
    pick_terminal(page, TERMINAL)
    directory_button(page).click()
    expect(page.get_by_role("menuitemradio", name=north.name)).to_be_visible()
    expect(page.get_by_role("menuitemradio", name=south.name)).to_be_visible()
    page.get_by_placeholder("Search folders").fill(south.name)
    expect(page.get_by_role("menuitemradio", name=north.name)).to_have_count(0)
    expect(page.get_by_role("menuitemradio", name=south.name)).to_be_visible()

    other = page_for(make_user())
    pick_terminal(other, TERMINAL)
    directory_button(other).click()
    expect(other.get_by_role("menuitemradio", name="No folder")).to_be_visible()
    expect(other.get_by_role("menuitemradio", name=re.compile("quay"))).to_have_count(0)


@pytest.fixture(scope="module")
def git_server():
    """A local HTTP server holding one Git repository with a README; yields its address."""
    served = Path(tempfile.mkdtemp(prefix="git-served-"))
    work = served / "work"
    work.mkdir()
    (work / "README.md").write_text("# Lighthouse log\n")
    for command in (
        ["init", "-q", "-b", "main"],
        ["add", "README.md"],
        ["-c", "user.name=Ada", "-c", "user.email=ada@example.com", "commit", "-q", "-m", "Start"],
        ["clone", "-q", "--bare", ".", str(served / "lighthouse.git")],
    ):
        subprocess.run(["git", *command], cwd=work, check=True, capture_output=True, timeout=30)
    # served the dumb way, as plain files
    subprocess.run(
        ["git", "update-server-info"],
        cwd=served / "lighthouse.git",
        check=True,
        capture_output=True,
        timeout=30,
    )
    handler = partial(SimpleHTTPRequestHandler, directory=str(served))
    handler.log_message = lambda *args: None
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        shutil.rmtree(served, ignore_errors=True)


def open_picker_in(page: Page, folder: Path) -> Locator:
    directory_button(page).click()
    page.get_by_role("menuitem", name="Open folder").click()
    dialog = page.get_by_role("dialog", name="Choose working directory")
    dialog.get_by_text(folder.name, exact=True).click()
    expect(dialog.get_by_text("charts", exact=True)).to_be_visible()
    return dialog


def clone_in(dialog: Locator, address: str) -> None:
    dialog.get_by_role("button", name="Clone repository").click()
    dialog.get_by_label("Repository URL").fill(address)
    dialog.get_by_role("button", name="Clone repository").click()


def test_a_repository_cloned_in_the_picker_becomes_the_working_directory(
    page_for, make_user, terminal, open_terminal, upstream, git_server
):
    page = page_for(make_user())
    pick_terminal(page, TERMINAL)
    folder = _directory(open_terminal, "clones")
    dialog = open_picker_in(page, folder)

    clone_in(dialog, f"{git_server}/lighthouse.git")

    expect(dialog).to_have_count(0, timeout=REPLY_TIMEOUT_MS)
    expect(directory_button(page)).to_contain_text("lighthouse")
    clone = folder / "lighthouse"
    assert (clone / "README.md").read_text() == "# Lighthouse log\n", "the clone holds no README"
    assert where_pwd_ran(page, upstream) == str(clone), "the command ran outside the clone"


@pytest.mark.parametrize(
    "address, reason",
    [
        ("file:///srv/lighthouse.git", "Enter an HTTP(S) or SSH repository URL"),
        ("{git_server}/missing.git", "not found"),
    ],
    ids=["not-http-or-ssh", "missing-repository"],
)
def test_a_repository_that_cannot_be_cloned_says_why(
    page_for, make_user, terminal, open_terminal, git_server, address, reason
):
    page = page_for(make_user())
    pick_terminal(page, TERMINAL)
    folder = _directory(open_terminal, "clones")
    dialog = open_picker_in(page, folder)

    clone_in(dialog, address.format(git_server=git_server))

    expect(dialog.get_by_role("alert")).to_contain_text(reason, timeout=REPLY_TIMEOUT_MS)
    expect(directory_button(page)).not_to_contain_text("lighthouse")
    assert sorted(entry.name for entry in folder.iterdir()) == ["charts"], (
        f"a failed clone left something in the folder: {list(folder.iterdir())}"
    )
