"""Journey: a file attached in the chat goes into the picked terminal when its Chat Uploads say so.

An admin connects a plain Open Terminal with Chat Uploads on Filesystem. A person picks it and
attaches a file from the message box's More menu: the file is written into the terminal and is
attached to the message as a file of the terminal, which the model is told by its path. A second
file of the same name goes beside the first as "name (1).ext" and leaves the first as it was.
With Chat Uploads on Default the same attachment is kept by Open WebUI and the terminal gets
nothing.

The docs promise that the file lands in the folder the file browser has open and shows there
straight away. On dev c15f28649 it is written to the terminal's working directory whatever the
file browser shows (finding `open-webui-chat-upload-misses-the-open-folder`); that test is red.

Such a terminal takes the file itself, so a model with File Upload switched off still lets a
file be attached once the terminal is picked; before that Upload Files does nothing and says
"Model(s) do not support file upload". Without the File Upload permission Upload Files does
nothing either, terminal or not, and says "You do not have permission to upload files.".

A fake terminal stands in for uploads that fail: one that names no working folder ("Could not
determine the upload folder.") and one that refuses the file ("Failed to upload file."). Both
leave nothing attached, and the next upload it accepts is attached.

The terminal with a machine per chat, whose first upload waits for the chat, is driven in
e2e/terminals/test_terminals_per_chat.py.

Discriminates: the other tests pass on the dev c15f28649 build. In frontend copies, a taken
name written over turns the name test red; every terminal taking chat uploads turns the
Default test red; Upload Files opening the file chooser whatever the model or the permission
turns those two tests red; a failed upload without its message, or one left attached, turns
the failed-upload test red.
"""

from __future__ import annotations

import re
import shutil
import uuid
from urllib.parse import parse_qs, urlparse

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.backends import wait_until
from harness.listener import json_answer
from harness.model_presets import preset_with_capabilities
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    configure_terminals,
    read_grant,
    serving_terminal,
)
from utils.cached_chat import ask, attach, choose_terminal, pick_terminal
from utils.chat_ui import (
    REPLY_TIMEOUT_MS,
    chat_input,
    close_menu,
    open_input_menu,
    split_screen,
)
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

CHAT_URL = re.compile(r"/c/([0-9a-f-]{36})$")
FILESYSTEM = "filesystem"
NO_MODEL_UPLOAD = "Model(s) do not support file upload"
NO_PERMISSION = "You do not have permission to upload files."
FAKE_HOME = "/home/user/"
UPLOAD_SECONDS = 15


@pytest.fixture
def person(make_user):
    return make_user()


@pytest.fixture
def connect(admin, preserve, open_terminal, person):
    """`connect(chat_uploads)` connects the Open Terminal for the person; returns its name."""
    preserve(TERMINAL_SERVERS_CONFIG)

    def save(chat_uploads: str) -> str:
        connection = open_terminal.connection(
            name=f"Quay office {uuid.uuid4().hex[:6]}",
            config={"access_grants": [read_grant(person.id)], "chat_uploads": chat_uploads},
        )
        with admin.client() as client:
            configure_terminals(client, connection)
        return connection["name"]

    return save


@pytest.fixture
def hold(open_terminal):
    """A folder of the test's own in the terminal's home, with one file in it."""
    folder = open_terminal.home / f"hold-{uuid.uuid4().hex[:6]}"
    folder.mkdir()
    (folder / "ballast.txt").write_text("stones\n")
    yield folder
    shutil.rmtree(folder, ignore_errors=True)


def _composer(page: Page) -> Locator:
    return page.locator("form").filter(has=chat_input(page))


def _attached(page: Page, name: str) -> Locator:
    """The file `name` above the message box, with its size after the name."""
    return _composer(page).get_by_role("button", name=re.compile(rf"^{re.escape(name)} \d"))


def _attach_beside_the_files_tab(page: Page, name: str, content: str) -> None:
    """Attach a file from the message box's More menu; the Files tab has a More of its own."""
    _composer(page).get_by_role("button", name="More", exact=True).click()
    with page.expect_file_chooser() as chooser:
        page.get_by_role("menu").get_by_role("menuitem", name="Upload Files").click()
    chooser.value.set_files({"name": name, "mimeType": "text/plain", "buffer": content.encode()})


def _stored_files(person, page: Page) -> list[dict]:
    expect(page).to_have_url(CHAT_URL, timeout=REPLY_TIMEOUT_MS)
    with person.client() as client:
        stored = client.get(f"/api/v1/chats/{CHAT_URL.search(page.url).group(1)}")
    assert stored.status_code == 200, stored.text
    messages = stored.json()["chat"]["history"]["messages"].values()
    return [file for message in messages if message["role"] == "user" for file in message["files"]]


def _upload_files_entry(page: Page, tooltip: str) -> Locator:
    """Open the More menu; returns its Upload Files entry, which says `tooltip` on hover."""
    menu = open_input_menu(page)
    entry = tooltip_button(menu, tooltip, role="menuitem")
    expect(entry).to_have_text("Upload Files")
    return entry


def _click_opens_no_file_chooser(page: Page, entry: Locator) -> bool:
    choosers = []
    page.on("filechooser", lambda chooser: choosers.append(chooser))
    entry.click()
    # the entry leaves its menu open when it does nothing; a chooser would be up by its close
    expect(entry).to_be_visible()
    close_menu(page)
    return not choosers


def test_an_attached_file_is_written_into_the_terminal_under_a_name_that_is_free(
    page_for, person, connect, open_terminal, upstream
):
    terminal_name = connect(FILESYSTEM)
    page = page_for(person)
    choose_terminal(page, terminal_name)
    tag = uuid.uuid4().hex[:6]
    first = open_terminal.home / f"manifest-{tag}.txt"
    second = open_terminal.home / f"manifest-{tag} (1).txt"
    try:
        attach(page, first.name, "ten casks")

        expect(_attached(page, first.name)).to_be_visible()
        assert wait_until(first.exists, UPLOAD_SECONDS), "the upload never reached the terminal"
        assert first.read_text() == "ten casks"

        attach(page, first.name, "eleven casks")

        expect(_attached(page, second.name)).to_be_visible()
        assert wait_until(second.exists, UPLOAD_SECONDS), "the second upload never arrived"
        assert second.read_text() == "eleven casks"
        assert first.read_text() == "ten casks", "the first upload was written over"
    finally:
        first.unlink(missing_ok=True)
        second.unlink(missing_ok=True)

    prompt = f"count the casks {uuid.uuid4().hex[:6]}"
    ask(page, upstream, prompt, reply.text("Twenty-one."))

    asked = next(filter(reply.answering(prompt), upstream.chat_requests()))
    told = [entry["content"] for entry in asked["messages"] if entry["role"] == "user"][-1]
    for path in (first, second):
        assert f'<file type="filesystem" id="{path}"' in told, (
            f"the model was not told the uploaded file {path}: {told!r}"
        )
    stored = _stored_files(person, page)
    assert [(file["type"], file["path"]) for file in stored] == [
        ("filesystem", str(first)),
        ("filesystem", str(second)),
    ], f"the message should carry both uploads as files of the terminal: {stored}"


def test_an_attached_file_lands_in_the_folder_the_file_browser_has_open(
    page_for, person, connect, hold, open_terminal
):
    terminal_name = connect(FILESYSTEM)
    page = page_for(person)
    pick_terminal(page, terminal_name)
    split_screen(page)
    files = page.get_by_role("region", name="File browser")
    files.get_by_text(hold.name, exact=True).click()
    expect(files.get_by_text("ballast.txt", exact=True)).to_be_visible()
    name = f"manifest-{uuid.uuid4().hex[:6]}.txt"

    _attach_beside_the_files_tab(page, name, "ten casks")

    expect(_attached(page, name)).to_be_visible()
    in_the_home = open_terminal.home / name
    arrived = wait_until(lambda: (hold / name).exists() or in_the_home.exists(), UPLOAD_SECONDS)
    assert arrived, "the upload never reached the terminal"
    try:
        # red on dev c15f28649: the file is written to the terminal's working directory
        # (finding open-webui-chat-upload-misses-the-open-folder)
        assert (hold / name).exists(), (
            "the upload is not in the folder the file browser has open; "
            f"in the terminal's home: {in_the_home.exists()}"
        )
    finally:
        in_the_home.unlink(missing_ok=True)
    expect(files.get_by_text(name, exact=True)).to_be_visible()


def test_with_chat_uploads_on_default_an_attached_file_is_kept_by_open_webui(
    page_for, person, connect, open_terminal, upstream
):
    terminal_name = connect("default")
    page = page_for(person)
    choose_terminal(page, terminal_name)
    name = f"manifest-{uuid.uuid4().hex[:6]}.txt"

    attach(page, name, "ten casks")

    expect(_attached(page, name)).to_be_visible()
    prompt = f"count the casks {uuid.uuid4().hex[:6]}"
    ask(page, upstream, prompt, reply.text("Ten."))
    stored = _stored_files(person, page)
    assert [(file["type"], file["name"]) for file in stored] == [("file", name)], (
        f"the message should carry the upload as a file of Open WebUI: {stored}"
    )
    with person.client() as client:
        kept = client.get(f"/api/v1/files/{stored[0]['id']}/content")
    assert kept.status_code == 200 and kept.text == "ten casks", kept.text
    on_the_terminal = [path.name for path in open_terminal.home.rglob(name)]
    assert on_the_terminal == [], f"the upload also went to the terminal: {on_the_terminal}"


def test_a_terminal_that_takes_chat_uploads_accepts_a_file_for_a_model_without_file_upload(
    page_for, person, admin, connect, open_terminal
):
    terminal_name = connect(FILESYSTEM)
    name = f"label-{uuid.uuid4().hex[:6]}.txt"
    with preset_with_capabilities(admin, {"file_upload": False}) as model_id:
        page = page_for(person, path=f"/?models={model_id}")

        refused = _click_opens_no_file_chooser(page, _upload_files_entry(page, NO_MODEL_UPLOAD))
        assert refused, "Upload Files opened a file chooser for a model with File Upload off"

        choose_terminal(page, terminal_name)
        attach(page, name, "dry cider")

        expect(_attached(page, name)).to_be_visible()
    uploaded = open_terminal.home / name
    try:
        assert wait_until(uploaded.exists, UPLOAD_SECONDS), "the upload never reached the terminal"
        assert uploaded.read_text() == "dry cider"
    finally:
        uploaded.unlink(missing_ok=True)


@pytest.fixture
def uploads_denied(admin, preserve):
    preserve("permissions")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["chat"]["file_upload"] = False
        saved = client.post("/api/v1/users/default/permissions", json=permissions)
    assert saved.status_code == 200, saved.text


def test_without_the_file_upload_permission_upload_files_does_nothing(
    uploads_denied, page_for, person, connect
):
    terminal_name = connect(FILESYSTEM)
    page = page_for(person)
    choose_terminal(page, terminal_name)

    refused = _click_opens_no_file_chooser(page, _upload_files_entry(page, NO_PERMISSION))

    assert refused, "Upload Files opened a file chooser without the File Upload permission"


def _uploads_to(server) -> list[str]:
    return [
        parse_qs(urlparse(entry.path).query)["directory"][0]
        for entry in server.requests_to("/files/upload")
    ]


def test_an_upload_the_terminal_cannot_take_says_why_and_leaves_nothing_attached(
    page_for, person, admin, preserve
):
    preserve(TERMINAL_SERVERS_CONFIG)
    name = f"Quay office {uuid.uuid4().hex[:6]}"
    with serving_terminal() as server:
        server.folder = {}
        server.upload = json_answer({"detail": "The disk is full"}, status=507)
        server.route("GET", "/openapi.json", json_answer({"openapi": "3.0.0", "paths": {}}))
        server.route("GET", "/api/config", json_answer({"features": {"terminal": True}}))
        server.route("GET", "/files/cwd", lambda _request: json_answer(server.folder))
        server.route("GET", "/files/list", json_answer({"entries": [], "writable": True}))
        server.route("POST", "/files/upload", lambda _request: server.upload)
        grants = {"access_grants": [read_grant(person.id)], "chat_uploads": FILESYSTEM}
        with admin.client() as client:
            configure_terminals(client, server.connection(name=name, config=grants))
        page = page_for(person)
        choose_terminal(page, name)

        attach(page, "manifest.txt", "ten casks")

        expect(page.get_by_text("Could not determine the upload folder.")).to_be_visible()
        expect(_attached(page, "manifest.txt")).to_have_count(0)
        assert _uploads_to(server) == [], "a file was sent to a terminal that named no folder"

        server.folder = {"cwd": FAKE_HOME, "home": FAKE_HOME}
        attach(page, "manifest.txt", "ten casks")

        expect(page.get_by_text("Failed to upload file.")).to_be_visible()
        expect(_attached(page, "manifest.txt")).to_have_count(0)
        assert _uploads_to(server) == [FAKE_HOME]

        server.upload = json_answer({"path": f"{FAKE_HOME}manifest.txt", "size": 9})
        attach(page, "manifest.txt", "ten casks")

        expect(_attached(page, "manifest.txt")).to_be_visible()
        assert wait_until(lambda: len(_uploads_to(server)) == 2, UPLOAD_SECONDS), (
            f"the terminal was sent {_uploads_to(server)}"
        )
        expect(_attached(page, "manifest.txt")).to_be_visible()
