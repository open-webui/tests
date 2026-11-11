"""Journey: what a person sees in the chat when the model shows or rewrites a terminal file.

A fake terminal holds a few files and offers the model `display_file` and `write_file`. The
scripted model calls them in a chat that has the terminal picked.

A file the model shows inline (`inline: true`, or any `display_file` call for a person whose
Terminal File Display setting is Inline) is a card in the reply: the file's name, its preview
read from the terminal, a Download button that saves the file and an Open button that opens it in
the Files tab. Clicking the name folds the preview away and back. The card belongs to the
terminal the file is on: once another terminal is picked for the chat it reads "Terminal
unavailable", its buttons are off and the other terminal is asked for nothing; picking the first
terminal again brings the preview back. A preview the terminal refuses reads "Preview failed" and
a refused download says "Download failed".

`display_file` naming a file uploaded to Open WebUI shows the same card from Open WebUI: no Open
button, nothing asked of the terminal, and it stays readable with no terminal picked.

Without `inline` and with the setting on Sidebar (the default) the reply has no card and the file
opens in a file tab beside the chat. A file the model rewrites with `write_file` is read again in
the tab that has it open; where the person has unsaved edits in it, their text stays and a notice
says the file was updated on disk, with Reload to take the model's version.

A terminal a person added in their own settings is called by their browser: the page sends the
terminal the `display_file` call without the page number, adds `inline: true` under the Inline
setting and shows the card, and otherwise opens the file tab.

The card keeps the terminal it found when the chat was opened until a reload (finding
`open-webui-file-card-keeps-terminal-until-reload`); the test that picks the terminal again
without a reload is red on dev c15f28649 and on 0.13.3, and passes in a frontend copy whose
card follows the picked terminal.

Discriminates: the other tests pass on the dev c15f28649 build. Each turns red in a frontend
copy that breaks what it reads: a name that does not fold the preview, an Open that does
nothing, a refused preview or download without its message, an Open button on an Open WebUI
file, the Inline setting showing no card or not keeping the file tab closed, the model's
`display_file` opening no file tab, a rewritten file not read again or read over unsaved
edits, the page number sent to a personal terminal or `inline` not sent to it. A card that
reads its terminal whichever one is picked turns the reopened-chat test red.
"""

from __future__ import annotations

import base64
import json
import re
import uuid
from urllib.parse import parse_qs, quote, urlparse

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.listener import json_answer
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    configure_terminals,
    read_grant,
    serving_terminal,
)
from utils.cached_chat import ask, calling, choose_terminal, pick_terminal
from utils.chat_ui import chat_input, last_reply, split_screen
from utils.tooltips import tooltip_button
from utils.user_terminals import listed_terminal, open_terminal_menu

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

ROOT = "/home/user/"
LOG = "/home/user/harbour-log.txt"
CHART = "/home/user/tide-chart.png"
CHART_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
UNAVAILABLE = "Terminal unavailable"


def _operation(operation_id: str, **properties: dict) -> dict:
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}, **properties},
        "required": ["path"],
    }
    return {
        "operationId": operation_id,
        "requestBody": {"required": True, "content": {"application/json": {"schema": schema}}},
        "responses": {"200": {"description": "ok"}},
    }


SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Open Terminal", "version": "1"},
    "paths": {
        "/files/display": {
            "post": _operation("display_file", inline={"type": "boolean"}, page={"type": "integer"})
        },
        "/files/write": {"post": _operation("write_file", content={"type": "string"})},
    },
}


def _asked_path(request) -> str:
    return parse_qs(urlparse(request.path).query).get("path", [""])[0]


def _serve_files(server, files: dict[str, bytes]) -> None:
    """Answer what the page and the model ask of a terminal that holds `files`."""

    def listing(_request):
        entries = [
            {"name": path.removeprefix(ROOT), "type": "file", "size": len(content)}
            for path, content in files.items()
        ]
        return json_answer({"entries": entries, "writable": True})

    def read(request):
        path = _asked_path(request)
        if path not in files:
            return json_answer({"detail": "File not found"}, status=404)
        return json_answer({"path": path, "content": files[path].decode()})

    def view(request):
        path = _asked_path(request)
        if server.refuses_downloads or path not in files:
            return json_answer({"detail": "The disk is not answering"}, status=500)
        content_type = "image/png" if path.endswith(".png") else "text/plain"
        return 200, {"Content-Type": content_type}, files[path]

    def display(request):
        path = json.loads(request.body)["path"]
        return json_answer({"exists": path in files, "path": path})

    def write(request):
        written = json.loads(request.body)
        files[written["path"]] = written["content"].encode()
        return json_answer({"path": written["path"], "size": len(written["content"])})

    server.refuses_downloads = False
    server.route("GET", "/openapi.json", json_answer(SPEC))
    server.route("GET", "/api/config", json_answer({"features": {"terminal": True}}))
    server.route("GET", "/files/cwd", json_answer({"cwd": ROOT, "home": ROOT}))
    server.route("POST", "/files/cwd", json_answer({"cwd": ROOT}))
    server.route("GET", "/files/list", listing)
    server.route("GET", "/files/read", read)
    server.route("GET", "/files/view", view)
    server.route("POST", "/files/display", display)
    server.route("POST", "/files/write", write)


@pytest.fixture
def person(make_user):
    return make_user()


@pytest.fixture
def log_text() -> str:
    return f"moored at berth {uuid.uuid4().hex[:6]}"


@pytest.fixture
def terminal(admin, preserve, person, log_text):
    """A terminal shared with the person, holding a log and a chart; yields the fake server."""
    preserve(TERMINAL_SERVERS_CONFIG)
    with serving_terminal() as server, serving_terminal() as other:
        server.files = {LOG: log_text.encode(), CHART: CHART_PNG}
        _serve_files(server, server.files)
        _serve_files(other, {})
        grants = {"access_grants": [read_grant(person.id)]}
        tag = uuid.uuid4().hex[:6]
        server.name = f"Harbour office {tag}"
        server.other = other
        other.name = f"Boat shed {tag}"
        with admin.client() as client:
            configure_terminals(
                client,
                server.connection(name=server.name, config=grants),
                other.connection(name=other.name, config=grants),
            )
        yield server


def _set_file_display(person, display: str) -> None:
    with person.client() as client:
        saved = client.post(
            "/api/v1/users/user/settings/update", json={"ui": {"terminalFileDisplay": display}}
        )
    assert saved.status_code == 200, saved.text


def _show(chat: Page, upstream, path: str, **arguments) -> Locator:
    """Have the model show `path`; returns the reply once it has ended."""
    prompt = f"show me {path.removeprefix(ROOT)} ({uuid.uuid4().hex[:6]})"
    done = f"Shown {uuid.uuid4().hex[:6]}."
    call = calling("display_file", {"path": path, **arguments})
    ask(chat, upstream, prompt, call, reply.text(done))
    return last_reply(chat)


def _card_name(shown: Locator, name: str) -> Locator:
    return shown.get_by_role("button", name=name, exact=True)


def _download(shown: Locator) -> Locator:
    return shown.get_by_role("button", name="Download", exact=True)


def _open(shown: Locator) -> Locator:
    return shown.get_by_role("button", name="Open", exact=True)


def _file_requests(server) -> list[str]:
    """What the terminal was asked about a file to show or download."""
    return [
        entry.path
        for entry in server.received
        if entry.path.split("?")[0] in ("/files/view", "/files/display")
        or entry.path.startswith(f"/files/read?path={quote(LOG, safe='')}")
    ]


def _switch_terminal(page: Page, picked: str, name: str) -> None:
    """Pick `name` in the Terminal menu of a chat that has `picked`; picking `picked` unpicks."""
    open_terminal_menu(page, picked)
    listed_terminal(page, name).click()
    expect(page.get_by_role("menu")).to_have_count(0)


def _personal_terminal(server, name: str) -> dict:
    return {
        "url": server.base_url,
        "key": "",
        "name": name,
        "enabled": False,
        "auth_type": "bearer",
        "path": "/openapi.json",
        "config": {"enable": True},
    }


def test_a_file_shown_inline_is_a_card_with_its_preview_download_and_open(
    page_for, person, terminal, upstream, log_text
):
    page = page_for(person)
    choose_terminal(page, terminal.name)

    shown = _show(page, upstream, LOG, inline=True)

    name = _card_name(shown, "harbour-log.txt")
    expect(name).to_have_attribute("aria-expanded", "true")
    expect(shown.get_by_text(log_text)).to_be_visible()

    name.click()
    expect(name).to_have_attribute("aria-expanded", "false")
    expect(shown.get_by_text(log_text)).to_have_count(0)
    name.click()
    expect(shown.get_by_text(log_text)).to_be_visible()

    with page.expect_download() as download:
        _download(shown).click()
    assert download.value.suggested_filename == "harbour-log.txt"
    saved = download.value.path().read_text()
    assert saved == log_text, f"the card's Download saved {saved!r}"

    _open(shown).click()
    files = page.get_by_role("region", name="File browser")
    expect(files.get_by_text(log_text)).to_be_visible()


def _reopen(page: Page) -> Locator:
    """Load the chat again; returns its last reply once the card is back."""
    page.reload()
    expect(chat_input(page)).to_be_visible()
    shown = last_reply(page)
    expect(_card_name(shown, "harbour-log.txt")).to_be_visible()
    return shown


def test_a_chat_opened_with_another_terminal_picked_reads_terminal_unavailable_on_the_card(
    page_for, person, terminal, upstream, log_text
):
    page = page_for(person)
    choose_terminal(page, terminal.name)
    shown = _show(page, upstream, LOG, inline=True)
    expect(shown.get_by_text(log_text)).to_be_visible()

    _switch_terminal(page, terminal.name, terminal.other.name)
    terminal.other.clear()
    shown = _reopen(page)

    expect(shown.get_by_text(UNAVAILABLE)).to_be_visible()
    expect(shown.get_by_text(log_text)).to_have_count(0)
    expect(_download(shown)).to_be_disabled()
    expect(_open(shown)).to_be_disabled()
    asked = _file_requests(terminal.other)
    assert asked == [], f"the card asked another terminal for its file: {asked}"

    _switch_terminal(page, terminal.other.name, terminal.name)
    shown = _reopen(page)

    expect(shown.get_by_text(log_text)).to_be_visible()
    expect(shown.get_by_text(UNAVAILABLE)).to_have_count(0)
    expect(_download(shown)).to_be_enabled()
    expect(_open(shown)).to_be_enabled()


def test_the_card_follows_the_terminal_picked_for_the_chat_without_a_reload(
    page_for, person, terminal, upstream, log_text
):
    page = page_for(person)
    choose_terminal(page, terminal.name)
    shown = _show(page, upstream, LOG, inline=True)
    expect(shown.get_by_text(log_text)).to_be_visible()
    _switch_terminal(page, terminal.name, terminal.other.name)
    shown = _reopen(page)
    expect(shown.get_by_text(UNAVAILABLE)).to_be_visible()

    _switch_terminal(page, terminal.other.name, terminal.name)

    # red on dev c15f28649 and 0.13.3: the card keeps the terminal it found when the chat opened
    expect(shown.get_by_text(log_text)).to_be_visible()
    expect(shown.get_by_text(UNAVAILABLE)).to_have_count(0)
    expect(_download(shown)).to_be_enabled()

    _switch_terminal(page, terminal.name, terminal.other.name)

    expect(shown.get_by_text(UNAVAILABLE)).to_be_visible()
    expect(_open(shown)).to_be_disabled()


def test_an_uploaded_file_the_model_shows_is_a_card_read_from_open_webui(
    page_for, person, terminal, upstream
):
    text = f"berth {uuid.uuid4().hex[:6]} is free"
    with person.client() as client:
        uploaded = client.post(
            "/api/v1/files/",
            params={"process": "false"},
            files={"file": ("berths.txt", text.encode(), "text/plain")},
        )
    assert uploaded.status_code == 200, uploaded.text
    page = page_for(person)
    choose_terminal(page, terminal.name)
    terminal.clear()

    shown = _show(page, upstream, uploaded.json()["id"])

    expect(_card_name(shown, "berths.txt")).to_have_attribute("aria-expanded", "true")
    expect(shown.get_by_text(text)).to_be_visible()
    expect(_open(shown)).to_have_count(0)
    with page.expect_download() as download:
        _download(shown).click()
    saved = download.value.path().read_text()
    assert saved == text, f"the card's Download saved {saved!r}"
    asked = _file_requests(terminal)
    assert asked == [], f"a file kept in Open WebUI was asked of the terminal: {asked}"

    _switch_terminal(page, terminal.name, terminal.name)
    page.reload()
    expect(chat_input(page)).to_be_visible()

    shown = last_reply(page)
    expect(tooltip_button(page.get_by_role("main"), "Terminal")).to_be_visible()
    expect(shown.get_by_text(text)).to_be_visible()
    expect(shown.get_by_text(UNAVAILABLE)).to_have_count(0)


def test_a_preview_and_a_download_the_terminal_refuses_say_so(page_for, person, terminal, upstream):
    page = page_for(person)
    choose_terminal(page, terminal.name)
    terminal.refuses_downloads = True

    refused = _show(page, upstream, CHART, inline=True)

    expect(_card_name(refused, "tide-chart.png")).to_be_visible()
    expect(refused.get_by_text("Preview failed")).to_be_visible()
    _download(refused).click()
    expect(page.get_by_text("Download failed")).to_be_visible()

    terminal.refuses_downloads = False
    shown = _show(page, upstream, CHART, inline=True)

    expect(shown.get_by_role("img", name="tide-chart.png")).to_be_visible()
    expect(shown.get_by_text("Preview failed")).to_have_count(0)


def test_a_file_shown_without_inline_opens_in_a_file_tab_unless_the_setting_is_inline(
    page_for, person, terminal, upstream, log_text
):
    _set_file_display(person, "inline")
    page = page_for(person)
    choose_terminal(page, terminal.name)

    inline = _show(page, upstream, LOG, page=2)

    expect(_card_name(inline, "harbour-log.txt")).to_be_visible()
    expect(inline.get_by_text(log_text)).to_be_visible()
    expect(page.get_by_role("region", name="File browser")).to_have_count(0)

    _set_file_display(person, "sidebar")
    page.reload()
    expect(chat_input(page)).to_be_visible()
    expect(_card_name(last_reply(page), "harbour-log.txt")).to_have_count(0)

    beside = _show(page, upstream, LOG)

    files = page.get_by_role("region", name="File browser")
    expect(files.get_by_text(log_text)).to_be_visible()
    expect(_card_name(beside, "harbour-log.txt")).to_have_count(0)
    asked = [json.loads(entry.body) for entry in terminal.requests_to("/files/display")]
    assert asked == [{"path": LOG}, {"path": LOG}], f"the terminal's display_file got {asked}"


def test_a_file_the_model_rewrites_is_read_again_and_unsaved_edits_are_kept(
    page_for, person, terminal, upstream, log_text
):
    page = page_for(person)
    pick_terminal(page, terminal.name)
    split_screen(page)
    files = page.get_by_role("region", name="File browser")
    files.get_by_text("harbour-log.txt", exact=True).click()
    expect(files.get_by_text(log_text)).to_be_visible()
    notice = files.get_by_role("status").filter(has_text="File updated on disk")

    first = f"left for the buoy {uuid.uuid4().hex[:6]}"
    ask(
        page,
        upstream,
        "note that we left",
        calling("write_file", {"path": LOG, "content": first}),
        reply.text("Noted."),
    )

    expect(files.get_by_text(first)).to_be_visible()
    expect(notice).to_have_count(0)

    files.get_by_text(first).click()
    page.keyboard.press("End")
    page.keyboard.type(" and my own remark")
    expect(files.get_by_text(f"{first} and my own remark")).to_be_visible()
    second = f"back at the quay {uuid.uuid4().hex[:6]}"
    ask(
        page,
        upstream,
        "note that we are back",
        calling("write_file", {"path": LOG, "content": second}),
        reply.text("Noted again."),
    )

    expect(notice).to_have_text(
        re.compile(r"^\s*File updated on disk\. Your local changes are preserved\.")
    )
    expect(files.get_by_text(f"{first} and my own remark")).to_be_visible()
    assert terminal.files[LOG].decode() == second, "the model's second write never reached the disk"

    notice.get_by_role("button", name="Reload").click()
    page.get_by_role("button", name="Confirm").click()

    expect(files.get_by_text(second)).to_be_visible()
    expect(notice).to_have_count(0)


@pytest.mark.parametrize("display", ["inline", "sidebar"])
def test_a_personal_terminal_is_asked_for_the_file_by_the_persons_browser(
    page_for, make_user, upstream, direct_integrations, log_text, display
):
    direct_integrations(True)
    owner = make_user(role="admin")
    name = f"My laptop {uuid.uuid4().hex[:6]}"
    with serving_terminal() as laptop:
        laptop.cors = True
        _serve_files(laptop, {LOG: log_text.encode()})
        laptop.route("GET", "/skills", json_answer([]))
        settings = {"terminalServers": [_personal_terminal(laptop, name)]}
        with owner.client() as client:
            saved = client.post("/api/v1/users/user/settings/update", json={"ui": settings})
        assert saved.status_code == 200, saved.text
        _set_file_display(owner, display)
        page = page_for(owner)
        choose_terminal(page, name)

        shown = _show(page, upstream, LOG, page=2)

        files = page.get_by_role("region", name="File browser")
        if display == "inline":
            expect(_card_name(shown, "harbour-log.txt")).to_be_visible()
            expect(shown.get_by_text(log_text)).to_be_visible()
            expect(files).to_have_count(0)
        else:
            expect(files.get_by_text(log_text)).to_be_visible()
            expect(_card_name(shown, "harbour-log.txt")).to_have_count(0)
        calls = [call for call in laptop.requests_to("/files/display") if call.method == "POST"]
        assert [bool(call.headers.get("origin")) for call in calls] == [True], (
            f"the browser should call the personal terminal's display_file once: {calls}"
        )
        expected = {"path": LOG, **({"inline": True} if display == "inline" else {})}
        asked = json.loads(calls[0].body)
        assert asked == expected, f"the terminal's display_file got {asked}"
