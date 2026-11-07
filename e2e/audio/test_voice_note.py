"""Journey: a voice note is recorded, transcribed and saved into the picked terminal.

Since e7c8e58e9 a chat with an Open Terminal picked offers Voice Note (Ctrl+Alt+M, or the New tab
menu). It records from the microphone into a small floating panel until Stop recording; the
recording then gets a name, plays back, can be sent to the admin's speech-to-text engine with
Transcribe, and Save writes two files to the terminal's working folder: the recording and a
Markdown note headed by the name, holding the transcript and a link to the recording. The note
then opens in a tab. A second note of the same name is saved beside the first, never over it.
Transcribe is only offered to someone allowed speech-to-text, a name with a slash is refused, and
Cancel or Esc throws the recording away without writing anything and turns the microphone off.

The microphone is Chromium's fake capture device from `voice_page_for`.

Red on dev f659c0d12, two bugs of the new voice note: the saved note never opens in a tab (the
panel drops its terminal as it closes, before it reports the saved path), and Esc does nothing
while the message box has focus, which it has after the shortcut (the editor marks Esc handled and
the panel skips handled keys).

Discriminates: on the f659c0d12 build patched to report the saved path before closing, the first
test passes (open-webui/open-webui#33886), and on one patched to focus the panel as it opens, the
Esc test passes (open-webui/open-webui#33887); with Transcribe offered whatever the permission and
the name check removed, the permission and slash tests fail; with a second note of the same name
replacing the first, the same-name test fails.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from e2e.audio.test_web_api_dictation import LIVE_TRACKS, RECORD_MICROPHONE_TRACKS
from e2e.conftest import APP_LOADED
from harness.audio_engine import TRANSCRIPT
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, read_grant
from utils.cached_chat import choose_terminal
from utils.chat_ui import REPLY_TIMEOUT_MS, chat_input

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

PERMISSIONS = "/api/v1/users/default/permissions"


def _set_speech_to_text(admin, allowed: bool) -> None:
    with admin.client() as client:
        permissions = client.get(PERMISSIONS).json()
        permissions["chat"].update(stt=allowed)
        saved = client.post(PERMISSIONS, json=permissions)
    assert saved.status_code == 200, saved.text


@pytest.fixture
def terminal_chat(voice_page_for, make_user, admin, preserve, open_terminal):
    """Returns `open_chat(stt=True)`: a new chat with the terminal picked and a microphone.

    `watch_microphone` keeps every microphone track the page opens in `window.microphoneTracks`."""
    preserve(TERMINAL_SERVERS_CONFIG, "permissions")
    person = make_user()
    connection = open_terminal.connection(config={"access_grants": [read_grant(person.id)]})
    with admin.client() as client:
        configure_terminals(client, connection)

    def open_chat(stt: bool = True, watch_microphone: bool = False) -> Page:
        _set_speech_to_text(admin, stt)
        page = voice_page_for(person)
        if watch_microphone:
            page.add_init_script(f"({RECORD_MICROPHONE_TRACKS})()")
            page.goto("/")
            page.wait_for_function(APP_LOADED, timeout=REPLY_TIMEOUT_MS)
        choose_terminal(page, connection["name"])
        return page

    return open_chat


def _files(open_terminal) -> set[str]:
    """The visible files in the terminal's home; the terminal keeps dotfiles of its own there."""
    return {path.name for path in open_terminal.home.iterdir() if not path.name.startswith(".")}


def _record(page: Page) -> Locator:
    """Start a voice note with its shortcut, record a second and stop; returns its panel."""
    page.keyboard.press("Control+Alt+M")
    panel = page.get_by_role("dialog", name="Voice Note")
    stop = panel.get_by_role("button", name="Stop recording")
    expect(stop).to_be_enabled()
    expect(panel.get_by_text("0:01")).to_be_visible()
    stop.click()
    expect(panel.get_by_role("textbox", name="File name")).to_have_value("Voice Note")
    return panel


def test_a_voice_note_is_transcribed_and_saved_into_the_terminal(
    terminal_chat, speech_engine, open_terminal
):
    """Red on dev f659c0d12 (open-webui/open-webui#33886): the note does not open in a tab."""
    page = terminal_chat()
    name = f"Harbour walk {uuid.uuid4().hex[:6]}"
    before = _files(open_terminal)
    panel = _record(page)
    expect(panel.get_by_text(str(open_terminal.home), exact=True)).to_be_visible()

    panel.get_by_role("button", name="Transcribe").click()
    expect(panel.get_by_role("textbox", name="Transcript")).to_have_value(TRANSCRIPT)
    assert len(speech_engine.transcription_requests()) == 1, "Transcribe sent no recording"
    panel.get_by_role("textbox", name="File name").fill(name)
    panel.get_by_role("button", name="Save", exact=True).click()

    expect(panel).to_have_count(0)
    added = _files(open_terminal) - before
    assert added == {f"{name}.webm", f"{name}.md"}, f"Save wrote {added}"
    assert (open_terminal.home / f"{name}.webm").stat().st_size > 0, "the recording is empty"
    note = (open_terminal.home / f"{name}.md").read_text()
    link = name.replace(" ", "%20")
    assert note == f"# {name}\n\n{TRANSCRIPT}\n\n[Audio](./{link}.webm)\n", (
        f"the voice note reads {note!r}"
    )
    expect(page.get_by_role("tabpanel", name=f"{name}.md")).to_contain_text(TRANSCRIPT)


def test_transcribe_is_only_offered_with_the_speech_to_text_permission(
    terminal_chat, speech_engine, open_terminal
):
    page = terminal_chat(stt=False)
    panel = _record(page)

    # with the permission, Transcribe shows beside Play
    expect(panel.get_by_role("button", name="Play")).to_be_visible()
    expect(panel.get_by_role("button", name="Transcribe")).to_have_count(0)
    name = f"Untranscribed {uuid.uuid4().hex[:6]}"
    panel.get_by_role("textbox", name="File name").fill(name)
    panel.get_by_role("button", name="Save", exact=True).click()

    expect(panel).to_have_count(0)
    note = (open_terminal.home / f"{name}.md").read_text()
    assert TRANSCRIPT not in note and note.startswith(f"# {name}\n\n[Audio]("), note
    assert speech_engine.transcription_requests() == []


def test_a_name_with_a_slash_is_refused_and_cancel_saves_nothing(
    terminal_chat, speech_engine, open_terminal
):
    page = terminal_chat()
    before = _files(open_terminal)
    panel = _record(page)

    panel.get_by_role("textbox", name="File name").fill("notes/harbour")
    panel.get_by_role("button", name="Save", exact=True).click()
    expect(panel.get_by_role("alert")).to_have_text("Enter a file name without slashes.")
    panel.get_by_role("button", name="Cancel").click()

    expect(panel).to_have_count(0)
    assert _files(open_terminal) == before, (
        f"a refused or cancelled voice note wrote {_files(open_terminal) - before}"
    )


def test_escape_while_recording_throws_the_note_away_and_frees_the_microphone(
    terminal_chat, open_terminal
):
    """Red on dev f659c0d12 (open-webui/open-webui#33887): Esc from the message box is lost."""
    page = terminal_chat(watch_microphone=True)
    before = _files(open_terminal)
    chat_input(page).click()
    page.keyboard.press("Control+Alt+M")
    panel = page.get_by_role("dialog", name="Voice Note")
    expect(panel.get_by_role("button", name="Stop recording")).to_be_enabled()
    page.wait_for_function(f"({LIVE_TRACKS})() > 0")

    page.keyboard.press("Escape")

    expect(panel).to_have_count(0)
    page.wait_for_function(f"({LIVE_TRACKS})() === 0")
    assert _files(open_terminal) == before, (
        f"a voice note left with Escape wrote {_files(open_terminal) - before}"
    )


def test_a_second_note_of_the_same_name_keeps_the_first(terminal_chat, open_terminal):
    page = terminal_chat(stt=False)
    name = f"Ferry times {uuid.uuid4().hex[:6]}"
    for _ in range(2):
        panel = _record(page)
        panel.get_by_role("textbox", name="File name").fill(name)
        panel.get_by_role("button", name="Save", exact=True).click()
        expect(panel).to_have_count(0)

    home = open_terminal.home
    saved = {path.name for path in home.iterdir() if path.name.startswith(name)}
    assert saved == {f"{name}.md", f"{name}.webm", f"{name} (1).md", f"{name} (1).webm"}, (
        f"two notes named {name!r} were saved as {saved}"
    )
    link = name.replace(" ", "%20")
    first = (home / f"{name}.md").read_text()
    assert first.endswith(f"[Audio](./{link}.webm)\n"), f"the first note was changed: {first!r}"
    second = (home / f"{name} (1).md").read_text()
    assert second.endswith(f"[Audio](./{link}%20%281%29.webm)\n"), (
        f"the second note does not link its own recording: {second!r}"
    )
