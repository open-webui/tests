"""Journey: adding audio to a note by uploading it, recording it or capturing it.

The microphone menu in the note header offers Upload Audio, Record and Capture Audio. An uploaded
audio file and a finished recording both go to the server, which transcribes them with the
admin's speech-to-text engine. The audio then sits in the note as a file chip in its text, the
note's own text stays as written, and the note stores the file, whose record holds the
transcript. Opening the chip, also after a reload, shows the transcript. Capture Audio asks the
browser for the audio of a screen or tab, and the capture is attached like a recording. Pressing
Escape while recording attaches nothing.

Discriminates: passes on dev 176d31d1d; in a frontend copy, each test fails when its behaviour
is cut: the note not saving after a file upload, a finished recording not being uploaded, Capture
Audio recording the microphone instead of the screen and Escape confirming the recording.
"""

from __future__ import annotations

import io
import math
import re
import struct
import time
import wave
from pathlib import Path
from typing import Generator

import pytest
from playwright.sync_api import Browser, Locator, Page, Playwright, expect

from conftest import AppConfig
from e2e.audio.conftest import SPOKEN_TURN
from e2e.conftest import _close_context, _dismiss_first_run_modals, _new_context, _signed_in_page
from harness.actors import Actor
from harness.audio_engine import TRANSCRIPT
from utils.note_content import note_files

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

NOTE_TEXT = "Notes from the call."
RECORDING_NAME = re.compile(r"^Recording-.*\.webm$")
DISPLAY_MEDIA_SPY = """
    window.displayMediaRequests = 0;
    const getDisplayMedia = navigator.mediaDevices.getDisplayMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getDisplayMedia = (...args) => {
        window.displayMediaRequests += 1;
        return getDisplayMedia(...args);
    };
"""


def _memo() -> bytes:
    frames = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * index / 16000)))
        for index in range(8000)
    )
    memo = io.BytesIO()
    with wave.open(memo, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(frames)
    return memo.getvalue()


def _create_note(owner: Actor) -> str:
    with owner.client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={
                "title": "Call",
                "data": {"content": {"md": NOTE_TEXT}},
                "access_grants": [],
            },
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _stored_files(owner: Actor, note_id: str) -> list[dict]:
    with owner.client() as client:
        note = client.get(f"/api/v1/notes/{note_id}").json()
    return note_files(note)


def _wait_for_stored_files(owner: Actor, note_id: str, timeout: float = 15.0) -> list[dict]:
    deadline = time.monotonic() + timeout
    files = _stored_files(owner, note_id)
    while not files and time.monotonic() < deadline:
        time.sleep(0.2)
        files = _stored_files(owner, note_id)
    assert files, "the note never stored the audio file"
    return files


def _note_editor(page: Page) -> Locator:
    return page.get_by_role("main").get_by_label("Write something...")


def _open_note(page: Page, note_id: str) -> None:
    page.goto(f"/notes/{note_id}")
    expect(_note_editor(page)).to_contain_text(NOTE_TEXT)


def _open_audio_menu(page: Page) -> Locator:
    """Opens the microphone menu in the note's header, whose first entry is named Record too."""
    microphone = page.get_by_role("main").get_by_role("button", name="Record", exact=True)
    expect(microphone, "the note header shows no microphone button").to_be_visible()
    microphone.click()
    return page.get_by_role("menu")


def _upload_memo(page: Page, tmp_path: Path) -> None:
    memo = tmp_path / "memo.wav"
    memo.write_bytes(_memo())
    _open_audio_menu(page)
    with page.expect_file_chooser() as chooser:
        page.get_by_role("menuitem", name="Upload Audio").click()
    chooser.value.set_files(memo)


def _file_chip(page: Page, name: str | re.Pattern) -> Locator:
    return page.get_by_role("main").get_by_text(name)


def _note_text(page: Page) -> Locator:
    """The note's own paragraphs, without the file chips that sit between them."""
    return _note_editor(page).locator("p")


def _assert_stored_with_transcript(owner: Actor, files: list[dict], content_type: str) -> None:
    assert len(files) == 1, files
    with owner.client() as client:
        stored = client.get(f"/api/v1/files/{files[0]['id']}").json()
    assert stored["meta"]["content_type"].startswith(content_type), stored["meta"]
    assert stored["data"]["content"] == TRANSCRIPT, stored["data"]


@pytest.fixture
def capturing_browser(
    playwright_instance: Playwright, config: AppConfig, tmp_path_factory
) -> Generator[Browser, None, None]:
    """A browser that lets a page capture the screen's audio without asking anyone."""
    capture = tmp_path_factory.mktemp("microphone") / "spoken-turn.wav"
    capture.write_bytes(SPOKEN_TURN)
    browser = playwright_instance.chromium.launch(
        headless=config.headless,
        args=[
            "--use-fake-ui-for-media-stream",
            "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-audio-capture={capture}",
            "--auto-select-desktop-capture-source=Entire screen",
        ],
    )
    yield browser
    browser.close()


# ---------------------------------------------------------------- upload


def test_uploading_audio_attaches_it_to_the_note_with_its_transcript(
    voice_page_for, make_user, speech_engine, tmp_path
):
    author = make_user()
    note_id = _create_note(author)
    page = voice_page_for(author)
    _open_note(page, note_id)
    _upload_memo(page, tmp_path)

    expect(_file_chip(page, "memo.wav")).to_be_visible()
    expect(_note_text(page)).to_have_text([NOTE_TEXT])
    _assert_stored_with_transcript(author, _wait_for_stored_files(author, note_id), "audio/wav")
    assert len(speech_engine.transcription_requests()) == 1


def test_an_attached_audio_file_opens_with_its_transcript_after_a_reload(
    voice_page_for, make_user, speech_engine, tmp_path
):
    author = make_user()
    note_id = _create_note(author)
    page = voice_page_for(author)
    _open_note(page, note_id)
    _upload_memo(page, tmp_path)
    _wait_for_stored_files(author, note_id)

    page.reload()
    _file_chip(page, "memo.wav").click()

    dialog = page.get_by_role("dialog")
    expect(dialog).to_contain_text("memo.wav")
    expect(dialog).to_contain_text(TRANSCRIPT)


# ---------------------------------------------------------------- record


def test_recording_with_the_microphone_attaches_the_recording_with_its_transcript(
    voice_page_for, make_user, speech_engine
):
    author = make_user()
    note_id = _create_note(author)
    page = voice_page_for(author)
    _open_note(page, note_id)

    _open_audio_menu(page).get_by_role("menuitem", name="Record", exact=True).click()
    expect(page.get_by_text("0:02", exact=True)).to_be_visible()
    page.get_by_role("button", name="Confirm recording").click()

    expect(_file_chip(page, RECORDING_NAME)).to_be_visible()
    expect(page.get_by_role("button", name="Confirm recording")).to_have_count(0)
    expect(_note_text(page)).to_have_text([NOTE_TEXT])
    _assert_stored_with_transcript(author, _wait_for_stored_files(author, note_id), "audio/webm")
    assert len(speech_engine.transcription_requests()) == 1


def test_escape_while_recording_attaches_nothing(voice_page_for, make_user, speech_engine):
    author = make_user()
    note_id = _create_note(author)
    page = voice_page_for(author)
    _open_note(page, note_id)
    _open_audio_menu(page).get_by_role("menuitem", name="Record", exact=True).click()
    expect(page.get_by_text("0:02", exact=True)).to_be_visible()

    page.keyboard.press("Escape")

    expect(page.get_by_role("button", name="Confirm recording")).to_have_count(0)
    page.wait_for_timeout(4000)  # a wrongly confirmed recording uploads within this time
    expect(_file_chip(page, RECORDING_NAME)).to_have_count(0)
    assert _stored_files(author, note_id) == []
    assert speech_engine.transcription_requests() == []


# ---------------------------------------------------------------- capture


def test_capturing_audio_attaches_the_capture_with_its_transcript(
    capturing_browser, config, make_user, speech_engine, request
):
    author = make_user()
    note_id = _create_note(author)
    _dismiss_first_run_modals(author)
    browser_context = _new_context(capturing_browser, config, author.base_url)
    browser_context.add_init_script(DISPLAY_MEDIA_SPY)
    try:
        page = _signed_in_page(browser_context, author.token, config.navigation_timeout)
        _open_note(page, note_id)

        _open_audio_menu(page)
        page.get_by_role("menuitem", name="Capture Audio").click()
        expect(page.get_by_text("0:02", exact=True)).to_be_visible()
        page.get_by_role("button", name="Confirm recording").click()

        expect(_file_chip(page, RECORDING_NAME)).to_be_visible()
        _assert_stored_with_transcript(
            author, _wait_for_stored_files(author, note_id), "audio/webm"
        )
        assert page.evaluate("window.displayMediaRequests") == 1
    finally:
        _close_context(browser_context, request.node, "capture")
