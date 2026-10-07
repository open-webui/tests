"""A browser with a microphone that speaks, and the speech engine it talks to.

Chromium's fake capture device plays `SPOKEN_TURN` on a loop: a second of a rising tone, loud
enough for the voice call's sound detection, then three seconds of silence, which the call takes
as the end of a turn. So a voice call in `voice_page_for(actor)` sends a recording for
transcription every few seconds for as long as it listens. `speech_engine` is the admin's
speech-to-text and text-to-speech pointed at `harness.audio_engine`, restored afterwards.
"""

from __future__ import annotations

import contextlib
import io
import math
import struct
import wave
from typing import Callable, Generator

import pytest
from playwright.sync_api import Browser, Page, Playwright

from conftest import AppConfig
from e2e.conftest import _close_context, _dismiss_first_run_modals, _new_context, _signed_in_page
from harness.actors import Actor
from harness.audio_engine import AudioEngine, serve_audio_engine, using_audio_engine

SAMPLE_RATE = 16000


def _spoken_turn() -> bytes:
    frames = bytearray()
    for index in range(SAMPLE_RATE):
        seconds = index / SAMPLE_RATE
        pitch = 300 + 600 * seconds
        frames += struct.pack("<h", int(20000 * math.sin(2 * math.pi * pitch * seconds)))
    frames += b"\x00\x00" * SAMPLE_RATE * 3
    recording = io.BytesIO()
    with wave.open(recording, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(bytes(frames))
    return recording.getvalue()


SPOKEN_TURN = _spoken_turn()


@pytest.fixture(scope="session")
def speaking_browser(
    playwright_instance: Playwright, config: AppConfig, tmp_path_factory
) -> Generator[Browser, None, None]:
    capture = tmp_path_factory.mktemp("microphone") / "spoken-turn.wav"
    capture.write_bytes(SPOKEN_TURN)
    browser = playwright_instance.chromium.launch(
        headless=config.headless,
        args=[
            "--use-fake-ui-for-media-stream",
            "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-audio-capture={capture}",
            "--autoplay-policy=no-user-gesture-required",
        ],
    )
    yield browser
    browser.close()


@pytest.fixture
def voice_page_for(
    speaking_browser: Browser, config: AppConfig, request: pytest.FixtureRequest
) -> Generator[Callable[[Actor], Page], None, None]:
    """`voice_page_for(actor)`: a signed-in page whose microphone plays `SPOKEN_TURN`.

    Further keywords go to the browser context, such as `reduced_motion="reduce"`.
    """
    opened = []

    def open_page(actor: Actor, **context_options) -> Page:
        _dismiss_first_run_modals(actor)
        browser_context = _new_context(speaking_browser, config, actor.base_url, **context_options)
        browser_context.grant_permissions(["microphone"])
        opened.append(browser_context)
        return _signed_in_page(browser_context, actor.token)

    yield open_page
    with contextlib.ExitStack() as closing:
        for index, browser_context in enumerate(opened):
            closing.callback(_close_context, browser_context, request.node, f"voice-{index}")


@pytest.fixture
def speech_engine(admin, listener, e2e_instance) -> Generator[AudioEngine, None, None]:
    engine = serve_audio_engine(listener)
    with admin.client() as client, using_audio_engine(client, engine):
        yield engine
