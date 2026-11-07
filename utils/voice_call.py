"""Starting a voice call on the chat page and reading what the call overlay shows."""

from __future__ import annotations

import time

from playwright.sync_api import Page, expect

from harness.audio_engine import AudioEngine
from utils.chat_ui import chat_input

TURN_TIMEOUT_MS = 20_000


def end_call_button(page: Page):
    """The call panel's End call button; the call's tab offers another to close it."""
    return page.get_by_role("tabpanel").get_by_role("button", name="End call")


def start_call(page: Page) -> None:
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("button", name="Voice mode").click()
    expect(end_call_button(page)).to_be_visible()


def show_chat(page: Page) -> None:
    """Switch from the call's tab to the chat's own; on a wide screen the call covers the chat."""
    # the chat's tab leads the tab bar and is named by the chat's changing title
    page.get_by_role("tablist", name="Tabs").get_by_role("tab").first.click()
    expect(chat_input(page)).to_be_visible()


def show_call(page: Page) -> None:
    """Switch back from the chat's tab to the running call's."""
    page.get_by_role("tablist", name="Tabs").get_by_role("tab", name="Voice").click()
    expect(end_call_button(page)).to_be_visible()


def call_status(page: Page, status: str):
    return page.get_by_text(status, exact=True)


def wait_for_transcriptions(engine: AudioEngine, count: int, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while len(engine.transcription_requests()) < count:
        if time.monotonic() > deadline:
            raise AssertionError(
                f"expected {count} recordings to be transcribed, got "
                f"{len(engine.transcription_requests())}"
            )
        time.sleep(0.1)


# a screen wake lock that counts what the page asks of it
RECORD_WAKE_LOCKS = """() => {
    window.wakeLocks = { requested: 0, released: 0 };
    const wakeLock = {
        request: async (type) => {
            window.wakeLocks.requested += 1;
            const sentinel = new EventTarget();
            sentinel.type = type;
            sentinel.released = false;
            sentinel.release = async () => {
                if (sentinel.released) return;
                sentinel.released = true;
                window.wakeLocks.released += 1;
            };
            return sentinel;
        }
    };
    Object.defineProperty(Navigator.prototype, 'wakeLock', {
        configurable: true,
        get: () => wakeLock
    });
}"""


def record_wake_locks(page: Page) -> None:
    """Count the page's screen wake lock requests and releases, from a reload on."""
    page.add_init_script(f"({RECORD_WAKE_LOCKS})()")
    page.reload()


def expect_screen_kept_on(page: Page, kept_on: bool) -> None:
    """Wait until the page holds a screen wake lock (`kept_on`) or holds none, having asked once."""
    held = 1 if kept_on else 0
    page.wait_for_function(
        f"() => window.wakeLocks.requested > 0 && "
        f"window.wakeLocks.requested - window.wakeLocks.released === {held}",
        timeout=TURN_TIMEOUT_MS,
    )
