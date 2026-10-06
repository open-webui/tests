"""Journey: the artifacts pane steps between the versions of a chat and opens one in full screen.

A chat whose two replies each hold an HTML block opens the pane on the newest one, headed
"Version 2 of 2". Previous version shows the first block's content and Next version brings the
newest back, the header following each step. Open in full screen makes the artifact's frame fill
the window, and leaving full screen shows the pane as it was.

Discriminates: passes on dev 30f3f6a8f; in a frontend build whose pane opens on the first version,
whose version arrows stay put and whose full screen does nothing, every test fails.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from utils.chat_ui import expect_reply, send
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


def html_reply(label: str) -> str:
    return f"Here it is.\n\n```html\n<h1>{label}</h1>\n```\n\nDone with {label}."


def artifacts_pane(page: Page):
    return page.locator("#artifacts-container")


def shown_artifact(page: Page):
    return artifacts_pane(page).frame_locator("iframe").locator("h1")


@pytest.fixture
def two_artifacts(page_for, make_user, upstream):
    """A chat whose two replies each hold an artifact, the pane open on the second."""
    page = page_for(make_user())
    upstream.queue(
        reply.text(html_reply("first card"), match=reply.answering("first page")),
        reply.text(html_reply("second card"), match=reply.answering("second page")),
    )
    send(page, "make me the first page")
    expect_reply(page, "Done with first card.")
    send(page, "make me the second page")
    expect_reply(page, "Done with second card.")
    return page


def test_the_pane_opens_on_the_newest_version_labelled_as_the_last_of_two(two_artifacts):
    page = two_artifacts

    expect(artifacts_pane(page)).to_contain_text("Version 2 of 2")
    expect(shown_artifact(page)).to_have_text("second card")


def test_previous_version_shows_the_first_block_and_next_version_the_newest_again(two_artifacts):
    page = two_artifacts
    expect(shown_artifact(page)).to_have_text("second card")

    artifacts_pane(page).get_by_role("button", name="Previous version").click()
    expect(artifacts_pane(page)).to_contain_text("Version 1 of 2")
    expect(shown_artifact(page)).to_have_text("first card")

    artifacts_pane(page).get_by_role("button", name="Next version").click()
    expect(artifacts_pane(page)).to_contain_text("Version 2 of 2")
    expect(shown_artifact(page)).to_have_text("second card")


def test_full_screen_makes_the_artifact_fill_the_window_and_leaving_it_brings_the_pane_back(
    two_artifacts,
):
    page = two_artifacts
    expect(shown_artifact(page)).to_have_text("second card")
    frame = artifacts_pane(page).locator("iframe")
    window = page.evaluate("({ width: innerWidth, height: innerHeight })")
    pane_width = frame.evaluate("element => element.getBoundingClientRect().width")
    assert pane_width < window["width"], "the frame already fills the window before full screen"

    tooltip_button(artifacts_pane(page), "Open in full screen").click()

    expect(shown_artifact(page)).to_have_text("second card")
    page.wait_for_function("document.fullscreenElement !== null")
    size = frame.evaluate("element => element.getBoundingClientRect().toJSON()")
    assert (size["width"], size["height"]) == (window["width"], window["height"]), (
        f"the artifact does not fill the window in full screen: {size}"
    )

    page.evaluate("document.exitFullscreen()")

    page.wait_for_function("document.fullscreenElement === null")
    expect(artifacts_pane(page)).to_contain_text("Version 2 of 2")
    expect(shown_artifact(page)).to_have_text("second card")
    assert frame.evaluate("element => element.getBoundingClientRect().width") == pane_width
