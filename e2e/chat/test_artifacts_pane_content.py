"""Journey: what the artifacts pane shows for the HTML, CSS, JavaScript and SVG blocks of a reply.

An SVG block opens the pane on the drawing itself, sanitised in the page rather than framed, and
offers no full screen. HTML, CSS and JavaScript blocks of one reply make one page: the CSS styles
the HTML and the script runs inside the page's own frame, which cannot reach the chat around it.
Two HTML blocks in one reply are two versions, and a block's Preview button moves the pane to
that block's version. The pane's Copy puts the whole page on the clipboard and Download saves it
as an HTML file. Versions and full screen of HTML artifacts are covered in
test_artifacts_versions.py.

Discriminates: passes on the dev ebc6add67 build. In its mutation build (the `rendering-front`
copy: SVG blocks no longer collected as artifacts, CSS and JavaScript blocks left out of the
page and Preview no longer selecting the block's version) the SVG, page and Preview tests go red;
the Copy and Download test stays green there as a control.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.chat_ui import expect_reply, last_reply, send
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

SUN = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 60" width="240" height="120">'
    '<circle cx="30" cy="30" r="20" fill="orange"/>'
    '<text x="60" y="35">Sunny day</text></svg>'
)
PAGE_BLOCKS = (
    "```html\n<h1 id='title'>Harbour page</h1>\n<p id='note'>waiting</p>\n```\n\n"
    "```css\n#title { color: rgb(0, 128, 0); }\n```\n\n"
    "```javascript\n"
    "document.getElementById('note').textContent = 'script ran';\n"
    "try { parent.document.title = 'changed by the artifact'; } catch (error) {\n"
    "  document.getElementById('note').textContent += ', chat out of reach';\n"
    "}\n```"
)


def fenced(lang: str, code: str) -> str:
    return f"```{lang}\n{code}\n```"


def ask(page: Page, upstream, answer: str, prompt: str) -> Locator:
    upstream.queue(reply.text(f"{answer}\n\nArtifact done.", match=reply.answering(prompt)))
    send(page, prompt)
    expect_reply(page, "Artifact done.")
    return last_reply(page)


def artifacts_pane(page: Page) -> Locator:
    return page.locator("#artifacts-container")


def artifact_page(page: Page):
    return artifacts_pane(page).frame_locator("iframe")


def test_an_svg_block_opens_the_pane_on_the_drawing(page_for, make_user, upstream):
    page = page_for(make_user())

    ask(page, upstream, fenced("svg", SUN), "draw me a sun")

    pane = artifacts_pane(page)
    expect(pane).to_contain_text("Version 1 of 1")
    drawing = pane.locator("svg").filter(has_text="Sunny day")
    expect(drawing).to_be_visible()
    expect(drawing.locator("circle")).to_have_attribute("fill", "orange")
    expect(pane.locator("iframe")).to_have_count(0)
    tooltips = pane.get_by_role("button").evaluate_all(
        "buttons => buttons.map(button => button.parentElement?._tippy?.props.content)"
    )
    assert "Download" in tooltips and "Open in full screen" not in tooltips, tooltips


def test_html_css_and_script_blocks_make_one_styled_page_that_cannot_reach_the_chat(
    page_for, make_user, upstream
):
    page = page_for(make_user())

    ask(page, upstream, PAGE_BLOCKS, "build me a harbour page")

    expect(artifacts_pane(page)).to_contain_text("Version 1 of 1")
    title = artifact_page(page).locator("#title")
    expect(title).to_have_text("Harbour page")
    expect(title).to_have_css("color", "rgb(0, 128, 0)")
    expect(artifact_page(page).locator("#note")).to_have_text("script ran, chat out of reach")
    assert page.title() != "changed by the artifact"


def test_preview_moves_the_pane_to_that_blocks_version(page_for, make_user, upstream):
    page = page_for(make_user())
    answer = (
        fenced("html", "<h1>first draft</h1>") + "\n\n" + fenced("html", "<h1>second draft</h1>")
    )

    box = ask(page, upstream, answer, "give me two drafts")

    pane = artifacts_pane(page)
    expect(pane).to_contain_text("Version 2 of 2")
    expect(artifact_page(page).locator("h1")).to_have_text("second draft")
    box.get_by_role("button", name="Preview", exact=True).first.click()
    expect(pane).to_contain_text("Version 1 of 2")
    expect(artifact_page(page).locator("h1")).to_have_text("first draft")


def test_copy_and_download_give_the_whole_page(page_for, make_user, upstream):
    page = page_for(make_user(), permissions=["clipboard-read", "clipboard-write"])
    ask(page, upstream, fenced("html", "<h1>saved card</h1>"), "make a card to keep")
    pane = artifacts_pane(page)
    expect(artifact_page(page).locator("h1")).to_have_text("saved card")
    chat_id = page.url.rsplit("/", 1)[-1]

    pane.get_by_role("button", name="Copy", exact=True).click()

    expect(pane.get_by_role("button", name="Copied", exact=True)).to_be_visible()
    copied = page.evaluate("navigator.clipboard.readText()")
    assert re.search(r"<!DOCTYPE html>.*<h1>saved card</h1>", copied, re.S), copied
    with page.expect_download() as download_info:
        tooltip_button(pane, "Download").click()
    download = download_info.value
    assert download.suggested_filename == f"artifact-{chat_id}-0.html"
    with open(download.path(), encoding="utf-8") as saved:
        assert "<h1>saved card</h1>" in saved.read()
