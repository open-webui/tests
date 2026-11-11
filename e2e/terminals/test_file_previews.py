"""Journey: each kind of file a person opens from a terminal's file browser shows in its own view.

A real Open Terminal runs with a folder of the test's own in its home. In a chat the person picks
the terminal, opens Files, enters the folder and clicks a file, which opens in a tab of its own.
A video and a sound file load into the browser's players. A PDF the model shows with a page
number opens on that page, and a page picked from the thumbnails comes into view. A
tab-separated file shows as a table and an empty CSV as its source, ready to type into. A
database runs a typed query and says what is wrong with a bad one, and a long table is paged a
hundred rows at a time. An SVG shows without the script inside it, zooms, and shows the script
in its source. A 3D model draws in a viewer and a broken one says why it cannot. A note shows
the picture beside it and a link in it opens the linked file, in a note that can be edited and
in a read-only one. A file whose bytes are no text says it could not be read, and a picture of
a kind the page has no viewer for is named as a binary file. The toolbar shows the file's size,
copies a text file and downloads the file as it is on the terminal. Resting on a file in the
list opens a card with its name, size and content; an HTML page there runs no script, and a
text file past 1 MiB or any file past 20 MiB says no preview is available. A comment pinned in
a PDF shows with its author and text.

Discriminates: all pass on dev c15f28649 with Open Terminal 0.17.7. On one frontend build of it
with these mutations each named test fails: the video and audio branches of the preview removed
(players); a requested page ignored by the page viewer (PDF); the delimiter of a .tsv read as a
comma and the empty-file switch to source removed (table); a typed query never run and the
page size raised to 1000 (database); the SVG sanitizer returning its input (SVG); the model
viewer skipped (3D model); relative pictures not resolved against the terminal and local links
not opened (note); the binary placeholder dropped (unreadable file); Copy copying nothing and
the size left out (toolbar); the hover card's limits removed and its scripts allowed (hover);
PDF annotations not read as comments (comment).
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import uuid

import pytest
from playwright.sync_api import Locator, expect

from harness import sample_documents as samples
from harness import slow_machine
from harness import upstream as reply
from harness.backends import wait_until
from harness.sample_documents import Note
from harness.terminal_server import TERMINAL_SERVERS_CONFIG
from utils.cached_chat import ask, calling
from utils.chat_ui import split_screen
from utils.clipboard import copied, loopback_clipboard
from utils.file_views import (
    back_to,
    comment_shown_by,
    file_browser,
    files_of,
    open_file,
    rest_elsewhere,
    row,
    tab,
)

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

# pdf.js, the 3D viewer and the SQLite engine are fetched and started on first use
VIEWER_LOAD_TIMEOUT_MS = slow_machine.scaled(5_000)
MEBIBYTE = 1024 * 1024
SVG_WITH_A_SCRIPT = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="120" height="60">'
    "<script>document.title = 'the picture ran a script'</script>"
    '<rect width="120" height="60" fill="teal"/><text x="10" y="35">Hollow Creek</text></svg>'
)
PAGE_WITH_A_SCRIPT = (
    "<html><body><h1>Cider report</h1>"
    "<script>document.body.append('the page ran a script')</script></body></html>"
)


@pytest.fixture
def folder(open_terminal):
    """A folder of this test's own in the terminal's home, which the whole session shares."""
    path = open_terminal.home / f"previews-{uuid.uuid4().hex[:6]}"
    path.mkdir()
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def person(make_user):
    return make_user()


@pytest.fixture
def browse(folder, person, page_for, admin, preserve, open_terminal):
    """`browse()` opens the folder in the person's file browser, once its files are written."""

    def open_folder() -> Locator:
        preserve(TERMINAL_SERVERS_CONFIG)
        return files_of(page_for, admin, person, open_terminal, folder.name)

    return open_folder


def _length(player: Locator) -> float:
    """Seconds of media the player has loaded, 0 until it knows."""
    return player.evaluate("player => Number.isFinite(player.duration) ? player.duration : 0")


def test_a_video_and_a_sound_file_load_into_the_browsers_players(folder, browse):
    (folder / "pressing.webm").write_bytes(samples.webm(seconds=1))
    (folder / "bell.wav").write_bytes(samples.wav(seconds=2))
    browser = browse()

    open_file(browser, "pressing.webm")
    # the players have no role or name of their own
    video = browser.locator("video")
    expect(video).to_be_visible()
    expect(video).to_have_js_property("controls", True)
    assert wait_until(lambda: _length(video) > 0, 10), "the video never loaded from the terminal"
    assert 0.5 < _length(video) < 1.5, f"the one-second video plays for {_length(video)} s"

    back_to(browser, folder.name)
    open_file(browser, "bell.wav")
    sound = browser.locator("audio")
    expect(sound).to_be_visible()
    expect(sound).to_have_js_property("controls", True)
    assert wait_until(lambda: _length(sound) > 0, 10), "the sound never loaded from the terminal"
    assert 1.9 < _length(sound) < 2.1, f"the two-second sound plays for {_length(sound)} s"


def test_a_pdf_the_model_shows_with_a_page_number_opens_on_that_page(
    folder, person, page_for, admin, preserve, open_terminal, upstream
):
    (folder / "ledger.pdf").write_bytes(
        samples.pdf(["Orchard ledger", "Pressing days", "Cellar stock"])
    )
    preserve(TERMINAL_SERVERS_CONFIG)
    browser = files_of(page_for, admin, person, open_terminal)
    page = browser.page
    split_screen(page)

    ask(
        page,
        upstream,
        f"show me the cellar page {uuid.uuid4().hex[:6]}",
        calling("display_file", {"path": f"{folder.name}/ledger.pdf", "page": 3}),
        reply.text("It is on page three."),
    )

    expect(tab(page, "ledger.pdf")).to_have_attribute("aria-selected", "true")
    viewer = file_browser(page).get_by_role("application", name="Page viewer")
    expect(viewer.get_by_text("Cellar stock")).to_be_in_viewport(timeout=VIEWER_LOAD_TIMEOUT_MS)
    expect(viewer.get_by_text("Orchard ledger")).not_to_be_in_viewport()


def test_a_page_picked_from_a_pdfs_thumbnails_comes_into_view(folder, browse):
    (folder / "ledger.pdf").write_bytes(
        samples.pdf(["Orchard ledger", "Pressing days", "Cellar stock"])
    )
    browser = browse()
    open_file(browser, "ledger.pdf")
    viewer = browser.get_by_role("application", name="Page viewer")
    expect(viewer.get_by_text("Orchard ledger")).to_be_in_viewport(timeout=VIEWER_LOAD_TIMEOUT_MS)
    pages = browser.get_by_role("complementary", name="Pages")
    expect(pages.get_by_role("button")).to_have_count(3)
    expect(pages.get_by_role("button", name="Page 1")).to_have_attribute("aria-current", "true")

    pages.get_by_role("button", name="Page 3").click()

    expect(viewer.get_by_text("Cellar stock")).to_be_in_viewport()
    expect(viewer.get_by_text("Orchard ledger")).not_to_be_in_viewport()
    expect(pages.get_by_role("button", name="Page 3")).to_have_attribute("aria-current", "true")


def test_a_tab_separated_file_shows_as_a_table_and_an_empty_sheet_as_its_source(folder, browse):
    (folder / "yield.tsv").write_text("tree\tyield, in crates\napple\t40\npear\t25\n")
    (folder / "empty.csv").write_text("")
    browser = browse()

    open_file(browser, "yield.tsv")
    table = browser.get_by_role("table")
    # a comma inside a field stays in its column
    expect(table.get_by_role("columnheader", name="yield, in crates")).to_be_visible()
    expect(table.get_by_role("row", name=re.compile(r"apple\s*40"))).to_be_visible()
    expect(table.get_by_role("row", name=re.compile(r"pear\s*25"))).to_be_visible()
    browser.get_by_role("button", name="Source", exact=True).click()
    expect(browser.get_by_role("textbox")).to_contain_text("apple\t40")

    back_to(browser, folder.name)
    open_file(browser, "empty.csv")
    # nothing to show as a table, so the file opens where a first row can be typed
    expect(browser.get_by_role("button", name="Preview", exact=True)).to_be_visible()
    editor = browser.get_by_role("textbox")
    expect(editor).to_be_visible()
    expect(browser.get_by_role("table")).to_have_count(0)
    editor.click()
    browser.page.keyboard.type("tree,yield")
    browser.get_by_role("button", name="Save", exact=True).click()
    assert wait_until(lambda: (folder / "empty.csv").read_text() == "tree,yield", 10), (
        f"the typed row was not saved: {(folder / 'empty.csv').read_text()!r}"
    )


def _database(path, casks: int) -> None:
    with sqlite3.connect(path) as database:
        database.execute("CREATE TABLE barrels (label TEXT, litres INTEGER)")
        database.executemany(
            "INSERT INTO barrels VALUES (?, ?)",
            [(f"cask {number}", number * 10) for number in range(1, casks + 1)],
        )


def test_a_query_typed_into_a_database_shows_its_rows_and_a_bad_one_what_is_wrong(folder, browse):
    _database(folder / "cellar.db", casks=5)
    browser = browse()
    open_file(browser, "cellar.db")
    table = browser.get_by_role("table")
    expect(table.get_by_role("row", name=re.compile(r"cask 5\s*50"))).to_be_visible(
        timeout=VIEWER_LOAD_TIMEOUT_MS
    )

    browser.get_by_role("button", name="SQL", exact=True).click()
    query = browser.get_by_role("textbox", name="SQL query")
    query.fill("SELECT label AS biggest FROM barrels WHERE litres > 30 ORDER BY litres DESC")
    browser.get_by_role("button", name="Run", exact=True).click()

    expect(table.get_by_role("columnheader", name="biggest")).to_be_visible()
    rows = table.get_by_role("row")
    expect(rows).to_have_count(3)
    expect(rows.nth(1)).to_contain_text("cask 5")
    expect(rows.nth(2)).to_contain_text("cask 4")

    query.fill("SELECT label FROM bottles")
    query.press("Control+Enter")
    expect(browser.get_by_text("no such table: bottles")).to_be_visible()


def test_a_long_table_of_a_database_is_paged_a_hundred_rows_at_a_time(folder, browse):
    _database(folder / "cellar.db", casks=130)
    browser = browse()
    open_file(browser, "cellar.db")
    table = browser.get_by_role("table")
    expect(table.get_by_role("row", name=re.compile(r"cask 100\s*1000"))).to_be_visible(
        timeout=VIEWER_LOAD_TIMEOUT_MS
    )
    # the heading and a hundred casks
    expect(table.get_by_role("row")).to_have_count(101)
    expect(browser.get_by_text("1 / 2 (130 rows)")).to_be_visible()
    expect(browser.get_by_role("button", name="Previous page")).to_be_disabled()

    browser.get_by_role("button", name="Next page").click()

    expect(browser.get_by_text("2 / 2 (130 rows)")).to_be_visible()
    expect(table.get_by_role("row")).to_have_count(31)
    expect(table.get_by_role("row", name=re.compile(r"130\s*cask 130\s*1300"))).to_be_visible()
    expect(browser.get_by_role("button", name="Next page")).to_be_disabled()

    browser.get_by_role("button", name="Previous page").click()
    expect(table.get_by_role("row", name=re.compile(r"cask 1\s*10$"))).to_be_visible()


def test_an_svg_picture_shows_without_the_script_inside_it(folder, browse):
    (folder / "logo.svg").write_text(SVG_WITH_A_SCRIPT)
    browser = browse()
    page = browser.page
    title = page.title()

    open_file(browser, "logo.svg")

    expect(browser.get_by_text("Hollow Creek", exact=True)).to_be_visible()
    # the picture is put into the page itself, so its script would run there
    expect(browser.locator("svg script")).to_have_count(0)
    assert page.title() == title, f"the picture's script ran in the page: {page.title()!r}"
    level = browser.get_by_role("button", name="Reset zoom")
    expect(level).to_have_text("100%")
    browser.get_by_role("button", name="Zoom in").click()
    expect(level).not_to_have_text("100%")

    browser.get_by_role("button", name="Source", exact=True).click()
    expect(browser.get_by_role("textbox")).to_contain_text("the picture ran a script")
    assert page.title() == title, f"the picture's script ran in the page: {page.title()!r}"


def test_a_3d_model_draws_in_a_viewer_and_a_broken_one_says_why_not(folder, browse):
    (folder / "wedge.stl").write_bytes(samples.stl())
    (folder / "broken.gltf").write_text("{ this is no model")
    browser = browse()

    open_file(browser, "wedge.stl")
    viewer = browser.get_by_label("3D model preview")
    expect(viewer).to_be_visible(timeout=VIEWER_LOAD_TIMEOUT_MS)
    expect(browser.get_by_role("button", name="Reset view")).to_be_visible()
    expect(browser.get_by_role("button", name="Zoom in")).to_be_enabled()
    expect(browser.get_by_text(re.compile("Error previewing file"))).to_have_count(0)

    back_to(browser, folder.name)
    open_file(browser, "broken.gltf")
    expect(browser.get_by_text(re.compile("^Error previewing file: .+"))).to_be_visible(
        timeout=VIEWER_LOAD_TIMEOUT_MS
    )


NOTE_WITH_A_PICTURE = "# Label proof\n\n![the label](art/label.png)\n\n[the yields](yield.tsv)\n"


def _note_with_a_picture(folder) -> None:
    (folder / "art").mkdir()
    (folder / "art" / "label.png").write_bytes(samples.png(64, 48))
    (folder / "yield.tsv").write_text("tree\tyield\napple\t40\n")
    (folder / "proof.md").write_text(NOTE_WITH_A_PICTURE)


def _expect_the_picture(browser: Locator) -> None:
    picture = browser.get_by_role("img", name="the label")
    expect(picture).to_be_visible()
    # the width of the file on the terminal: a picture that failed to load has none
    expect(picture).to_have_js_property("naturalWidth", 64)


def test_a_note_shows_the_picture_beside_it(folder, browse):
    _note_with_a_picture(folder)
    browser = browse()

    open_file(browser, "proof.md")

    expect(browser.get_by_role("heading", name="Label proof")).to_be_visible()
    _expect_the_picture(browser)


def test_a_read_only_note_shows_its_picture_and_its_link_opens_the_linked_file(folder, browse):
    _note_with_a_picture(folder)
    (folder / "proof.md").chmod(0o444)
    browser = browse()
    page = browser.page

    open_file(browser, "proof.md")
    expect(browser.get_by_role("button", name="Save", exact=True)).to_be_disabled()
    _expect_the_picture(browser)

    browser.get_by_role("link", name="the yields").click()

    expect(tab(page, "yield.tsv")).to_have_attribute("aria-selected", "true")
    expect(file_browser(page).get_by_role("row", name=re.compile(r"apple\s*40"))).to_be_visible()


def test_a_file_that_is_no_text_says_so_and_an_unknown_picture_is_named_as_binary(folder, browse):
    (folder / "dump.bin").write_bytes(bytes(range(256)) * 4)
    (folder / "scan.tiff").write_bytes(b"II*\x00" + bytes(range(128, 256)) * 4)
    (folder / "plain.txt").write_text("a readable line")
    browser = browse()

    open_file(browser, "plain.txt")
    expect(browser.get_by_role("textbox")).to_have_text("a readable line")

    back_to(browser, folder.name)
    open_file(browser, "dump.bin")
    expect(browser.get_by_text("Could not read file.", exact=True)).to_be_visible()
    expect(browser.get_by_role("textbox")).to_have_count(0)

    back_to(browser, folder.name)
    open_file(browser, "scan.tiff")
    expect(browser.get_by_role("textbox")).to_have_text("[Binary file: image/tiff]")


def test_the_toolbar_shows_a_files_size_copies_its_text_and_downloads_it(folder, browse, person):
    farm = {"farm": "Hollow Creek", "crates": [12, 8]}
    written = json.dumps(farm, indent=2)
    (folder / "farm.json").write_text(written)
    browser = browse()
    page = browser.page
    open_file(browser, "farm.json")
    expect(browser.get_by_text("Hollow Creek")).to_be_visible()

    expect(browser.get_by_text(f"{len(written)}.0 B", exact=True)).to_be_visible()

    clipboard = loopback_clipboard(page, person)
    browser.get_by_role("button", name="Copy", exact=True).click()
    expect(page.get_by_text("Copied to clipboard")).to_be_visible()
    assert copied(clipboard) == written, f"Copy put this on the clipboard: {copied(clipboard)!r}"

    with page.expect_download() as download:
        browser.get_by_role("button", name="Download", exact=True).click()
    assert download.value.suggested_filename == "farm.json"
    with open(download.value.path()) as saved:
        assert saved.read() == written, "the downloaded file differs from the one on the terminal"


def _hover_card(browser: Locator, name: str) -> Locator:
    """Rest on the file in the list; returns the preview card that opens."""
    rest_elsewhere(browser.page)
    row(browser, name).get_by_text(name, exact=True).hover()
    card = browser.page.get_by_label("File preview", exact=True)
    expect(card).to_be_visible()
    return card


def test_resting_on_a_file_shows_a_card_with_its_content_but_runs_no_script(folder, browse):
    (folder / "recipe.txt").write_text("three parts apple, one part pear")
    (folder / "report.html").write_text(PAGE_WITH_A_SCRIPT)
    browser = browse()

    card = _hover_card(browser, "recipe.txt")
    expect(card.get_by_text("recipe.txt", exact=True)).to_be_visible()
    expect(card.get_by_text("32.0 B", exact=True)).to_be_visible()
    expect(card.get_by_text("three parts apple, one part pear")).to_be_visible()
    # a card only shows: nothing in it can be typed into
    expect(card.get_by_role("textbox")).to_have_attribute("contenteditable", "false")

    card = _hover_card(browser, "report.html")
    shown = card.frame_locator("iframe")
    expect(shown.get_by_role("heading", name="Cider report")).to_be_visible()
    expect(shown.get_by_text("the page ran a script")).to_have_count(0)

    # the same page opened in its tab does run its script, so the card is what held it back
    open_file(browser, "report.html")
    opened = browser.frame_locator("iframe")
    expect(opened.get_by_role("heading", name="Cider report")).to_be_visible()
    expect(opened.get_by_text("the page ran a script")).to_be_visible()


def test_a_file_too_large_for_the_card_says_no_preview_is_available(folder, browse):
    (folder / "small.log").write_text("pressed 40 crates\n" * 10)
    (folder / "long.log").write_text("pressed 40 crates\n" * (MEBIBYTE // 18 + 1))
    (folder / "small.png").write_bytes(samples.png(64, 48))
    with open(folder / "huge.png", "wb") as huge:
        huge.write(samples.png(64, 48))
        huge.truncate(20 * MEBIBYTE + 1)
    browser = browse()

    expect(_hover_card(browser, "small.log").get_by_text("pressed 40 crates").first).to_be_visible()
    card = _hover_card(browser, "long.log")
    expect(card.get_by_text("Preview not available", exact=True)).to_be_visible()

    card = _hover_card(browser, "small.png")
    expect(card.get_by_role("img", name="small.png")).to_have_js_property("naturalWidth", 64)
    card = _hover_card(browser, "huge.png")
    expect(card.get_by_text("Preview not available", exact=True)).to_be_visible()
    expect(card.get_by_role("img", name="huge.png")).to_have_count(0)


def test_a_comment_pinned_in_a_pdf_shows_with_its_author(folder, browse):
    (folder / "ledger.pdf").write_bytes(
        samples.pdf(["Orchard ledger"], Note("Mara Lindqvist", "Check this sum against the till"))
    )
    browser = browse()
    open_file(browser, "ledger.pdf")
    viewer = browser.get_by_role("application", name="Page viewer")
    expect(viewer.get_by_text("Orchard ledger")).to_be_visible(timeout=VIEWER_LOAD_TIMEOUT_MS)

    comment = comment_shown_by(
        browser.page, viewer.get_by_role("button", name="Comment: Mara Lindqvist")
    )

    expect(comment.get_by_text("Mara Lindqvist", exact=True)).to_be_visible()
    expect(comment.get_by_text("Check this sum against the till", exact=True)).to_be_visible()
