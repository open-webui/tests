"""Journey: a person clicks files in the terminal's file browser and each opens in its own view.

A real Open Terminal runs with files in its home: a Markdown note, a CSV sheet, a JSON document
and a broken one, a PNG image, a SQLite database and a Jupyter notebook. In a chat the person
picks the terminal from the input's Terminal menu, opens Files and clicks each file. The note
shows rendered in the Markdown editor and as its source after the Source toggle; the sheet shows
as a table, every row of a long one included; the document shows as a tree whose nodes fold and
the broken one opens as its source; the image shows with working zoom buttons whose percentage
changes; the database lists its table with its rows; the notebook shows its cells and their
outputs. Ticking a task in a note and saving writes the tick into the file. A note renders its
math, its alert and its Mermaid diagram, read only and writable alike.

Since b6a77a0f1 a writable note opens in a rich-text editor: text typed there is saved as
Markdown with its front matter kept, Ctrl+S saves too, an edit typed and erased again saves the
note exactly as it was written, and the Source view edits the raw text. Since 6ac83ce14 and
d4e8493cc an Excel workbook opens in a spreadsheet view: each sheet has a tab of its own, the
formula bar shows a selected cell's formula, a double-clicked cell takes an edit that Save writes
into the workbook while the other sheet stays as it was, a formula cell refuses an edit, a large
sheet finds a cell far down and scrolls to it, and a sheet past the preview's 200,000 cells says
the preview failed. A cell with a comment shows the comment and its author on hover, and a legacy
Excel 2.x .xls sheet shows as a grid too.

test_a_markdown_file_renders_math_alerts_and_diagrams is red on dev f659c0d12: the editor shows
all three as raw text, a regression from 0.13.0.

Discriminates: passes on dev 30f3f6a8f; on a build with the file preview's Markdown, CSV, JSON,
notebook and SQLite branches switched off and its three zoom button handlers removed, every test
here fails. The task, math, comment and .xls tests pass on dev 3b8205505 and fail on a build where
a ticked task is not saved, documents render without the KaTeX extension and .xlsx and .xls no
longer open in the comment-aware preview. The tests added for dev f659c0d12 pass on its build
(three runs, the math test red as said), and each fails on one frontend build of it with these
mutations: the editor serializing every note without its front matter, the note's Save button doing
nothing, sheet tabs not switching, the formula bar showing values only, formula cells taking edits,
find never moving to a match, a workbook save never uploading, undo and redo doing nothing, no
preview size limit and the CSV view cut at 1000 rows.
"""

from __future__ import annotations

import io
import json
import re
import sqlite3
import stat
import struct
import uuid
import zlib

import pytest
from playwright.sync_api import expect

from harness import slow_machine
from harness.backends import wait_until
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, read_grant
from utils.cached_chat import pick_terminal

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

# the page parses a workbook and lays out its whole grid in the browser, seconds on a slow CPU
WORKBOOK_LOAD_TIMEOUT_MS = slow_machine.scaled(5_000)


NOTEBOOK = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": ["# Cider log"]},
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": 1,
            "source": ["print(6 * 7)"],
            "outputs": [{"output_type": "stream", "name": "stdout", "text": ["42\n"]}],
        },
    ],
}
FARM = {"farm": {"name": "Hollow Creek", "acres": 12}, "crops": ["rye", "oats"]}
CHORES = "# Chores\n\n- [ ] prune the pears\n- [x] press the cider\n"
FROST = (
    "# Frost watch\n\n"
    "$$E = mc^2$$\n\n"
    "> [!WARNING]\n> Cover the seedlings tonight.\n\n"
    "```mermaid\ngraph TD\n  Pick[Pick apples] --> Press[Press juice]\n```\n"
)


@pytest.fixture
def home_files(open_terminal):
    """The files the tests open, written into the terminal's home before the browser lists it."""
    home = open_terminal.home
    (home / "plan-notes.md").write_text("# Harvest plan\n\nBring **twelve** baskets.\n")
    (home / "orchard.csv").write_text("tree,yield\napple,40\npear,25\n")
    (home / "farm.json").write_text(json.dumps(FARM))
    (home / "broken.json").write_text('{"farm": ')
    (home / "label.png").write_bytes(png())
    (home / "log.ipynb").write_text(json.dumps(NOTEBOOK))
    (home / "chores.md").write_text(CHORES)
    (home / "frost.md").write_text(FROST)
    (home / "ledger.xls").write_bytes(biff2_xls([["barrel", "litres"], ["cider", 120.0]]))
    (home / "stock.db").unlink(missing_ok=True)
    with sqlite3.connect(home / "stock.db") as database:
        database.execute("CREATE TABLE barrels (label TEXT, litres INTEGER)")
        database.executemany("INSERT INTO barrels VALUES (?, ?)", [("cider", 120), ("perry", 80)])


@pytest.fixture
def browser(home_files, page_for, make_user, admin, preserve, open_terminal):
    """The File browser region of a chat with the Open Terminal picked."""
    preserve(TERMINAL_SERVERS_CONFIG)
    person = make_user()
    connection = open_terminal.connection(config={"access_grants": [read_grant(person.id)]})
    with admin.client() as client:
        configure_terminals(client, connection)
    page = page_for(person)
    pick_terminal(page, connection["name"])
    return page.get_by_role("region", name="File browser")


def open_file(browser, name: str) -> None:
    browser.get_by_text(name, exact=True).click()


def png(width: int = 64, height: int = 48) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    rows = b"".join(b"\x00" + b"\x20\x80\xe0" * width for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def xlsx_with_a_comment() -> bytes:
    openpyxl = pytest.importorskip("openpyxl")
    from openpyxl.comments import Comment

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Barrels"
    sheet.append(["barrel", "litres"])
    sheet.append(["cider", 120])
    sheet["B2"].comment = Comment("Topped up after racking", "Mara")
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def workbook(sheets: dict[str, list[list]]) -> bytes:
    """An .xlsx with one sheet per entry, its rows written from A1."""
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.Workbook(write_only=True)
    for title, rows in sheets.items():
        sheet = book.create_sheet(title)
        for row in rows:
            sheet.append(row)
    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()


def read_sheets(path) -> dict[str, list[tuple]]:
    """Every sheet's rows as stored, formulas as their text."""
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.load_workbook(path)
    return {sheet.title: list(sheet.iter_rows(values_only=True)) for sheet in book.worksheets}


HARVEST = {
    "Apples": [["variety", "crates"], ["Bramley", 12], ["Cox", 8], ["total", "=B2+B3"]],
    "Pears": [["variety", "crates"], ["Conference", 5]],
}


def unique(name: str) -> str:
    """`name` with a tag of its own, as the terminal's home is shared by the whole session."""
    stem, _, extension = name.rpartition(".")
    return f"{stem}-{uuid.uuid4().hex[:6]}.{extension}"


def open_workbook(browser, home, name: str, sheets: dict[str, list[list]]):
    """Write the workbook, open it and return its worksheet grid once the first cell shows."""
    (home / name).write_bytes(workbook(sheets))
    browser.get_by_label("Refresh", exact=True).first.click()
    open_file(browser, name)
    grid = browser.get_by_role("grid", name="Worksheet")
    first = next(iter(sheets.values()))[0][0]
    # cells by their text: naming each of a sheet's thousand cells by role is slow
    expect(grid.get_by_text(str(first), exact=True)).to_be_visible(timeout=WORKBOOK_LOAD_TIMEOUT_MS)
    return grid


def oversized_workbook(also: dict[str, list[list]] | None = None) -> bytes:
    """A first sheet whose two cells span A1:CV2001, 200,100 cells, just past the preview's
    200,000, then the sheets in `also`."""
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.Workbook()
    book.active.title = "Census"
    book.active["A1"] = "census"
    book.active["CV2001"] = "last"
    for title, rows in (also or {}).items():
        sheet = book.create_sheet(title)
        for row in rows:
            sheet.append(row)
    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()


def biff2_xls(rows: list[list[str | float]]) -> bytes:
    """A legacy Excel 2.x worksheet: the bare BIFF record stream, strings and numbers."""

    def record(kind: int, data: bytes = b"") -> bytes:
        return struct.pack("<HH", kind, len(data)) + data

    cells = b""
    for row_number, row in enumerate(rows):
        for column_number, value in enumerate(row):
            where = struct.pack("<HH", row_number, column_number) + b"\x00\x00\x00"
            if isinstance(value, str):
                text = value.encode("latin-1")
                cells += record(0x0004, where + bytes([len(text)]) + text)
            else:
                cells += record(0x0003, where + struct.pack("<d", value))
    return record(0x0009, struct.pack("<HH", 0x0002, 0x0010)) + cells + record(0x000A)


def test_a_markdown_file_shows_rendered_and_as_source(browser):
    open_file(browser, "plan-notes.md")

    expect(browser.get_by_role("heading", name="Harvest plan")).to_be_visible()
    expect(browser.locator("strong", has_text="twelve")).to_be_visible()

    browser.get_by_role("button", name="Source").click()

    expect(browser.get_by_text("# Harvest plan")).to_be_visible()
    expect(browser.get_by_role("heading", name="Harvest plan")).to_have_count(0)


def test_ticking_a_markdown_task_saves_it_to_the_file(browser, open_terminal):
    open_file(browser, "chores.md")
    task = (
        browser.get_by_role("listitem").filter(has_text="prune the pears").get_by_role("checkbox")
    )
    expect(task).not_to_be_checked()

    task.click()

    expect(task).to_be_checked()
    # since b6a77a0f1 a note opens in the editor, where a tick is an edit until it is saved
    browser.get_by_role("button", name="Save", exact=True).click()
    chores = open_terminal.home / "chores.md"
    ticked = CHORES.replace("[ ] prune", "[x] prune")
    assert wait_until(lambda: chores.read_text() == ticked, 10), (
        f"the ticked task never reached the file: {chores.read_text()!r}"
    )


def test_a_markdown_file_renders_math_alerts_and_diagrams(browser):
    # red on dev f659c0d12: since b6a77a0f1 a writable note opens in the editor, which shows all
    # three as raw text (regression from 0.13.0, open-webui/open-webui#33900)
    open_file(browser, "frost.md")

    expect(browser.locator("math")).to_have_count(1)
    expect(browser.get_by_text("WARNING", exact=True)).to_be_visible()
    expect(browser.get_by_text("Cover the seedlings tonight.")).to_be_visible()
    expect(browser.get_by_text("[!WARNING]")).to_have_count(0)
    expect(browser.get_by_text("Pick apples", exact=True)).to_be_visible()
    expect(browser.get_by_text("graph TD")).to_have_count(0)
    expect(browser.get_by_text("$$")).to_have_count(0)


def test_a_read_only_markdown_file_renders_math_alerts_and_diagrams(browser, open_terminal):
    frost = open_terminal.home / "frost.md"
    frost.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    try:
        browser.get_by_label("Refresh", exact=True).first.click()
        open_file(browser, "frost.md")

        expect(browser.locator("math")).to_have_count(1)
        expect(browser.get_by_text("WARNING", exact=True)).to_be_visible()
        expect(browser.get_by_text("Cover the seedlings tonight.")).to_be_visible()
        expect(browser.get_by_text("[!WARNING]")).to_have_count(0)
        expect(browser.get_by_text("Pick apples", exact=True)).to_be_visible()
        expect(browser.get_by_text("graph TD")).to_have_count(0)
        expect(browser.get_by_text("$$")).to_have_count(0)
        expect(browser.get_by_role("button", name="Save", exact=True)).to_be_disabled()
    finally:
        frost.chmod(stat.S_IRUSR | stat.S_IWUSR)


def test_a_spreadsheet_comment_shows_on_its_cell(browser, open_terminal):
    (open_terminal.home / "barrels.xlsx").write_bytes(xlsx_with_a_comment())
    browser.get_by_label("Refresh", exact=True).first.click()
    open_file(browser, "barrels.xlsx")

    cell = browser.get_by_role("gridcell", name="120")
    expect(cell).to_contain_text("◥", timeout=WORKBOOK_LOAD_TIMEOUT_MS)
    expect(browser.get_by_role("gridcell", name="cider")).not_to_contain_text("◥")

    cell.hover()

    comment = browser.page.get_by_role("dialog", name="Comment")
    expect(comment).to_contain_text("Topped up after racking")
    expect(comment).to_contain_text("Mara")


def test_a_legacy_xls_sheet_shows_as_a_table(browser):
    open_file(browser, "ledger.xls")

    grid = browser.get_by_role("grid", name="Worksheet")
    expect(grid.get_by_text("litres", exact=True)).to_be_visible()
    expect(grid.get_by_role("row", name=re.compile(r"cider\s*120"))).to_be_visible()


def test_a_csv_file_shows_as_a_table(browser):
    open_file(browser, "orchard.csv")

    table = browser.get_by_role("table")
    expect(table.locator("th", has_text="tree")).to_be_visible()
    expect(table.locator("th", has_text="yield")).to_be_visible()
    expect(table.get_by_role("row", name=re.compile(r"apple\s*40"))).to_be_visible()
    expect(table.get_by_role("row", name=re.compile(r"pear\s*25"))).to_be_visible()


def test_a_json_file_shows_as_a_tree(browser):
    open_file(browser, "farm.json")

    expect(browser.get_by_text('"Hollow Creek"')).to_be_visible()
    expect(browser.get_by_text('"oats"')).to_be_visible()
    expect(browser.get_by_text("JSON parse error")).to_have_count(0)

    browser.get_by_text("farm", exact=True).click()

    expect(browser.get_by_text("{ {2} }")).to_be_visible()
    expect(browser.get_by_text('"Hollow Creek"')).to_have_count(0)


def test_an_invalid_json_file_opens_as_its_source(browser):
    # since b5eaa59d0 a document that does not parse opens in the code editor
    open_file(browser, "broken.json")

    expect(browser.get_by_text('{"farm":')).to_be_visible()
    expect(browser.get_by_role("textbox")).to_be_editable()


def test_an_image_zooms_in_out_and_back(browser):
    open_file(browser, "label.png")

    expect(browser.get_by_role("img", name="label.png")).to_be_visible()
    level = browser.get_by_role("button", name="Reset zoom")
    expect(level).to_have_text("100%")

    browser.get_by_role("button", name="Zoom in").click()
    expect(level).not_to_have_text("100%")
    zoomed_in = int(level.inner_text().rstrip("%"))
    assert zoomed_in > 100

    browser.get_by_role("button", name="Reset zoom").click()
    expect(level).to_have_text("100%")

    browser.get_by_role("button", name="Zoom out").click()
    expect(level).not_to_have_text("100%")
    assert int(level.inner_text().rstrip("%")) < 100


def test_a_sqlite_database_shows_its_table_and_rows(browser):
    open_file(browser, "stock.db")

    expect(browser.get_by_role("button", name="barrels")).to_be_visible()
    table = browser.get_by_role("table")
    expect(table.locator("th", has_text="litres")).to_be_visible()
    expect(table.get_by_role("row", name=re.compile(r"cider\s*120"))).to_be_visible()
    expect(table.get_by_role("row", name=re.compile(r"perry\s*80"))).to_be_visible()


def test_a_notebook_shows_its_cells_and_outputs(browser):
    open_file(browser, "log.ipynb")

    expect(browser.get_by_role("heading", name="Cider log")).to_be_visible()
    expect(browser.get_by_text("print(6 * 7)")).to_be_visible()
    expect(browser.get_by_text("42", exact=True)).to_be_visible()


def test_a_workbook_shows_each_sheet_from_its_tab(browser, open_terminal):
    grid = open_workbook(browser, open_terminal.home, unique("harvest.xlsx"), HARVEST)
    sheets = browser.get_by_role("tablist", name="Worksheets")

    expect(sheets.get_by_role("tab")).to_have_text(["Apples", "Pears"])
    expect(sheets.get_by_role("tab", name="Apples")).to_have_attribute("aria-selected", "true")
    expect(grid.get_by_role("row", name=re.compile(r"Bramley\s*12"))).to_be_visible()
    expect(browser.get_by_role("status").filter(has_text="Used range")).to_have_text(
        "Used range: A1:B4"
    )

    sheets.get_by_role("tab", name="Pears").click()

    expect(sheets.get_by_role("tab", name="Pears")).to_have_attribute("aria-selected", "true")
    # switching the sheet lays its grid out anew
    expect(grid.get_by_role("row", name=re.compile(r"Conference\s*5"))).to_be_visible(
        timeout=WORKBOOK_LOAD_TIMEOUT_MS
    )
    expect(grid.get_by_text("Bramley", exact=True)).to_have_count(0)


def test_a_selected_cell_shows_its_formula_in_the_formula_bar(browser, open_terminal):
    grid = open_workbook(browser, open_terminal.home, unique("totals.xlsx"), HARVEST)

    # the formula has no saved result, so its cell is empty and known only by its address
    grid.locator("[data-cell='B4']").click()

    expect(browser.get_by_label("Selected range")).to_have_text("B4")
    expect(browser.get_by_label("Formula or value")).to_have_value("=B2+B3")

    grid.get_by_text("Cox", exact=True).click()

    expect(browser.get_by_label("Selected range")).to_have_text("A3")
    expect(browser.get_by_label("Formula or value")).to_have_value("Cox")


def test_an_edited_cell_is_saved_into_the_workbook(browser, open_terminal):
    path = open_terminal.home / unique("edit-harvest.xlsx")
    grid = open_workbook(browser, open_terminal.home, path.name, HARVEST)
    written = path.read_bytes()

    grid.get_by_text("Cox", exact=True).dblclick()
    cell_input = browser.get_by_role("textbox", name="Edit cell A3")
    expect(cell_input).to_have_value("Cox")
    cell_input.fill("Russet")
    cell_input.press("Enter")
    # each edit lays the sheet's grid out anew
    expect(grid.get_by_text("Russet", exact=True)).to_be_visible(timeout=WORKBOOK_LOAD_TIMEOUT_MS)
    bramley = grid.get_by_role("row", name=re.compile(r"Bramley"))
    bramley.get_by_text("12", exact=True).dblclick()
    browser.get_by_role("textbox", name="Edit cell B2").fill("15")
    browser.get_by_role("textbox", name="Edit cell B2").press("Enter")
    expect(bramley.get_by_text("15", exact=True)).to_be_visible(timeout=WORKBOOK_LOAD_TIMEOUT_MS)
    browser.get_by_role("button", name="Save", exact=True).click()

    assert wait_until(lambda: path.read_bytes() != written, 10), "Save never wrote the workbook"
    stored = read_sheets(path)
    assert stored["Apples"] == [
        ("variety", "crates"),
        ("Bramley", 15),
        ("Russet", 8),
        ("total", "=B2+B3"),
    ], f"the saved Apples sheet holds {stored['Apples']}"
    assert stored["Pears"] == [("variety", "crates"), ("Conference", 5)], (
        f"saving changed the untouched Pears sheet to {stored['Pears']}"
    )


def test_a_formula_cell_cannot_be_edited(browser, open_terminal):
    grid = open_workbook(browser, open_terminal.home, unique("locked.xlsx"), HARVEST)

    # the formula has no saved result, so its cell is empty and known only by its address
    grid.locator("[data-cell='B4']").dblclick()

    expect(browser.get_by_role("status").filter(has_text="read-only")).to_have_text(
        "Formula cells, table headers, and protected sheets are read-only."
    )
    expect(browser.get_by_role("textbox", name="Edit cell B4")).to_have_count(0)


def test_a_large_sheet_shows_its_last_row_and_finds_a_cell(browser, open_terminal):
    rows = [["lot", "barrel", "litres", "cellar"]]
    rows += [[f"lot-{n}", f"barrel-{n}", n, "north"] for n in range(1, 200)]
    grid = open_workbook(browser, open_terminal.home, unique("cellar.xlsx"), {"Lots": rows})

    expect(browser.get_by_role("status").filter(has_text="Used range")).to_have_text(
        "Used range: A1:D200"
    )
    browser.get_by_role("button", name="Find in sheet").click()
    find = browser.get_by_role("search", name="Find in sheet").get_by_role("textbox")
    find.fill("barrel-199")
    find.press("Enter")

    expect(browser.get_by_text("1 / 1", exact=True)).to_be_visible()
    expect(grid.get_by_text("barrel-199", exact=True)).to_be_in_viewport()
    expect(browser.get_by_label("Selected range")).to_have_text("B200")


def test_a_sheet_past_the_preview_limit_says_the_preview_failed(browser, open_terminal):
    path = open_terminal.home / unique("census.xlsx")
    path.write_bytes(oversized_workbook())
    browser.get_by_label("Refresh", exact=True).first.click()

    open_file(browser, path.name)

    expect(browser.get_by_text("Preview failed", exact=True)).to_be_visible(
        timeout=WORKBOOK_LOAD_TIMEOUT_MS
    )
    expect(browser.get_by_role("grid", name="Worksheet")).to_have_count(0)


def test_a_long_csv_file_shows_every_row(browser, open_terminal):
    lines = ["tree,yield"] + [f"tree-{n},{n}" for n in range(1, 501)]
    path = open_terminal.home / unique("orchard-census.csv")
    path.write_text("\n".join(lines) + "\n")
    browser.get_by_label("Refresh", exact=True).first.click()

    open_file(browser, path.name)

    table = browser.get_by_role("table")
    expect(table.get_by_text("tree-500", exact=True)).to_be_attached()
    expect(table.get_by_role("row")).to_have_count(501)
    last = table.get_by_text("tree-500", exact=True)
    last.scroll_into_view_if_needed()
    expect(last).to_be_visible()


LETTER = "---\ntitle: Letter\n---\n# Letter\n\nDear **Mara**,\n\n- [ ] post it\n"


def test_a_markdown_file_edited_in_the_editor_is_saved_as_markdown(browser, open_terminal):
    letter = open_terminal.home / unique("letter.md")
    letter.write_text(LETTER)
    browser.get_by_label("Refresh", exact=True).first.click()
    open_file(browser, letter.name)
    expect(browser.get_by_role("button", name="Editor")).to_have_attribute("aria-pressed", "true")
    expect(browser.get_by_role("heading", name="Letter")).to_be_visible()
    expect(browser.get_by_text("title: Letter")).to_have_count(0)

    browser.get_by_text("Dear", exact=False).click()
    browser.page.keyboard.press("End")
    browser.page.keyboard.type(" thanks")
    browser.get_by_role("button", name="Save", exact=True).click()

    saved = LETTER.replace("**Mara**,", "**Mara**, thanks")
    assert wait_until(lambda: letter.read_text() == saved, 10), (
        f"the edited note on disk reads {letter.read_text()!r}"
    )


def test_an_edit_undone_saves_the_markdown_as_it_was_written(browser, open_terminal):
    shopping = open_terminal.home / unique("shopping.md")
    written = "Shopping\n========\n\n* eggs\n* flour\n"
    shopping.write_text(written)
    before = shopping.stat().st_mtime_ns
    browser.get_by_label("Refresh", exact=True).first.click()
    open_file(browser, shopping.name)

    browser.get_by_text("flour", exact=True).click()
    browser.page.keyboard.press("End")
    browser.page.keyboard.type("s")
    browser.page.keyboard.press("Backspace")
    browser.page.keyboard.press("Control+s")

    assert wait_until(lambda: shopping.stat().st_mtime_ns != before, 10), "Ctrl+S saved nothing"
    assert shopping.read_text() == written, (
        f"an unchanged note was rewritten as {shopping.read_text()!r}"
    )


def test_the_source_view_edits_and_saves_the_raw_markdown(browser, open_terminal):
    plan = open_terminal.home / unique("raw-plan.md")
    plan.write_text("# Plan\n")
    browser.get_by_label("Refresh", exact=True).first.click()
    open_file(browser, plan.name)

    browser.get_by_role("button", name="Source").click()
    expect(browser.get_by_role("button", name="Source")).to_have_attribute("aria-pressed", "true")
    browser.get_by_text("# Plan", exact=True).click()
    browser.page.keyboard.press("End")
    browser.page.keyboard.type(" B")
    browser.get_by_role("button", name="Save", exact=True).click()

    assert wait_until(lambda: plan.read_text() == "# Plan B\n", 10), (
        f"the note on disk reads {plan.read_text()!r}"
    )
    browser.get_by_role("button", name="Editor").click()
    expect(browser.get_by_role("heading", name="Plan B")).to_be_visible()


def test_an_edit_undone_in_the_sheet_is_not_saved(browser, open_terminal):
    path = open_terminal.home / unique("undo-harvest.xlsx")
    grid = open_workbook(browser, open_terminal.home, path.name, HARVEST)

    grid.get_by_text("Cox", exact=True).dblclick()
    browser.get_by_role("textbox", name="Edit cell A3").fill("Russet")
    browser.get_by_role("textbox", name="Edit cell A3").press("Enter")
    expect(grid.get_by_text("Russet", exact=True)).to_be_visible(timeout=WORKBOOK_LOAD_TIMEOUT_MS)
    browser.get_by_role("button", name="Undo").click()

    expect(grid.get_by_text("Cox", exact=True)).to_be_visible(timeout=WORKBOOK_LOAD_TIMEOUT_MS)
    expect(grid.get_by_text("Russet", exact=True)).to_have_count(
        0, timeout=WORKBOOK_LOAD_TIMEOUT_MS
    )
    browser.get_by_role("button", name="Redo").click()
    expect(grid.get_by_text("Russet", exact=True)).to_be_visible(timeout=WORKBOOK_LOAD_TIMEOUT_MS)


@pytest.mark.xfail(strict=True, reason="gap: a CSV file opens as a plain table, not as a sheet")
def test_a_csv_file_opens_in_the_spreadsheet_view_like_a_workbook(browser):
    open_file(browser, "orchard.csv")

    grid = browser.get_by_role("grid", name="Worksheet")
    expect(grid.get_by_text("apple", exact=True)).to_be_visible(timeout=WORKBOOK_LOAD_TIMEOUT_MS)


@pytest.mark.xfail(
    strict=True, reason="gap: a sheet past the preview limit only says Preview failed"
)
def test_a_sheet_past_the_preview_limit_says_it_is_too_large(browser, open_terminal):
    path = open_terminal.home / unique("census.xlsx")
    path.write_bytes(oversized_workbook())
    browser.get_by_label("Refresh", exact=True).first.click()

    open_file(browser, path.name)

    expect(browser.get_by_text("too large to preview")).to_be_visible(
        timeout=WORKBOOK_LOAD_TIMEOUT_MS
    )


@pytest.mark.xfail(
    strict=True, reason="gap: one sheet past the preview limit hides the workbook's other sheets"
)
def test_the_other_sheets_still_open_when_one_is_past_the_preview_limit(browser, open_terminal):
    path = open_terminal.home / unique("census.xlsx")
    path.write_bytes(oversized_workbook(also={"Summary": [["parish", "people"], ["Ashby", 412]]}))
    browser.get_by_label("Refresh", exact=True).first.click()
    open_file(browser, path.name)
    summary = browser.get_by_role("tablist", name="Worksheets").get_by_role("tab", name="Summary")
    expect(summary).to_be_visible(timeout=WORKBOOK_LOAD_TIMEOUT_MS)

    summary.click()

    grid = browser.get_by_role("grid", name="Worksheet")
    expect(grid.get_by_role("row", name=re.compile(r"Ashby\s*412"))).to_be_visible()
