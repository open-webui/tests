"""Journey: a person clicks files in the terminal's file browser and each opens in its own view.

A real Open Terminal runs with files in its home: a Markdown note, a CSV sheet, a JSON document
and a broken one, a PNG image, a SQLite database and a Jupyter notebook. In a chat the person
picks the terminal from the input's Terminal menu and clicks each file in the File browser. The
note shows rendered and its source after the Source toggle; the sheet shows as a table; the
document shows as a tree whose nodes fold and the broken one shows the parse error; the image
shows with working zoom buttons whose percentage changes; the database lists its table with its
rows; the notebook shows its cells and their outputs. Ticking a task in a Markdown note writes
the tick into the file, and a note renders its math, its alert and its Mermaid diagram. An Excel
workbook marks the cell that has a comment and shows the comment with its author on hover, and a
legacy Excel 2.x .xls sheet shows as a table.

Discriminates: passes on dev 30f3f6a8f; on a build with the file preview's Markdown, CSV, JSON,
notebook and SQLite branches switched off and its three zoom button handlers removed, every test
here fails. The task, math, comment and .xls tests pass on dev 3b8205505 and fail on a build where
a ticked task is not saved, documents render without the KaTeX extension and .xlsx and .xls no
longer open in the comment-aware preview.
"""

from __future__ import annotations

import io
import json
import re
import sqlite3
import struct
import zlib

import pytest
from playwright.sync_api import expect

from harness.backends import wait_until
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, read_grant
from utils.cached_chat import pick_terminal

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


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
    chores = open_terminal.home / "chores.md"
    ticked = CHORES.replace("[ ] prune", "[x] prune")
    assert wait_until(lambda: chores.read_text() == ticked, 10), (
        f"the ticked task never reached the file: {chores.read_text()!r}"
    )


def test_a_markdown_file_renders_math_alerts_and_diagrams(browser):
    open_file(browser, "frost.md")

    expect(browser.locator("math")).to_have_count(1)
    expect(browser.get_by_text("WARNING", exact=True)).to_be_visible()
    expect(browser.get_by_text("Cover the seedlings tonight.")).to_be_visible()
    expect(browser.get_by_text("[!WARNING]")).to_have_count(0)
    expect(browser.get_by_text("Pick apples", exact=True)).to_be_visible()
    expect(browser.get_by_text("graph TD")).to_have_count(0)
    expect(browser.get_by_text("$$")).to_have_count(0)


def test_a_spreadsheet_comment_shows_on_its_cell(browser, open_terminal):
    (open_terminal.home / "barrels.xlsx").write_bytes(xlsx_with_a_comment())
    browser.get_by_label("Refresh", exact=True).first.click()
    open_file(browser, "barrels.xlsx")

    cell = browser.get_by_role("cell", name="120")
    expect(cell).to_contain_text("◥")
    expect(browser.get_by_role("cell", name="cider")).not_to_contain_text("◥")

    cell.hover()

    comment = browser.page.get_by_role("dialog", name="Comment")
    expect(comment).to_contain_text("Topped up after racking")
    expect(comment).to_contain_text("Mara")


def test_a_legacy_xls_sheet_shows_as_a_table(browser):
    open_file(browser, "ledger.xls")

    table = browser.get_by_role("table")
    expect(table.get_by_role("cell", name="litres")).to_be_visible()
    expect(table.get_by_role("row", name=re.compile(r"cider\s*120"))).to_be_visible()


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


def test_an_invalid_json_file_shows_the_parse_error(browser):
    open_file(browser, "broken.json")

    expect(browser.get_by_text("JSON parse error")).to_be_visible()
    expect(browser.get_by_text('{"farm":')).to_be_visible()


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
