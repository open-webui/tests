"""Journey: a person clicks files in the terminal's file browser and each opens in its own view.

A real Open Terminal runs with files in its home: a Markdown note, a CSV sheet, a JSON document
and a broken one, a PNG image, a SQLite database and a Jupyter notebook. In a chat the person
picks the terminal from the input's Terminal menu and clicks each file in the File browser. The
note shows rendered and its source after the Source toggle; the sheet shows as a table; the
document shows as a tree whose nodes fold and the broken one shows the parse error; the image
shows with working zoom buttons whose percentage changes; the database lists its table with its
rows; the notebook shows its cells and their outputs.

Discriminates: passes on dev 30f3f6a8f; on a build with the file preview's Markdown, CSV, JSON,
notebook and SQLite branches switched off and its three zoom button handlers removed, every test
here fails.
"""

from __future__ import annotations

import json
import re
import sqlite3
import struct
import zlib

import pytest
from playwright.sync_api import expect

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
    page.goto("/")
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


def test_a_markdown_file_shows_rendered_and_as_source(browser):
    open_file(browser, "plan-notes.md")

    expect(browser.get_by_role("heading", name="Harvest plan")).to_be_visible()
    expect(browser.locator("strong", has_text="twelve")).to_be_visible()

    browser.get_by_role("button", name="Source").click()

    expect(browser.get_by_text("# Harvest plan")).to_be_visible()
    expect(browser.get_by_role("heading", name="Harvest plan")).to_have_count(0)


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
