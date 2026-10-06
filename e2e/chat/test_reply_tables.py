"""Journey: a table in a reply is drawn as a table that copies as markdown and exports as CSV.

A reply holding a markdown table shows it as a table with its header and rows. The table's Copy
button puts the table's markdown on the clipboard, and Export to CSV downloads the same rows as
a CSV file, each cell quoted, a quote inside a cell doubled and a comma kept inside its cell.

Discriminates: passes on dev 30f3f6a8f; in a frontend build whose table copy and CSV export write
nothing both tests fail.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.chat_ui import expect_reply, last_reply, send
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

TABLE = (
    "| Harbour | High tide | Note |\n"
    "| --- | --- | --- |\n"
    '| Portree | 12:10 | calm, "good" holding |\n'
    "| Mallaig | 12:40 | busy |"
)
ANSWER = f"Here are the tides.\n\n{TABLE}\n\nSafe sailing."


@pytest.fixture
def table_reply(page_for, make_user, upstream) -> Page:
    page = page_for(make_user(), permissions=["clipboard-read", "clipboard-write"])
    upstream.queue(reply.text(ANSWER, match=reply.answering("tide table")))
    send(page, "give me a tide table")
    expect_reply(page, "Safe sailing.")
    return page


def _table(page: Page) -> Locator:
    return last_reply(page).get_by_role("table")


def _table_buttons(page: Page) -> Locator:
    table = _table(page)
    table.hover()
    return table.locator("xpath=ancestor::div[contains(@class, 'group')][1]")


def test_the_table_is_drawn_and_copies_as_its_markdown(table_reply):
    table = _table(table_reply)
    expect(table.get_by_role("columnheader")).to_have_text(["Harbour", "High tide", "Note"])
    expect(table.get_by_role("row")).to_have_count(3)
    expect(table.get_by_role("cell", name="Mallaig")).to_be_visible()

    tooltip_button(_table_buttons(table_reply), "Copy").click()

    table_reply.wait_for_function("navigator.clipboard.readText().then(text => text !== '')")
    assert table_reply.evaluate("navigator.clipboard.readText()") == TABLE


def test_export_to_csv_downloads_the_rows_with_cells_quoted(table_reply):
    with table_reply.expect_download() as download_info:
        tooltip_button(_table_buttons(table_reply), "Export to CSV").click()
    download = download_info.value

    assert download.suggested_filename.endswith(".csv")
    with open(download.path(), encoding="utf-8-sig") as saved:
        exported = saved.read()
    assert exported == (
        '"Harbour","High tide","Note"\n'
        '"Portree","12:10","calm, ""good"" holding"\n'
        '"Mallaig","12:40","busy"'
    )
