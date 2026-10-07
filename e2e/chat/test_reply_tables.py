"""Journey: a table in a reply is drawn as a table that copies, exports and scrolls in its own box.

A reply holding a markdown table shows it as a table with its header and rows. The table's Copy
button puts the table's markdown on the clipboard, and Export to CSV downloads the same rows as
a CSV file, each cell quoted, a quote inside a cell doubled and a comma kept inside its cell. A
table wider than the reply scrolls sideways inside its own box while the reply around it stays
put and the page never scrolls sideways, and bold words, code and links inside cells are drawn
as such, each cell aligned as its column says.

Discriminates: passes on dev 30f3f6a8f; in a frontend build whose table copy and CSV export write
nothing the copy and export tests fail. On dev ebc6add67, in its mutation build (the
`rendering-front` copy: the table's box no longer scrolling) the wide table test goes red and
the cell test stays green.
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


WIDE_HEADERS = [f"Column {number} with a long heading" for number in range(1, 13)]
WIDE_TABLE = (
    "| " + " | ".join(WIDE_HEADERS) + " |\n"
    "| " + " | ".join("---" for _ in WIDE_HEADERS) + " |\n"
    "| " + " | ".join(f"value{number}_written_without_any_spaces" for number in range(1, 13)) + " |"
)
# the table scrolls inside the box drawn around it
SCROLLED_SIDEWAYS = "table => table.parentElement.scrollLeft"


def test_a_table_wider_than_the_reply_scrolls_sideways_inside_its_own_box(
    page_for, make_user, upstream
):
    page = page_for(make_user())
    upstream.queue(reply.text(f"{WIDE_TABLE}\n\nWide enough.", match=reply.answering("wide table")))
    send(page, "give me a wide table")
    expect_reply(page, "Wide enough.")
    table = _table(page)
    last_heading = table.get_by_role("columnheader").last
    expect(table.get_by_role("columnheader")).to_have_text(WIDE_HEADERS)

    closing_words = last_reply(page).get_by_text("Wide enough.")
    reply_edge = last_reply(page).evaluate("box => box.getBoundingClientRect().right")
    heading_edge = last_heading.evaluate("cell => cell.getBoundingClientRect().left")
    assert heading_edge > reply_edge, "the last column already fits inside the reply"
    words_edge = left_edge(closing_words)

    last_heading.scroll_into_view_if_needed()

    expect(last_heading).to_be_in_viewport()
    assert table.evaluate(SCROLLED_SIDEWAYS) > 0, "the table did not scroll inside its box"
    assert left_edge(closing_words) == words_edge, "the reply around the table moved sideways"
    page_width = page.evaluate("[document.documentElement.scrollWidth, innerWidth]")
    assert page_width[0] <= page_width[1], f"the page itself scrolls sideways: {page_width}"


def left_edge(element: Locator) -> float:
    return element.evaluate("element => element.getBoundingClientRect().left")


def test_inline_markdown_inside_cells_is_drawn(page_for, make_user, upstream):
    page = page_for(make_user())
    table = (
        "| Item | Detail |\n| :--- | ---: |\n"
        "| **Kettle** | `220 V` and [manual](https://example.com/manual) |"
    )
    upstream.queue(reply.text(f"{table}\n\nCells done.", match=reply.answering("cell table")))
    send(page, "give me a cell table")
    expect_reply(page, "Cells done.")

    cells = _table(page).get_by_role("cell")
    expect(cells.first.get_by_role("strong")).to_have_text("Kettle")
    expect(cells.last.get_by_role("code")).to_have_text("220 V")
    expect(cells.last.get_by_role("link", name="manual")).to_have_attribute(
        "href", "https://example.com/manual"
    )
    assert cells.last.evaluate("cell => getComputedStyle(cell).textAlign") == "right"
    expect(_table(page)).not_to_contain_text("**")
