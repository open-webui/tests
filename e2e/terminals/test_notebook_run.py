"""Journey: a person runs the cells of a notebook opened from a terminal's file browser.

The terminal is the full Open Terminal image (`full_open_terminal`), the one with notebook
kernels; the tests skip where `OPEN_TERMINAL_FULL_IMAGE` is not set. The person opens a notebook
in Files. A cell's Run button starts a kernel on the terminal, shows what the cell printed and
its run number, and the terminal saves the output into the notebook. A cell that is
double-clicked can be retyped and runs as typed with Ctrl+Enter. Run All runs every code cell
in order, and a cell that fails shows its error where the output goes. Restart starts a fresh
kernel that has forgotten what the cells set, and Stop ends the kernel, after which Restart and
Stop are gone until a cell is run again. A read-only notebook shows its cells with nothing to
run or edit.

Discriminates: all pass on dev c15f28649 with the full image of Open Terminal 0.17.7. On
frontend builds of it with these mutations each named test fails: a cell's run number never
put under it (run); a retyped cell run with its old source (retyped cell); Run All stopping
after the first cell (Run All); Restart neither stopping nor starting a kernel, and on another
build Stop doing nothing (Restart and Stop); the run controls shown for a read-only file
(read-only).
"""

from __future__ import annotations

import json
import uuid

import pytest
from playwright.sync_api import Locator, Response, expect

from harness import slow_machine
from harness.backends import wait_until
from harness.terminal_server import TERMINAL_SERVERS_CONFIG
from utils.file_views import back_to, files_of, open_file

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

# the first cell of a notebook waits for its kernel to start on the terminal
KERNEL_TIMEOUT_MS = slow_machine.scaled(10_000)
RUN_CELL = "Run cell (⌘+Enter)"


def _notebook(*sources: str) -> str:
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": ["# Cellar sums"]},
        *(
            {
                "cell_type": "code",
                "metadata": {},
                "source": [source],
                "outputs": [],
                "execution_count": None,
            }
            for source in sources
        ),
    ]
    kernel = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return json.dumps(
        {"cells": cells, "metadata": {"kernelspec": kernel}, "nbformat": 4, "nbformat_minor": 5}
    )


@pytest.fixture
def open_notebook(full_open_terminal, make_user, page_for, admin, preserve):
    """`open_notebook(*sources)` writes a notebook with those code cells into a folder of the
    test's own on the terminal and opens it in a person's file browser; returns the browser and
    the notebook's path. `with_locked_copy=True` puts a read-only copy, locked.ipynb, beside it."""

    def write_and_open(*sources: str, with_locked_copy: bool = False) -> tuple[Locator, str]:
        folder = f"notebooks-{uuid.uuid4().hex[:6]}"
        path = f"{folder}/sums.ipynb"
        full_open_terminal.write_file(path, _notebook(*sources))
        if with_locked_copy:
            locked = f"{folder}/locked.ipynb"
            full_open_terminal.write_file(locked, _notebook(*sources))
            with full_open_terminal.client() as client:
                done = client.post(
                    "/execute", params={"wait": 10}, json={"command": f"chmod 444 {locked}"}
                )
            assert done.status_code == 200, f"locking the notebook failed: {done.text}"
        preserve(TERMINAL_SERVERS_CONFIG)
        browser = files_of(page_for, admin, make_user(), full_open_terminal, folder)
        open_file(browser, "sums.ipynb")
        expect(browser.get_by_role("heading", name="Cellar sums")).to_be_visible()
        return browser, path

    return write_and_open


def _saved(terminal, path: str) -> str:
    with terminal.client() as client:
        read = client.get("/files/read", params={"path": path})
    assert read.status_code == 200, f"reading the notebook back failed: {read.text}"
    return read.json()["content"]


def _kernel_started(response: Response) -> bool:
    return response.request.method == "POST" and response.url.endswith("/notebooks")


def _run_buttons(browser: Locator) -> Locator:
    return browser.get_by_title(RUN_CELL)


def test_a_cell_runs_on_the_terminal_and_shows_what_it_printed(open_notebook, full_open_terminal):
    tag = uuid.uuid4().hex[:6]
    browser, path = open_notebook(f"print('{tag}', 6 * 7)")
    expect(browser.get_by_text("[ ]", exact=True)).to_be_visible()
    expect(browser.get_by_text(f"{tag} 42")).to_have_count(0)
    # no kernel runs until a cell is run
    expect(browser.get_by_role("button", name="Run All", exact=True)).to_be_visible()
    expect(browser.get_by_role("button", name="Restart", exact=True)).to_have_count(0)

    _run_buttons(browser).click()

    expect(browser.get_by_text(f"{tag} 42", exact=True)).to_be_visible(timeout=KERNEL_TIMEOUT_MS)
    expect(browser.get_by_text("[1]", exact=True)).to_be_visible()
    expect(browser.get_by_role("button", name="Restart", exact=True)).to_be_visible()
    expect(browser.get_by_role("button", name="Stop", exact=True)).to_be_visible()
    assert wait_until(lambda: f"{tag} 42" in _saved(full_open_terminal, path), 10), (
        "the notebook on the terminal does not hold the cell's output"
    )


def test_a_retyped_cell_runs_with_ctrl_enter_as_it_was_typed(open_notebook):
    tag = uuid.uuid4().hex[:6]
    browser, _ = open_notebook(f"print('{tag}', 'as written')")
    page = browser.page

    browser.get_by_text(f"print('{tag}', 'as written')").dblclick()
    # the cell's editor takes the focus only once its language has loaded
    expect(page.locator("[contenteditable=true]:focus")).to_contain_text(f"'{tag}', 'as written'")
    page.keyboard.press("Control+a")
    # in one piece, so the editor closes no bracket of its own
    page.keyboard.insert_text(f"print('{tag}', 2 + 3)")
    page.keyboard.press("Control+Enter")

    expect(browser.get_by_text(f"{tag} 5", exact=True)).to_be_visible(timeout=KERNEL_TIMEOUT_MS)
    expect(browser.get_by_text(f"{tag} as written")).to_have_count(0)
    expect(browser.get_by_text(f"print('{tag}', 2 + 3)")).to_be_visible()


def test_run_all_runs_every_cell_in_order_and_a_failing_cell_shows_its_error(open_notebook):
    tag = uuid.uuid4().hex[:6]
    browser, _ = open_notebook("crates = 40", f"print('{tag}', crates + 2)", "crates / 0")

    browser.get_by_role("button", name="Run All", exact=True).click()

    # the second cell needs what the first one set, so they ran in order on one kernel
    expect(browser.get_by_text(f"{tag} 42", exact=True)).to_be_visible(timeout=KERNEL_TIMEOUT_MS)
    expect(browser.get_by_text("ZeroDivisionError").first).to_be_visible()
    for number in (1, 2):
        expect(browser.get_by_text(f"[{number}]", exact=True)).to_be_visible()
    expect(browser.get_by_role("button", name="Run All", exact=True)).to_be_enabled()


def test_restart_forgets_what_the_cells_set_and_stop_ends_the_kernel(open_notebook):
    tag = uuid.uuid4().hex[:6]
    browser, _ = open_notebook("crates = 40", f"print('{tag}', crates + 2)")
    run = _run_buttons(browser)
    restart = browser.get_by_role("button", name="Restart", exact=True)
    run.first.click()
    expect(browser.get_by_text("[1]", exact=True)).to_be_visible(timeout=KERNEL_TIMEOUT_MS)
    run.last.click()
    expect(browser.get_by_text(f"{tag} 42", exact=True)).to_be_visible()

    # Restart stays on screen while the old kernel stops, so wait for the new one's start
    with browser.page.expect_response(_kernel_started, timeout=KERNEL_TIMEOUT_MS) as started:
        restart.click()
    assert started.value.ok, f"the new kernel did not start: {started.value.status}"
    expect(restart).to_be_visible()
    run.last.click()

    expect(browser.get_by_text("name 'crates' is not defined").first).to_be_visible(
        timeout=KERNEL_TIMEOUT_MS
    )
    expect(browser.get_by_text(f"{tag} 42", exact=True)).to_have_count(0)

    browser.get_by_role("button", name="Stop", exact=True).click()
    expect(restart).to_have_count(0)
    expect(browser.get_by_role("button", name="Stop", exact=True)).to_have_count(0)
    # the next run starts a kernel again
    run.first.click()
    expect(restart).to_be_visible(timeout=KERNEL_TIMEOUT_MS)


def test_a_read_only_notebook_shows_its_cells_with_nothing_to_run_or_edit(open_notebook):
    tag = uuid.uuid4().hex[:6]
    browser, path = open_notebook(f"print('{tag}', 6 * 7)", with_locked_copy=True)
    source = browser.get_by_text(f"print('{tag}', 6 * 7)")
    # the notebook that can be written offers its cell to run
    expect(source).to_be_visible()
    expect(_run_buttons(browser)).to_have_count(1)
    expect(browser.get_by_role("button", name="Run All", exact=True)).to_be_visible()

    back_to(browser, path.split("/")[0])
    open_file(browser, "locked.ipynb")

    expect(source).to_be_visible()
    expect(_run_buttons(browser)).to_have_count(0)
    expect(browser.get_by_role("button", name="Run All", exact=True)).to_have_count(0)
    source.dblclick()
    browser.page.keyboard.insert_text("typed into a read-only cell")
    expect(browser.get_by_text("typed into a read-only cell")).to_have_count(0)
    expect(source).to_be_visible()
