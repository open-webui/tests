"""The automations list and an automation's own page, located the way a person sees them.

The list: each automation is a row named by its title, showing its schedule and holding the
Active switch and the Automation Menu. The row's former `Open automation` label is gone from the
markup; the title is a link inside the row now.

An automation's page is its editor since dev 9723b3a6a: the Name and Instructions boxes, then
Properties whose values (Schedule, Model, Destination, Terminal) are the dropdowns of the Create
dialog shown as plain values, then the Activity list of its runs. A Save button shows in the
page's header while a field differs from the saved automation; Delete moved into its More menu.

The Create dialog's and the page's Terminal dropdown: `terminal_button`, `open_terminal_menu`,
`toggle_terminal`, `close_terminal_menu` and `working_directory`.
"""

from __future__ import annotations

import re

from playwright.sync_api import Locator, Page, expect

from harness.backends import wait_until
from utils.workspace_list import item_rows

RUN_ENTRY = re.compile(r"^\s*(Ran|Failed)\b")
RUN_TIMEOUT_MS = 30_000


def automation_row(page: Page, name: str) -> Locator:
    return item_rows(page).filter(has_text=name)


def wait_until_loaded(dialog: Locator, save: str = "Create") -> None:
    """The dialog resets its fields once its terminals, folders and channels have loaded."""
    expect(dialog.get_by_role("button", name=save, exact=True)).to_be_enabled()


def automation_page(page: Page) -> Locator:
    """The open automation's page: the page tab panel holding its Instructions box."""
    return page.get_by_role("tabpanel").filter(
        has=page.get_by_role("textbox", name="Instructions", exact=True)
    )


def instructions(page: Page) -> Locator:
    return automation_page(page).get_by_role("textbox", name="Instructions", exact=True)


def name_box(page: Page) -> Locator:
    return automation_page(page).get_by_role("textbox", name="Name", exact=True)


def open_automation(page: Page, automation: dict) -> None:
    page.goto(f"/automations/{automation['id']}")
    expect(instructions(page)).to_have_value(automation["data"]["prompt"])


def property_row(page: Page, label: str) -> Locator:
    """The row of `label` among the page's Properties, holding its value."""
    return automation_page(page).get_by_text(label, exact=True).locator("xpath=..")


def save_changes(page: Page) -> None:
    """Press the page's Save and wait until the server has the edit and the button is gone."""
    save = automation_page(page).get_by_role("button", name="Save", exact=True)
    with page.expect_response(
        lambda response: "/update" in response.url and response.request.method == "POST"
    ) as saved:
        save.click()
    assert saved.value.ok, saved.value.text()
    expect(save).to_be_hidden()


def run_entries(page: Page) -> Locator:
    """The page's Activity entries for runs, each reading Ran or Failed and its time."""
    return automation_page(page).get_by_role("button").filter(has_text=RUN_ENTRY)


def run_now(page: Page, owner, automation_id: str, runs: int = 1) -> None:
    """Press Run now, wait until the server lists `runs` runs, then reload: the page reads its
    runs only once, two seconds after Run now."""
    triggered = page.get_by_text("Automation triggered")
    # the previous run's toast covers the header and stays while the pointer rests on it
    page.mouse.move(0, 0)
    expect(triggered).to_be_hidden(timeout=RUN_TIMEOUT_MS)
    automation_page(page).get_by_role("button", name="Run now", exact=True).click()
    expect(triggered).to_be_visible()
    with owner.client() as client:

        def listed() -> int:
            found = client.get(f"/api/v1/automations/{automation_id}/runs")
            assert found.status_code == 200, found.text
            return len(found.json())

        assert wait_until(lambda: listed() >= runs, RUN_TIMEOUT_MS / 1000), (
            f"Run now never recorded run {runs}: the server lists {listed()}"
        )
    page.reload()
    expect(run_entries(page)).to_have_count(runs)


def expect_no_runs(page: Page) -> None:
    """The Activity list has loaded with only the automation's creation in it."""
    expect(automation_page(page).get_by_text("Automation created")).to_be_visible()
    expect(run_entries(page)).to_have_count(0)


def open_page_menu_entry(page: Page, name: str) -> None:
    """Pick `name` (Duplicate, Delete) from the page header's More menu."""
    automation_page(page).get_by_role("button", name="More", exact=True).click()
    menu = page.get_by_role("menu")
    menu.get_by_role("button", name=name, exact=True).or_(
        menu.get_by_role("menuitem", name=name, exact=True)
    ).click()


def terminal_button(scope: Locator, shown: str = "Terminal") -> Locator:
    """The Terminal dropdown, which reads the picked terminal's name once one is picked."""
    return scope.get_by_role("button", name=shown, exact=True).first


def open_terminal_menu(page: Page, scope: Locator, shown: str = "Terminal") -> Locator:
    terminal_button(scope, shown).click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_text("Terminal", exact=True)).to_be_visible()
    return menu


def toggle_terminal(page: Page, menu: Locator, name: str) -> None:
    """Click the terminal in the open dropdown, which closes it: picks the terminal, or takes
    it off when it was the picked one."""
    menu.get_by_role("menuitem", name=name).click()
    expect(page.get_by_role("menu")).to_have_count(0)


def close_terminal_menu(page: Page, scope: Locator, shown: str = "Terminal") -> None:
    terminal_button(scope, shown).click()
    # a dropdown opened while this one still fades out would be a second menu on the page
    expect(page.get_by_role("menu")).to_have_count(0)


def working_directory(menu: Locator) -> Locator:
    """The Working Directory field of the open Terminal dropdown; it has no label of its own."""
    return menu.get_by_placeholder("/home/user/project")
