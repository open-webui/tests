"""Journey: an admin reviews the changes on an Admin Settings page before they are saved.

Since 0.13.4 (f2a30994c, #34534) a settings page shows next to Save how many settings differ from
what was loaded or last saved, and clicking that count or Save lists every changed setting with its
old and new value to confirm, with passwords and API keys kept hidden. Cancelling saves nothing and
keeps the edits on the page; confirming saves exactly the listed changes and the count goes away.
Web Search walks it with Save, a number, a list and an API key; General walks it with the count,
a switch turned back and a text field.

Discriminates: passes on dev f80a4f782 (5 of 5); in a frontend build of it whose Web Search page
marks no field as sensitive the Web Search test fails (both keys show in plain text), and in one
whose General page keeps the loaded values as the baseline after a save the General test fails
(the count stays after saving).
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.web_retrieval import RETRIEVAL_CONFIG, save_web_settings
from utils.admin_settings import SAVED, review_dialog

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

ADMIN_CONFIG = "/api/v1/auths/admin/config"
DOMAIN_FILTER = "Enter domains separated by commas (e.g., example.com,site.org,!excludedsite.com)"
WATERMARK = "Enter a watermark for the response. Leave empty for none."
OLD_KEY = "brave-old-key-4c1d7a"
NEW_KEY = "brave-new-key-9e2b51"


def change_count(settings: Locator) -> Locator:
    return settings.get_by_role("button", name=re.compile(r"^\d+ Changes?$"))


def change_row(dialog: Locator, label: str) -> Locator:
    """The smallest block of the review list whose name line starts with `label`."""
    names_it = f"div[starts-with(normalize-space(.), '{label}')]"
    return dialog.locator(f"xpath=.//div[{names_it}]").last


def expect_change(dialog: Locator, label: str, old: str, new: str) -> None:
    row = change_row(dialog, label)
    expect(row, f"the review lists no change of {label}").to_be_visible()
    text = row.inner_text()
    assert old in text and new in text, (
        f"{label} should read {old!r} -> {new!r}, the list shows {text!r}"
    )
    assert text.index(old) < text.index(new), f"{label} shows the new value first: {text!r}"


def stored_web(admin) -> dict:
    with admin.client() as client:
        current = client.get(RETRIEVAL_CONFIG[0])
    assert current.status_code == 200, current.text
    return current.json()["web"]


def stored_admin_config(admin) -> dict:
    with admin.client() as client:
        current = client.get(ADMIN_CONFIG)
    assert current.status_code == 200, current.text
    return current.json()


def confirm(page: Page, settings: Locator) -> None:
    dialog = review_dialog(page)
    with page.expect_response(lambda response: response.request.method == "POST") as saved:
        dialog.get_by_role("button", name="Save", exact=True).click()
    assert saved.value.ok, f"saving the reviewed changes failed: {saved.value.status}"
    expect(dialog).to_be_hidden()
    expect(page.get_by_text(SAVED).first).to_be_visible()
    expect(change_count(settings), "the change count stays after saving").to_have_count(0)


def cancel(page: Page) -> None:
    dialog = review_dialog(page)
    dialog.get_by_role("button", name="Cancel", exact=True).click()
    expect(dialog).to_be_hidden()


def test_web_search_lists_each_change_hides_the_key_and_saves_only_on_confirm(
    page_for, admin, make_user, preserve
):
    preserve(RETRIEVAL_CONFIG)
    with admin.client() as client:
        save_web_settings(
            client,
            ENABLE_WEB_SEARCH=True,
            WEB_SEARCH_ENGINE="brave",
            BRAVE_SEARCH_API_KEY=OLD_KEY,
            WEB_SEARCH_RESULT_COUNT=3,
            WEB_SEARCH_DOMAIN_FILTER_LIST=[],
        )
    before = stored_web(admin)
    page = page_for(make_user(role="admin"), path="/admin/settings/web")
    settings = page.get_by_role("dialog")
    result_count = settings.get_by_placeholder("Search Result Count")
    expect(result_count).to_have_value("3")
    expect(change_count(settings), "a count shows before anything was edited").to_have_count(0)

    result_count.fill("5")
    expect(change_count(settings)).to_have_text("1 Change")
    settings.get_by_placeholder(DOMAIN_FILTER).fill("harbour.example")
    expect(change_count(settings)).to_have_text("2 Changes")
    settings.get_by_placeholder("Enter Brave Search API Key").fill(NEW_KEY)
    expect(change_count(settings)).to_have_text("3 Changes")

    settings.get_by_role("button", name="Save", exact=True).click()
    dialog = review_dialog(page)
    expect(dialog).to_be_visible()
    expect_change(dialog, "Search Result Count", "3", "5")
    expect_change(dialog, "Domain Filter List", "Empty", "harbour.example")
    key_row = change_row(dialog, "Brave Search API Key")
    expect(key_row, "the review leaves out the changed API key").to_be_visible()
    shown = dialog.inner_text()
    assert OLD_KEY not in shown and NEW_KEY not in shown, f"the review shows the API key: {shown!r}"
    for key in (OLD_KEY, NEW_KEY):
        holders = [field for field in key_row.locator("input").all() if field.input_value() == key]
        assert holders, f"the review does not show that the key changed to or from {key!r}"
        for field in holders:
            masking = field.evaluate("input => getComputedStyle(input).webkitTextSecurity")
            assert masking == "disc", f"the API key shows in the review in plain text ({masking})"

    cancel(page)
    expect(change_count(settings), "cancelling dropped the edits").to_have_text("3 Changes")
    expect(result_count).to_have_value("5")
    assert stored_web(admin) == before, "cancelling the review saved the changes anyway"

    settings.get_by_role("button", name="Save", exact=True).click()
    confirm(page, settings)
    after = stored_web(admin)
    expected = {
        **before,
        "WEB_SEARCH_RESULT_COUNT": 5,
        "WEB_SEARCH_DOMAIN_FILTER_LIST": ["harbour.example"],
        "BRAVE_SEARCH_API_KEY": NEW_KEY,
    }
    differing = {key: after.get(key) for key in expected if after.get(key) != expected[key]}
    assert not differing, f"saving stored other values than the reviewed ones: {differing}"


def test_general_counts_a_reverted_switch_out_and_saves_only_the_listed_change(
    page_for, admin, make_user, preserve
):
    preserve("admin_config")
    with admin.client() as client:
        current = client.get(ADMIN_CONFIG).json()
        reset = client.post(
            ADMIN_CONFIG, json={**current, "ENABLE_MESSAGE_RATING": True, "RESPONSE_WATERMARK": ""}
        )
    assert reset.status_code == 200, reset.text
    before = stored_admin_config(admin)
    page = page_for(make_user(role="admin"), path="/admin/settings/general")
    settings = page.get_by_role("dialog")
    rating = settings.get_by_role("switch", name="Message Rating", exact=True)
    expect(rating).to_be_checked()
    expect(change_count(settings), "a count shows before anything was edited").to_have_count(0)

    rating.click()
    expect(change_count(settings)).to_have_text("1 Change")
    watermark = "Checked by the harbour office."
    settings.get_by_placeholder(WATERMARK).fill(watermark)
    expect(change_count(settings)).to_have_text("2 Changes")
    rating.click()
    expect(rating).to_be_checked()
    expect(change_count(settings), "a switch turned back still counts").to_have_text("1 Change")

    change_count(settings).click()
    dialog = review_dialog(page)
    expect(dialog).to_be_visible()
    expect_change(dialog, "Response Watermark", "Empty", watermark)
    expect(dialog.get_by_text("Message Rating")).to_have_count(0)

    cancel(page)
    expect(settings.get_by_placeholder(WATERMARK)).to_have_value(watermark)
    assert stored_admin_config(admin) == before, "cancelling the review saved the change anyway"

    change_count(settings).click()
    expect(review_dialog(page)).to_be_visible()
    confirm(page, settings)
    after = stored_admin_config(admin)
    differing = {
        key: after.get(key)
        for key in {*before, *after}
        if key != "RESPONSE_WATERMARK" and after.get(key) != before.get(key)
    }
    assert after["RESPONSE_WATERMARK"] == watermark, after["RESPONSE_WATERMARK"]
    assert not differing, f"saving changed settings that were not in the review: {differing}"
