"""Admin connection settings that were saved wrong: the Add Connection form.

* open-webui/open-webui#30957, fix `301aa8c5b` (PR open-webui/open-webui#30958): after a
  connection was added, the Add Connection form cleared its URL, key and a few fields but kept
  the headers, the enabled switch, the connection and API types, the provider and the API
  version, so the next connection added was silently saved with the first one's.
* open-webui/open-webui#31377, fix `e924fa7b5` (PR open-webui/open-webui#31378): after the
  connection dialog refused headers that are not a JSON object, its Save button stayed disabled
  with a spinner until the page was reloaded, so the corrected headers could not be saved.

Discriminates: passes on the efe63bd34 build (the headers test on dev a5bc78300); with `301aa8c5b`
reverted the add test fails (the second connection carries the first one's headers, provider and
API type), with `e924fa7b5` reverted the headers test fails (Save stays disabled after the error).

The tests of a person's own direct connections (#31383 and the switched-off half of #30957) went
with the feature, which Open WebUI removed in 4720ff19e.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.second_provider import OPENAI_CONFIG
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

# refused at once, so the model lists the app fetches after a save fail fast
FIRST_URL, SECOND_URL = "http://127.0.0.1:9/first", "http://127.0.0.1:9/second"


def add_connection_form(page: Page) -> Locator:
    form = page.get_by_role("dialog").filter(has=page.get_by_role("heading", name="Add Connection"))
    expect(form).to_be_visible()
    return form


def test_the_next_admin_connection_does_not_inherit_the_advanced_settings(
    page_for, make_user, admin, preserve
):
    preserve(OPENAI_CONFIG)
    page = page_for(make_user(role="admin"), path="/admin/settings/connections")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("tab", selected=True)).to_be_visible()

    tooltip_button(settings, "Add Connection").click()
    form = add_connection_form(page)
    form.get_by_role("combobox", name="URL").fill(FIRST_URL)
    form.get_by_role("button", name="API Type").click()
    form.get_by_role("button", name="Advanced").click()
    form.get_by_placeholder("Enter additional headers in JSON format").fill('{"X-Team": "first"}')
    form.get_by_role("combobox", name="Provider").select_option(label="LiteLLM")
    with page.expect_response(lambda response: "/openai/config/update" in response.url):
        form.get_by_role("button", name="Save").click()
    expect(form).to_be_hidden()

    tooltip_button(settings, "Add Connection").click()
    form = add_connection_form(page)
    form.get_by_role("combobox", name="URL").fill(SECOND_URL)
    with page.expect_response(lambda response: "/openai/config/update" in response.url):
        form.get_by_role("button", name="Save").click()

    with admin.client() as client:
        config = client.get(OPENAI_CONFIG[0]).json()
    urls = config["OPENAI_API_BASE_URLS"]
    first = config["OPENAI_API_CONFIGS"][str(urls.index(FIRST_URL))]
    second = config["OPENAI_API_CONFIGS"][str(urls.index(SECOND_URL))]
    assert first.get("headers") == {"X-Team": "first"}
    assert first.get("provider") == "litellm"
    assert first.get("api_type") == "responses"
    carried = {key: second.get(key) for key in ("headers", "provider", "api_type")}
    assert carried == {"headers": None, "provider": None, "api_type": None}, (
        f"the second connection was saved with the first one's settings: {carried}"
    )


# --------------------------------------------------------------------------- invalid headers


def refuse_then_save_headers(page, form, url: str, is_save) -> None:
    """Fill the form with headers that are no JSON object, then fix them and save."""
    form.get_by_role("combobox", name="URL").fill(url)
    form.get_by_role("button", name="Advanced").click()
    headers = form.get_by_placeholder("Enter additional headers in JSON format")
    headers.fill("[1, 2]")
    save = form.get_by_role("button", name="Save")
    save.click()
    expect(page.get_by_text("Headers must be a valid JSON object")).to_be_visible()
    expect(save, "#31377: Save stayed disabled after the headers error").to_be_enabled()
    headers.fill('{"X-Team": "crew"}')
    with page.expect_response(is_save):
        save.click()
    expect(form).to_be_hidden()


def test_an_admin_connection_saves_once_its_headers_are_fixed(page_for, make_user, admin, preserve):
    preserve(OPENAI_CONFIG)
    page = page_for(make_user(role="admin"), path="/admin/settings/connections")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("tab", selected=True)).to_be_visible()

    tooltip_button(settings, "Add Connection").click()
    refuse_then_save_headers(
        page,
        add_connection_form(page),
        FIRST_URL,
        lambda response: "/openai/config/update" in response.url,
    )

    with admin.client() as client:
        config = client.get(OPENAI_CONFIG[0]).json()
    index = config["OPENAI_API_BASE_URLS"].index(FIRST_URL)
    assert config["OPENAI_API_CONFIGS"][str(index)]["headers"] == {"X-Team": "crew"}
