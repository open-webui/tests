"""Regression: invalid external loader headers silently kept the Documents settings from saving.

Fix f8e15f92d (open-webui/open-webui#30434, issue open-webui/open-webui#30294). The Documents
tab only checked the external document loader's headers while the External engine was selected.
With invalid headers left in the field and another engine picked, Save skipped the check, the
headers then failed to parse while the settings were sent, and nothing was saved and nothing was
said. The headers are now checked whenever they are set, so Save names the problem.

Discriminates: passes on dev efe63bd34, fails on a frontend build with f8e15f92d reverted (Save
with the Default engine shows no error and saves nothing).
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.web_retrieval import RETRIEVAL_CONFIG
from utils.admin_settings import submit

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

HEADERS_ERROR = "Headers must be a valid JSON object"
SAVED = "Settings saved successfully!"


@pytest.fixture
def documents(make_user, page_for, preserve) -> tuple[Page, Locator]:
    """A fresh admin on the Documents tab, the retrieval settings restored afterwards."""
    preserve(RETRIEVAL_CONFIG)
    page = page_for(make_user(role="admin"), path="/admin/settings/documents")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("tab", selected=True)).to_be_visible()
    return page, settings


def engine(settings: Locator) -> Locator:
    return settings.get_by_role("combobox").filter(
        has=settings.page.get_by_role("option", name="External", exact=True)
    )


def enter_external_loader(settings: Locator, headers: str) -> None:
    engine(settings).select_option("external")
    settings.get_by_placeholder("Enter External Document Loader URL").fill("http://127.0.0.1:9")
    settings.get_by_placeholder("Enter additional headers in JSON format").fill(headers)


def test_invalid_headers_are_reported_when_another_engine_is_selected(documents):
    page, settings = documents
    enter_external_loader(settings, "{not json")
    engine(settings).select_option("")

    submit(settings)

    expect(page.get_by_text(HEADERS_ERROR).first).to_be_visible()
    expect(page.get_by_text(SAVED)).to_have_count(0)


def test_headers_that_are_not_an_object_are_reported_for_any_engine(documents):
    page, settings = documents
    enter_external_loader(settings, '["X-Team", "harbour"]')
    engine(settings).select_option("tika")
    settings.get_by_placeholder("Enter Tika Server URL").fill("http://127.0.0.1:9")

    submit(settings)

    expect(page.get_by_text(HEADERS_ERROR).first).to_be_visible()


def test_invalid_headers_are_reported_with_the_external_engine(documents):
    page, settings = documents
    enter_external_loader(settings, "{not json")

    submit(settings)

    expect(page.get_by_text(HEADERS_ERROR).first).to_be_visible()
    expect(page.get_by_text(SAVED)).to_have_count(0)


def test_valid_headers_save_with_another_engine_selected(documents):
    page, settings = documents
    enter_external_loader(settings, '{"X-Team": "harbour"}')
    engine(settings).select_option("")

    submit(settings)

    expect(page.get_by_text(SAVED).first).to_be_visible()
    expect(page.get_by_text(HEADERS_ERROR)).to_have_count(0)
