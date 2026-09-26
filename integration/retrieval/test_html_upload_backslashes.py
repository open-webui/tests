"""Regression: backslashes in an uploaded HTML file were read as escape codes.

Issue open-webui/open-webui#31440. With the built-in loaders, `.html` and `.htm` files are opened
with the `unicode_escape` codec, so `C:\\new\\table` is stored with a line break and a tab, and a
page holding `C:\\Users\\...` cannot be processed at all. Plain text uploads keep backslashes.

Discriminates: fails on dev ac00d40e3 (the stored text holds a line break and a tab, and the
second page ends `failed`), passes with the HTML branch reading the detected text encoding.
"""

from __future__ import annotations

import httpx
import pytest

from harness.web_retrieval import RETRIEVAL_CONFIG

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


@pytest.fixture
def built_in_loaders(preserve, admin):
    preserve(RETRIEVAL_CONFIG)
    with admin.client() as client:
        saved = client.post(RETRIEVAL_CONFIG[1], json={"CONTENT_EXTRACTION_ENGINE": ""})
        assert saved.status_code == 200, saved.text
    yield


def _stored(client: httpx.Client, filename: str, text: str, content_type: str) -> dict:
    uploaded = client.post(
        "/api/v1/files/",
        params={"process": "true", "process_in_background": "false"},
        files={"file": (filename, text.encode(), content_type)},
    )
    assert uploaded.status_code == 200, uploaded.text
    return client.get(f"/api/v1/files/{uploaded.json()['id']}").json()["data"]


def test_a_windows_path_in_html_keeps_its_backslashes(built_in_loaders, make_user):
    with make_user().client() as client:
        data = _stored(client, "paths.html", "<p>Path C:\\new\\table</p>", "text/html")

    assert data.get("status") == "completed", data.get("error")
    assert data["content"].strip() == "Path C:\\new\\table", ascii(data["content"])


def test_an_html_page_with_a_users_path_is_processed(built_in_loaders, make_user):
    with make_user().client() as client:
        data = _stored(client, "home.html", "<p>C:\\Users\\xavier</p>", "text/html")

    assert data.get("status") == "completed", data.get("error")
    assert data["content"].strip() == "C:\\Users\\xavier"


def test_a_text_file_keeps_its_backslashes(built_in_loaders, make_user):
    with make_user().client() as client:
        data = _stored(client, "paths.txt", "Path C:\\new\\table", "text/plain")

    assert data["content"].strip() == "Path C:\\new\\table"


def test_non_ascii_html_is_read_as_written(built_in_loaders, make_user):
    with make_user().client() as client:
        data = _stored(client, "cafe.html", "<p>café 日本</p>", "text/html")

    assert data["content"].strip() == "café 日本"
