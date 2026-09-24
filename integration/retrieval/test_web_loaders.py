"""Regressions in attaching a link and searching the web, fixed in v0.11.1.

- Attached page content (issue #28378, commit 9c21d4ed3b): `process_web` returned the text only
  under `file.data.content` while the caller reads the top-level `content`, so the model got an
  empty attachment.
- Unreadable links blamed on the knowledge base (issue #28361, PR #28362, commit 121f2404e):
  fetching and saving sat in one `try` whose handler said "Error querying knowledge base". A link
  that cannot be read now names the link. A closed port still gets the old message: the default
  loader swallows the connection error and the empty page fails at the vector save (strict
  `xfail` below).
- Microsoft Web IQ loader (issue #28688, commits 6dcc2d5269 + 140d2cf4b5): its constructor did
  not take the `api_base_url` that `get_web_loader` always passes, so Web IQ never loaded a page.
- Content-type sniffing (commit 886248de36): any type merely containing `xml` counted as text,
  so Office archives (`...openxmlformats...`) went through the HTML loader and arrived as
  mojibake instead of being extracted.
- Web search errors (PR #28942, commit fca3be541; PR #28948): the handler put the exception
  object itself into `HTTPException.detail`, which does not serialise, so a failed search was a
  500 with an empty body.

Twin of unit/retrieval/test_web_loaders.py, which keeps the Tavily loader construction (its API
base URL is environment-only, so no local stand-in can be named at boot), the construction of
every engine and the YouTube transcript errors (the transcript API talks to YouTube).

Discriminates: passes on dev bbfa876af; dropping the top-level `content` fails the attached page,
one `try` for fetch and save fails the unreadable link, a Web IQ constructor without
`api_base_url` fails the Web IQ page, substring sniffing fails every Office type and passing the
exception as the detail fails the failed search.
"""

from __future__ import annotations

import io
import zipfile

import pytest

from harness.actors import create_user
from harness.instance import free_port
from harness.listener import json_answer, text_answer
from harness.web_retrieval import LOCAL_WEB_FETCH, save_web_settings, web_settings_restored

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

PAGE_TEXT = "Herons nest in colonies called heronries"
WORD_TEXT = "Quarterly heron census"
KNOWLEDGE_BASE_ERROR = "knowledge base"
DOCX_PARTS = {
    "[Content_Types].xml": (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/'
        'vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>'
    ),
    "_rels/.rels": (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Target="word/document.xml" Type="http://schemas.'
        'openxmlformats.org/officeDocument/2006/relationships/officeDocument"/></Relationships>'
    ),
    "word/document.xml": (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body><w:p><w:r><w:t>{WORD_TEXT}</w:t></w:r></w:p></w:body></w:document>"
    ),
}
OFFICE_TYPES = [
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document; charset=binary",
]


def word_document() -> bytes:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
        for name, part in DOCX_PARTS.items():
            package.writestr(name, part)
    return archive.getvalue()


@pytest.fixture(scope="module")
def fetching_instance(instance_with):
    return instance_with(LOCAL_WEB_FETCH)


@pytest.fixture
def web_admin(fetching_instance):
    with fetching_instance.client() as client, web_settings_restored(client):
        yield client


@pytest.fixture
def page(listener) -> str:
    listener.route("GET", "/herons", text_answer(f"<html><body><p>{PAGE_TEXT}</p></body></html>"))
    return f"{listener.base_url}/herons"


def attach(client, url: str, **options):
    process = "false" if options.pop("preview", False) else "true"
    return client.post(
        f"/api/v1/retrieval/process/web?process={process}", json={"url": url, **options}
    )


def test_an_attached_page_carries_its_text_at_the_top_level(web_admin, page):
    attached = attach(web_admin, page)

    assert attached.status_code == 200, attached.text
    body = attached.json()
    assert body.get("content") == PAGE_TEXT, "the attachment reached the model empty (#28378)"
    assert body["file"]["data"]["content"] == PAGE_TEXT
    assert (body["filename"], body["file"]["meta"]["source"]) == (page, page)


def test_a_previewed_page_carries_its_text(web_admin, page):
    previewed = attach(web_admin, page, preview=True)

    assert previewed.json() == {"status": True, "content": PAGE_TEXT}


def test_a_link_that_cannot_be_read_is_named_in_the_error(web_admin):
    link = "http://169.254.169.254/latest/meta-data"  # on the default block list, never fetched

    refused = attach(web_admin, link)

    assert refused.status_code == 400
    assert link in refused.json()["detail"]
    assert KNOWLEDGE_BASE_ERROR not in refused.json()["detail"].lower()


@pytest.mark.xfail(raises=AssertionError, strict=True, reason="a closed port is a save error")
def test_a_link_on_a_closed_port_is_named_in_the_error(web_admin):
    link = f"http://127.0.0.1:{free_port()}/gone"

    refused = attach(web_admin, link)

    assert refused.status_code == 400
    assert link in refused.json()["detail"], refused.json()["detail"]


def test_a_refused_collection_keeps_its_own_status(fetching_instance, page):
    with create_user(fetching_instance).client() as client:
        refused = attach(client, page, collection_name="knowledge-bases")

    assert refused.status_code == 403, refused.text


def test_a_page_is_read_through_microsoft_web_iq(web_admin, listener, page):
    listener.route("POST", "/v3/browse", json_answer({"content": "read by web iq", "url": page}))
    save_web_settings(
        web_admin,
        WEB_LOADER_ENGINE="microsoft_web_iq",
        MICROSOFT_WEB_IQ_API_BASE_URL=f"{listener.base_url}/v3",
        MICROSOFT_WEB_IQ_API_KEY="web-iq-key",
    )

    previewed = attach(web_admin, page, preview=True)

    assert previewed.status_code == 200, previewed.text
    assert previewed.json()["content"] == "read by web iq", "the Web IQ loader was never built"
    assert listener.requests_to("/v3/browse")[0].headers["x-apikey"] == "web-iq-key"


@pytest.mark.parametrize("content_type", OFFICE_TYPES)
def test_an_office_document_link_is_extracted_not_read_as_html(web_admin, listener, content_type):
    listener.route("GET", "/report.docx", (200, {"Content-Type": content_type}, word_document()))

    previewed = attach(web_admin, f"{listener.base_url}/report.docx", preview=True)

    assert previewed.status_code == 200, previewed.text
    assert WORD_TEXT in previewed.json()["content"], (
        f"a {content_type} link went through the HTML loader instead of the extractor"
    )


@pytest.mark.parametrize("content_type", ["text/plain; charset=utf-8", "application/json"])
def test_a_text_link_still_goes_to_the_web_loader(web_admin, listener, content_type):
    listener.route("GET", "/notes", text_answer(f'"{PAGE_TEXT}"', content_type=content_type))

    previewed = attach(web_admin, f"{listener.base_url}/notes", preview=True)

    assert PAGE_TEXT in previewed.json()["content"]


def search(client):
    return client.post("/api/v1/retrieval/process/web/search", json={"queries": ["herons"]})


def test_a_failed_web_search_answers_with_a_readable_message(web_admin):
    save_web_settings(web_admin, ENABLE_WEB_SEARCH=True, WEB_SEARCH_ENGINE="")

    failed = search(web_admin)

    assert failed.status_code == 400, f"HTTP {failed.status_code}: {failed.text!r}"
    assert "searching the web" in failed.json()["detail"]


def test_a_disabled_web_search_is_refused(web_admin):
    save_web_settings(web_admin, ENABLE_WEB_SEARCH=False)

    assert search(web_admin).status_code == 403
