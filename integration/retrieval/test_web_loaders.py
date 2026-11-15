"""Regressions in attaching a link and searching the web, fixed in v0.11.1.

- Attached page content (issue #28378, commit 9c21d4ed3b): `process_web` returned the text only
  under `file.data.content` while the caller reads the top-level `content`, so the model got an
  empty attachment.
- Unreadable links blamed on the knowledge base (issue #28361, PR #28362, commit 121f2404e):
  fetching and saving sat in one `try` whose handler said "Error querying knowledge base". A link
  that cannot be read now names the link. A closed port still got the old message (#31347, PR
  #31351): the default loader swallows the connection error and the empty page fails at the
  vector save. The chat's own attach route answered a generic error for it (PR #31354). Both
  tests pass on dev efe63bd34 and fail with 420b4a279 and 91fb33ef5 reverted.
- Microsoft Web IQ loader (issue #28688, commits 6dcc2d5269 + 140d2cf4b5): its constructor did
  not take the `api_base_url` that `get_web_loader` always passes, so Web IQ never loaded a page.
- Content-type sniffing (commit 886248de36): any type merely containing `xml` counted as text,
  so Office archives (`...openxmlformats...`) went through the HTML loader and arrived as
  mojibake instead of being extracted.
- Web search errors (PR #28942, commit fca3be541; PR #28948): the handler put the exception
  object itself into `HTTPException.detail`, which does not serialise, so a failed search was a
  500 with an empty body.

- Tavily web loader (PR #27636, commit 104a0f2f1): `SafeTavilyLoader` required an
  `api_base_url` that `get_web_loader` never passed, so choosing Tavily raised before any
  request. Tavily's API base URL is environment-only, so the instance boots pointed at a local
  stand-in. The external loader and an unknown engine are read the same way.

- Internal certificate authority (issue #34479): with Verify SSL Certificate on, the certificate
  check the Tavily, Playwright and Web IQ loaders run before reading a page trusted only the
  public authorities, so a page signed by the CA named in `AIOHTTP_CLIENT_SSL_CERT_FILE` was
  refused. The page's port is honoured since #34476, which this test needs to reach it.

- urllib3-future socket options (PR #26796, commit 7ef0530b2, issue #26791): the connect-time
  SSRF guard passed every socket option straight to `setsockopt()`, which takes three arguments.
  urllib3-future, which shadows urllib3 once a tool or function installs it, declares its
  defaults in a four-element per-protocol form, so every synchronous fetch failed on connect and
  a linked document was no longer recognised as one. A function installed by the admin sets
  those defaults here, the way installing urllib3-future does.

Twin of unit/retrieval/test_web_loaders.py and of the socket option part of
unit/retrieval/test_web_loader_configuration.py; the YouTube transcript errors are in
integration/retrieval/test_youtube_transcripts.py.

Discriminates: passes on dev bbfa876af; dropping the top-level `content` fails the attached page,
one `try` for fetch and save fails the unreadable link, a Web IQ constructor without
`api_base_url` fails the Web IQ page, substring sniffing fails every Office type and passing the
exception as the detail fails the failed search. On dev ef67cc3fa, a required `api_base_url`
on the Tavily loader fails the Tavily page, and passing every socket option to `setsockopt` whole
fails the document link under urllib3-future's defaults. The internal CA case fails on dev
8b11a2018 and on #34476 alone (the page is refused), and passes with the #34479 fix on top.
"""

from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

import pytest

from harness.actors import admin_of, create_user
from harness.instance import free_port
from harness.listener import json_answer, listening, text_answer
from harness.object_storage import self_signed_certificate
from harness.plugins import installed_function
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
def tavily_api():
    """A stand-in for Tavily's extract endpoint, named at boot (its base URL is env-only)."""
    with listening() as service:
        yield service


@pytest.fixture(scope="module")
def fetching_instance(instance_with, tavily_api):
    return instance_with({**LOCAL_WEB_FETCH, "TAVILY_API_BASE_URL": tavily_api.base_url})


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


def test_a_link_on_a_closed_port_is_named_in_the_error(web_admin):
    link = f"http://127.0.0.1:{free_port()}/gone"

    refused = attach(web_admin, link)

    assert refused.status_code == 400
    assert link in refused.json()["detail"], refused.json()["detail"]


def test_the_chat_attach_route_names_a_link_on_a_closed_port(web_admin):
    link = f"http://127.0.0.1:{free_port()}/gone"

    refused = web_admin.post("/api/v1/retrieval/process/url", json={"url": link})

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


def test_a_page_is_read_through_tavily(web_admin, tavily_api, page):
    extracted = {"results": [{"url": page, "raw_content": "read by tavily"}]}
    tavily_api.route("POST", "/extract", json_answer(extracted))
    save_web_settings(
        web_admin,
        WEB_LOADER_ENGINE="tavily",
        TAVILY_API_KEY="tvly-key",
        TAVILY_EXTRACT_DEPTH="advanced",
    )

    previewed = attach(web_admin, page, preview=True)

    assert previewed.status_code == 200, previewed.text
    assert previewed.json()["content"] == "read by tavily", "the Tavily loader was never built"
    extract = tavily_api.requests_to("/extract")[-1]
    assert (extract.json()["urls"], extract.json()["extract_depth"]) == (page, "advanced")
    assert extract.headers["Authorization"] == "Bearer tvly-key"


@pytest.fixture(scope="module")
def internal_ca():
    """(certificate, key) of an authority of its own, which signs the pages it serves."""
    with tempfile.TemporaryDirectory(prefix="owui-internal-ca-") as directory:
        yield self_signed_certificate(Path(directory))


@pytest.mark.slow
def test_tavily_reads_a_page_signed_by_the_configured_internal_ca(
    instance_with, tavily_api, internal_ca
):
    trusting = instance_with(
        {
            **LOCAL_WEB_FETCH,
            "TAVILY_API_BASE_URL": tavily_api.base_url,
            "AIOHTTP_CLIENT_SSL_CERT_FILE": str(internal_ca[0]),
        }
    )
    with (
        trusting.client() as client,
        web_settings_restored(client),
        listening(certificate=internal_ca) as internal,
        listening(tls=True) as untrusted,
    ):
        internal_page, untrusted_page = (
            f"{internal.base_url}/herons",
            f"{untrusted.base_url}/herons",
        )
        extracted = {"results": [{"url": internal_page, "raw_content": "read by tavily"}]}
        tavily_api.route("POST", "/extract", json_answer(extracted))
        save_web_settings(
            client,
            WEB_LOADER_ENGINE="tavily",
            TAVILY_API_KEY="tvly-key",
            ENABLE_WEB_LOADER_SSL_VERIFICATION=True,
        )

        asked_before = len(tavily_api.requests_to("/extract"))
        previewed = attach(client, internal_page, preview=True)
        refused = attach(client, untrusted_page, preview=True)

    assert previewed.status_code == 200, (
        f"the internal CA was not trusted (#34479): {previewed.text}"
    )
    assert previewed.json()["content"] == "read by tavily"
    assert refused.status_code == 400, "a page signed by an unknown authority was read"
    asked = tavily_api.requests_to("/extract")[asked_before:]
    assert [request.json()["urls"] for request in asked] == [internal_page]


def test_a_page_is_read_through_the_external_loader(web_admin, listener, page):
    listener.route(
        "POST", "/loader", json_answer([{"page_content": "read by the external loader"}])
    )
    save_web_settings(
        web_admin,
        WEB_LOADER_ENGINE="external",
        EXTERNAL_WEB_LOADER_URL=f"{listener.base_url}/loader",
        EXTERNAL_WEB_LOADER_API_KEY="loader-key",
    )

    previewed = attach(web_admin, page, preview=True)

    assert previewed.status_code == 200, previewed.text
    assert previewed.json()["content"] == "read by the external loader"
    assert listener.requests_to("/loader")[0].json()["urls"] == [page]


def test_an_unknown_loader_engine_is_refused(web_admin, page):
    save_web_settings(web_admin, WEB_LOADER_ENGINE="not-an-engine")

    refused = attach(web_admin, page, preview=True)

    assert refused.status_code == 400, refused.text
    assert page in refused.json()["detail"]


@pytest.mark.parametrize("content_type", OFFICE_TYPES)
def test_an_office_document_link_is_extracted_not_read_as_html(web_admin, listener, content_type):
    listener.route("GET", "/report.docx", (200, {"Content-Type": content_type}, word_document()))

    previewed = attach(web_admin, f"{listener.base_url}/report.docx", preview=True)

    assert previewed.status_code == 200, previewed.text
    assert WORD_TEXT in previewed.json()["content"], (
        f"a {content_type} link went through the HTML loader instead of the extractor"
    )


URLLIB3_FUTURE_DEFAULTS = """
import socket
from urllib3.connection import HTTPConnection

# urllib3-future's per-protocol defaults, in place so every new connection takes them
HTTPConnection.default_socket_options[:] = [
    (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1, "tcp"),
    (socket.IPPROTO_UDP, 1, 1, "udp"),
]


class Filter:
    pass
"""
STOCK_URLLIB3_DEFAULTS = """
import socket
from urllib3.connection import HTTPConnection

HTTPConnection.default_socket_options[:] = [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]


class Filter:
    pass
"""


@pytest.fixture
def urllib3_future_defaults(fetching_instance):
    """The instance's urllib3 set to urllib3-future's socket options, put back afterwards."""
    owner = admin_of(fetching_instance)
    with installed_function(owner, URLLIB3_FUTURE_DEFAULTS, active=False):
        pass
    try:
        yield
    finally:
        with installed_function(owner, STOCK_URLLIB3_DEFAULTS, active=False):
            pass


def test_a_document_link_is_extracted_under_urllib3_futures_socket_options(
    web_admin, listener, urllib3_future_defaults
):
    listener.route("GET", "/report.docx", (200, {"Content-Type": OFFICE_TYPES[0]}, word_document()))

    previewed = attach(web_admin, f"{listener.base_url}/report.docx", preview=True)

    assert previewed.status_code == 200, previewed.text
    assert WORD_TEXT in previewed.json()["content"], (
        "the document was not recognised: the fetch that reads its type failed on connect (#26791)"
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
