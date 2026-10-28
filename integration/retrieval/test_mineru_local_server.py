"""MinerU in local mode, against a 4.x server and a 3.x one.

MinerU 4.x support (#32422, 81f5b8adf, v0.12.2): a server that answers `/v1/health` is taken for
4.x, and a document goes through its V1 API (upload, parse job, poll, download) in place of the 3.x
`/file_parse` form. The job gets only the fields the V1 API knows (the tier and an OCR mode made
from `enable_ocr`), a page range only for a PDF, and the pictures the V1 API embeds in the text
are left out of what is stored. MinerU is a local service here that answers the way each version
does.

Regression: MinerU in local mode failed every document on a 4.x server started with `--api-key`.
Issue open-webui/open-webui#32836: local mode sent none of its V1 requests with the API Key from
the Documents settings, so a key-protected server answered 401 to the first upload and nothing
was parsed. The key now goes along as a Bearer token whenever one is set (3e0564d11). Like the real
server, the stand-in leaves `/v1/health` open and refuses every other V1 call without the key.

Discriminates: the key test passes with fix fed382957 on dev ca8f451cc and fails on dev ca8f451cc
(MinerU answers 401 to the upload and the file has no text); the test without a key passes on
both. The journey tests pass on dev 7b4d4b51f and each fails under one mutation of the loader:
passing every MinerU parameter into the job (the job fields test), sending the page range for any
file (the page range test), reading the job's first answer without polling (the polling test),
taking a failed job for a completed one (the failed job test), storing the text as MinerU sent it
(the picture test) and taking every server for 4.x (the 3.x test).
"""

from __future__ import annotations

import uuid
from typing import Iterator

import pytest

from harness.listener import ReceivedRequest, json_answer, text_answer
from harness.web_retrieval import RETRIEVAL_CONFIG

pytestmark = [pytest.mark.api, pytest.mark.requires_source]

PARSED = "parsed by mineru"
JOB_ID = "job-1"
UPLOADED_FILE = "file-1"
MARKDOWN_FILE = "markdown-1"
PDF = ("pdf", b"%PDF-1.4 report", "application/pdf")
DOCX = (
    "docx",
    b"PK\x03\x04 report",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
)


def job_answer(status: str, error: str = "") -> dict:
    file_result = (
        {"error": {"message": error}}
        if error
        else {"output_files": {"markdown": {"file_id": MARKDOWN_FILE}}, "parse": {}}
    )
    return {"job_id": JOB_ID, "status": status, "tier": "flash", "files": [file_result]}


def mineru_v1_server(
    listener,
    api_key: str = "",
    statuses: tuple[str, ...] = ("completed",),
    markdown: str = PARSED,
    error: str = "",
) -> None:
    """Route `listener` like a MinerU 4.x server, requiring `api_key` when one is given.

    The parse job answers the `statuses` in turn, one per call, and then stays at the last one.
    """
    job_states: Iterator[str] = iter(statuses)
    last_status = statuses[-1]

    def next_job(_request: ReceivedRequest):
        return json_answer(job_answer(next(job_states, last_status), error))

    answers = [
        ("POST", "/v1/uploads", lambda _request: json_answer({"id": "upload-1"})),
        ("PUT", "/v1/uploads/upload-1/content", lambda _request: json_answer({})),
        (
            "POST",
            "/v1/uploads/upload-1/complete",
            lambda _request: json_answer({"file": {"id": UPLOADED_FILE}}),
        ),
        ("POST", "/v1/parse/jobs", next_job),
        ("GET", f"/v1/parse/jobs/{JOB_ID}", next_job),
        (
            "GET",
            f"/v1/files/{MARKDOWN_FILE}/content",
            lambda _request: text_answer(markdown, "text/markdown"),
        ),
    ]
    listener.route("GET", "/v1/health", json_answer({"status": "ok"}))
    for method, path, answer in answers:

        def handler(request: ReceivedRequest, answer=answer):
            if api_key and request.headers.get("Authorization") != f"Bearer {api_key}":
                return json_answer({"error": {"code": "invalid_api_key"}}, status=401)
            return answer(request)

        listener.route(method, path, handler)


@pytest.fixture
def use_mineru(admin, preserve, listener):
    """Returns `use(api_key, params, extensions)`: MinerU in local mode on `listener`."""
    preserve(RETRIEVAL_CONFIG)
    client = admin.client()

    def use(api_key: str = "", params: dict | None = None, extensions: tuple = ("pdf",)):
        saved = client.post(
            RETRIEVAL_CONFIG[1],
            json={
                "CONTENT_EXTRACTION_ENGINE": "mineru",
                "MINERU_API_MODE": "local",
                "MINERU_API_URL": listener.base_url,
                "MINERU_API_KEY": api_key,
                "MINERU_PARAMS": {"tier": "flash"} if params is None else params,
                "MINERU_FILE_EXTENSIONS": list(extensions),
                "BYPASS_EMBEDDING_AND_RETRIEVAL": True,
            },
        )
        assert saved.status_code == 200, saved.text
        return client

    yield use
    client.close()


def upload(client, document: tuple = PDF) -> dict:
    """Upload and process in the foreground; returns the stored file."""
    extension, content, content_type = document
    uploaded = client.post(
        "/api/v1/files/",
        params={"process": "true", "process_in_background": "false"},
        files={"file": (f"report-{uuid.uuid4().hex[:6]}.{extension}", content, content_type)},
    )
    assert uploaded.status_code == 200, uploaded.text
    return client.get(f"/api/v1/files/{uploaded.json()['id']}").json()


def job_request(listener) -> dict:
    [job] = listener.requests_to("/v1/parse/jobs")
    return job.json()


@pytest.mark.regression
def test_a_key_protected_mineru_server_parses_the_document(use_mineru, listener):
    api_key = f"mineru-{uuid.uuid4().hex[:8]}"
    mineru_v1_server(listener, api_key)

    stored = upload(use_mineru(api_key))

    rejected = [
        request.path
        for request in listener.received
        if request.path != "/v1/health"
        and request.headers.get("Authorization") != f"Bearer {api_key}"
    ]
    assert stored["data"].get("content") == PARSED, (
        f"the document was not parsed; MinerU got calls without the key (#32836): {rejected}"
    )
    assert not rejected


@pytest.mark.regression
def test_without_a_key_mineru_is_called_without_authorization(use_mineru, listener):
    mineru_v1_server(listener, api_key="")

    stored = upload(use_mineru(""))

    assert stored["data"].get("content") == PARSED, stored["data"]
    v1_calls = [request for request in listener.received if request.path != "/v1/health"]
    assert v1_calls, "MinerU was never called"
    assert all("Authorization" not in request.headers for request in v1_calls), [
        request.headers for request in v1_calls
    ]


@pytest.mark.journey
@pytest.mark.parametrize(
    ("ocr_setting", "ocr_mode"),
    [({"enable_ocr": True}, "ocr"), ({"enable_ocr": False}, "auto"), ({}, "auto")],
    ids=["ocr-on", "ocr-off", "ocr-unset"],
)
def test_the_parse_job_gets_only_the_tier_and_ocr_mode(use_mineru, listener, ocr_setting, ocr_mode):
    mineru_v1_server(listener)
    params = {
        "tier": "pro",
        "language": "de",
        "model_version": "vlm",
        "enable_formula": False,
        "enable_table": False,
        **ocr_setting,
    }

    stored = upload(use_mineru(params=params))

    assert stored["data"].get("content") == PARSED, stored["data"]
    assert job_request(listener) == {
        "files": [{"source": {"type": "file_id", "file_id": UPLOADED_FILE}}],
        "tier": "pro",
        "ocr_mode": ocr_mode,
    }, "the MinerU 4.x parse job should carry only the tier and the OCR mode"


@pytest.mark.journey
def test_a_page_range_is_sent_for_a_pdf_only(use_mineru, listener):
    params = {"tier": "flash", "page_ranges": "2-3"}
    client = use_mineru(params=params, extensions=("pdf", "docx"))

    mineru_v1_server(listener)
    upload(client, PDF)
    [pdf_entry] = job_request(listener)["files"]
    listener.received.clear()

    stored_docx = upload(client, DOCX)
    [docx_entry] = job_request(listener)["files"]

    assert pdf_entry.get("page_range") == "2-3", f"the PDF was sent without its pages: {pdf_entry}"
    assert stored_docx["data"].get("content") == PARSED, stored_docx["data"]
    assert "page_range" not in docx_entry, f"a Word file was sent a page range: {docx_entry}"


@pytest.mark.journey
def test_a_queued_job_is_polled_until_it_completes(use_mineru, listener):
    mineru_v1_server(listener, statuses=("queued", "running", "completed"))

    stored = upload(use_mineru())

    polls = listener.requests_to(f"/v1/parse/jobs/{JOB_ID}")
    assert stored["data"].get("content") == PARSED, (
        f"the text was not stored after {len(polls)} polls: {stored['data']}"
    )
    assert len(polls) == 2, f"the job should be asked twice more, was asked {len(polls)} times"


@pytest.mark.journey
def test_a_failed_job_leaves_the_file_unparsed_with_minerus_error(use_mineru, listener):
    reason = f"page 3 cannot be read {uuid.uuid4().hex[:6]}"
    mineru_v1_server(listener, statuses=("running", "failed"), error=reason)

    stored = upload(use_mineru())

    assert stored["data"].get("status") == "failed", stored["data"]
    assert reason in (stored["data"].get("error") or ""), stored["data"]
    assert not stored["data"].get("content"), stored["data"]
    assert not listener.requests_to(f"/v1/files/{MARKDOWN_FILE}/content")


@pytest.mark.journey
def test_embedded_pictures_are_left_out_of_the_stored_text(use_mineru, listener):
    markdown = (
        "Quarterly report\n\n"
        "![Revenue by region](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB)\n\n"
        '<img src="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ" alt="Company logo" width="40">\n\n'
        "Revenue grew in every region."
    )
    mineru_v1_server(listener, markdown=markdown)

    stored = upload(use_mineru())

    content = stored["data"].get("content") or ""
    for kept in ("Quarterly report", "Revenue by region", "Revenue grew in every region."):
        assert kept in content, f"{kept!r} is missing from the stored text: {content!r}"
    for picture in ("data:image", "base64", "<img", "Company logo"):
        assert picture not in content, f"the picture was stored as text ({picture!r}): {content!r}"


@pytest.mark.journey
def test_a_3x_server_gets_the_parameters_as_a_file_parse_form(use_mineru, listener):
    listener.route(
        "POST", "/file_parse", json_answer({"results": {"report": {"md_content": PARSED}}})
    )
    params = {"tier": "flash", "enable_ocr": True, "language": "de", "page_ranges": "2-3"}

    stored = upload(use_mineru(params=params))

    assert stored["data"].get("content") == PARSED, stored["data"]
    [parsed] = listener.requests_to("/file_parse")
    assert b'name="language"\r\n\r\nde' in parsed.body, parsed.body[:600]
    v1_calls = [request.path for request in listener.received if request.path.startswith("/v1/")]
    assert v1_calls == ["/v1/health"], f"a 3.x server was sent V1 calls: {v1_calls}"
