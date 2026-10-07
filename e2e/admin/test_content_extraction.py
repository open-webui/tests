"""Journey: the Content Extraction Engine picked in Admin Settings > Documents reads the files.

With Tika and its server URL saved, a document a user attaches to a chat is sent to the Tika
server, at the endpoint of the Tika version picked, and the text Tika returns is what the model
reads with the question. A plain text file is still read as it is, without Tika. The External
document loader gets the file with the admin's API key, the file's name and the admin's extra
headers, and Docling gets it with its API key; the text each returns is what the model reads. The
same holds for the Datalab Marker API (its key and the switches the admin turned on), Document
Intelligence (its key and the model typed), Mistral OCR (uploading the file first, or sending it
inline with Use Base64 on), PaddleOCR-vl (its token) and MinerU (the parameters typed).

The Datalab Marker test is red on dev ebc6add67: the loader writes the text Marker returned under
the hard-coded /app/backend/data/uploads/marker_output before handing it on, so an install outside
the official container, which has no such folder, never gets the text to the model
(open-webui/open-webui#32025).

Discriminates: passes on dev ebc6add67 apart from the Datalab Marker test; in a frontend build
whose Documents form sends the stored engine settings back in place of the edited ones, the Tika,
External and Docling tests fail; in a backend whose document loader never picks the Datalab
Marker, Document Intelligence, Mistral OCR, PaddleOCR-vl or MinerU engine, the test of each of
them fails.
"""

from __future__ import annotations

import base64
import json
import re

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.listener import ReceivedRequest, json_answer
from harness.web_retrieval import RETRIEVAL_CONFIG
from utils.chat_ui import chat_input, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

EXTRACTED = "The tide table lists high water at 06:42 and 19:05."
QUESTION = "When is high water according to the report?"
ENDPOINTS = {"3": ("/tika/text", "X-TIKA:content"), "4": ("/tika/json/md", "tk:content")}


def serve_tika(listener, version: str) -> str:
    path, content_key = ENDPOINTS[version]

    def extract(_request: ReceivedRequest):
        return json_answer({content_key: EXTRACTED, "Content-Type": "application/pdf"})

    listener.route("PUT", path, extract)
    return path


def pick_engine(page: Page, engine: str):
    page.goto("/admin/settings/documents")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("tab", selected=True)).to_be_visible()
    settings.get_by_role("combobox").filter(
        has=page.get_by_role("option", name="Tika", exact=True)
    ).select_option(engine)
    return settings


def save(page: Page, settings) -> None:
    settings.get_by_role("button", name="Save", exact=True).click()
    expect(page.get_by_text("Settings saved successfully!").first).to_be_visible()


def save_tika(page: Page, url: str, version: str) -> None:
    settings = pick_engine(page, "tika")
    settings.get_by_placeholder("Enter Tika Server URL").fill(url)
    settings.get_by_role("combobox").filter(
        has=page.get_by_role("option", name="Tika 4.x")
    ).select_option(version)
    save(page, settings)


def ask_about(page: Page, upstream, name: str, mime_type: str, content: bytes) -> str:
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("main").get_by_label("More").click()
    with page.expect_file_chooser() as chooser:
        page.get_by_role("menu").get_by_role("button", name="Upload Files").click()
    chooser.value.set_files({"name": name, "mimeType": mime_type, "buffer": content})
    expect(page.get_by_role("button", name=name)).to_be_visible()
    upstream.queue(reply.text("At 06:42.", match=reply.answering(QUESTION)))
    send(page, QUESTION)
    expect_reply(page, "At 06:42.")
    return json.dumps(next(filter(reply.answering(QUESTION), upstream.chat_requests())))


@pytest.mark.parametrize("version", ["3", "4"])
def test_a_document_is_read_through_the_tika_version_picked(
    version, page_for, make_user, upstream, listener, preserve
):
    preserve(RETRIEVAL_CONFIG)
    path = serve_tika(listener, version)
    save_tika(page_for(make_user(role="admin")), listener.base_url, version)

    sent = ask_about(
        page_for(make_user()), upstream, "report.pdf", "application/pdf", b"%PDF-1.4 scanned"
    )

    assert EXTRACTED in sent
    assert [request.method for request in listener.requests_to(path)] == ["PUT"]


def test_a_text_file_is_read_without_tika(page_for, make_user, upstream, listener, preserve):
    preserve(RETRIEVAL_CONFIG)
    serve_tika(listener, "3")
    save_tika(page_for(make_user(role="admin")), listener.base_url, "3")
    note = "The harbour master says high water is at 06:42."

    sent = ask_about(page_for(make_user()), upstream, "note.txt", "text/plain", note.encode())

    assert note in sent
    assert listener.requests_to("/tika/text") == []


def test_the_external_loader_gets_the_file_with_the_admins_key_and_headers(
    page_for, make_user, upstream, listener, preserve
):
    preserve(RETRIEVAL_CONFIG)
    listener.route("PUT", "/process", json_answer({"page_content": EXTRACTED, "metadata": {}}))
    admin_page = page_for(make_user(role="admin"))
    settings = pick_engine(admin_page, "external")
    settings.get_by_placeholder("Enter External Document Loader URL").fill(listener.base_url)
    settings.get_by_placeholder("Enter External Document Loader API Key").fill("loader-key-7")
    settings.get_by_placeholder("Enter additional headers in JSON format").fill(
        '{"X-Harbour": "north quay"}'
    )
    save(admin_page, settings)

    sent = ask_about(
        page_for(make_user()), upstream, "report.pdf", "application/pdf", b"%PDF-1.4 scanned"
    )

    assert EXTRACTED in sent
    [loaded] = listener.requests_to("/process")
    assert loaded.headers["Authorization"] == "Bearer loader-key-7"
    assert loaded.headers["X-Harbour"] == "north quay"
    assert loaded.headers["X-Filename"].endswith("report.pdf")


def test_docling_converts_the_file_with_its_api_key(
    page_for, make_user, upstream, listener, preserve
):
    preserve(RETRIEVAL_CONFIG)
    listener.route(
        "POST",
        "/v1/convert/file",
        json_answer({"status": "success", "document": {"md_content": EXTRACTED}}),
    )
    admin_page = page_for(make_user(role="admin"))
    settings = pick_engine(admin_page, "docling")
    settings.get_by_placeholder("Enter Docling Server URL").fill(listener.base_url)
    settings.get_by_placeholder("Enter Docling API Key").fill("docling-key-3")
    save(admin_page, settings)

    sent = ask_about(
        page_for(make_user()), upstream, "report.pdf", "application/pdf", b"%PDF-1.4 scanned"
    )

    assert EXTRACTED in sent
    [converted] = listener.requests_to("/v1/convert/file")
    assert converted.headers["X-Api-Key"] == "docling-key-3"


def save_engine(page: Page, engine: str, fields: dict[str, str]):
    settings = pick_engine(page, engine)
    for placeholder, value in fields.items():
        settings.get_by_placeholder(placeholder, exact=True).fill(value)
    return settings


def turn_on(settings, name: str) -> None:
    switch = settings.get_by_role("switch", name=name, exact=True)
    switch.click()
    expect(switch).to_have_attribute("aria-checked", "true")


def ask_about_report(page_for, make_user, upstream) -> str:
    return ask_about(
        page_for(make_user()), upstream, "report.pdf", "application/pdf", b"%PDF-1.4 scanned"
    )


def test_datalab_marker_gets_the_file_with_the_key_and_the_switches_turned_on(
    page_for, make_user, upstream, listener, preserve
):
    preserve(RETRIEVAL_CONFIG)
    listener.route("POST", "/marker", json_answer({"success": True, "output": EXTRACTED}))
    admin_page = page_for(make_user(role="admin"))
    settings = save_engine(
        admin_page,
        "datalab_marker",
        {
            "Enter Datalab Marker API Base URL": f"{listener.base_url}/marker",
            "Enter Datalab Marker API Key": "marker-key-5",
        },
    )
    turn_on(settings, "Force OCR")
    save(admin_page, settings)

    sent = ask_about_report(page_for, make_user, upstream)

    [marked] = listener.requests_to("/marker")
    assert marked.headers["X-Api-Key"] == "marker-key-5"
    assert b'name="force_ocr"\r\n\r\ntrue' in marked.body, "Force OCR was not sent as turned on"
    assert b'name="use_llm"\r\n\r\nfalse' in marked.body
    assert EXTRACTED in sent, (
        "the text Marker returned did not reach the model: the loader writes its output under the "
        "hard-coded /app/backend/data/uploads/marker_output, which a non-Docker install lacks "
        "(open-webui/open-webui#32025)"
    )


def test_document_intelligence_analyzes_the_file_with_the_key_and_the_model_typed(
    page_for, make_user, upstream, listener, preserve
):
    preserve(RETRIEVAL_CONFIG)
    analyze = "/documentintelligence/documentModels/prebuilt-read:analyze"

    def accept(_request: ReceivedRequest):
        location = f"{listener.base_url}/operations/42"
        return 202, {"Operation-Location": location, "Retry-After": "0"}, b""

    listener.route("POST", analyze, accept)
    listener.route(
        "GET",
        "/operations/42",
        json_answer(
            {
                "status": "succeeded",
                "analyzeResult": {
                    "apiVersion": "2024-11-30",
                    "modelId": "prebuilt-read",
                    "contentFormat": "markdown",
                    "content": EXTRACTED,
                    "pages": [],
                },
            }
        ),
    )
    admin_page = page_for(make_user(role="admin"))
    settings = save_engine(
        admin_page,
        "document_intelligence",
        {
            "Enter Document Intelligence Endpoint": listener.base_url,
            "Enter Document Intelligence Key": "azure-key-9",
            "Enter Document Intelligence Model": "prebuilt-read",
        },
    )
    save(admin_page, settings)

    sent = ask_about_report(page_for, make_user, upstream)

    assert EXTRACTED in sent
    [analyzed] = listener.requests_to(analyze)
    assert analyzed.headers["Ocp-Apim-Subscription-Key"] == "azure-key-9"
    assert b"%PDF-1.4 scanned" in analyzed.body


def serve_mistral(listener) -> None:
    pages = [{"index": 0, "markdown": EXTRACTED}]
    listener.route("POST", "/v1/files", json_answer({"id": "file-1"}))
    listener.route("GET", "/v1/files/file-1/url", json_answer({"url": "https://signed.test/f"}))
    listener.route("POST", "/v1/ocr", json_answer({"pages": pages}))
    listener.route("DELETE", "/v1/files/file-1", json_answer({"id": "file-1", "deleted": True}))


def save_mistral(page: Page, listener, use_base64: bool) -> None:
    settings = save_engine(
        page,
        "mistral_ocr",
        {
            "Enter Mistral API Base URL": f"{listener.base_url}/v1",
            "Enter Mistral API Key": "mistral-key-2",
        },
    )
    if use_base64:
        turn_on(settings, "Use Base64")
    save(page, settings)


def test_mistral_ocr_uploads_the_file_reads_it_and_deletes_it(
    page_for, make_user, upstream, listener, preserve
):
    preserve(RETRIEVAL_CONFIG)
    serve_mistral(listener)
    save_mistral(page_for(make_user(role="admin")), listener, use_base64=False)

    sent = ask_about_report(page_for, make_user, upstream)

    assert EXTRACTED in sent
    for path in ("/v1/files", "/v1/files/file-1/url", "/v1/ocr", "/v1/files/file-1"):
        [call] = listener.requests_to(path)
        assert call.headers["Authorization"] == "Bearer mistral-key-2"
    [ocr] = listener.requests_to("/v1/ocr")
    assert ocr.json()["document"]["document_url"] == "https://signed.test/f"


def test_mistral_ocr_sends_the_file_inline_with_use_base64_on(
    page_for, make_user, upstream, listener, preserve
):
    preserve(RETRIEVAL_CONFIG)
    serve_mistral(listener)
    save_mistral(page_for(make_user(role="admin")), listener, use_base64=True)

    sent = ask_about_report(page_for, make_user, upstream)

    assert EXTRACTED in sent
    [ocr] = listener.requests_to("/v1/ocr")
    assert ocr.headers["Authorization"] == "Bearer mistral-key-2"
    encoded = base64.b64encode(b"%PDF-1.4 scanned").decode()
    assert ocr.json()["document"]["document_url"] == f"data:application/pdf;base64,{encoded}"
    assert listener.requests_to("/v1/files") == []


def test_paddleocr_vl_parses_the_file_with_its_token(
    page_for, make_user, upstream, listener, preserve
):
    preserve(RETRIEVAL_CONFIG)
    result = {"layoutParsingResults": [{"markdown": {"text": EXTRACTED}}]}
    listener.route("POST", "/layout-parsing", json_answer({"result": result}))
    admin_page = page_for(make_user(role="admin"))
    settings = save_engine(
        admin_page,
        "paddleocr_vl",
        {
            "Enter PaddleOCR-vl API Base URL": listener.base_url,
            "Enter PaddleOCR-vl API Token": "paddle-token-4",
        },
    )
    save(admin_page, settings)

    sent = ask_about_report(page_for, make_user, upstream)

    assert EXTRACTED in sent
    [parsed] = listener.requests_to("/layout-parsing")
    assert parsed.headers["Authorization"] == "token paddle-token-4"
    assert parsed.json()["fileType"] == 0


def test_mineru_parses_the_file_with_the_parameters_the_admin_typed(
    page_for, make_user, upstream, listener, preserve
):
    preserve(RETRIEVAL_CONFIG)
    listener.route(
        "POST", "/file_parse", json_answer({"results": {"report": {"md_content": EXTRACTED}}})
    )
    admin_page = page_for(make_user(role="admin"))
    settings = save_engine(admin_page, "mineru", {"http://localhost:8000": listener.base_url})
    settings.get_by_placeholder(re.compile('"enable_formula"')).fill(
        '{"enable_ocr": true, "language": "de"}'
    )
    save(admin_page, settings)

    sent = ask_about_report(page_for, make_user, upstream)

    assert EXTRACTED in sent
    [parsed] = listener.requests_to("/file_parse")
    assert b'name="enable_ocr"\r\n\r\nTrue' in parsed.body
    assert b'name="language"\r\n\r\nde' in parsed.body
    assert b'name="return_md"\r\n\r\ntrue' in parsed.body
