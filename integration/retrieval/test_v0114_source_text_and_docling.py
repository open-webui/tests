"""Regressions in source-text routing, Docling conversion and entity handling of uploads, v0.11.4.

- `.ino` sketches (fix `acd03b147`, PR #29673, issue #29670): browsers send `.ino` as
  `application/octet-stream` and the extension was missing from `known_source_ext`, so an
  Arduino sketch went to the Tika or Docling server instead of the plain text reader and the
  upload failed. It reads as text now, like the `.cpp` and `.h` files beside it.
- Docling failures inside an HTTP 200 (fix `0837f310b`, PR #30107, issue #29808): a file Docling
  refused came back as a 200 with `status: failure`, so the upload stored and indexed the
  `<No text content found>` placeholder in place of the file. A failed or skipped conversion now
  fails the upload with Docling's own messages, and a null `md_content` no longer raises.
- HTML entities in uploaded text (fix `1bfa59acb`, PR #29736, issue #29732): every document went
  through `ftfy.fix_text`, whose default also decodes entities, so `&nbsp;` and `&gt;` in a file
  were rewritten before being stored. Entity decoding is off; ftfy's mojibake repair stays, and
  since PR #31655 (issue #17087) it keeps curly quotes and full-width punctuation as written.
- Docling page numbers (issue open-webui/open-webui#33603): a PDF Docling split into pages
  showed the model only the 0-based page index, so the model cited and opened the page before the
  one it meant. Since `fe2219d6f` the model is given an uploaded file's page 1-based, for every
  loader. The chunks keep the 0-based index, which the citation view turns into the page a person
  sees by itself. A file under the Full Text Threshold is sent whole as one passage with no page,
  by design, so the threshold is off here and the file is searched by its chunks. File Context is
  on for the model, since native mode otherwise only gives it the file tools.

Tika and Docling are local services; each upload is processed before the response returns.

Twin of unit/retrieval/test_v0114_source_text_and_docling.py.

Discriminates: passes on dev bbfa876af; removing `ino` from `known_source_ext` fails the sketch
cases (the extraction server is called), dropping the conversion status check fails the refused
and skipped cases, reading `md_content` with a `''` default fails the null case and decoding
entities again fails the entity case. The page number case passes on dev fe2219d6f and fails on
fe2219d6f^ (the model is given pages 0 and 2). The mojibake case passes on dev 015dbc861 and
fails on dev a5bc78300, before PR #31655 (the repaired quote comes back straightened).
"""

from __future__ import annotations

import json
import re
import uuid

import pytest

from harness import upstream as reply
from harness.actors import admin_of
from harness.chat import ask
from harness.listener import json_answer

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

RETRIEVAL_CONFIG = ("/api/v1/retrieval/config", "/api/v1/retrieval/config/update")
OCTET_STREAM = "application/octet-stream"
SKETCH = "void setup() { pinMode(13, OUTPUT); }\n"
ENTITY_TEXT = "cost &gt; 5\nA&nbsp;B\nC &amp; D\n<xml>markup &gt; stays after a literal tag</xml>\n"


@pytest.fixture
def extraction(admin, preserve, listener):
    """(admin client, `select(engine)`) for Tika, Docling or the built-in reader."""
    preserve(RETRIEVAL_CONFIG)
    listener.route("PUT", "/tika/text", json_answer({"X-TIKA:content": "extracted by tika"}))
    with admin.client() as client:

        def select(engine: str) -> None:
            saved = client.post(
                RETRIEVAL_CONFIG[1],
                json={
                    "CONTENT_EXTRACTION_ENGINE": engine,
                    "TIKA_SERVER_URL": listener.base_url,
                    "DOCLING_SERVER_URL": listener.base_url,
                },
            )
            assert saved.status_code == 200, saved.text

        yield client, select


def upload(client, filename: str, content: bytes, content_type: str) -> dict:
    uploaded = client.post(
        "/api/v1/files/?process_in_background=false",
        files={"file": (filename, content, content_type)},
    )
    assert uploaded.status_code == 200, uploaded.text
    stored = client.get(f"/api/v1/files/{uploaded.json()['id']}")
    assert stored.status_code == 200, stored.text
    return stored.json()["data"]


def serve_docling(listener, answer: dict) -> None:
    listener.route("POST", "/v1/convert/file", json_answer(answer))


@pytest.mark.parametrize("engine", ["tika", "docling"])
@pytest.mark.parametrize("filename", ["blink.ino", "helper.cpp", "helper.h"])
def test_a_source_file_is_read_as_text_not_sent_for_extraction(
    extraction, listener, engine, filename
):
    client, select = extraction
    select(engine)

    data = upload(client, filename, SKETCH.encode(), OCTET_STREAM)

    assert data["content"] == SKETCH, f"{filename} was not stored as written: {data}"
    assert listener.received == [], f"{filename} went to the {engine} server"


def test_a_binary_file_still_goes_to_the_extraction_server(extraction, listener):
    client, select = extraction
    select("tika")

    data = upload(client, "blob.bin", b"\x00\x01\x02", OCTET_STREAM)

    assert data["content"] == "extracted by tika"
    assert listener.requests_to("/tika/text")


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        (
            {
                "status": "failure",
                "errors": [
                    {"error_message": None},
                    {"error_message": "File format not allowed: drawing.dxf"},
                ],
                "document": {},
            },
            "File format not allowed: drawing.dxf",
        ),
        ({"status": "skipped", "errors": [], "document": {}}, "skipped"),
        ({"status": "failure", "errors": [], "document": {}}, "no error message"),
    ],
    ids=["refused", "skipped", "failed-silently"],
)
def test_a_failed_docling_conversion_fails_the_upload(extraction, listener, answer, reason):
    client, select = extraction
    select("docling")
    serve_docling(listener, answer)

    data = upload(client, "drawing.dxf", b"binary-ish", OCTET_STREAM)

    assert data["status"] == "failed", (
        f"a conversion Docling reported as {answer['status']} was stored as {data} (#29808)"
    )
    assert reason in data["error"]


@pytest.mark.parametrize(
    ("markdown", "stored"),
    [("# Title\n\nBody", "# Title\n\nBody"), (None, "<No text content found>")],
    ids=["markdown", "null-markdown"],
)
def test_a_successful_docling_conversion_is_stored(extraction, listener, markdown, stored):
    client, select = extraction
    select("docling")
    serve_docling(listener, {"status": "success", "document": {"md_content": markdown}})

    data = upload(client, "drawing.dxf", b"binary-ish", OCTET_STREAM)

    assert (data["status"], data["content"]) == ("completed", stored), json.dumps(data)


def test_uploaded_entities_are_stored_exactly_as_written(extraction):
    client, select = extraction
    select("")

    data = upload(client, "notes.txt", ENTITY_TEXT.encode(), "text/plain")

    assert data["content"] == ENTITY_TEXT, (
        "HTML entities in the file were decoded before storage, so the model reads a "
        "rewritten copy (#29732)"
    )


def test_mojibake_is_still_repaired(extraction):
    client, select = extraction
    select("")

    data = upload(client, "mojibake.txt", "donâ€™t".encode(), "text/plain")

    assert data["content"] == "don’t"


@pytest.fixture(scope="module")
def pages_shown(instance_with):
    """An instance that gives the model each source chunk's page."""
    return instance_with({"RAG_SOURCE_METADATA_KEYS": "page"})


@pytest.mark.slow
def test_docling_pages_reach_the_model_as_the_page_a_person_sees(pages_shown, preserve, listener):
    preserve(RETRIEVAL_CONFIG, on=pages_shown)
    tag = uuid.uuid4().hex[:8]
    # page 2 is blank, so the page numbers come from Docling's JSON
    serve_docling(
        listener,
        {
            "status": "success",
            "document": {
                "md_content": f"harbour {tag} on the first page\fharbour {tag} on the third page",
                "json_content": {
                    "texts": [
                        {"content_layer": "body", "prov": [{"page_no": 1}]},
                        {"content_layer": "body", "prov": [{"page_no": 3}]},
                    ]
                },
            },
        },
    )
    with admin_of(pages_shown).client() as client:
        saved = client.post(
            RETRIEVAL_CONFIG[1],
            json={
                "CONTENT_EXTRACTION_ENGINE": "docling",
                "DOCLING_SERVER_URL": listener.base_url,
                # a file under the threshold is sent whole, without its pages
                "RAG_FULL_TEXT_CHAR_THRESHOLD": 0,
            },
        )
        assert saved.status_code == 200, saved.text
        uploaded = client.post(
            "/api/v1/files/?process_in_background=false",
            files={"file": ("survey.pdf", b"%PDF-1.4", "application/pdf")},
        )
        assert uploaded.status_code == 200, uploaded.text
        file_id = uploaded.json()["id"]
        model_id = f"pages-{tag}"
        created = client.post(
            "/api/v1/models/create",
            json={
                "id": model_id,
                "base_model_id": reply.MOCK_MODEL_ID,
                "name": model_id,
                "meta": {"capabilities": {"file_context": True}},
                "params": {},
            },
        )
        assert created.status_code == 200, created.text
        try:
            client.get("/api/models", params={"refresh": "true"}).raise_for_status()
            prompt = f"Which page mentions harbour {tag}?"
            pages_shown.upstream.queue(reply.text("page 3", match=reply.answering(prompt)))
            _, message = ask(
                client,
                prompt,
                model=model_id,
                chat_files=[
                    {"type": "file", "id": file_id, "name": "survey.pdf", "status": "uploaded"}
                ],
            )
        finally:
            client.post("/api/v1/models/model/delete", json={"id": model_id})
    assert message.get("done") and not message.get("error"), message

    sent = [
        json.dumps(body["messages"])
        for body in pages_shown.upstream.chat_requests()
        if prompt in json.dumps(body["messages"])
    ]
    assert sent, "the chat never reached the model"
    shown = sorted(int(page) for page in re.findall(r'<source [^>]*page=\\"(\d+)\\"', sent[-1]))
    assert shown == [1, 3], (
        f"the model was given pages {shown} for a PDF whose text is on pages 1 and 3, so it cites "
        "and opens the page before the right one (#33603)"
    )

    # the citation view adds one to the stored page itself, so the chunks must stay 0-based
    stored = sorted(
        meta.get("page")
        for source in message.get("sources") or []
        for meta in source.get("metadata") or []
        if meta.get("file_id") == file_id
    )
    assert stored == [0, 2], (
        f"the reply's citations carry pages {stored}; the citation view, which adds one itself, "
        "would show a page after the right one"
    )
