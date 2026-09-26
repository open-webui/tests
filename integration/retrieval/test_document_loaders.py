"""Journey: the text Open WebUI extracts from each kind of document a user uploads.

With no extraction engine configured, an upload is read by the built-in loader for its type:
plain text, Markdown, JSON and other source files as they are; CSV one row at a time as
`column: value` lines; HTML without its markup; Word documents, spreadsheets and slide decks
through their own readers; PDFs page by page, or as one document when the admin chooses single
mode. An Office archive that unpacks far larger than the file is refused before it is opened.
The stored file text and its chunks are what a chat later retrieves from.

Discriminates: in a backend copy, the CSV loader joining values without their column turned the
CSV test red; the PDF loader ignoring `PDF_LOADER_MODE` turned the single mode test red; the
archive size check skipped turned the oversized archive test red; and the spreadsheet branch
sent to the plain text loader turned the spreadsheet test red.
"""

from __future__ import annotations

import io
import zipfile

import httpx
import pytest

from harness.web_retrieval import RETRIEVAL_CONFIG

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


@pytest.fixture
def local_loaders(preserve, admin):
    """The built-in loaders, with `settings(**changes)` for further document settings."""
    preserve(RETRIEVAL_CONFIG)
    client = admin.client()

    def settings(**changes) -> None:
        saved = client.post(RETRIEVAL_CONFIG[1], json=changes)
        assert saved.status_code == 200, saved.text

    settings(
        CONTENT_EXTRACTION_ENGINE="",
        BYPASS_EMBEDDING_AND_RETRIEVAL=False,
        PDF_LOADER_MODE="page",
        PDF_EXTRACT_IMAGES=False,
    )
    yield settings
    client.close()


def upload(client: httpx.Client, filename: str, content: bytes, content_type: str) -> dict:
    """Upload and process in the foreground; returns the stored file."""
    uploaded = client.post(
        "/api/v1/files/",
        params={"process": "true", "process_in_background": "false"},
        files={"file": (filename, content, content_type)},
    )
    assert uploaded.status_code == 200, uploaded.text
    stored = client.get(f"/api/v1/files/{uploaded.json()['id']}")
    assert stored.status_code == 200, stored.text
    return stored.json()


def extracted(client: httpx.Client, filename: str, content: bytes, content_type: str) -> str:
    stored = upload(client, filename, content, content_type)
    assert stored["data"].get("status") == "completed", (
        f"{filename} was not read: {stored['data'].get('error')}"
    )
    return stored["data"]["content"]


def chunk_metadata(client: httpx.Client, file_id: str) -> list[dict]:
    answered = client.post(
        "/api/v1/retrieval/query/doc",
        json={"collection_name": f"file-{file_id}", "query": "heron", "k": 50},
    )
    assert answered.status_code == 200, answered.text
    return answered.json()["metadatas"][0]


# --- building documents ---------------------------------------------------------------------------


def docx_bytes(*paragraphs: str) -> bytes:
    docx = pytest.importorskip("docx")
    document = docx.Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def xlsx_bytes(rows: list[list[str]]) -> bytes:
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Sightings"
    for row in rows:
        sheet.append(row)
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def pptx_bytes(*slides: tuple[str, str]) -> bytes:
    pptx = pytest.importorskip("pptx")
    deck = pptx.Presentation()
    for title, body in slides:
        slide = deck.slides.add_slide(deck.slide_layouts[1])
        slide.shapes.title.text = title
        slide.placeholders[1].text = body
    buffer = io.BytesIO()
    deck.save(buffer)
    return buffer.getvalue()


def pdf_bytes(*pages: str) -> bytes:
    """A PDF with one line of Helvetica text per page, built by hand."""
    count = len(pages)
    font_id = 3 + 2 * count
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids ["
        + b" ".join(f"{3 + 2 * index} 0 R".encode() for index in range(count))
        + f"] /Count {count} >>".encode(),
    ]
    for index, text in enumerate(pages):
        stream = f"BT /F1 18 Tf 72 720 Td ({text}) Tj ET".encode()
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {4 + 2 * index} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
        )
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    output = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(output))
        output += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(output)
    output += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    output += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    output += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return output


# --- text files ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename, content_type",
    [
        ("notes.txt", "text/plain"),
        ("notes.md", "text/markdown"),
        ("notes.json", "application/json"),
        ("notes.py", "application/octet-stream"),
    ],
)
def test_text_and_source_files_are_stored_as_written(
    local_loaders, make_user, filename, content_type
):
    text = '{"bird": "heron", "note": "stands in the reeds"}\n# second line\n'
    with make_user().client() as client:
        content = extracted(client, filename, text.encode(), content_type)

    assert content == text


def test_a_csv_is_read_row_by_row_with_its_columns(local_loaders, make_user):
    table = "bird,habitat\nheron,reed bed\ngrebe,open water\n"
    with make_user().client() as client:
        content = extracted(client, "birds.csv", table.encode(), "text/csv")

    assert content == "bird: heron\nhabitat: reed bed bird: grebe\nhabitat: open water"


def test_html_is_read_without_its_markup(local_loaders, make_user):
    page = (
        "<html><head><title>Lake</title><style>p { color: red }</style></head>"
        "<body><h1>Herons</h1><p>They wait <b>very</b> still.</p></body></html>"
    )
    with make_user().client() as client:
        content = extracted(client, "lake.html", page.encode(), "text/html")

    assert "<" not in content and "Herons" in content, content
    assert "They wait very still." in content


# --- office documents ---------------------------------------------------------------------------


def test_a_word_document_is_read(local_loaders, make_user):
    document = docx_bytes("Heron survey", "Two herons nested by the jetty.")
    with make_user().client() as client:
        content = extracted(client, "survey.docx", document, DOCX)

    assert "Heron survey" in content and "Two herons nested by the jetty." in content


def test_a_spreadsheet_is_read_cell_by_cell(local_loaders, make_user):
    workbook = xlsx_bytes([["bird", "count"], ["heron", "4"], ["grebe", "7"]])
    with make_user().client() as client:
        content = extracted(client, "counts.xlsx", workbook, XLSX)

    for cell in ("bird", "count", "heron", "grebe", "7"):
        assert cell in content, f"{cell!r} missing from the spreadsheet text: {content!r}"
    assert "PK" not in content[:4], "the spreadsheet was read as raw bytes"


def test_a_slide_deck_is_read(local_loaders, make_user):
    deck = pptx_bytes(("Herons", "Grey and patient"), ("Grebes", "Crested and loud"))
    with make_user().client() as client:
        content = extracted(client, "birds.pptx", deck, PPTX)

    for text in ("Herons", "Grey and patient", "Grebes", "Crested and loud"):
        assert text in content, content


def test_an_archive_that_unpacks_far_larger_is_refused(local_loaders, make_user):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", b"\0" * (24 * 1024 * 1024))
    with make_user().client() as client:
        stored = upload(client, "bomb.docx", buffer.getvalue(), DOCX)

    assert stored["data"].get("status") == "failed", stored["data"]
    assert "too large after decompression" in (stored["data"].get("error") or "")


# --- PDFs ---------------------------------------------------------------------------------------


def test_a_pdf_is_read_page_by_page(local_loaders, make_user):
    document = pdf_bytes("Herons wait at dawn", "Grebes dive at dusk")
    with make_user().client() as client:
        stored = upload(client, "birds.pdf", document, "application/pdf")
        metadata = chunk_metadata(client, stored["id"])

    assert stored["data"]["content"] == "Herons wait at dawn Grebes dive at dusk"
    assert sorted(entry["page"] for entry in metadata) == [0, 1]
    assert {entry["total_pages"] for entry in metadata} == {2}


def test_single_mode_reads_a_pdf_as_one_document(local_loaders, make_user):
    local_loaders(PDF_LOADER_MODE="single")
    document = pdf_bytes("Herons wait at dawn", "Grebes dive at dusk")
    with make_user().client() as client:
        stored = upload(client, "birds.pdf", document, "application/pdf")
        metadata = chunk_metadata(client, stored["id"])

    assert stored["data"]["content"] == "Herons wait at dawn\n\fGrebes dive at dusk"
    assert [entry.get("page") for entry in metadata] == [None], metadata
