"""Regression: a document container that unpacks far beyond its stored size is refused.

open-webui 0.11.1 fix `2a0274a0a`: Office, OpenDocument and EPUB uploads (`docx`, `xlsx`,
`pptx`, `odt`, `epub`) are zip containers, and the loader handed them straight to their
extractor, which inflated every member into the worker's memory. A few kilobytes on disk could
declare gigabytes. The fix reads the zip directory first and refuses the file with "Document
archive is too large after decompression" when the declared total exceeds
`min(max(10 MiB, stored size * 100), FILE_MAX_SIZE)`, with the admin's file size limit now
passed through to the loader.

#33460 (`4adf3083f`) then made the check recognise a zip by its content instead of its extension
or declared type, so a zip uploaded under any name, a `.pdf` included, is checked too, while a
real PDF still reaches its reader.

Issue #34944: the check ran for every extraction engine, so a remote engine (External, Tika,
Docling and the rest), to which Open WebUI only forwards the file, never got a valid workbook
that declares more than the admin's file size limit unpacked. The check now applies only where
Open WebUI unpacks the file itself, which includes a remote engine handing a file type it does
not take to the built-in loaders.

Twin of unit/security/test_archive_expansion_limit.py. Bounded by construction: every archive
declares at most 16 MiB, so the pre-fix run inflates no more than that, and the sparse workbook
(98 MB unpacked) only ever goes to the local stand-in engine or is refused.

Discriminates: passes on dev 3cda94d40; with the size check removed from `Loader._get_loader`
the 16 MiB archives reach their extractor and fail (or pass) for other reasons, with
`FILE_MAX_SIZE` no longer passed to the loader a 1 MiB limit refuses nothing, and on 4adf3083f^
(archives still told apart by extension) the zip named `scan.pdf` is not refused. The external
engine test fails on dev d89cd809c (#34944 unfixed: the workbook is refused and the engine never
called) and passes with the check moved to the built-in loaders.
"""

from __future__ import annotations

import functools
import io
import random
import zipfile

import pytest
from pypdf import PdfReader, PdfWriter

from harness.listener import json_answer
from harness.web_retrieval import RETRIEVAL_CONFIG
from integration.retrieval.test_document_loaders import pdf_bytes

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

MIB = 1024 * 1024
TOO_LARGE = "too large after decompression"

ARCHIVE_FORMATS = {
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "odt": "application/vnd.oasis.opendocument.text",
    "epub": "application/epub+zip",
}


def _archive_declaring(declared_bytes: int) -> bytes:
    """A real zip whose one member inflates to `declared_bytes` of zeros."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        with archive.open("word/document.xml", "w") as member:
            for _ in range(declared_bytes // MIB):
                member.write(b"\0" * MIB)
    return buffer.getvalue()


def _processed(client, filename: str, content_type: str, payload: bytes) -> dict:
    """Upload and process in the request, then return the stored file's `data`."""
    uploaded = client.post(
        "/api/v1/files/?process=true&process_in_background=false",
        files={"file": (filename, payload, content_type)},
    )
    assert uploaded.status_code == 200, uploaded.text
    return client.get(f"/api/v1/files/{uploaded.json()['id']}").json()["data"]


@pytest.fixture
def file_size_limit(admin):
    """Set the admin's upload limit in MiB; the original is put back afterwards."""
    with admin.client() as client:
        original = client.get("/api/v1/retrieval/config").json()["FILE_MAX_SIZE"]

        def set_limit(mebibytes: int) -> None:
            updated = client.post(
                "/api/v1/retrieval/config/update", json={"FILE_MAX_SIZE": mebibytes}
            )
            assert updated.status_code == 200, updated.text

        yield set_limit
        # An empty string is how the endpoint clears the limit.
        restored = client.post(
            "/api/v1/retrieval/config/update",
            json={"FILE_MAX_SIZE": "" if original is None else original},
        )
        assert restored.status_code == 200, restored.text


@pytest.mark.parametrize("extension", sorted(ARCHIVE_FORMATS))
def test_an_archive_declaring_16_mib_is_refused_before_extraction(extension, make_user):
    bomb = _archive_declaring(16 * MIB)
    assert len(bomb) < MIB, "the archive must stay small on disk to be a bomb"

    with make_user().client() as client:
        data = _processed(client, f"report.{extension}", ARCHIVE_FORMATS[extension], bomb)

    assert data.get("status") == "failed" and TOO_LARGE in (data.get("error") or ""), (
        f"a .{extension} of {len(bomb)} bytes declaring 16 MiB reached its extractor instead of "
        f"being refused: {data}"
    )


def test_the_content_type_alone_marks_an_archive(make_user):
    """Renaming the upload does not get it past the check."""
    bomb = _archive_declaring(16 * MIB)
    with make_user().client() as client:
        data = _processed(client, "payload.bin", ARCHIVE_FORMATS["xlsx"], bomb)

    assert TOO_LARGE in (data.get("error") or ""), data


def test_the_admins_file_size_limit_tightens_the_ceiling(make_user, file_size_limit):
    archive = _archive_declaring(4 * MIB)
    with make_user().client() as client:
        unlimited = _processed(client, "sheet.xlsx", ARCHIVE_FORMATS["xlsx"], archive)
        file_size_limit(1)
        limited = _processed(client, "sheet.xlsx", ARCHIVE_FORMATS["xlsx"], archive)

    assert TOO_LARGE not in (unlimited.get("error") or ""), (
        "4 MiB is under the 10 MiB floor and must not be refused without a tighter limit"
    )
    assert TOO_LARGE in (limited.get("error") or ""), (
        f"a 1 MiB upload limit did not reach the archive check, so a 4 MiB expansion passed: "
        f"{limited}"
    )


def test_a_real_word_document_still_reaches_its_extractor(make_user):
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("Quarterly revenue summary.")
    buffer = io.BytesIO()
    document.save(buffer)

    with make_user().client() as client:
        data = _processed(client, "memo.docx", ARCHIVE_FORMATS["docx"], buffer.getvalue())

    # Only extraction matters here; the embedding step after it is not the loader's.
    assert TOO_LARGE not in (data.get("error") or ""), data
    assert "Quarterly revenue summary." in data.get("content", ""), data


def test_a_zip_named_as_a_pdf_is_still_refused(make_user):
    bomb = _archive_declaring(16 * MIB)
    with make_user().client() as client:
        data = _processed(client, "scan.pdf", "application/pdf", bomb)

    assert data.get("status") == "failed" and TOO_LARGE in (data.get("error") or ""), (
        f"a zip declaring 16 MiB got past the archive check by being named scan.pdf: {data}"
    )


def _with_zip_attached(document: bytes) -> bytes:
    """The PDF with a small zip embedded as a file attachment, as PDF editors allow."""
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(document)))
    writer.add_attachment("figures.zip", _archive_declaring(1 * MIB))
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.mark.parametrize("attached", [False, True], ids=["plain", "zip-attached"])
def test_a_real_pdf_still_reaches_its_extractor(attached, make_user):
    document = pdf_bytes("Quarterly revenue summary.")
    if attached:
        document = _with_zip_attached(document)
        assert zipfile.is_zipfile(io.BytesIO(document)), "the attachment must read as a zip"

    with make_user().client() as client:
        data = _processed(client, "report.pdf", "application/pdf", document)

    assert TOO_LARGE not in (data.get("error") or ""), data
    assert "Quarterly revenue summary." in data.get("content", ""), data


def test_an_unreadable_archive_is_left_to_the_extractor(make_user):
    with make_user().client() as client:
        data = _processed(client, "broken.docx", ARCHIVE_FORMATS["docx"], b"not a zip at all")

    assert TOO_LARGE not in (data.get("error") or ""), data


@functools.cache
def _sparse_workbook() -> bytes:
    """The workbook from #34944: a pivot cache of mostly empty cells, 98 MB unpacked in 3.7 MB."""
    seeded = random.Random(34944)
    padding = b"<c/>" * 20
    records = b"".join(
        b"<r><c>" + seeded.randbytes(2).hex().encode() + b"</c>" + padding + b"</r>"
        for _ in range(1_000_000)
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("xl/pivotCache/pivotCacheRecords1.xml", records)
    return buffer.getvalue()


@pytest.fixture
def document_settings(preserve, admin, file_size_limit):
    """`update(**settings)` saves document settings, with Max Upload Size at 80 MB."""
    preserve(RETRIEVAL_CONFIG)
    file_size_limit(80)
    client = admin.client()

    def update(**settings) -> None:
        saved = client.post(RETRIEVAL_CONFIG[1], json=settings)
        assert saved.status_code == 200, saved.text

    yield update
    client.close()


def test_the_external_engine_gets_a_workbook_too_large_to_unpack_locally(
    document_settings, listener, make_user
):
    workbook = _sparse_workbook()
    unpacked = sum(entry.file_size for entry in zipfile.ZipFile(io.BytesIO(workbook)).infolist())
    assert len(workbook) < 5 * MIB and 80 * MIB < unpacked < len(workbook) * 100, (
        "the workbook must be over the 80 MB limit unpacked yet under 100 times its size"
    )
    listener.route(
        "PUT",
        "/process",
        json_answer({"page_content": "Plan revenue by cost centre.", "metadata": {}}),
    )
    document_settings(
        CONTENT_EXTRACTION_ENGINE="external",
        EXTERNAL_DOCUMENT_LOADER_URL=listener.base_url,
        EXTERNAL_DOCUMENT_LOADER_API_KEY="loader-key",
    )

    with make_user().client() as client:
        data = _processed(client, "plan.xlsx", ARCHIVE_FORMATS["xlsx"], workbook)

    received = listener.requests_to("/process")
    assert TOO_LARGE not in (data.get("error") or ""), (
        f"a workbook the external engine reads was refused before it was sent there: {data}"
    )
    assert [request.body == workbook for request in received] == [True], (
        f"the external engine did not get the uploaded workbook once: {len(received)} requests"
    )
    assert data.get("content") == "Plan revenue by cost centre.", (
        f"the text the external engine extracted was not stored: {data}"
    )


def test_the_built_in_loaders_still_refuse_the_workbook_and_a_zip_bomb(
    document_settings, make_user
):
    document_settings(CONTENT_EXTRACTION_ENGINE="")
    bomb = _archive_declaring(16 * MIB)
    assert len(bomb) * 100 < 16 * MIB, "the bomb must unpack to over 100 times its size"

    with make_user().client() as client:
        workbook = _processed(client, "plan.xlsx", ARCHIVE_FORMATS["xlsx"], _sparse_workbook())
        zip_bomb = _processed(client, "report.docx", ARCHIVE_FORMATS["docx"], bomb)

    assert TOO_LARGE in (workbook.get("error") or ""), (
        f"the built-in loader unpacked a workbook over the 80 MB limit: {workbook}"
    )
    assert TOO_LARGE in (zip_bomb.get("error") or ""), (
        f"the built-in loader unpacked a zip bomb: {zip_bomb}"
    )


def test_a_remote_engine_passing_a_file_type_to_the_built_in_loaders_keeps_the_check(
    document_settings, listener, make_user
):
    """Mistral OCR takes only PDFs, so a workbook goes to the built-in loader and is checked."""
    document_settings(
        CONTENT_EXTRACTION_ENGINE="mistral_ocr",
        MISTRAL_OCR_API_BASE_URL=listener.base_url,
        MISTRAL_OCR_API_KEY="ocr-key",
    )

    with make_user().client() as client:
        data = _processed(client, "plan.xlsx", ARCHIVE_FORMATS["xlsx"], _sparse_workbook())

    assert TOO_LARGE in (data.get("error") or ""), (
        f"a workbook the engine does not take was unpacked over the 80 MB limit: {data}"
    )
    assert listener.received == [], "Mistral OCR was sent a file type it does not take"
