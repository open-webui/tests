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

Issue #34944: a valid workbook whose pivot table covers whole columns stores about a million
mostly empty pivot cache records, so a 4 MB file declared about 100 MB unpacked and was refused
for every extraction engine. The pivot cache records are not sheet text, so Open WebUI now
empties them in a copy before extraction and leaves them out of the size check: the built-in
loaders read the workbook, a remote engine gets the small copy, and the stored file stays as
uploaded. An archive whose other parts are over the limit is still refused.

Twin of unit/security/test_archive_expansion_limit.py. Bounded by construction: every archive
declares at most 16 MiB, so the pre-fix run inflates no more than that. The pivot workbook only
ever reaches the reader that refuses it on the pre-fix ref, and the one over the limit is refused.

Discriminates: passes on dev 3cda94d40; with the size check removed from `Loader._get_loader`
the 16 MiB archives reach their extractor and fail (or pass) for other reasons, with
`FILE_MAX_SIZE` no longer passed to the loader a 1 MiB limit refuses nothing, and on 4adf3083f^
(archives still told apart by extension) the zip named `scan.pdf` is not refused. The two pivot
workbook tests fail on dev d89cd809c (#34944 unfixed: the workbook is refused as too large and
the external engine is never called) and pass with the pivot cache records emptied.
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
from integration.retrieval.test_document_loaders import pdf_bytes, xlsx_bytes

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


SPREADSHEET_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
RELATIONSHIPS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_RELATIONSHIPS = "http://schemas.openxmlformats.org/package/2006/relationships"
SPREADSHEET_PARTS = "application/vnd.openxmlformats-officedocument.spreadsheetml"
PIVOT_RECORDS = "xl/pivotCache/pivotCacheRecords1.xml"
PIVOT_FIELDS = 21
SIGHTINGS = [["Harbour", "Herons"], ["North quay", 1200], ["South quay", 800]]


def _relationship(rel_id: str, kind: str, target: str) -> str:
    return (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<Relationships xmlns="{PACKAGE_RELATIONSHIPS}"><Relationship Id="{rel_id}" '
        f'Type="{RELATIONSHIPS}/{kind}" Target="{target}"/></Relationships>'
    )


@functools.cache
def _pivot_workbook() -> bytes:
    """A valid workbook whose pivot table covers whole columns, as in #34944.

    Excel then caches a million mostly empty records: 4 MB on disk, about 100 MB unpacked.
    """
    seeded = random.Random(34944)
    empty_fields = b"<m/>" * (PIVOT_FIELDS - 1)
    records = (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<pivotCacheRecords xmlns="{SPREADSHEET_MAIN}" count="1000000">'.encode()
        + b"".join(
            b'<r><n v="' + str(seeded.randrange(65536)).encode() + b'"/>' + empty_fields + b"</r>"
            for _ in range(1_000_000)
        )
        + b"</pivotCacheRecords>"
    )
    cache_fields = "".join(
        f'<cacheField name="F{index}" numFmtId="0"><sharedItems containsBlank="1"/></cacheField>'
        for index in range(1, PIVOT_FIELDS + 1)
    )
    cache_definition = (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<pivotCacheDefinition xmlns="{SPREADSHEET_MAIN}" xmlns:r="{RELATIONSHIPS}" r:id="rId1" '
        f'refreshOnLoad="1" recordCount="1000000"><cacheSource type="worksheet">'
        f'<worksheetSource ref="A1:U1000001" sheet="Sightings"/></cacheSource>'
        f'<cacheFields count="{PIVOT_FIELDS}">{cache_fields}</cacheFields></pivotCacheDefinition>'
    )
    pivot_table = (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<pivotTableDefinition xmlns="{SPREADSHEET_MAIN}" name="PivotTable1" cacheId="1" '
        f'dataCaption="Values"><location ref="A3:A4" firstHeaderRow="1" firstDataRow="1" '
        f'firstDataCol="0"/><pivotFields count="{PIVOT_FIELDS}">'
        + '<pivotField dataField="1" showAll="0"/>'
        + '<pivotField showAll="0"/>' * (PIVOT_FIELDS - 1)
        + '</pivotFields><dataFields count="1"><dataField name="Sum of F1" fld="0"/></dataFields>'
        "</pivotTableDefinition>"
    )
    parts = {
        "xl/worksheets/_rels/sheet2.xml.rels": _relationship(
            "rId1", "pivotTable", "../pivotTables/pivotTable1.xml"
        ),
        "xl/pivotTables/pivotTable1.xml": pivot_table,
        "xl/pivotTables/_rels/pivotTable1.xml.rels": _relationship(
            "rId1", "pivotCacheDefinition", "../pivotCache/pivotCacheDefinition1.xml"
        ),
        "xl/pivotCache/pivotCacheDefinition1.xml": cache_definition,
        "xl/pivotCache/_rels/pivotCacheDefinition1.xml.rels": _relationship(
            "rId1", "pivotCacheRecords", "pivotCacheRecords1.xml"
        ),
        PIVOT_RECORDS: records,
    }
    content_types = "".join(
        f'<Override PartName="/{part}" ContentType="{SPREADSHEET_PARTS}.{kind}+xml"/>'
        for part, kind in [
            ("xl/pivotTables/pivotTable1.xml", "pivotTable"),
            ("xl/pivotCache/pivotCacheDefinition1.xml", "pivotCacheDefinition"),
            (PIVOT_RECORDS, "pivotCacheRecords"),
        ]
    )
    additions = {
        "[Content_Types].xml": ("</Types>", content_types),
        "xl/workbook.xml": (
            "</workbook>",
            f'<pivotCaches><pivotCache xmlns:r="{RELATIONSHIPS}" cacheId="1" '
            f'r:id="rIdPivotCache1"/></pivotCaches>',
        ),
        "xl/_rels/workbook.xml.rels": (
            "</Relationships>",
            f'<Relationship Id="rIdPivotCache1" Type="{RELATIONSHIPS}/pivotCacheDefinition" '
            f'Target="pivotCache/pivotCacheDefinition1.xml"/>',
        ),
    }

    openpyxl = pytest.importorskip("openpyxl")
    sheets = openpyxl.load_workbook(io.BytesIO(xlsx_bytes(SIGHTINGS)))
    sheets.create_sheet("Pivot")
    plain = io.BytesIO()
    sheets.save(plain)
    buffer = io.BytesIO()
    with (
        zipfile.ZipFile(plain) as source,
        zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as workbook,
    ):
        for name in source.namelist():
            content = source.read(name)
            if name in additions:
                closing_tag, addition = additions[name]
                content = content.replace(closing_tag.encode(), (addition + closing_tag).encode())
            workbook.writestr(name, content)
        for name, content in parts.items():
            workbook.writestr(name, content)
    return buffer.getvalue()


def _unpacked_size(archive: bytes) -> int:
    return sum(entry.file_size for entry in zipfile.ZipFile(io.BytesIO(archive)).infolist())


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


def _uploaded_and_processed(client, filename: str, payload: bytes) -> dict:
    """Like `_processed`, but returns the whole stored file, its id included."""
    uploaded = client.post(
        "/api/v1/files/?process=true&process_in_background=false",
        files={"file": (filename, payload, ARCHIVE_FORMATS["xlsx"])},
    )
    assert uploaded.status_code == 200, uploaded.text
    return client.get(f"/api/v1/files/{uploaded.json()['id']}").json()


def test_the_built_in_loaders_read_a_workbook_with_a_large_pivot_cache(
    document_settings, make_user
):
    workbook = _pivot_workbook()
    assert len(workbook) < 5 * MIB and 80 * MIB < _unpacked_size(workbook) < len(workbook) * 100, (
        "the workbook must be over the 80 MB limit unpacked yet under 100 times its size"
    )
    document_settings(CONTENT_EXTRACTION_ENGINE="")

    with make_user().client() as client:
        stored = _uploaded_and_processed(client, "sightings.xlsx", workbook)
        downloaded = client.get(f"/api/v1/files/{stored['id']}/content")

    data = stored["data"]
    assert TOO_LARGE not in (data.get("error") or ""), (
        f"a valid workbook with a large pivot cache was refused: {data}"
    )
    for cell in ("North quay", "1200", "South quay", "800"):
        assert cell in (data.get("content") or ""), f"the sheet text was not extracted: {data}"
    assert downloaded.status_code == 200, downloaded.text
    assert downloaded.content == workbook, "the stored workbook no longer matches the upload"


def test_the_external_engine_gets_the_workbook_without_its_pivot_cache_records(
    document_settings, listener, make_user
):
    openpyxl = pytest.importorskip("openpyxl")
    workbook = _pivot_workbook()
    listener.route(
        "PUT",
        "/process",
        json_answer({"page_content": "Herons by harbour.", "metadata": {}}),
    )
    document_settings(
        CONTENT_EXTRACTION_ENGINE="external",
        EXTERNAL_DOCUMENT_LOADER_URL=listener.base_url,
        EXTERNAL_DOCUMENT_LOADER_API_KEY="loader-key",
    )

    with make_user().client() as client:
        data = _uploaded_and_processed(client, "sightings.xlsx", workbook)["data"]

    received = listener.requests_to("/process")
    assert TOO_LARGE not in (data.get("error") or ""), (
        f"a workbook the external engine reads was refused before it was sent there: {data}"
    )
    assert len(received) == 1, f"the external engine was called {len(received)} times"
    assert received[0].headers["X-Filename"].endswith("sightings.xlsx"), received[0].headers
    sent = zipfile.ZipFile(io.BytesIO(received[0].body))
    original = zipfile.ZipFile(io.BytesIO(workbook))
    assert sent.namelist() == original.namelist(), "the sent workbook lost or gained parts"
    changed = [name for name in original.namelist() if sent.read(name) != original.read(name)]
    assert changed == [PIVOT_RECORDS], (
        f"parts other than the pivot cache records changed: {changed}"
    )
    assert _unpacked_size(received[0].body) < MIB, "the pivot cache records were sent in full"
    sent_sheets = openpyxl.load_workbook(io.BytesIO(received[0].body))
    assert [list(row) for row in sent_sheets["Sightings"].iter_rows(values_only=True)] == SIGHTINGS
    assert data.get("content") == "Herons by harbour.", (
        f"the text the external engine extracted was not stored: {data}"
    )


def test_a_pivot_workbook_whose_other_parts_are_too_large_is_still_refused(
    document_settings, listener, make_user
):
    """The pivot cache is left out of the size check, everything else still counts."""
    padded = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(_pivot_workbook())) as source,
        zipfile.ZipFile(padded, "w", zipfile.ZIP_DEFLATED) as target,
    ):
        for entry in source.infolist():
            target.writestr(entry, source.read(entry))
        with target.open("xl/media/image1.bin", "w") as member:
            for _ in range(81):
                member.write(b"\0" * MIB)
    listener.route("PUT", "/process", json_answer({"page_content": "unused", "metadata": {}}))
    document_settings(
        CONTENT_EXTRACTION_ENGINE="external",
        EXTERNAL_DOCUMENT_LOADER_URL=listener.base_url,
        EXTERNAL_DOCUMENT_LOADER_API_KEY="loader-key",
    )

    with make_user().client() as client:
        data = _processed(client, "sightings.xlsx", ARCHIVE_FORMATS["xlsx"], padded.getvalue())

    assert data.get("status") == "failed" and TOO_LARGE in (data.get("error") or ""), (
        f"a workbook with 81 MiB outside its pivot cache got past the 80 MB limit: {data}"
    )
    assert listener.received == [], "a workbook over the limit was sent to the engine"
