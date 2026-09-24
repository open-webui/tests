"""Regression: a document container that unpacks far beyond its stored size is refused.

open-webui 0.11.1 fix `2a0274a0a`: Office, OpenDocument and EPUB uploads (`docx`, `xlsx`,
`pptx`, `odt`, `epub`) are zip containers, and the loader handed them straight to their
extractor, which inflated every member into the worker's memory. A few kilobytes on disk could
declare gigabytes. The fix reads the zip directory first and refuses the file with "Document
archive is too large after decompression" when the declared total exceeds
`min(max(10 MiB, stored size * 100), FILE_MAX_SIZE)`, with the admin's file size limit now
passed through to the loader.

Twin of unit/security/test_archive_expansion_limit.py. Bounded by construction: every archive
declares at most 16 MiB, so the pre-fix run inflates no more than that.

Discriminates: passes on dev bbfa876af; with the size check removed from `Loader._get_loader`
the 16 MiB archives reach their extractor and fail (or pass) for other reasons, and with
`FILE_MAX_SIZE` no longer passed to the loader a 1 MiB limit refuses nothing.
"""

from __future__ import annotations

import io
import zipfile

import pytest

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


def test_a_pdf_is_not_run_through_the_archive_check(make_user):
    bomb = _archive_declaring(16 * MIB)
    with make_user().client() as client:
        data = _processed(client, "scan.pdf", "application/pdf", bomb)

    assert TOO_LARGE not in (data.get("error") or ""), data


def test_an_unreadable_archive_is_left_to_the_extractor(make_user):
    with make_user().client() as client:
        data = _processed(client, "broken.docx", ARCHIVE_FORMATS["docx"], b"not a zip at all")

    assert TOO_LARGE not in (data.get("error") or ""), data
