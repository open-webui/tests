"""Dependency contract: unstructured (import name ``unstructured``).

``UnstructuredLoader`` in ``retrieval/loaders/local.py`` imports the partitioner for a format by
name and calls it on the uploaded file::

    module = import_module(f'unstructured.partition.{file_format}')
    elements = getattr(module, f'partition_{file_format}')(filename=path, **kwargs)
    Document(page_content=str(element),
             metadata={**element.metadata.to_dict(), 'category': element.category,
                       'element_id': element.id})

``retrieval/loaders/main.py`` sends .doc, .ppt, .pptx, .xls, .xlsx, .rst, .xml, .epub, .msg and
.odt that way; .msg goes out with ``process_attachments=False`` and, when
``unstructured.file_utils.filetype.detect_filetype`` says the file is a plain RFC 822 mail
(``EML``), to ``partition_email`` instead. .doc/.ppt versus .docx/.pptx is also decided by
``detect_filetype(path).name``.

This module pins that surface: every partitioner the loader names exists and takes
``filename``, the .msg pair takes ``process_attachments``, ``detect_filetype`` names what the
loader branches on, and the elements an XML and an email partition return carry the text and
data model the loader reads. The formats themselves are read end to end over HTTP in
integration/deps/test_document_extraction.py.

Discriminates: with ``detect_filetype`` answering ``UNK`` for every file, or ``partition_xml``
returning no elements (a pytest plugin patching the installed library), the matching tests
go red.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import importlib
import io

import pytest

pytestmark = pytest.mark.depcheck

LOADER_FORMATS = ["doc", "ppt", "pptx", "xlsx", "rst", "xml", "epub", "msg", "email", "odt"]

TEXT = "The harbour lighthouse budget was approved."

MAIL = (
    "From: Alice <alice@example.com>\n"
    "To: Bob <bob@example.com>\n"
    "Subject: Quarterly numbers\n"
    "MIME-Version: 1.0\n"
    'Content-Type: text/plain; charset="utf-8"\n'
    "\n"
    f"{TEXT}\n"
)


def _partitioner(depcheck, file_format: str):
    depcheck.load("unstructured")
    # a partition module that no longer imports is breakage, not an absent package
    module = importlib.import_module(f"unstructured.partition.{file_format}")
    return getattr(module, f"partition_{file_format}")


def _detect_filetype(depcheck):
    return depcheck.resolve(depcheck.load("unstructured"), "file_utils.filetype.detect_filetype")


@pytest.mark.parametrize("file_format", LOADER_FORMATS)
def test_every_partitioner_the_loader_names_takes_a_filename(depcheck, file_format):
    partition = _partitioner(depcheck, file_format)

    assert callable(partition)
    depcheck.assert_params(partition, ["filename"])


@pytest.mark.parametrize("file_format", ["msg", "email"])
def test_the_mail_partitioners_take_process_attachments(depcheck, file_format):
    depcheck.assert_params(_partitioner(depcheck, file_format), ["process_attachments"])


def test_detect_filetype_tells_a_mail_saved_as_msg_apart(depcheck, tmp_path):
    mail = tmp_path / "mail.msg"
    mail.write_text(MAIL)

    assert _detect_filetype(depcheck)(str(mail)).name == "EML"


def test_detect_filetype_names_a_pptx(depcheck, tmp_path):
    pptx = depcheck.load("pptx")
    presentation = pptx.Presentation()
    presentation.slides.add_slide(presentation.slide_layouts[5]).shapes.title.text = TEXT
    path = tmp_path / "slides.pptx"
    presentation.save(str(path))

    # the loader compares the lower-cased name against "ppt"
    assert _detect_filetype(depcheck)(str(path)).name.lower() == "pptx"


def test_partition_xml_returns_elements_the_loader_can_read(depcheck, tmp_path):
    path = tmp_path / "notes.xml"
    path.write_text(f"<?xml version='1.0'?><notes><note>{TEXT}</note></notes>")

    elements = _partitioner(depcheck, "xml")(filename=str(path))

    assert TEXT in "\n\n".join(map(str, elements))
    for element in elements:
        assert isinstance(element.metadata.to_dict(), dict)
        assert isinstance(element.category, str)
        assert isinstance(element.id, str)


def test_partition_email_reads_the_body_without_attachments(depcheck, tmp_path):
    path = tmp_path / "mail.eml"
    path.write_text(MAIL)

    elements = _partitioner(depcheck, "email")(filename=str(path), process_attachments=False)

    assert TEXT in "\n\n".join(map(str, elements))


def test_partition_xlsx_reads_a_workbook(depcheck, tmp_path):
    """The encryption check through msoffcrypto, then pandas on openpyxl."""
    openpyxl = depcheck.load("openpyxl")
    workbook = openpyxl.Workbook()
    workbook.active.append(["note", TEXT])
    buffer = io.BytesIO()
    workbook.save(buffer)
    path = tmp_path / "budget.xlsx"
    path.write_bytes(buffer.getvalue())

    elements = _partitioner(depcheck, "xlsx")(filename=str(path))

    assert TEXT in "\n\n".join(map(str, elements))
