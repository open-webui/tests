"""Dependency contract: docx2txt.

``docx2txt`` is the plain-text extractor behind LangChain's
``Docx2txtLoader``, which the Open WebUI backend uses to ingest ``.docx``
uploads in the retrieval pipeline
(``retrieval/loaders/main.py`` -> ``Docx2txtLoader(file_path)``). The
backend never imports ``docx2txt`` directly — it reaches it transitively
through ``langchain_community`` — so a breaking change to its single
public entry point (``docx2txt.process``) would surface as empty/failed
Word-document ingestion rather than at import time.

This module pins that one load-bearing function and its signature, then
exercises the real extraction path offline by building minimal but valid
``.docx`` packages in memory (a ``.docx`` is a zip of OOXML parts) and
asserting the text comes back. No network, no temp-file dependence for the
core contract.

Pattern mirrors test_requests.py. Uses ``depcheck`` from conftest.py.
"""

from __future__ import annotations

import inspect
import zipfile
from io import BytesIO

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "docx2txt"
DIST_NAME = "docx2txt"


# ---------------------------------------------------------------------------
# Helpers: build a minimal valid .docx (OOXML zip) in memory.
# ---------------------------------------------------------------------------

_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" '
    'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" '
    'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
    "</Types>"
)

_ROOT_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" '
    'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
    'Target="word/document.xml"/>'
    "</Relationships>"
)


def _make_docx(paragraphs: list[str]) -> BytesIO:
    """Assemble a valid single-section .docx containing the given paragraphs."""
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    )
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CONTENT_TYPES)
        z.writestr("_rels/.rels", _ROOT_RELS)
        z.writestr("word/document.xml", document_xml)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Import + version
# ---------------------------------------------------------------------------


def test_import(depcheck):
    """`docx2txt` must import (skip cleanly if absent)."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "docx2txt"


def test_version_reported(depcheck):
    """The installed distribution version must resolve."""
    assert depcheck.dist_version(DIST_NAME) is not None


# ---------------------------------------------------------------------------
# Symbol-existence + signature (the single public entry point).
# ---------------------------------------------------------------------------


def test_process_exists_and_callable(depcheck):
    """`docx2txt.process` is what LangChain's Docx2txtLoader calls; it must
    exist and be callable."""
    mod = depcheck.load(IMPORT_NAME)
    assert hasattr(mod, "process")
    assert callable(mod.process)


def test_process_signature(depcheck):
    """process(docx, img_dir=None) — the loader calls process(path). The first
    positional parameter (the docx source) must remain; img_dir stays optional."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.process)
    params = list(sig.parameters)
    assert params, "docx2txt.process lost its parameters"
    assert params[0] in ("docx", "docx_file", "file"), f"unexpected first param: {params}"
    # img_dir must remain optional (no extra required positional args).
    required = [
        n
        for n, p in sig.parameters.items()
        if p.default is inspect.Parameter.empty
        and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    assert len(required) == 1, f"process now requires extra args: {required}"


# ---------------------------------------------------------------------------
# Behavioural contracts (OFFLINE) — real extraction from in-memory .docx.
# ---------------------------------------------------------------------------


def test_behaviour_extracts_single_paragraph(depcheck):
    """A one-paragraph .docx must extract exactly that text — the core contract
    LangChain relies on for Word ingestion."""
    mod = depcheck.load(IMPORT_NAME)
    docx = _make_docx(["Hello Open WebUI extraction test"])
    text = mod.process(docx)
    assert isinstance(text, str)
    assert "Hello Open WebUI extraction test" in text


def test_behaviour_extracts_multiple_paragraphs(depcheck):
    """Multiple paragraphs must all appear in the extracted text (order
    preserved), so multi-paragraph documents are fully ingested."""
    mod = depcheck.load(IMPORT_NAME)
    paras = ["First paragraph here", "Second paragraph follows", "Third and final"]
    text = mod.process(_make_docx(paras))
    for p in paras:
        assert p in text, f"missing paragraph: {p!r}"
    # Order: first paragraph appears before the last.
    assert text.index(paras[0]) < text.index(paras[-1])


def test_behaviour_accepts_file_like_object(depcheck):
    """The loader can hand process() a file-like (BytesIO); zip-based reading
    must work on an in-memory stream, not only a path."""
    mod = depcheck.load(IMPORT_NAME)
    bio = _make_docx(["Stream based content"])
    text = mod.process(bio)
    assert "Stream based content" in text


def test_behaviour_accepts_path(depcheck, tmp_path):
    """Docx2txtLoader(file_path) passes a filesystem path; process() must read
    a .docx from disk. (tmp_path is pytest's per-test temp dir — local, no net.)"""
    mod = depcheck.load(IMPORT_NAME)
    p = tmp_path / "doc.docx"
    p.write_bytes(_make_docx(["Path based content"]).getvalue())
    text = mod.process(str(p))
    assert "Path based content" in text


def test_behaviour_empty_document_returns_string(depcheck):
    """A .docx with no text paragraphs must still return a (possibly empty)
    string, not raise — so empty uploads don't crash ingestion."""
    mod = depcheck.load(IMPORT_NAME)
    docx = _make_docx([])
    text = mod.process(docx)
    assert isinstance(text, str)


def test_behaviour_unicode_content_preserved(depcheck):
    """Non-ASCII text (the common case for real documents) must survive
    extraction intact."""
    mod = depcheck.load(IMPORT_NAME)
    sample = "Grüße café — 日本語 текст"
    text = mod.process(_make_docx([sample]))
    # Each distinctive token should survive; em-dash handling can vary so check
    # the surrounding unicode words.
    for token in ("Grüße", "café", "日本語", "текст"):
        assert token in text, f"unicode token lost: {token!r}"
