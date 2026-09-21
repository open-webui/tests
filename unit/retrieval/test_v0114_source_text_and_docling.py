"""Regressions in source-text routing and Docling conversion, open-webui v0.11.4.

Three fixes in `open_webui/retrieval/loaders/main.py`, pinned at the loader
dispatch and conversion layer with the extraction servers stubbed:

* `.ino` sketches (fix `acd03b147`, PR #29673, issue #29670): browsers send
  `.ino` as `application/octet-stream` and the extension was missing from
  `known_source_ext`, so an Arduino sketch was handed to the Tika/Docling
  extraction server instead of the plain text reader, and the upload failed.
  `.ino` now routes to the text loader like the `.cpp`/`.h` files beside it.
* Docling conversion failures (fix `0837f310b`, PR #30107, issue #29808): a
  file Docling refuses or fails to convert came back inside an HTTP 200 body,
  so the status check saw success; the upload either crashed with a TypeError
  or quietly stored the `<No text content found>` placeholder and indexed it
  in place of the file. A failure/skipped status now raises with Docling's
  own error messages, and a nullable `md_content` no longer trips the
  page-break split.
* HTML entities in uploaded text (fix `1bfa59acb`, PR #29736, issue #29732):
  `Loader.load` ran every document through `ftfy.fix_text`, whose default
  also decodes HTML entities, so `&nbsp;` and `&gt;` in a file were rewritten
  before being stored and read by the model. Entity unescaping is now off;
  every other ftfy repair stays.

Discriminates: passes on dev 344ea5306, fails on each fix's parent (a `.ino`
upload resolves to the extraction loader, a Docling refusal returns a
placeholder document instead of raising, a null `md_content` raises TypeError,
and `&nbsp;` in an uploaded file is rewritten to a non-breaking space).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.regression

OCTET_STREAM = "application/octet-stream"


@pytest.fixture(scope="session")
def loaders_main_module(owui_module):
    """`open_webui.retrieval.loaders.main` (Loader dispatch, DoclingLoader)."""
    return owui_module("open_webui.retrieval.loaders.main")


def _write(tmp_path, name: str, content: bytes) -> str:
    path = tmp_path / name
    path.write_bytes(content)
    return str(path)


# =============================================================================
# 1. .ino sketches reach the plain text reader (fix acd03b147, PR #29673)
# =============================================================================


@pytest.mark.parametrize(
    "engine,server_key",
    [("tika", "TIKA_SERVER_URL"), ("docling", "DOCLING_SERVER_URL")],
)
def test_ino_upload_routes_to_the_text_loader(
    loaders_main_module, tmp_path, engine, server_key
):
    sketch = _write(tmp_path, "blink.ino", b"void setup() {}\n")
    loader = loaders_main_module.Loader(engine=engine, **{server_key: "http://extractor.local"})

    resolved = loader._get_loader("blink.ino", OCTET_STREAM, sketch)

    assert type(resolved).__name__ == "TextLoader", (
        f"an Arduino sketch under {engine} went to {type(resolved).__name__} instead of "
        "the plain text reader, so the extraction server was asked to parse C++ source"
    )


def test_ino_is_a_known_source_extension(loaders_main_module):
    assert loaders_main_module.Loader()._is_text_file("ino", OCTET_STREAM), (
        "a .ino file with a browser-sent content type is no longer recognized as text"
    )


def test_cpp_and_h_beside_the_sketch_still_route_to_text(loaders_main_module, tmp_path):
    server = {"TIKA_SERVER_URL": "http://extractor.local"}
    for filename in ("helper.cpp", "helper.h", "helper.c"):
        path = _write(tmp_path, filename, b"int x;\n")
        loader = loaders_main_module.Loader(engine="tika", **server)
        resolved = loader._get_loader(filename, OCTET_STREAM, path)
        assert type(resolved).__name__ == "TextLoader", filename


def test_a_binary_extension_still_reaches_the_extraction_server(loaders_main_module, tmp_path):
    blob = _write(tmp_path, "blob.bin", b"\x00\x01")
    loader = loaders_main_module.Loader(
        engine="tika", TIKA_SERVER_URL="http://extractor.local"
    )

    resolved = loader._get_loader("blob.bin", OCTET_STREAM, blob)

    assert type(resolved).__name__ == "TikaLoader", (
        "the media skip widened into binary files, which the text reader cannot read"
    )


def test_an_ino_upload_extracts_the_sketch_verbatim(loaders_main_module, tmp_path):
    sketch_text = "void setup() { pinMode(13, OUTPUT); }\n"
    path = _write(tmp_path, "blink.ino", sketch_text.encode())
    loader = loaders_main_module.Loader(
        engine="tika", TIKA_SERVER_URL="http://extractor.local"
    )

    documents = loader.load("blink.ino", OCTET_STREAM, path)

    assert [doc.page_content for doc in documents] == [sketch_text]


# =============================================================================
# 2. Docling conversions that failed inside an HTTP 200 (fix 0837f310b, PR #30107)
# =============================================================================


class _FakeDoclingResponse:
    def __init__(self, payload: dict) -> None:
        self.ok = True
        self._payload = payload
        self.reason = "OK"
        self.text = ""

    def json(self) -> dict:
        return self._payload


@pytest.fixture
def docling(loaders_main_module, tmp_path, monkeypatch):
    """A DoclingLoader whose HTTP boundary is a stub, recording each conversion."""

    class _Recorder:
        def __init__(self) -> None:
            self.responses: list[_FakeDoclingResponse] = []

        def stub(self, payload: dict) -> None:
            self.responses.append(_FakeDoclingResponse(payload))

    recorder = _Recorder()
    file_path = _write(tmp_path, "doc.dxf", b"binary-ish")

    def fake_post(url, **kwargs):
        assert url.endswith("/v1/convert/file"), url
        return recorder.responses.pop(0)

    monkeypatch.setattr(loaders_main_module.requests, "post", fake_post)

    def _loader():
        return loaders_main_module.DoclingLoader(
            url="http://docling.local", file_path=file_path, mime_type=OCTET_STREAM
        )

    return SimpleNamespace(loader=_loader, stub=recorder.stub)


def test_a_refused_conversion_raises_with_doclings_reason(docling):
    docling.stub(
        {
            "status": "failure",
            "errors": [{"error_message": "File format not allowed: doc.dxf"}],
            "document": {},
        }
    )

    with pytest.raises(Exception, match="File format not allowed: doc.dxf"):
        docling.loader().load()


def test_a_skipped_conversion_raises(docling):
    docling.stub({"status": "skipped", "errors": [], "document": {}})

    with pytest.raises(Exception, match="skipped"):
        docling.loader().load()


def test_a_conversion_failure_without_messages_names_the_status(docling):
    docling.stub({"status": "failure", "errors": [], "document": {}})

    with pytest.raises(Exception, match="status failure - no error message"):
        docling.loader().load()


def test_several_docling_errors_are_all_surfaced(docling):
    docling.stub(
        {
            "status": "failure",
            "errors": [{"error_message": None}, {"error_message": "second failure"}],
            "document": {},
        }
    )

    with pytest.raises(Exception, match="second failure"):
        docling.loader().load()


# ---------------------------------------------------------------------------
# nearby: successful conversions, and the nullable md_content the fix also covers
# ---------------------------------------------------------------------------


def test_a_successful_conversion_still_returns_the_markdown(docling):
    docling.stub({"status": "success", "document": {"md_content": "# Title\n\nBody"}})

    documents = docling.loader().load()

    assert [doc.page_content for doc in documents] == ["# Title\n\nBody"]


def test_a_null_md_content_yields_the_placeholder_not_a_crash(docling):
    """Docling returns JSON null when it was not asked for markdown output."""
    docling.stub({"status": "success", "document": {"md_content": None}})

    documents = docling.loader().load()

    assert [doc.page_content for doc in documents] == ["<No text content found>"]


def test_an_empty_markdown_keeps_the_placeholder(docling):
    docling.stub({"status": "success", "document": {"md_content": ""}})

    documents = docling.loader().load()

    assert [doc.page_content for doc in documents] == ["<No text content found>"]


def test_the_docling_loader_does_not_log_warnings_per_conversion(docling, caplog):
    docling.stub(
        {"status": "failure", "errors": [{"error_message": "not allowed"}], "document": {}}
    )
    caplog.set_level(logging.WARNING)

    with pytest.raises(Exception):
        docling.loader().load()

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# =============================================================================
# 3. HTML entities in uploaded text survive (fix 1bfa59acb, PR #29736)
# =============================================================================

ENTITY_TEXT = "cost &gt; 5\nA&nbsp;B\nC &amp; D\n<xml>markup &gt; stays after a literal tag</xml>\n"


def test_uploaded_entities_reach_the_model_exactly_as_written(loaders_main_module, tmp_path):
    path = _write(tmp_path, "notes.txt", ENTITY_TEXT.encode())
    loader = loaders_main_module.Loader(engine="")

    documents = loader.load("notes.txt", "text/plain", path)

    assert [doc.page_content for doc in documents] == [ENTITY_TEXT], (
        "HTML entities in an uploaded file were decoded before storage, so the model "
        "reads a rewritten copy of the file"
    )


def test_mojibake_repair_still_runs(loaders_main_module, tmp_path):
    """Disabling entity unescaping must not switch off ftfy's other repairs."""
    path = _write(tmp_path, "mojibake.txt", "donâ€™t".encode("utf-8"))
    loader = loaders_main_module.Loader(engine="")

    documents = loader.load("mojibake.txt", "text/plain", path)

    assert [doc.page_content for doc in documents] == ["don't"], (
        "the mojibake repair is off: the windows-1252 curly quote was left broken"
    )


def test_plain_text_is_unchanged_by_the_extraction_pipeline(loaders_main_module, tmp_path):
    text = "plain ascii, 100% unremarkable\nsecond line\n"
    path = _write(tmp_path, "plain.txt", text.encode())
    loader = loaders_main_module.Loader(engine="")

    documents = loader.load("plain.txt", "text/plain", path)

    assert [doc.page_content for doc in documents] == [text]
