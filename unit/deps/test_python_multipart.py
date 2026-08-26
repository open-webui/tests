"""Dependency contract: python-multipart (import name ``multipart``).

``python-multipart`` is the streaming multipart/form-data parser that
Starlette (and therefore FastAPI) uses to decode request bodies into the
``UploadFile`` / ``Form(...)`` / ``File(...)`` objects the Open WebUI
backend relies on for every file upload and form-encoded endpoint (file
ingestion, audio upload, image upload, OpenAI/Ollama proxy multipart
bodies, etc.). The backend never imports ``multipart`` directly — it is a
*transitive* dependency reached through ``starlette.formparsers`` — so a
breaking bump would surface as a 500/422 deep in request parsing rather
than at import time.

This module therefore pins the public parsing surface that Starlette
binds to (``MultipartParser`` / ``FormParser`` / ``parse_form`` and the
``Field`` / ``File`` result objects), plus an offline behavioural
round-trip that feeds a real multipart body through the parser and checks
the decoded field/file callbacks — no network, no server.

Pattern mirrors test_requests.py: symbol-existence + signature checks for
the API surface, plus offline behavioural contracts. Uses the
``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import inspect
import re
from io import BytesIO

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "multipart"
DIST_NAME = "python-multipart"

# Public symbols Starlette's formparsers resolve on the package (and that
# any direct consumer would use). These are the load-bearing parser
# entry-points and result classes.
TOP_LEVEL_SYMBOLS = [
    "MultipartParser",  # starlette.formparsers.MultiPartParser wraps this
    "FormParser",  # high-level content-type dispatching parser
    "QuerystringParser",  # urlencoded form bodies
    "OctetStreamParser",  # raw binary bodies
    "BaseParser",  # shared base
    "parse_form",  # convenience one-shot parse
    "create_form_parser",  # factory used to build a FormParser from headers
]

# Symbols on the multipart.multipart submodule (where the real classes
# live; the top-level package re-exports a subset).
SUBMODULE_SYMBOLS = [
    "MultipartParser",
    "FormParser",
    "QuerystringParser",
    "OctetStreamParser",
    "Field",  # decoded form field result object
    "File",  # decoded file result object
    "parse_form",
    "create_form_parser",
    "MultipartParseError",  # raised on malformed bodies
    "FormParserError",
    "QuerystringParseError",
]


# ---------------------------------------------------------------------------
# Import + version
# ---------------------------------------------------------------------------


def test_import(depcheck):
    """`multipart` must import (skip cleanly if absent in this env)."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "multipart"


def test_version_reported(depcheck):
    """The installed distribution (python-multipart) version must resolve."""
    assert depcheck.dist_version(DIST_NAME) is not None


def test_submodule_imports(depcheck):
    """The real parser classes live in multipart.multipart; Starlette imports
    `from multipart.multipart import parse_options_header` etc."""
    sub = depcheck.load("multipart.multipart")
    assert sub.__name__ in ("multipart.multipart", "python_multipart.multipart")


# ---------------------------------------------------------------------------
# Symbol-existence checks (API surface)
# ---------------------------------------------------------------------------


def test_top_level_symbols_exist(depcheck):
    """Every parser entry-point Starlette binds must exist on the package."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, TOP_LEVEL_SYMBOLS)


def test_submodule_symbols_exist(depcheck):
    """The multipart.multipart submodule must expose the parser + result
    classes (Field/File) and the error types."""
    sub = depcheck.load("multipart.multipart")
    depcheck.assert_symbols(sub, SUBMODULE_SYMBOLS)


def test_parse_options_header_available(depcheck):
    """Starlette calls multipart.multipart.parse_options_header to split the
    Content-Type into the mimetype + boundary params. Pin it."""
    sub = depcheck.load("multipart.multipart")
    assert hasattr(sub, "parse_options_header")
    assert callable(sub.parse_options_header)


def test_parser_entrypoints_callable(depcheck):
    """The parser classes/functions must be constructable/callable."""
    mod = depcheck.load(IMPORT_NAME)
    for name in ("MultipartParser", "FormParser", "parse_form", "create_form_parser"):
        depcheck.assert_callable(mod, name)


# ---------------------------------------------------------------------------
# Signature contracts — Starlette constructs these with specific kwargs.
# ---------------------------------------------------------------------------


def test_multipart_parser_signature(depcheck):
    """starlette builds MultipartParser(boundary, callbacks=..., max_size=...).
    Those parameters must remain accepted."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.MultipartParser.__init__, ["boundary", "callbacks"])


def test_form_parser_signature(depcheck):
    """FormParser(content_type, on_field, on_file, ...) is the high-level entry
    Starlette's create_form_parser path uses; pin the core callback params."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.FormParser.__init__,
        ["content_type", "on_field", "on_file"],
    )


def test_parse_form_signature(depcheck):
    """parse_form(headers, input_stream, on_field, on_file, chunk_size=...) is
    the one-shot helper; pin its parameters (used in our behavioural test)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.parse_form,
        ["headers", "input_stream", "on_field", "on_file"],
    )


def test_create_form_parser_signature(depcheck):
    """create_form_parser(headers, on_field, on_file, ...) returns a FormParser
    configured from the request headers — the path Starlette ultimately uses."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.create_form_parser,
        ["headers", "on_field", "on_file"],
    )


def test_multipart_parser_has_write_and_finalize(depcheck):
    """Starlette feeds chunks via parser.write(chunk) and finishes with
    parser.finalize(). Both must exist on the parser class."""
    mod = depcheck.load(IMPORT_NAME)
    for name in ("write", "finalize"):
        assert hasattr(mod.MultipartParser, name), f"MultipartParser.{name} missing"
        assert callable(getattr(mod.MultipartParser, name))


# ---------------------------------------------------------------------------
# Field / File result-object contracts.
# ---------------------------------------------------------------------------


def test_field_result_object_contract(depcheck):
    """A decoded form Field exposes field_name + value + write/finalize. These
    back Starlette's `Form(...)` extraction."""
    sub = depcheck.load("multipart.multipart")
    field = sub.Field("name")
    for name in ("field_name", "value", "write", "finalize"):
        assert hasattr(field, name), f"Field.{name} missing"
    assert field.field_name == b"name" or field.field_name == "name"


def test_file_result_object_contract(depcheck):
    """A decoded File exposes field_name / file_name / file_object / size and
    write/finalize — the shape Starlette maps into UploadFile."""
    sub = depcheck.load("multipart.multipart")
    # File requires a config dict (it reads on_disk/buffer limits from it).
    f = sub.File(b"upload.bin", config={})
    for name in ("field_name", "file_name", "file_object", "write", "finalize", "size"):
        assert hasattr(f, name), f"File.{name} missing"
    try:
        f.finalize()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Behavioural contracts (OFFLINE) — feed a real multipart body and assert
# the decoded fields/files match. This is exactly the parse Starlette does
# for an UploadFile + Form endpoint.
# ---------------------------------------------------------------------------

_BOUNDARY = b"----WebKitFormBoundaryABC123"


def _build_multipart_body() -> bytes:
    """A two-part body: one text field, one file part (CRLF line endings as
    required by RFC 7578 / the wire format browsers send)."""
    b = _BOUNDARY
    return (
        b"--" + b + b"\r\n"
        b'Content-Disposition: form-data; name="title"\r\n'
        b"\r\n"
        b"hello world\r\n"
        b"--" + b + b"\r\n"
        b'Content-Disposition: form-data; name="upload"; filename="a.txt"\r\n'
        b"Content-Type: text/plain\r\n"
        b"\r\n"
        b"file-contents-here\r\n"
        b"--" + b + b"--\r\n"
    )


def test_behaviour_multipart_parser_roundtrip(depcheck):
    """Drive MultipartParser directly with callbacks and reconstruct a field +
    a file part from the raw bytes — the low-level contract Starlette depends
    on (part headers -> data -> end-of-part events)."""
    sub = depcheck.load("multipart.multipart")

    parts: list[dict] = []
    cur: dict = {}

    def on_part_begin():
        cur.clear()
        cur["headers"] = {}
        cur["data"] = bytearray()

    def on_header_field(data, start, end):
        cur.setdefault("_hf", bytearray())
        cur["_hf"] += data[start:end]

    def on_header_value(data, start, end):
        cur.setdefault("_hv", bytearray())
        cur["_hv"] += data[start:end]

    def on_header_end():
        cur["headers"][bytes(cur.pop("_hf", b"")).lower()] = bytes(cur.pop("_hv", b""))

    def on_part_data(data, start, end):
        cur["data"] += data[start:end]

    def on_part_end():
        parts.append({"headers": dict(cur["headers"]), "data": bytes(cur["data"])})

    callbacks = {
        "on_part_begin": on_part_begin,
        "on_header_field": on_header_field,
        "on_header_value": on_header_value,
        "on_header_end": on_header_end,
        "on_part_data": on_part_data,
        "on_part_end": on_part_end,
    }
    parser = sub.MultipartParser(_BOUNDARY, callbacks=callbacks)
    body = _build_multipart_body()
    parser.write(body)
    parser.finalize()

    assert len(parts) == 2, f"expected 2 parts, got {len(parts)}: {parts}"
    # First part: the text field.
    assert parts[0]["data"] == b"hello world"
    assert b"content-disposition" in parts[0]["headers"]
    # Second part: the file content.
    assert parts[1]["data"] == b"file-contents-here"
    assert b"text/plain" in parts[1]["headers"].get(b"content-type", b"")


def test_behaviour_parse_form_field_and_file(depcheck):
    """Use the high-level parse_form() helper (headers + stream -> on_field /
    on_file callbacks) — the same composition create_form_parser builds for
    Starlette. Assert the field value and file name/contents decode."""
    mod = depcheck.load(IMPORT_NAME)
    body = _build_multipart_body()
    headers = {
        "Content-Type": b"multipart/form-data; boundary=" + _BOUNDARY,
        "Content-Length": str(len(body)).encode(),
    }

    fields: list = []
    files: list = []

    def on_field(field):
        fields.append((field.field_name, field.value))

    def on_file(file):
        file.finalize()
        file.file_object.seek(0)
        files.append((file.field_name, file.file_name, file.file_object.read()))

    mod.parse_form(headers, BytesIO(body), on_field, on_file)

    # Exactly one field and one file decoded.
    assert len(fields) == 1, f"fields={fields}"
    assert len(files) == 1, f"files={files}"
    fname, fval = fields[0]
    assert fname == b"title"
    assert fval == b"hello world"
    file_field, file_name, file_data = files[0]
    assert file_field == b"upload"
    assert file_name == b"a.txt"
    assert file_data == b"file-contents-here"


def test_behaviour_querystring_parser_roundtrip(depcheck):
    """urlencoded form bodies route through QuerystringParser (Starlette's
    non-multipart form path). Decode a=1&b=two and assert the fields."""
    sub = depcheck.load("multipart.multipart")
    fields: list = []
    names: dict = {}

    def on_field_name(data, start, end):
        names.setdefault("n", bytearray())
        names["n"] += data[start:end]

    def on_field_data(data, start, end):
        names.setdefault("v", bytearray())
        names["v"] += data[start:end]

    def on_field_end():
        fields.append((bytes(names.pop("n", b"")), bytes(names.pop("v", b""))))

    parser = sub.QuerystringParser(
        callbacks={
            "on_field_name": on_field_name,
            "on_field_data": on_field_data,
            "on_field_end": on_field_end,
        }
    )
    parser.write(b"a=1&b=two")
    parser.finalize()
    decoded = dict(fields)
    assert decoded.get(b"a") == b"1"
    assert decoded.get(b"b") == b"two"


def test_behaviour_parse_options_header_splits_boundary(depcheck):
    """Starlette extracts the boundary via parse_options_header(content_type).
    Verify it returns (mimetype, {b'boundary': ...})."""
    sub = depcheck.load("multipart.multipart")
    mime, options = sub.parse_options_header(b"multipart/form-data; boundary=" + _BOUNDARY)
    assert mime == b"multipart/form-data"
    assert options.get(b"boundary") == _BOUNDARY


def test_behaviour_malformed_body_does_not_crash_interpreter(depcheck):
    """A truncated/garbage body must surface as a parse error or simply yield
    no complete parts — never a segfault or silent partial accept that
    Starlette would treat as a valid form."""
    sub = depcheck.load("multipart.multipart")
    seen = []
    parser = sub.MultipartParser(
        _BOUNDARY,
        callbacks={"on_part_end": lambda: seen.append(1)},
    )
    try:
        parser.write(b"--" + _BOUNDARY + b"\r\nthis is not a valid part header")
        parser.finalize()
    except sub.MultipartParseError:
        return  # acceptable: explicit parse error
    # If no error was raised, the incomplete part must not have been emitted.
    assert seen == [], "incomplete multipart part was emitted as complete"


def test_starlette_binds_python_multipart(depcheck):
    """Sanity: confirm the transitive linkage actually holds — Starlette's
    formparsers module references this package, so this contract guards the
    real consumer and not a coincidental install."""
    fp = depcheck.try_load("starlette.formparsers")
    if fp is None:
        pytest.skip("starlette not installed; surface tests still cover the API")
    src = inspect.getsource(fp)
    assert "multipart" in src, "starlette.formparsers no longer references multipart"


def test_requirements_pin_is_at_or_above_the_security_floor(open_webui_backend):
    """The checkout under test must not pin python-multipart below 0.0.32.

    A security advisory in the multipart parsing path (#26991) was fixed by moving to
    0.0.32. A floor rather than an exact
    version, so an ordinary bump stays quiet and only a downgrade past the fix
    is reported.
    """
    requirements = (open_webui_backend / "requirements.txt").read_text(encoding="utf-8")
    pinned = re.search(
        r"^python\-multipart(?:\[[^\]]*\])?==([0-9][^\s#]*)", requirements, re.MULTILINE
    )

    assert pinned, "python-multipart is not pinned in requirements.txt"
    assert _version_tuple(pinned.group(1)) >= _version_tuple("0.0.32"), (
        f"python-multipart is pinned at {pinned.group(1)}, below the 0.0.32 that fixed "
        "a security advisory in the multipart parsing path (#26991)"
    )


def _version_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", raw))
