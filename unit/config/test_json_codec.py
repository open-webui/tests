"""The orjson codec and the streamed-chunk line reader, both fixed in v0.11.1.

Three regressions:

* PR #27819 (commit 1d6735ff0), "Reliable streaming with unusual characters".
  stdlib json escapes U+2028, U+2029 and U+0085; orjson writes them raw. Python treats all
  three as line boundaries, so one of them inside a serialized payload split the SSE frame
  that the consumer reassembles with ``splitlines()``. ``ORJSONCodec.dumps`` now escapes
  them back before returning.
* Commit 78ed5a0235, "JSON options honoured again". ``ORJSONCodec.dumps``/``loads`` swallowed
  ``*args``/``**kwargs``, so ``indent``, ``sort_keys``, ``ensure_ascii`` and ``object_hook``
  had no effect. Any argument outside orjson's own defaults now falls back to engineio's
  stdlib-backed codec.
* Commit a33fa05adc, "Replies arriving in oversized pieces". ``stream_chunks_handler``
  returned the raw aiohttp reader when ``CHAT_STREAM_RESPONSE_CHUNK_MAX_BUFFER_SIZE`` was
  unset, so on default settings a line over aiohttp's own limit aborted the stream with
  ``LineTooLong``. The handler now always assembles lines itself.

Discriminates: passes on v0.11.1, fails on v0.11.0 (raw separators survive serialization,
dumps/loads kwargs are ignored, and the unset-buffer stream is handed back unwrapped).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json as stdlib_json
from pathlib import Path
from unittest import mock

import aiohttp
import pytest

pytestmark = pytest.mark.regression

LINE_SEPARATORS = ("\u2028", "\u2029", "\x85")


@pytest.fixture(scope="module")
def orjson_codec(open_webui_backend: Path, owui_module):
    """``ORJSONCodec`` from the checkout, loaded with ``ENABLE_ORJSON`` forced on.

    The codec picks its branch at import time and the flag defaults to off, so the module is
    executed a second time under its own name. It holds no state anything else shares.
    """
    pytest.importorskip("orjson")
    env = owui_module("open_webui.env")
    source = Path(open_webui_backend) / "open_webui" / "utils" / "json_codec.py"
    if not source.is_file():
        pytest.skip("open_webui/utils/json_codec.py not present in this checkout")

    spec = importlib.util.spec_from_file_location("owui_json_codec_orjson_test", source)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.object(env, "ENABLE_ORJSON", True):
        spec.loader.exec_module(module)
    return module.JSONCodec


# ── 36: line separators must not survive serialization ────────────────────────────────


@pytest.mark.parametrize("separator", LINE_SEPARATORS)
def test_sse_frame_survives_line_separator_in_payload(orjson_codec, separator):
    """Narrow: a payload carrying one of the three separators still reassembles as one frame."""
    payload = {"choices": [{"delta": {"content": f"before{separator}after"}}]}
    frame = f"data: {orjson_codec.dumps(payload)}\n\n"

    data_lines = [line for line in frame.splitlines() if line.startswith("data: ")]

    assert len(data_lines) == 1
    assert stdlib_json.loads(data_lines[0][len("data: ") :]) == payload


@pytest.mark.parametrize("separator", LINE_SEPARATORS)
def test_dumps_emits_no_raw_line_separator(orjson_codec, separator):
    """Narrow: the escaping is on the serialized text itself, not only on the SSE framing."""
    serialized = orjson_codec.dumps({"text": f"a{separator}b"})

    assert separator not in serialized
    assert len(serialized.splitlines()) == 1


def test_all_line_separators_escaped_in_one_payload(orjson_codec):
    """Broad: no serialized output ever contains a character Python reads as a line break."""
    serialized = orjson_codec.dumps({str(i): f"x{sep}y" for i, sep in enumerate(LINE_SEPARATORS)})

    assert not any(sep in serialized for sep in LINE_SEPARATORS)
    assert len(serialized.splitlines()) == 1


@pytest.mark.parametrize("separator", LINE_SEPARATORS)
def test_line_separators_round_trip(orjson_codec, separator):
    """Nearby: escaping them must not change the decoded value."""
    payload = {"text": f"a{separator}b"}

    assert orjson_codec.loads(orjson_codec.dumps(payload)) == payload
    assert stdlib_json.loads(orjson_codec.dumps(payload)) == payload


def test_ordinary_text_is_untouched(orjson_codec):
    """Nearby: text without a separator keeps orjson's compact raw-UTF-8 output."""
    assert orjson_codec.dumps({"text": "héllo\nworld"}) == '{"text":"héllo\\nworld"}'


# ── 37: dumps/loads options must be honoured ──────────────────────────────────────────


def test_dumps_honours_indent(orjson_codec):
    """Narrow: ``indent`` was silently dropped."""
    serialized = orjson_codec.dumps({"a": 1, "b": 2}, indent=2)

    assert "\n" in serialized
    assert serialized == stdlib_json.dumps({"a": 1, "b": 2}, indent=2)


def test_dumps_honours_sort_keys(orjson_codec):
    """Narrow: ``sort_keys`` was silently dropped."""
    assert orjson_codec.dumps({"b": 1, "a": 2}, sort_keys=True) == '{"a": 2, "b": 1}'


def test_dumps_honours_ensure_ascii(orjson_codec):
    """Narrow: orjson always writes raw UTF-8, so ``ensure_ascii=True`` had no effect."""
    assert orjson_codec.dumps({"text": "é"}, ensure_ascii=True) == '{"text": "\\u00e9"}'


def test_loads_honours_object_hook(orjson_codec):
    """Narrow: ``loads`` kwargs were dropped too."""
    result = orjson_codec.loads('{"a": 1}', object_hook=lambda d: {**d, "hooked": True})

    assert result == {"a": 1, "hooked": True}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"indent": 4},
        {"sort_keys": True},
        {"ensure_ascii": True},
        {"separators": (" | ", " -> ")},
        {"indent": 2, "sort_keys": True},
    ],
)
def test_dumps_with_options_matches_stdlib(orjson_codec, kwargs):
    """Broad: any option outside orjson's defaults must produce exactly stdlib's output."""
    payload = {"b": [1, {"z": "é", "a": None}], "a": True}

    assert orjson_codec.dumps(payload, **kwargs) == stdlib_json.dumps(payload, **kwargs)


@pytest.mark.parametrize("kwargs", [{}, {"separators": (",", ":")}, {"ensure_ascii": False}])
def test_compact_defaults_keep_the_fast_path(orjson_codec, kwargs):
    """Nearby: orjson's own defaults stay compact, spelled explicitly or not."""
    assert orjson_codec.dumps({"a": 1, "b": [2, 3]}, **kwargs) == '{"a":1,"b":[2,3]}'


def test_dumps_falls_back_for_unsupported_types(orjson_codec):
    """Nearby: the existing engineio fallback for what orjson rejects still runs."""
    assert orjson_codec.loads(orjson_codec.dumps({1: "int key"})) == {"1": "int key"}
    assert orjson_codec.dumps({"x": {1, 2}}, default=sorted) == '{"x": [1, 2]}'


def test_loads_accepts_plain_payloads(orjson_codec):
    """Nearby: the no-kwargs path is unchanged."""
    assert orjson_codec.loads('{"a": [1, 2], "b": null}') == {"a": [1, 2], "b": None}
    assert orjson_codec.loads(b'{"a": 1}') == {"a": 1}


# ── 20: oversized streamed chunks on default settings ─────────────────────────────────


def _reader(limit: int, chunks: list[bytes]) -> aiohttp.StreamReader:
    """A real aiohttp reader over pre-fed chunks, with aiohttp's own line limit lowered."""
    reader = aiohttp.StreamReader(
        mock.Mock(_reading_paused=False), limit=limit, loop=asyncio.get_running_loop()
    )
    for chunk in chunks:
        reader.feed_data(chunk)
    reader.feed_eof()
    return reader


def _run(misc_module, limit, chunks, max_buffer_size, want_reader_identity=False):
    """Drive the handler over a real reader; returns the chunks, or (chunks, is_raw_reader)."""

    async def main():
        reader = _reader(limit, chunks)
        with mock.patch.object(
            misc_module, "CHAT_STREAM_RESPONSE_CHUNK_MAX_BUFFER_SIZE", max_buffer_size
        ):
            stream = misc_module.stream_chunks_handler(reader)
        is_raw_reader = stream is reader
        out = [chunk async for chunk in stream]
        return (out, is_raw_reader) if want_reader_identity else out

    return asyncio.run(main())


def test_oversized_line_survives_with_no_buffer_limit_configured(misc_module):
    """Narrow: unset limit used to hand back the raw reader, whose line limit aborts the stream."""
    long_line = b"data: " + b"x" * 4096
    chunks = [long_line[:1000], long_line[1000:] + b"\n", b"data: [DONE]\n"]

    out = _run(misc_module, limit=64, chunks=chunks, max_buffer_size=None)

    assert out == [long_line + b"\n", b"data: [DONE]\n"]


@pytest.mark.parametrize("max_buffer_size", [None, 0, -1])
def test_disabled_buffer_limit_never_returns_the_raw_reader(misc_module, max_buffer_size):
    """Broad: every spelling of "no limit" must still go through the line assembler."""
    line = b"data: " + b"y" * 500 + b"\n"

    out, is_raw_reader = _run(
        misc_module,
        limit=64,
        chunks=[line],
        max_buffer_size=max_buffer_size,
        want_reader_identity=True,
    )

    assert not is_raw_reader
    assert out == [line]


def test_lines_split_across_chunks_are_reassembled(misc_module):
    """Nearby: the normal case, a line arriving in pieces, with a limit configured."""
    chunks = [b'data: {"a"', b': 1}\ndata: {"b"', b": 2}\n"]

    out = _run(misc_module, limit=1024, chunks=chunks, max_buffer_size=10_000)

    assert out == [b'data: {"a": 1}\n', b'data: {"b": 2}\n']


def test_oversized_line_is_dropped_when_a_limit_is_configured(misc_module):
    """Nearby: a configured limit still drops the offending line and keeps the rest."""
    chunks = [b"small\n", b"data: " + b"z" * 200 + b"\n", b"after\n"]

    out = _run(misc_module, limit=1024, chunks=chunks, max_buffer_size=50)

    joined = b"".join(out)

    assert b"small" in joined
    assert b"after" in joined
    assert b"z" * 200 not in joined


def test_empty_stream_yields_nothing(misc_module):
    """Nearby: an empty upstream response is not turned into a spurious chunk."""
    assert _run(misc_module, limit=1024, chunks=[], max_buffer_size=None) == []
    assert _run(misc_module, limit=1024, chunks=[b""], max_buffer_size=10_000) == []
