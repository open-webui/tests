"""The orjson codec's options and the stream line reader's limit settings, both fixed in v0.11.1.

* Commit 78ed5a0235: `ORJSONCodec.dumps` and `loads` swallowed their options, so `indent`,
  `sort_keys`, `ensure_ascii`, `separators` and `object_hook` had no effect. Any option outside
  orjson's own defaults now falls back to engineio's stdlib-backed codec.
* Commit a33fa05adc: `stream_chunks_handler` handed back aiohttp's raw reader when
  `CHAT_STREAM_RESPONSE_CHUNK_MAX_BUFFER_SIZE` meant "no limit", whose own line limit then aborted
  long lines. It now always assembles lines itself.

The integration twin pins what users see (a note's JSON, a separator in a streamed reply, one
oversized line on default settings). This file covers what it cannot reach: every option
spelling, `loads` options, which no route passes, and the "no limit" spellings other than unset.
The codec picks its implementation at import, so it is probed in a child interpreter booted with
`ENABLE_ORJSON=true` rather than by re-executing the module here.

Discriminates: passes on bbfa876af; dropping the option fallback from `ORJSONCodec` fails the
option and `object_hook` tests, and returning the raw reader for a disabled limit fails the
disabled-limit tests.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys

import aiohttp
import pytest
from aiohttp.base_protocol import BaseProtocol

pytestmark = pytest.mark.regression

PAYLOAD = {"b": [1, {"z": "é", "a": None}], "a": True}
OPTIONS = {
    "indent": {"indent": 4},
    "sort_keys": {"sort_keys": True},
    "ensure_ascii": {"ensure_ascii": True},
    "separators": {"separators": (" | ", " -> ")},
    "indent_and_sort_keys": {"indent": 2, "sort_keys": True},
}

CODEC_PROBE = """
import ast, json, sys

sys.path.insert(0, sys.argv[1])
from open_webui.utils.json_codec import JSONCodec

payload, options = ast.literal_eval(sys.argv[2])
print(json.dumps({
    "codec": JSONCodec.__name__,
    "with_options": {name: JSONCodec.dumps(payload, **kwargs) for name, kwargs in options.items()},
    "compact": JSONCodec.dumps(payload),
    "hooked": JSONCodec.loads('{"a": 1}', object_hook=lambda item: {**item, "hooked": True}),
    "int_keys": JSONCodec.loads(JSONCodec.dumps({1: "int key"})),
    "with_default": JSONCodec.dumps({"x": {2, 1}}, default=sorted),
}))
"""


@pytest.fixture(scope="module")
def orjson_codec(open_webui_backend) -> dict:
    """What the checkout's codec returns for each probe, with `ENABLE_ORJSON=true`."""
    probed = subprocess.run(
        [sys.executable, "-c", CODEC_PROBE, str(open_webui_backend), repr((PAYLOAD, OPTIONS))],
        env={**os.environ, "ENABLE_ORJSON": "true"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert probed.returncode == 0, f"the codec probe failed:\n{probed.stderr[-3000:]}"
    results = json.loads(probed.stdout.splitlines()[-1])
    assert results["codec"] == "ORJSONCodec", (
        f"ENABLE_ORJSON=true selected {results['codec']}; retarget the probe at the orjson codec"
    )
    return results


@pytest.mark.parametrize("name", sorted(OPTIONS))
def test_dumps_with_an_option_matches_stdlib(orjson_codec, name):
    expected = json.dumps(PAYLOAD, **OPTIONS[name])

    assert orjson_codec["with_options"][name] == expected, (
        f"ORJSONCodec.dumps ignored {OPTIONS[name]} and wrote orjson's compact output"
    )


def test_loads_honours_object_hook(orjson_codec):
    assert orjson_codec["hooked"] == {"a": 1, "hooked": True}


def test_dumps_without_options_stays_compact_raw_utf8(orjson_codec):
    assert orjson_codec["compact"] == json.dumps(PAYLOAD, separators=(",", ":"), ensure_ascii=False)


def test_what_orjson_rejects_still_falls_back_to_stdlib(orjson_codec):
    assert orjson_codec["int_keys"] == {"1": "int key"}
    assert orjson_codec["with_default"] == '{"x": [1, 2]}'


def _read_lines(misc_module, monkeypatch, chunks: list[bytes], max_buffer_size) -> list[bytes]:
    """Feed a real aiohttp reader, whose own line limit is 64 bytes, through the handler."""
    monkeypatch.setattr(misc_module, "CHAT_STREAM_RESPONSE_CHUNK_MAX_BUFFER_SIZE", max_buffer_size)

    async def read() -> list[bytes]:
        loop = asyncio.get_running_loop()
        reader = aiohttp.StreamReader(BaseProtocol(loop), limit=64, loop=loop)
        for chunk in chunks:
            reader.feed_data(chunk)
        reader.feed_eof()
        return [line async for line in misc_module.stream_chunks_handler(reader)]

    return asyncio.run(read())


@pytest.mark.parametrize("max_buffer_size", [None, 0, -1])
def test_a_long_line_survives_every_spelling_of_no_limit(misc_module, monkeypatch, max_buffer_size):
    long_line = b"data: " + b"y" * 500 + b"\n"

    lines = _read_lines(
        misc_module, monkeypatch, [long_line[:200], long_line[200:]], max_buffer_size
    )

    assert lines == [long_line]


def test_a_configured_limit_drops_only_the_oversized_line(misc_module, monkeypatch):
    chunks = [b"small\n", b"data: " + b"z" * 200 + b"\n", b"after\n"]

    assert _read_lines(misc_module, monkeypatch, chunks, 50) == [b"small\n", b"after\n"]


def test_an_empty_stream_yields_nothing(misc_module, monkeypatch):
    assert _read_lines(misc_module, monkeypatch, [], None) == []
