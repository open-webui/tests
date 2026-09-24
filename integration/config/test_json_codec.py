"""Regression: streamed replies and stored JSON under the orjson codec, and oversized stream lines.

Three open-webui v0.11.1 fixes:

* PR #27819 (commit 1d6735ff0): stdlib json escapes U+2028, U+2029 and U+0085, orjson writes them
  raw. Python reads all three as line breaks, so under `ENABLE_ORJSON` a chunk the server
  re-serialized (every chunk, once a stream filter is installed) split into lines that no longer
  parse for a consumer reading the stream with `splitlines()`. `dumps` now escapes them.
* Commit 78ed5a0235: the orjson codec swallowed `dumps` and `loads` options, so a note stored
  with structured markdown came back as one compact line instead of the indented JSON block the
  note sanitizer asks for.
* Commit a33fa05adc: with `CHAT_STREAM_RESPONSE_CHUNK_MAX_BUFFER_SIZE` unset (the default) the
  provider stream was read through aiohttp's own line reader, whose limit aborted any reply that
  arrived as one line over 128 KiB.

The codec's option handling across every spelling stays in unit/config/test_json_codec.py.

Twin of unit/config/test_json_codec.py.

Discriminates: passes on bbfa876af; dropping the separator escaping from `ORJSONCodec.dumps` fails
the separator test, dropping its option fallback fails the note test, and handing back the raw
reader for an unset buffer size fails the oversized line test. The plain-text test passes on both.
"""

from __future__ import annotations

import json

import pytest

from harness import upstream as reply
from harness.actors import admin_of
from harness.chat import ask
from harness.plugins import installed_function
from harness.upstream import MOCK_MODEL_ID

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

ORJSON = {"ENABLE_ORJSON": "true"}
LINE_SEPARATORS = ("\u2028", "\u2029", "\x85")

# Makes the server parse and re-serialize every streamed chunk.
PASS_THROUGH_STREAM_FILTER = """
class Filter:
    def stream(self, event):
        return event
"""


@pytest.fixture
def orjson_instance(instance_with):
    return instance_with(ORJSON)


def _streamed_frames(instance, pieces: list[str]) -> list[str]:
    """The data lines of a filtered streamed reply, split the way `str.splitlines()` splits."""
    instance.upstream.queue(reply.text(pieces))
    admin = admin_of(instance)
    request = {
        "model": MOCK_MODEL_ID,
        "messages": [{"role": "user", "content": "say it"}],
        "stream": True,
    }
    with installed_function(admin, PASS_THROUGH_STREAM_FILTER, is_global=True):
        with admin.client() as client:
            streamed = client.post("/api/chat/completions", json=request)
    assert streamed.status_code == 200, streamed.text
    return [
        line.removeprefix("data:").strip()
        for line in streamed.text.splitlines()
        if line.strip() and line.strip() != "data: [DONE]"
    ]


def _content(frames: list[str]) -> str:
    chunks = [json.loads(frame) for frame in frames]
    return "".join(
        choice["delta"].get("content") or "" for chunk in chunks for choice in chunk["choices"]
    )


def test_line_separators_in_a_reply_keep_every_frame_parseable(orjson_instance):
    pieces = [f"before{separator}after " for separator in LINE_SEPARATORS]

    frames = _streamed_frames(orjson_instance, pieces)

    unparseable = []
    for frame in frames:
        try:
            json.loads(frame)
        except ValueError:
            unparseable.append(frame)
    assert unparseable == [], (
        "a line separator in the reply was written raw, so the frame split into lines that do "
        f"not parse (#27819): {unparseable}"
    )
    assert _content(frames) == "".join(pieces)


def test_plain_text_streams_through_the_codec_unchanged(orjson_instance):
    pieces = ["héllo ", "wörld\n", "done"]

    assert _content(_streamed_frames(orjson_instance, pieces)) == "".join(pieces)


def test_a_note_with_structured_markdown_is_stored_as_indented_json(orjson_instance):
    structured = {"steps": ["mix", "bake"], "serves": 4}

    with admin_of(orjson_instance).client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={"title": "Recipe", "data": {"content": {"md": structured}}},
        )
    assert created.status_code == 200, created.text

    stored = created.json()["data"]["content"]["md"]
    body = stored.removeprefix("```json\n").removesuffix("\n```")
    assert json.loads(body) == structured
    assert len(body.splitlines()) > 1, (
        "the note's JSON was stored on one line: the orjson codec dropped the indent option "
        f"the sanitizer passed: {stored!r}"
    )


def test_a_reply_arriving_as_one_oversized_line_is_stored_whole(user, upstream):
    oversized = "x" * 200_000
    upstream.queue(reply.text(oversized))

    with user.client() as client:
        _, message = ask(client, "one very long line please")

    assert message["content"] == oversized, (
        "a reply that arrived as one line over aiohttp's limit was cut off on default settings, "
        f"because the stream was read through the raw reader: {len(message['content'])} chars"
    )
