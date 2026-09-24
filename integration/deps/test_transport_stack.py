"""Dependency smoke: how responses and live updates travel, each library through its feature.

starlette-compress compresses a response in the encoding the client asks for, with Brotli,
gzip and zstandard doing the work. /api/changelog is a large public response that shows it, and
its body is CHANGELOG.md turned into HTML by Markdown and split into versions by BeautifulSoup.
aiohttp decodes a Brotli-encoded provider reply with brotlicffi (Brotli when that is absent).
python-socketio carries the chat events to the browser, and pycrdt merges the live edits two tabs
make to one note.

Discriminates: passes on dev bbfa876af; in a backend copy, dropping `CompressMiddleware` fails
every encoding, skipping `markdown.markdown` empties the changelog, a provider session with
`auto_decompress=False` hands the Brotli bytes to the stream parser, emitting chat events to a
room other than `user:{id}` starves the socket of them and not applying the stored updates before
`ydoc.get_update()` sends the second tab an empty document.
"""

from __future__ import annotations

import contextlib
import json
import re
import time
from typing import Iterator

import brotli
import httpx
import pycrdt
import pytest

from harness import raw_provider
from harness import upstream as reply
from harness.chat import ask
from harness.instance import resolve_backend
from harness.raw_provider import RAW_MODEL_ID, chunk, sse
from harness.second_provider import OPENAI_CONFIG
from harness.socket_client import SocketSession, connected

pytestmark = [pytest.mark.depcheck, pytest.mark.api, pytest.mark.requires_source]

VERSION_HEADING = re.compile(r"^## \[(?P<version>[^\]]+)\] - (?P<date>.+)$", re.MULTILINE)
SECTION_HEADING = re.compile(r"^### (?P<section>.+)$", re.MULTILINE)
STATE_WAIT = 10.0


def _changelog(instance, encoding: str) -> httpx.Response:
    with httpx.Client(base_url=instance.base_url, timeout=60.0) as client:
        return client.get("/api/changelog", headers={"Accept-Encoding": encoding})


@pytest.mark.parametrize("encoding", ["br", "gzip", "zstd"])
def test_the_changelog_is_compressed_in_the_encoding_asked_for(instance, encoding):
    plain = _changelog(instance, "identity")
    compressed = _changelog(instance, encoding)

    assert compressed.status_code == 200, compressed.text
    assert compressed.headers.get("content-encoding") == encoding
    assert compressed.num_bytes_downloaded < plain.num_bytes_downloaded
    assert compressed.json() == plain.json()


def _changelog_source() -> str:
    """The CHANGELOG.md the server reads: the checkout's, else the one packaged with it."""
    backend = resolve_backend()
    for candidate in (backend.parent / "CHANGELOG.md", backend / "open_webui" / "CHANGELOG.md"):
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    raise AssertionError(f"no CHANGELOG.md next to {backend}")


def test_the_changelog_is_read_from_the_markdown(instance):
    source = _changelog_source()
    headings = list(VERSION_HEADING.finditer(source))
    newest_body = source[headings[0].end() : headings[1].start()]

    changelog = _changelog(instance, "identity").json()

    assert list(changelog) == [heading["version"] for heading in headings[:5]]
    assert [entry["date"] for entry in changelog.values()] == [
        heading["date"].strip() for heading in headings[:5]
    ]
    newest = changelog[headings[0]["version"]]
    sections = [match["section"].strip().lower() for match in SECTION_HEADING.finditer(newest_body)]
    assert [name for name in newest if name != "date"] == sections
    items = [item for name in sections for item in newest[name]]
    assert items and all(item["raw"].startswith("<li>") for item in items)
    assert any("<strong>" in item["raw"] for item in items), "the bold markdown was not rendered"


@pytest.fixture
def raw(admin, preserve, listener) -> raw_provider.RawProvider:
    preserve(OPENAI_CONFIG)
    return raw_provider.connect(admin, listener)


def test_a_brotli_encoded_provider_reply_is_decoded(admin, raw, listener):
    body = sse(chunk({"role": "assistant", "content": ""}), chunk({"content": "decoded"}))
    headers = {"Content-Type": "text/event-stream", "Content-Encoding": "br"}
    listener.route("POST", "/v1/chat/completions", (200, headers, brotli.compress(body)))

    with admin.client() as client:
        _, message = ask(client, "hello?", model=RAW_MODEL_ID)

    sent = listener.requests_to("/v1/chat/completions")[-1]
    assert "br" in sent.headers.get("Accept-Encoding", ""), "the provider was not offered Brotli"
    assert message["content"] == "decoded"


def test_a_socket_joins_its_account_and_receives_the_chat_events(make_user, upstream):
    account = make_user()
    upstream.queue(reply.text("over the socket"))

    with connected(account) as socket, account.client() as client:
        joined = socket.client.call("user-join", {"auth": {"token": account.token}}, timeout=30)
        turn, _ = ask(client, "hello?")
        socket.wait_for(turn.chat_id, "chat:completion", done=True)
        pushed = json.dumps(socket.events_of(turn.chat_id))

    assert joined == {"id": account.id, "name": account.name}
    assert "over the socket" in pushed, "the streamed reply never reached the socket"


def _text_of(state: list[int]) -> str:
    document = pycrdt.Doc()
    document.apply_update(bytes(state))
    return str(document.get("content", type=pycrdt.Text))


def _edit(text: str) -> list[int]:
    """The update a tab sends after typing `text` into an empty document."""
    document = pycrdt.Doc()
    document["content"] = content = pycrdt.Text()
    content += text
    return list(document.get_update())


@contextlib.contextmanager
def _document_tab(account, document_id: str) -> Iterator[tuple[SocketSession, list[list[int]]]]:
    """A socket joined to the note's live document, keeping every state the server sends it."""
    with connected(account) as socket:
        states: list[list[int]] = []
        socket.client.on("ydoc:document:state", lambda payload: states.append(payload["state"]))
        socket.call("ydoc:document:join", {"document_id": document_id})
        yield socket, states


def _a_state_reads(states: list[list[int]], text: str) -> bool:
    deadline = time.monotonic() + STATE_WAIT
    while time.monotonic() < deadline:
        if any(_text_of(state) == text for state in list(states)):
            return True
        time.sleep(0.05)
    return False


def test_two_tabs_editing_one_note_see_the_document_the_server_merged(make_user):
    account = make_user()
    with account.client() as client:
        note = client.post(
            "/api/v1/notes/create", json={"title": "shared", "data": {"content": {"md": ""}}}
        )
    assert note.status_code == 200, note.text
    document_id = f"note:{note.json()['id']}"

    with (
        _document_tab(account, document_id) as (first, _),
        _document_tab(account, document_id) as (second, states),
    ):
        first.call("ydoc:document:update", {"document_id": document_id, "update": _edit("hi")})
        second.call("ydoc:document:state", {"document_id": document_id})

        assert _a_state_reads(states, "hi"), "the second tab never got the merged document"
