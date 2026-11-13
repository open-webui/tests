"""Journey: what a terminal answers reaches the person who asked, through Open WebUI's proxy.

The file browser, previews, downloads and the port preview all call the terminal through
`/api/v1/terminals/<id>/<path>`. This module pins what comes back on a plain terminal: the
terminal's own status and body, which of its headers are kept (every cookie it sets among them,
since edecad351 for open-webui/open-webui#34092), that downloads and live updates
arrive while the terminal is still sending, what a broken-off answer and a connection without
an address give, the chat form of the address, and the port preview's rewriting of redirects
and pages. The request side (credentials, cookies, forged headers, traversal, the admin API)
is pinned in `integration/security/test_terminal_*`.

The terminal that answers piece by piece is a `listener`, since the fake terminal sends each
answer whole.

Discriminates: passes on dev c15f28649. In backend copies of it, one edit each:
* the chat id of the address no longer read (`request.path_params.get('chat_id')` dropped):
  the chat address test fails (no `x-session-id`);
* a terminal status of 400 or more answered as 502: the status test fails for all six;
* nothing dropped from the terminal's headers (`STRIPPED_RESPONSE_HEADERS` empty): the header
  test fails (the packed answer cannot be read);
* `text/event-stream`, then `application/octet-stream`, taken out of the streamed types: the
  live update case, then the download case, fails (nothing arrives while the terminal holds);
* every answer streamed: the whole JSON answer test fails (no length, sent as it comes);
* the check for an address dropped: the empty address test fails (502 in place of 503);
* the last `except Exception` arm of the proxy dropped: the broken-off answer test fails (500);
* the terminal's headers collected in a plain dict before the case-insensitive one, as before
  edecad351: both cases of the cookie test fail on dev 485103cde (only the last cookie arrives);
* the `Location` rewrite dropped: the port redirect test fails;
* the page rewrite dropped (`_port_browser_html` not called): the port page test fails.
"""

from __future__ import annotations

import gzip
import threading
import uuid
from typing import Iterator

import httpx
import pytest

from harness.chat_history import seed_chat
from harness.listener import json_answer, text_answer
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    FakeTerminalServer,
    configure_terminals,
    read_grant,
    serving_terminal,
)

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

# how long a piece may take from the terminal to the person
PIECE_BOUND_SECONDS = 15


@pytest.fixture
def terminal() -> Iterator[FakeTerminalServer]:
    with serving_terminal() as server:
        yield server


@pytest.fixture
def person(make_user):
    return make_user()


def _shared(admin, person, server, **fields) -> str:
    """The admin saves `server` as the only terminal, granted to `person`; returns its id."""
    connection = server.connection(config={"access_grants": [read_grant(person.id)]}, **fields)
    with admin.client() as client:
        configure_terminals(client, connection)
    server.clear()
    return connection["id"]


@pytest.fixture
def terminal_id(admin, preserve, person, terminal) -> str:
    preserve(TERMINAL_SERVERS_CONFIG)
    return _shared(admin, person, terminal)


def test_a_call_under_a_chats_address_names_that_chat_to_the_terminal(
    person, terminal, terminal_id
):
    terminal.route("GET", "/files/cwd", json_answer({"cwd": "/home/user"}))
    with person.client() as client:
        chat_id, _ = seed_chat(client, [{"role": "user", "content": "where am I"}])
        in_chat = client.get(f"/api/v1/terminals/{terminal_id}/chats/{chat_id}/files/cwd")
        outside = client.get(f"/api/v1/terminals/{terminal_id}/files/cwd")

    assert (in_chat.status_code, in_chat.json()) == (200, {"cwd": "/home/user"}), (
        f"the call under the chat's address was not answered: {in_chat.status_code} {in_chat.text}"
    )
    assert outside.status_code == 200, outside.text
    named = [request.headers.get("x-session-id") for request in terminal.requests_to("/files/cwd")]
    assert named == [chat_id, None], (
        f"the terminal was told the chats {named}; the first call was made under chat {chat_id} "
        "and the second under none"
    )


@pytest.mark.parametrize("status", [401, 403, 404, 409, 500, 503])
def test_a_refusal_or_failure_of_the_terminal_reaches_the_person_as_it_is(
    person, terminal, terminal_id, status
):
    reason = {"detail": f"the terminal says {status} {uuid.uuid4().hex[:8]}"}
    terminal.route("POST", "/files/write", json_answer(reason, status=status))

    with person.client() as client:
        answered = client.post(
            f"/api/v1/terminals/{terminal_id}/files/write", json={"path": "a", "content": "b"}
        )

    assert (answered.status_code, answered.json()) == (status, reason), (
        f"the terminal answered {status} {reason}; the person got {answered.status_code} "
        f"{answered.text}"
    )


def test_the_terminals_own_headers_arrive_except_those_of_its_connection(
    person, terminal, terminal_id
):
    report = f"quarterly report {uuid.uuid4().hex}".encode()
    terminal.route(
        "GET",
        "/files/view",
        (
            200,
            {
                "Content-Type": "text/plain; charset=utf-8",
                "Content-Encoding": "gzip",
                "Content-Disposition": 'attachment; filename="report.txt"',
                "Cache-Control": "no-store",
                "ETag": '"report-1"',
                "X-Terminal-Note": "kept",
            },
            gzip.compress(report),
        ),
    )

    with person.client() as client:
        answered = client.get(f"/api/v1/terminals/{terminal_id}/files/view")

    assert answered.status_code == 200, answered.text
    kept = {
        name: answered.headers.get(name)
        for name in ("content-type", "content-disposition", "cache-control", "etag")
    }
    assert kept == {
        "content-type": "text/plain; charset=utf-8",
        "content-disposition": 'attachment; filename="report.txt"',
        "cache-control": "no-store",
        "etag": '"report-1"',
    }, f"the terminal's headers did not arrive as it sent them: {dict(answered.headers)}"
    assert answered.headers.get("x-terminal-note") == "kept", dict(answered.headers)
    assert answered.content == report, (
        "the packed answer of the terminal did not arrive as the text it holds: "
        f"{answered.content[:60]!r}"
    )
    assert "content-encoding" not in answered.headers, (
        f"the unpacked answer is still labelled as packed: {dict(answered.headers)}"
    )
    assert answered.headers["content-length"] == str(len(report)), dict(answered.headers)
    servers = answered.headers.get_list("server")
    assert servers == ["uvicorn"], (
        f"the answer names the servers {servers}; the terminal's own server line passed through"
    )
    assert len(answered.headers.get_list("date")) == 1, (
        f"the answer carries the terminal's date as well: {answered.headers.get_list('date')}"
    )


class _RepeatedHeaders(list):
    """Header pairs a terminal sends as they are, a name more than once included."""

    def items(self):
        return iter(self)


@pytest.mark.parametrize(
    "path, content_type",
    [("files/session", "application/json"), ("files/download", "application/octet-stream")],
    ids=["buffered", "streamed"],
)
def test_every_cookie_the_terminal_sets_reaches_the_person(
    person, terminal, terminal_id, path, content_type
):
    """Fixed on dev in edecad351 (open-webui/open-webui#34092): only the last cookie arrived."""
    tag = uuid.uuid4().hex[:8]
    cookies = [f"workspace={tag}; Path=/", f"layout=split-{tag}; Path=/", "theme=dark; Path=/"]
    terminal.route(
        "GET",
        f"/{path}",
        (
            200,
            _RepeatedHeaders(
                [("Content-Type", content_type), *(("Set-Cookie", cookie) for cookie in cookies)]
            ),
            b"{}",
        ),
    )

    with person.client() as client:
        answered = client.get(f"/api/v1/terminals/{terminal_id}/{path}")

    assert answered.status_code == 200, answered.text
    assert answered.headers.get_list("set-cookie") == cookies, (
        f"the terminal set the cookies {cookies}; the person got "
        f"{answered.headers.get_list('set-cookie')}"
    )


def _held_answer(content_type: str, first: bytes, rest: bytes, release: threading.Event):
    """A terminal's answer whose second piece waits for `release`."""

    def pieces() -> Iterator[bytes]:
        yield first
        release.wait(PIECE_BOUND_SECONDS * 2)
        yield rest

    return lambda _request: (200, {"Content-Type": content_type}, pieces())


@pytest.mark.parametrize(
    "path, content_type",
    [
        ("files/download", "application/octet-stream"),
        ("files/events", "text/event-stream"),
        ("files/photo", "image/png"),
        ("files/manual", "application/pdf"),
    ],
    ids=["download", "live updates", "image", "pdf"],
)
def test_a_download_or_live_update_arrives_while_the_terminal_is_still_sending(
    admin, preserve, person, listener, path, content_type
):
    preserve(TERMINAL_SERVERS_CONFIG)
    terminal_id = _shared(admin, person, _AsTerminal(listener))
    first, rest = f"first {uuid.uuid4().hex}".encode(), f"rest {uuid.uuid4().hex}".encode()
    release = threading.Event()
    listener.route("GET", f"/{path}", _held_answer(content_type, first, rest, release))

    try:
        with (
            person.client() as client,
            client.stream(
                "GET",
                f"/api/v1/terminals/{terminal_id}/{path}",
                timeout=httpx.Timeout(120.0, read=PIECE_BOUND_SECONDS),
            ) as answered,
        ):
            pieces = answered.iter_raw()
            try:
                early = next(pieces)
            except httpx.ReadTimeout:
                early = None
            assert early == first, (
                f"nothing of the {content_type} answer reached the person while the terminal was "
                f"still sending it; got {early!r}"
            )
            assert answered.status_code == 200, answered.status_code
            assert answered.headers["content-type"] == content_type, dict(answered.headers)
            release.set()
            assert b"".join(pieces) == rest, "the rest of the answer did not follow"
    finally:
        release.set()


class _AsTerminal:
    """A `listener` saved as a terminal connection, for answers sent piece by piece."""

    def __init__(self, listener):
        self.base_url = listener.base_url
        self._listener = listener

    def connection(self, **fields) -> dict:
        return {
            "id": f"terminal-{uuid.uuid4().hex[:8]}",
            "name": "Piecewise terminal",
            "enabled": True,
            "url": self.base_url,
            "path": "/openapi.json",
            "key": "",
            "auth_type": "none",
            "forward_cookies": False,
            **fields,
        }

    def clear(self) -> None:
        with self._listener.lock:
            self._listener.received.clear()


def test_a_json_answer_arrives_whole_once_the_terminal_has_sent_all_of_it(
    admin, preserve, person, listener
):
    preserve(TERMINAL_SERVERS_CONFIG)
    terminal_id = _shared(admin, person, _AsTerminal(listener))
    release = threading.Event()
    release.set()
    listener.route(
        "GET", "/files/list", _held_answer("application/json", b'{"entries": [', b"]}", release)
    )

    with person.client() as client:
        answered = client.get(f"/api/v1/terminals/{terminal_id}/files/list")

    assert (answered.status_code, answered.json()) == (200, {"entries": []}), (
        f"an answer the terminal sent in two pieces arrived as {answered.status_code} "
        f"{answered.text!r}"
    )
    assert answered.headers.get("content-length") == str(len(b'{"entries": []}')), (
        f"the answer was not passed on as one body of known length: {dict(answered.headers)}"
    )


def test_a_connection_saved_without_an_address_says_so(admin, preserve, person, terminal):
    preserve(TERMINAL_SERVERS_CONFIG)
    terminal.route("GET", "/files/cwd", json_answer({"cwd": "/home/user"}))
    without_address = _shared(admin, person, terminal, url="")
    with person.client() as client:
        refused = client.get(f"/api/v1/terminals/{without_address}/files/cwd")
    with_address = _shared(admin, person, terminal)
    with person.client() as client:
        answered = client.get(f"/api/v1/terminals/{with_address}/files/cwd")

    assert (refused.status_code, refused.json()) == (
        503,
        {"error": "Terminal server URL not configured"},
    ), f"a terminal saved without an address answered {refused.status_code} {refused.text}"
    assert answered.status_code == 200, f"the same terminal with its address: {answered.text}"


def test_an_answer_the_terminal_breaks_off_is_reported_as_a_proxy_error(
    instance, admin, preserve, person, listener
):
    preserve(TERMINAL_SERVERS_CONFIG)
    terminal_id = _shared(admin, person, _AsTerminal(listener))

    def broken_off() -> Iterator[bytes]:
        yield b'{"entries": ['
        raise ConnectionResetError("the terminal died")

    listener.route("GET", "/files/list", (200, {"Content-Type": "application/json"}, broken_off()))
    listener.route("GET", "/files/cwd", json_answer({"cwd": "/home/user"}))
    log_start = instance.log_size()

    with person.client() as client:
        failed = client.get(f"/api/v1/terminals/{terminal_id}/files/list")
        afterwards = client.get(f"/api/v1/terminals/{terminal_id}/files/cwd")

    assert failed.status_code == 502, (
        f"an answer the terminal broke off came back as {failed.status_code} {failed.text!r}"
    )
    assert failed.json()["error"].startswith("Terminal proxy error: "), failed.text
    assert "Terminal proxy error: " in instance.log_since(log_start), (
        "the server log says nothing about the broken-off answer"
    )
    assert afterwards.status_code == 200, (
        f"the terminal was not reached again after the broken-off answer: {afterwards.text}"
    )


PORT = 3000


@pytest.mark.parametrize(
    "location, arrives_as",
    [
        ("/login?next=%2Fhome#top", "{prefix}login?next=%2Fhome#top"),
        ("settings", "{prefix}app/settings"),
        (f"http://localhost:{PORT}/done", "{prefix}done"),
        (f"http://localhost:{PORT + 1}/other-port", f"http://localhost:{PORT + 1}/other-port"),
        ("https://example.com/away", "https://example.com/away"),
    ],
    ids=["root path", "relative", "same port", "another port", "another site"],
)
def test_a_redirect_of_a_previewed_port_stays_inside_the_preview(
    person, terminal, terminal_id, location, arrives_as
):
    terminal.redirect(f"/proxy/{PORT}/app/start", location)
    prefix = f"/api/v1/terminals/{terminal_id}/proxy/{PORT}/"

    with person.client() as client:
        answered = client.get(f"{prefix}app/start", follow_redirects=False)

    assert answered.status_code == 302, f"{answered.status_code} {answered.text}"
    assert answered.headers.get_list("location") == [arrives_as.format(prefix=prefix)], (
        f"the app on port {PORT} redirected to {location}; the person's browser was sent to "
        f"{answered.headers.get_list('location')}"
    )


PAGE = """<!doctype html>
<html><head><title>Dev server</title>
<link rel="stylesheet" href="/static/site.css">
<script src='/static/app.js'></script>
<script src="//cdn.example.com/lib.js"></script>
</head><body>
<a href="docs/intro.html">Intro</a>
<a href="https://example.com/">Elsewhere</a>
<form action="/submit" method="post"></form>
<img src="/img/logo.png">
</body></html>"""


def test_a_page_of_a_previewed_port_loads_its_own_files_through_the_preview(
    person, terminal, terminal_id
):
    terminal.route("GET", f"/proxy/{PORT}/", text_answer(PAGE))
    terminal.route("GET", f"/proxy/{PORT}/data.json", json_answer({"href": "/static/site.css"}))
    terminal.route("GET", "/files/view", text_answer(PAGE))
    prefix = f"/api/v1/terminals/{terminal_id}/proxy/{PORT}/"

    with person.client() as client:
        page = client.get(prefix)
        data = client.get(f"{prefix}data.json")
        outside = client.get(f"/api/v1/terminals/{terminal_id}/files/view")

    assert page.status_code == 200, page.text
    for rewritten in (
        f'href="{prefix}static/site.css"',
        f"src='{prefix}static/app.js'",
        f'action="{prefix}submit"',
        f'src="{prefix}img/logo.png"',
    ):
        assert rewritten in page.text, (
            f"a root path of the page still points at Open WebUI itself, {rewritten} is missing: "
            f"{page.text[-420:]}"
        )
    for untouched in (
        'src="//cdn.example.com/lib.js"',
        'href="docs/intro.html"',
        'href="https://example.com/"',
    ):
        assert untouched in page.text, f"{untouched} was changed: {page.text[-420:]}"
    head, _, body = page.text.partition("<title>")
    assert head.startswith("<!doctype html>\n<html><head><script>"), (
        f"the page does not start with the preview's script in its head: {head[:80]!r}"
    )
    assert f'"prefix": "{prefix}"' in head and f'"origin": "http://localhost:{PORT}"' in head, (
        f"the preview's script does not know its address and port: {head[:200]!r}"
    )
    assert "terminal-browser-navigate" in head and "<script>" not in body, (
        "the script that keeps links inside the preview was not added once, to the head"
    )
    assert data.json() == {"href": "/static/site.css"}, (
        f"an answer of the port that is no page was rewritten: {data.text}"
    )
    assert outside.text == PAGE, (
        f"a page read outside a port preview was rewritten: {outside.text[:200]!r}"
    )
