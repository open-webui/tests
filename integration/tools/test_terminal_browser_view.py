"""Journey: a Browser tab's pages load through Open WebUI with a token of its own and no sign-in.

Opening, reading or changing a Browser tab answers with the terminal's tab and a token. The
page never gets the terminal's own token: Open WebUI hands out one it signed, and the tab's
frame loads `/api/v1/terminals/<id>/browser/view/<tab>/<token>/...` with it, without a session,
since a sandboxed frame has none. Open WebUI then asks the terminal with the terminal's token,
the connection's key and the person's id and chat, and passes on only the frame's accept,
origin, preflight, last-event-id and range headers; a terminal that takes the person's session
and cookies gets those of the request that opened the tab. The token opens that one tab on
that one terminal for as long as its person may use the terminal: after the account is deleted
or set to pending it answers 401, and after the terminal is unshared or switched off or the chat
is deleted the terminal is not asked any more. The frame's origin is opaque, so these paths skip
Open WebUI's own CORS answers and the terminal's reach the frame. The tab's three sockets (`ws`,
`stream`, `vnc`) are relayed both ways, text and binary; any other name is refused, a terminal
that refuses the socket ends it with 4004 (401, 403, 404) or 1011, and an open socket ends
within the 10 s re-check once its person lost the terminal or the account. The page tells a
viewer taken over by another window (4000) from a lost connection by the close code, so the
terminal's code has to reach it; on dev it gets 1011 (red, see the close-code test).

The terminal is the fake of `harness/terminal_server.py`.

Discriminates: on dev c15f28649, in a backend copy: passing the terminal's answer on unchanged,
or asking the terminal with the page's token in place of the terminal's, fails every test here;
forwarding none of the frame's headers fails the header test and the CORS test; dropping the
server check of a token fails the other-tab test, dropping the session check that and the
socket-name test, an account not checked for its role the account test; resolving the terminal
without the token's chat fails the lost-access test (deleted chat) and the header test; running
the app's CORS answers on view paths fails the CORS test; relaying any socket name fails the
socket-name test; closing every refused socket with 1011 fails the 401, 403 and 404 cases; a
re-check that never ends the relay fails the open-socket test, and so does opening a tab's
socket as the admin (it opens again after the access is gone); asking the terminal for a frame
or a socket without the cookies the tab was opened with fails the sign-in test. The close-code
test is red on dev and passes with the terminal's 4xxx close code passed on to the page.
"""

from __future__ import annotations

import contextlib
import json
import time
import uuid
from dataclasses import dataclass

import httpx
import pytest
from websockets.exceptions import InvalidStatus
from websockets.sync.client import connect

from harness.chat_history import seed_chat
from harness.listener import json_answer, text_answer
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    TerminalSession,
    close_of,
    configure_terminals,
    read_grant,
    serving_terminal,
)

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

KEY = "sk-browser-view"
EXPIRED = (401, "Browser session expired")
# the socket re-check runs every 10 s
RECHECK_BOUND_SECONDS = 20
BINARY_FRAME = b"\xff\xd8\xff\x00jpeg"
# the fake files a socket's first text message as a shell's sign-in and echoes the later ones
FIRST_MESSAGE = '{"type": "viewport"}'


@dataclass
class Tab:
    """A Browser tab as the page holds it: the terminal's id for it and Open WebUI's token."""

    terminal_id: str
    id: str
    token: str
    chat_id: str
    answer: dict

    @property
    def capability(self) -> str:
        return _capability(self.id)

    def view(self, path: str = "", token: str | None = None) -> str:
        return (
            f"/api/v1/terminals/{self.terminal_id}/browser/view/{self.id}/{token or self.token}"
            f"/{path}"
        )

    def upstream(self, path: str = "") -> str:
        return f"/browser/view/{self.id}/{self.capability}/{path}"


def _capability(tab_id: str) -> str:
    return f"capability-{tab_id}"


def _page(tab_id: str) -> str:
    return f"<html><body><h1>Page of tab {tab_id}</h1></body></html>"


@pytest.fixture
def terminal():
    """A fake terminal whose Browser tabs each serve a page under `frame/http/example.com/`."""
    with serving_terminal() as server:
        server.route("GET", "/openapi.json", json_answer({"openapi": "3.1.0", "paths": {}}))

        def open_tab(_request):
            tab_id = uuid.uuid4().hex[:12]
            tab = {
                "id": tab_id,
                "token": _capability(tab_id),
                "url": "http://example.com/",
                "title": f"Tab {tab_id}",
                "mode": "proxy",
                "chrome_available": False,
            }
            server.route("GET", f"/browser/sessions/{tab_id}", json_answer(tab))
            server.route(
                "PATCH", f"/browser/sessions/{tab_id}", json_answer({**tab, "mode": "chrome"})
            )
            server.route(
                "GET",
                f"/browser/view/{tab_id}/{_capability(tab_id)}/frame/http/example.com/",
                text_answer(_page(tab_id)),
            )
            return json_answer(tab)

        server.route("POST", "/browser/sessions", open_tab)
        yield server


@pytest.fixture
def share_terminal(admin, preserve, terminal):
    """`share_terminal(*people)`: the fake terminal's connection, readable by those people."""
    preserve(TERMINAL_SERVERS_CONFIG)

    def share(*people, **fields) -> dict:
        grants = [read_grant(person.id) for person in people]
        settings = {"key": KEY, "auth_type": "bearer", **fields}
        connection = terminal.connection(config={"access_grants": grants}, **settings)
        with admin.client() as client:
            configure_terminals(client, connection)
        return connection

    return share


def _open_tab(person, terminal_id: str) -> Tab:
    """Open a Browser tab in a saved chat of the person, as the Browser tab of a chat does."""
    with person.client() as client:
        chat_id = seed_chat(client, [{"role": "user", "content": "open the page"}])[0]
        opened = client.post(
            f"/api/v1/terminals/{terminal_id}/browser/sessions",
            headers={"X-Session-Id": chat_id},
            json={"url": "http://example.com/"},
        )
    assert opened.status_code == 200, f"opening the Browser tab failed: {opened.text}"
    answer = opened.json()
    return Tab(terminal_id, answer["id"], answer["token"], chat_id, answer)


def _signed_out(instance) -> httpx.Client:
    return httpx.Client(base_url=instance.base_url, timeout=60.0)


def _socket_url(person, tab: Tab, kind: str, token: str | None = None) -> str:
    return person.base_url.replace("http://", "ws://", 1) + tab.view(kind, token)


def _answer(response) -> tuple[int, str | None]:
    """The status and the message of a refusal, whichever key carries it."""
    try:
        body = response.json()
    except ValueError:
        return response.status_code, response.text
    return response.status_code, body.get("detail") or body.get("error")


def _view_requests(terminal) -> list[str]:
    with terminal.lock:
        return [
            f"{entry.method} {entry.path}"
            for entry in terminal.received
            if entry.path.startswith("/browser/view/")
        ]


def test_the_page_gets_a_token_of_open_webui_for_a_tab_and_never_the_terminals_own(
    instance, make_user, share_terminal, terminal
):
    person = make_user()
    terminal_id = share_terminal(person)["id"]
    tab = _open_tab(person, terminal_id)
    chat = {"X-Session-Id": tab.chat_id}
    sessions = f"/api/v1/terminals/{terminal_id}/browser/sessions"
    terminal.route("GET", "/browser/sessions", json_answer({"tabs": 1}))

    with person.client() as client:
        read = client.get(f"{sessions}/{tab.id}", headers=chat)
        changed = client.patch(f"{sessions}/{tab.id}", headers=chat, json={"mode": "chrome"})
        listed = client.get(sessions, headers=chat)
    answers = {"opening": tab.answer, "reading": read.json(), "changing": changed.json()}

    assert listed.json() == {"tabs": 1}, f"an answer that names no tab was changed: {listed.text}"
    for step, answer in answers.items():
        assert answer["token"] and tab.capability not in json.dumps(answer), (
            f"{step} the tab handed the page the terminal's own token: {answer}"
        )
        assert {**answer, "token": tab.capability} == {
            **tab.answer,
            "token": tab.capability,
            "mode": "chrome" if step == "changing" else "proxy",
        }, f"{step} the tab changed more than its token: {answer}"
    with _signed_out(instance) as frame:
        for step, answer in answers.items():
            terminal.clear()
            loaded = frame.get(tab.view("frame/http/example.com/", answer["token"]))
            assert (loaded.status_code, loaded.text) == (200, _page(tab.id)), (
                f"the token from {step} the tab did not load its page: "
                f"{loaded.status_code} {loaded.text}"
            )
            assert _view_requests(terminal) == [f"GET {tab.upstream('frame/http/example.com/')}"], (
                f"the terminal was asked for {_view_requests(terminal)} with the token "
                f"from {step} the tab"
            )


def test_the_terminal_is_asked_as_the_tabs_person_with_only_the_frames_own_headers(
    instance, make_user, share_terminal, terminal
):
    person = make_user()
    tab = _open_tab(person, share_terminal(person)["id"])
    tag = uuid.uuid4().hex[:8]
    part = (206, {"Content-Type": "video/mp4", "Content-Range": "bytes 0-3/9"}, b"\x00\x01\x02\x03")
    terminal.route("GET", tab.upstream("frame/http/example.com/clip.mp4"), part)
    terminal.route("POST", tab.upstream("frame/http/example.com/form"), text_answer(f"sent {tag}"))
    terminal.clear()

    with _signed_out(instance) as frame:
        clip = frame.get(
            tab.view("frame/http/example.com/clip.mp4"),
            params={"quality": tag},
            headers={
                "Accept": "video/*",
                "Origin": "null",
                "Range": "bytes=0-3",
                "Last-Event-ID": f"event-{tag}",
                "Authorization": f"Bearer page-{tag}",
                "Cookie": f"token=cookie-{tag}",
                "X-User-Id": f"someone-{tag}",
                "X-Page-Header": f"page-{tag}",
            },
        )
        posted = frame.post(
            tab.view("frame/http/example.com/form"),
            content=f"field={tag}",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    assert (clip.status_code, clip.content) == (206, b"\x00\x01\x02\x03"), (
        f"the terminal's partial answer arrived as {clip.status_code} {clip.content!r}"
    )
    assert clip.headers["content-range"] == "bytes 0-3/9", dict(clip.headers)
    assert (posted.status_code, posted.text) == (200, f"sent {tag}"), (
        f"the frame's form post answered {posted.status_code} {posted.text}"
    )
    asked, form = [entry for entry in terminal.received if entry.path.startswith("/browser/view/")]
    assert asked.path == f"{tab.upstream('frame/http/example.com/clip.mp4')}?quality={tag}", (
        f"the terminal was asked for {asked.path}"
    )
    sent = {
        name: asked.headers.get(name)
        for name in ("accept", "origin", "range", "last-event-id", "authorization")
    }
    assert sent == {
        "accept": "video/*",
        "origin": "null",
        "range": "bytes=0-3",
        "last-event-id": f"event-{tag}",
        "authorization": f"Bearer {KEY}",
    }, f"the terminal was asked with {sent}"
    assert (asked.headers.get("x-user-id"), asked.headers.get("x-session-id")) == (
        person.id,
        tab.chat_id,
    ), f"the terminal was asked as {asked.headers}"
    assert tag not in json.dumps({**asked.headers, "last-event-id": ""}) and not asked.cookies, (
        f"a header or cookie of the page reached the terminal: {asked.headers}"
    )
    assert (form.method, form.body, form.headers.get("content-type")) == (
        "POST",
        f"field={tag}".encode(),
        "application/x-www-form-urlencoded",
    ), f"the frame's form post reached the terminal as {form}"


def test_a_frame_reaches_the_terminal_with_the_sign_in_its_tab_was_opened_with(
    instance, make_user, share_terminal, terminal
):
    """A terminal that takes the person's own session and cookies gets both for the frame too."""
    person = make_user()
    tab_terminal = share_terminal(person, auth_type="session", forward_cookies=True)
    tag = uuid.uuid4().hex[:8]
    with person.client() as client:
        chat_id = seed_chat(client, [{"role": "user", "content": "open the page"}])[0]
        opened = client.post(
            f"/api/v1/terminals/{tab_terminal['id']}/browser/sessions",
            headers={"X-Session-Id": chat_id, "Cookie": f"sso=opened-{tag}"},
            json={"url": "http://example.com/"},
        )
    assert opened.status_code == 200, f"opening the Browser tab failed: {opened.text}"
    tab = Tab(tab_terminal["id"], opened.json()["id"], opened.json()["token"], chat_id, {})
    terminal.clear()

    with _signed_out(instance) as frame:
        loaded = frame.get(
            tab.view("frame/http/example.com/"), headers={"Cookie": f"sso=frame-{tag}"}
        )
    socket_url = instance.base_url.replace("http://", "ws://", 1) + tab.view("stream")
    with connect(socket_url, proxy=None, open_timeout=10) as socket:
        socket.send(FIRST_MESSAGE)
        socket.send("ping")
        echoed = socket.recv(timeout=10)

    assert (loaded.status_code, loaded.text) == (200, _page(tab.id)), (
        f"the tab's page did not load: {loaded.status_code} {loaded.text}"
    )
    [asked] = [entry for entry in terminal.received if entry.path.startswith("/browser/view/")]
    [viewer] = terminal.sessions
    assert echoed == "ping", f"the tab's socket did not relay: {echoed!r}"
    signed_in = {
        "page": (asked.headers.get("authorization"), asked.cookies.get("sso")),
        "socket": (
            viewer.headers.get("authorization"),
            viewer.headers.get("cookie", "").partition("sso=")[2].split(";")[0],
        ),
    }
    assert signed_in == dict.fromkeys(signed_in, (f"Bearer {person.token}", f"opened-{tag}")), (
        f"the terminal was asked for the tab's frame with {signed_in}"
    )


def test_a_token_opens_only_its_own_tab_on_its_own_terminal(
    admin, instance, make_user, preserve, terminal
):
    person = make_user()
    preserve(TERMINAL_SERVERS_CONFIG)
    grants = {"access_grants": [read_grant(person.id)]}
    first = terminal.connection(key=KEY, auth_type="bearer", config=grants)
    second = terminal.connection(key=KEY, auth_type="bearer", config=grants)
    with admin.client() as client:
        configure_terminals(client, first, second)
    tab = _open_tab(person, first["id"])
    other_tab = _open_tab(person, first["id"])
    twin = Tab(second["id"], tab.id, tab.token, tab.chat_id, tab.answer)
    page = "frame/http/example.com/"
    resigned = tab.token[:-4] + ("AAAA" if not tab.token.endswith("AAAA") else "BBBB")

    with _signed_out(instance) as frame:
        own = frame.get(tab.view(page))
        others_own = frame.get(other_tab.view(page))
        terminal.clear()
        refused = {
            "another tab's token": frame.get(tab.view(page, other_tab.token)),
            "a token used for another tab": frame.get(other_tab.view(page, tab.token)),
            "a token used on another terminal": frame.get(twin.view(page)),
            "a token with another signature": frame.get(tab.view(page, resigned)),
            "the person's session token": frame.get(tab.view(page, person.token)),
        }

    assert (own.status_code, own.text) == (200, _page(tab.id)), f"{own.status_code} {own.text}"
    assert (others_own.status_code, others_own.text) == (200, _page(other_tab.id)), (
        f"the second tab's own token did not load it: {others_own.status_code} {others_own.text}"
    )
    answers = {label: _answer(response) for label, response in refused.items()}
    assert answers == dict.fromkeys(refused, EXPIRED), (
        f"a token opened more than its tab: {answers}"
    )
    assert _view_requests(terminal) == [], (
        f"the terminal was asked for a refused token: {_view_requests(terminal)}"
    )


def test_a_token_stops_working_once_its_person_is_deleted_or_set_to_pending(
    admin, instance, make_user, share_terminal, terminal
):
    kept, deleted, pending = make_user(), make_user(), make_user()
    terminal_id = share_terminal(kept, deleted, pending)["id"]
    tabs = {
        "kept": _open_tab(kept, terminal_id),
        "deleted": _open_tab(deleted, terminal_id),
        "pending": _open_tab(pending, terminal_id),
    }
    page = "frame/http/example.com/"

    with _signed_out(instance) as frame:
        before = {label: frame.get(tab.view(page)).status_code for label, tab in tabs.items()}
        with admin.client() as client:
            removing = client.delete(f"/api/v1/users/{deleted.id}")
            demoting = client.post(f"/api/v1/users/{pending.id}/update", json={"role": "pending"})
        assert removing.status_code == 200, f"deleting the account failed: {removing.text}"
        assert demoting.status_code == 200, f"demoting the account failed: {demoting.text}"
        terminal.clear()
        after = {label: _answer(frame.get(tab.view(page))) for label, tab in tabs.items()}

    assert before == dict.fromkeys(tabs, 200), f"the tabs did not load to begin with: {before}"
    assert after == {
        "kept": (200, _page(tabs["kept"].id)),
        "deleted": EXPIRED,
        "pending": EXPIRED,
    }, f"the tabs answered {after} after one account was deleted and one set to pending"
    assert _view_requests(terminal) == [f"GET {tabs['kept'].upstream(page)}"], (
        f"the terminal was asked for {_view_requests(terminal)}"
    )


def test_a_token_does_not_outlive_its_persons_access_to_the_terminal_or_the_chat(
    admin, instance, make_user, preserve, terminal
):
    kept, unshared, chatless, switched_off = (make_user() for _ in range(4))
    preserve(TERMINAL_SERVERS_CONFIG)
    shared = terminal.connection(
        key=KEY,
        auth_type="bearer",
        config={"access_grants": [read_grant(person.id) for person in (kept, unshared, chatless)]},
    )
    retiring = terminal.connection(
        key=KEY, auth_type="bearer", config={"access_grants": [read_grant(switched_off.id)]}
    )
    with admin.client() as client:
        configure_terminals(client, shared, retiring)
    tabs = {
        "kept": _open_tab(kept, shared["id"]),
        "unshared": _open_tab(unshared, shared["id"]),
        "chat deleted": _open_tab(chatless, shared["id"]),
        "switched off": _open_tab(switched_off, retiring["id"]),
    }
    page = "frame/http/example.com/"

    with _signed_out(instance) as frame:
        before = {label: frame.get(tab.view(page)).status_code for label, tab in tabs.items()}
        with admin.client() as client:
            still_shared = {
                **shared,
                "config": {"access_grants": [read_grant(kept.id), read_grant(chatless.id)]},
            }
            configure_terminals(client, still_shared, {**retiring, "enabled": False})
        with chatless.client() as client:
            deleting = client.delete(f"/api/v1/chats/{tabs['chat deleted'].chat_id}")
        assert deleting.status_code == 200, f"deleting the chat failed: {deleting.text}"
        terminal.clear()
        after = {label: _answer(frame.get(tab.view(page))) for label, tab in tabs.items()}

    assert before == dict.fromkeys(tabs, 200), f"the tabs did not load to begin with: {before}"
    assert after == {
        "kept": (200, _page(tabs["kept"].id)),
        "unshared": (404, f"Terminal server '{shared['id']}' not found"),
        "chat deleted": (403, "Chat access denied"),
        "switched off": (404, f"Terminal server '{retiring['id']}' not found"),
    }, f"the tabs answered {after} after their people lost the terminal or the chat"
    assert _view_requests(terminal) == [f"GET {tabs['kept'].upstream(page)}"], (
        f"the terminal was asked for {_view_requests(terminal)}"
    )


def test_a_frames_cross_origin_calls_are_answered_by_the_terminal_and_not_by_open_webui(
    instance, make_user, share_terminal, terminal
):
    person = make_user()
    tab = _open_tab(person, share_terminal(person)["id"])
    page = "frame/http/example.com/"
    allowed = {"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "GET, POST"}
    terminal.route("OPTIONS", tab.upstream(page), (204, allowed, b""))
    terminal.route("GET", tab.upstream(page), (200, {**allowed, "Content-Type": "text/html"}, b""))
    terminal.route("GET", "/files/cwd", json_answer({"cwd": "/home/user"}))
    preflight = {
        "Origin": "null",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "x-frame-header",
    }

    with _signed_out(instance) as frame:
        terminal.clear()
        asked_first = frame.options(tab.view(page), headers=preflight)
        loaded = frame.get(tab.view(page), headers={"Origin": "null"})
        view_requests = _view_requests(terminal)
        [upstream_preflight] = [entry for entry in terminal.received if entry.method == "OPTIONS"]
        terminal.clear()
        elsewhere = frame.options(
            f"/api/v1/terminals/{tab.terminal_id}/files/cwd", headers=preflight
        )
        asked_elsewhere = [entry.path for entry in terminal.received]

    assert elsewhere.headers.get("access-control-allow-credentials") and not asked_elsewhere, (
        "Open WebUI did not answer a preflight outside the view paths itself: "
        f"{elsewhere.status_code} {dict(elsewhere.headers)}, terminal asked for {asked_elsewhere}"
    )
    assert view_requests == [f"OPTIONS {tab.upstream(page)}", f"GET {tab.upstream(page)}"], (
        f"the terminal was asked for {view_requests}"
    )
    sent = {name: upstream_preflight.headers.get(name.lower()) for name in preflight}
    assert sent == preflight, f"the frame's preflight reached the terminal as {sent}"
    for label, response in (("preflight", asked_first), ("page", loaded)):
        answered = {
            name: response.headers.get_list(name)
            for name in (*allowed, "Access-Control-Allow-Credentials")
        }
        assert answered == {
            **{name: [value] for name, value in allowed.items()},
            "Access-Control-Allow-Credentials": [],
        }, (
            f"the frame's {label} was not answered with the terminal's CORS headers alone: "
            f"{response.status_code} {dict(response.headers)}"
        )
    assert asked_first.status_code == 204, f"the preflight answered {asked_first.status_code}"


@pytest.mark.parametrize("kind", ["ws", "stream", "vnc"])
def test_a_tabs_socket_is_relayed_both_ways_to_the_terminal(
    make_user, share_terminal, terminal, kind
):
    person = make_user()
    tab = _open_tab(person, share_terminal(person)["id"])
    tag = uuid.uuid4().hex[:8]

    def viewer(session: TerminalSession) -> None:
        # the fake files the first text message as the sign-in of a shell
        first = session.authenticated()
        session.send(json.dumps({"type": "open", "for": first}))
        session.send(BINARY_FRAME)
        typed = session.receive()
        session.send(typed[::-1])
        session.close(1000)

    terminal.script_sessions(viewer, "/browser/view/*")
    terminal.clear()
    with connect(_socket_url(person, tab, kind), proxy=None, open_timeout=10) as socket:
        socket.send(json.dumps({"url": f"ws://example.com/{tag}"}))
        opened = json.loads(socket.recv(timeout=10))
        frame = socket.recv(timeout=10)
        socket.send(b"\x01" + tag.encode())
        echoed = socket.recv(timeout=10)
        closed = close_of(socket, timeout=10)

    [upstream] = terminal.sessions
    assert upstream.failure is None, f"the scripted viewer failed: {upstream.failure!r}"
    assert upstream.path == tab.upstream(kind), f"the terminal's socket was {upstream.path}"
    assert opened == {"type": "open", "for": {"url": f"ws://example.com/{tag}"}}, (
        f"the page's first message came back as {opened}"
    )
    assert frame == BINARY_FRAME, f"the terminal's binary frame arrived as {frame!r}"
    assert echoed == tag.encode()[::-1] + b"\x01", f"the page's binary message came back {echoed!r}"
    assert closed is not None, "the page's socket stayed open after the terminal closed its own"
    sent = {
        name: upstream.headers.get(name) for name in ("x-user-id", "x-session-id", "authorization")
    }
    assert sent == {
        "x-user-id": person.id,
        "x-session-id": tab.chat_id,
        "authorization": f"Bearer {KEY}",
    }, f"the terminal's socket was opened with {sent}"


def test_only_a_tabs_three_sockets_open_and_only_with_its_own_token(
    make_user, share_terminal, terminal
):
    person = make_user()
    terminal_id = share_terminal(person)["id"]
    tab, other_tab = _open_tab(person, terminal_id), _open_tab(person, terminal_id)
    terminal.clear()

    with connect(_socket_url(person, tab, "stream"), proxy=None, open_timeout=10) as socket:
        socket.send(FIRST_MESSAGE)
        socket.send("ping")
        opened = socket.recv(timeout=10)
    opened_on_terminal = [session.path for session in terminal.sessions]
    terminal.clear()
    refused = {}
    for label, url in {
        "another socket name": _socket_url(person, tab, "events"),
        "the shell's path": _socket_url(person, tab, "api/terminals/shell"),
        "another tab's token": _socket_url(person, tab, "stream", other_tab.token),
        "the person's session token": _socket_url(person, tab, "stream", person.token),
    }.items():
        try:
            with connect(url, proxy=None, open_timeout=10) as socket:
                closed = close_of(socket, timeout=5)
            refused[label] = f"opened, then {closed}"
        except InvalidStatus as error:
            refused[label] = error.response.status_code

    assert opened == "ping" and opened_on_terminal == [tab.upstream("stream")], (
        f"the tab's stream did not open: {opened!r}, terminal sockets {opened_on_terminal}"
    )
    assert refused == dict.fromkeys(refused, 403), f"a socket outside the tab's three: {refused}"
    assert not terminal.sessions and not terminal.received, (
        f"the terminal was asked for a refused socket: {terminal.sessions} {terminal.received}"
    )


@pytest.mark.parametrize(
    "status,expected",
    [(401, 4004), (403, 4004), (404, 4004), (500, 1011)],
    ids=["unauthorized", "forbidden", "not-found", "server-error"],
)
def test_a_tabs_socket_the_terminal_refuses_ends_with_the_code_for_its_status(
    make_user, share_terminal, terminal, status, expected
):
    person = make_user()
    tab = _open_tab(person, share_terminal(person)["id"])
    terminal.refuse_sessions(status, "/browser/view/*")
    terminal.clear()

    with connect(_socket_url(person, tab, "stream"), proxy=None, open_timeout=10) as socket:
        closed = close_of(socket, timeout=10)

    assert _view_requests(terminal) == [f"GET {tab.upstream('stream')}"], (
        f"the terminal was asked for {_view_requests(terminal)}"
    )
    assert closed is not None, f"the page's socket stayed open after the terminal's {status}"
    assert closed[0] == expected, f"a {status} ended the page's socket with {closed}"


def test_a_viewer_the_terminal_closes_ends_with_the_terminals_code(
    make_user, share_terminal, terminal
):
    """Red on dev c15f28649 and on 0.13.3: the page gets 1011 whatever the terminal's code, so
    a Chrome tab shown in two windows is taken back and forth between them without end. Draft:
    "issue: A Browser tab in Chrome mode opened in two windows keeps reconnecting in both"."""
    person = make_user()
    tab = _open_tab(person, share_terminal(person)["id"])

    def taken_over(session: TerminalSession) -> None:
        session.send('{"type": "state"}')
        session.close(4000, "Browser opened in another viewer")

    terminal.script_sessions(taken_over, "/browser/view/*")
    with connect(_socket_url(person, tab, "stream"), proxy=None, open_timeout=10) as socket:
        state = socket.recv(timeout=10)
        closed = close_of(socket, timeout=10)

    [upstream] = terminal.sessions
    assert upstream.failure is None, f"the scripted viewer failed: {upstream.failure!r}"
    assert state == '{"type": "state"}', f"the viewer's state arrived as {state!r}"
    assert closed is not None, "the page's socket stayed open after the terminal closed its own"
    assert closed[0] == 4000, (
        f"the terminal closed the viewer with 4000 (opened in another viewer), "
        f"the page's socket ended with {closed}"
    )


def _terminal_side(terminal, tab: Tab) -> TerminalSession:
    [session] = [entry for entry in terminal.sessions if entry.path == tab.upstream("stream")]
    return session


def _is_live(socket) -> bool:
    probe = f"probe-{uuid.uuid4().hex[:8]}"
    socket.send(probe)
    return socket.recv(timeout=10) == probe


@pytest.mark.slow
def test_an_open_tab_socket_ends_once_its_person_lost_the_terminal_or_the_account(
    admin, make_user, share_terminal, terminal
):
    kept, unshared, pending = make_user(), make_user(), make_user()
    connection = share_terminal(kept, unshared, pending)
    tabs = {
        "kept": (kept, _open_tab(kept, connection["id"])),
        "unshared": (unshared, _open_tab(unshared, connection["id"])),
        "pending": (pending, _open_tab(pending, connection["id"])),
    }
    terminal.clear()
    with contextlib.ExitStack() as opened:
        sockets = {
            label: opened.enter_context(
                connect(_socket_url(person, tab, "stream"), proxy=None, open_timeout=10)
            )
            for label, (person, tab) in tabs.items()
        }
        for socket in sockets.values():
            socket.send(FIRST_MESSAGE)
        live_before = {label: _is_live(socket) for label, socket in sockets.items()}
        with admin.client() as client:
            still_shared = {
                **connection,
                "config": {"access_grants": [read_grant(kept.id), read_grant(pending.id)]},
            }
            configure_terminals(client, still_shared)
            demoting = client.post(f"/api/v1/users/{pending.id}/update", json={"role": "pending"})
        assert demoting.status_code == 200, f"demoting the account failed: {demoting.text}"
        deadline = time.monotonic() + RECHECK_BOUND_SECONDS
        ended = {
            label: close_of(sockets[label], max(deadline - time.monotonic(), 0.5)) is not None
            for label in ("unshared", "pending")
        }
        kept_live = _is_live(sockets["kept"])
        left_on_terminal = {
            label: _terminal_side(terminal, tab).ended.wait(timeout=5)
            for label, (_, tab) in tabs.items()
            if label != "kept"
        }
        reopened = {}
        for label in ("unshared", "pending"):
            person, tab = tabs[label]
            try:
                with connect(_socket_url(person, tab, "stream"), proxy=None, open_timeout=10):
                    reopened[label] = "opened"
            except InvalidStatus as error:
                reopened[label] = error.response.status_code

    assert live_before == dict.fromkeys(tabs, True), f"the sockets did not relay: {live_before}"
    assert reopened == {"unshared": 403, "pending": 403}, (
        f"a tab's socket opened again after its person lost access: {reopened}"
    )
    assert ended == {"unshared": True, "pending": True}, (
        f"a tab's socket outlived its person's access for {RECHECK_BOUND_SECONDS} s: {ended}"
    )
    assert kept_live, "the socket of a person whose access never changed stopped relaying"
    assert left_on_terminal == {"unshared": True, "pending": True}, (
        f"Open WebUI kept the terminal's side of an ended socket open: {left_on_terminal}"
    )
