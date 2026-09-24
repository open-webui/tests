"""The terminal proxy keeps callers off the server's admin API and ends sessions whose access ends.

Two 0.11.4 fixes. `51bb8cb14`: the proxy passed requests on to the terminal server's
administrative endpoints (`api/v1/policies`, `api/v1/status`, `api/v1/terminals`), followed
whatever redirect the server answered with, and let through tab and CR, which URL parsers strip
so that `.<TAB>.` turns into `..`. The fix refuses those paths relative to the configured base,
stops following redirects and adds tab, CR and LF to the sanitizer's refusals (LF never matches
the route). `a1189a2d7`: an interactive terminal session checked access once, at the handshake,
so demoting the user or switching off or removing the connection left the open shell running;
a watchdog now re-checks every 10 seconds and closes the session. Twin of
unit/security/test_terminal_proxy_restrictions.py.

Discriminates: passes on dev bbfa876af; with the ADMIN_API_PATHS check removed the admin-path
tests fail (the fake terminal gets the request), with `allow_redirects=False` removed the
redirect test fails (the redirect target is fetched), with tab, CR and LF dropped from the
sanitizer the tab and CR tests fail, and without the watchdog task the revoked sessions stay
open past the bound.
"""

from __future__ import annotations

import time
from contextlib import ExitStack
from typing import Iterator

import pytest

from harness.listener import json_answer
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    FakeTerminalServer,
    close_of,
    configure_terminals,
    read_grant,
    serving_terminal,
    terminal_session,
)

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

# The watchdog's 10 s recheck interval plus slack.
RECHECK_BOUND_SECONDS = 13


@pytest.fixture
def terminal() -> Iterator[FakeTerminalServer]:
    with serving_terminal() as server:
        yield server


@pytest.fixture
def proxy_under(admin, preserve, terminal):
    """`proxy_under(base_path)` saves a connection rooted there and returns a GET sender."""
    preserve(TERMINAL_SERVERS_CONFIG)
    client = admin.client()

    def sender(base_path: str = ""):
        connection = terminal.connection(url=terminal.base_url + base_path)
        configure_terminals(client, connection)
        terminal.clear()
        return lambda path: client.get(f"/api/v1/terminals/{connection['id']}/{path}")

    yield sender
    client.close()


@pytest.mark.parametrize("base_path", ["", "/root"])
@pytest.mark.parametrize(
    "path",
    [
        "api/v1/policies",
        "api/v1/status",
        "api/v1/terminals",
        "api/v1/policies/policy-1",
        "api/v1/terminals/session-1",
        "api%2Fv1%2F%2E%2Fpolicies",
    ],
)
def test_the_terminal_admin_api_is_never_proxied(proxy_under, terminal, base_path, path):
    response = proxy_under(base_path)(path)
    assert response.status_code == 403, (
        f"{path} reached the terminal server's administrative API through a user's proxy "
        "(51bb8cb14)"
    )
    assert terminal.received == []


@pytest.mark.parametrize("path", ["api/v1/policiesx", "api/config", "files/list"])
def test_paths_beside_the_admin_api_still_proxy(proxy_under, terminal, path):
    terminal.route("GET", f"/{path}", json_answer({"ok": True}))
    assert proxy_under()(path).status_code == 200
    assert terminal.requests_to(f"/{path}")


@pytest.mark.parametrize("character", ["%09", "%0D"], ids=["tab", "cr"])
def test_characters_a_url_parser_strips_are_refused(proxy_under, terminal, character):
    response = proxy_under()(f"a/.{character}./etc")
    assert response.status_code == 400, (
        f"a path with {character} between two dots was proxied; once a parser strips it the "
        "dots resolve as a traversal out of the terminal's root (51bb8cb14)"
    )
    assert terminal.received == []


def test_a_newline_never_reaches_the_terminal(proxy_under, terminal):
    # The route's path pattern stops at a newline, so this 404s before the sanitizer runs.
    assert proxy_under()("a/.%0A./etc").status_code == 404
    assert terminal.received == []


def test_a_redirect_from_the_terminal_is_not_followed(proxy_under, terminal, listener):
    listener.route("GET", "/internal", json_answer({"secret": True}))
    terminal.redirect("/moved", f"{listener.base_url}/internal")
    response = proxy_under()("moved")
    assert response.status_code == 302
    assert listener.requests_to("/internal") == [], (
        "the proxy followed the terminal server's redirect to another service (51bb8cb14)"
    )


def _is_live(session) -> bool:
    session.send("still there?")
    return session.recv(timeout=10) == "still there?"


def _granted_to(*actors) -> dict:
    return {"access_grants": [read_grant(actor.id) for actor in actors]}


def test_open_sessions_end_once_access_is_revoked(admin, preserve, make_user, terminal):
    preserve(TERMINAL_SERVERS_CONFIG)
    kept, demoted, switched_off, removed = (make_user() for _ in range(4))
    shared = terminal.connection(config=_granted_to(kept, demoted))
    retiring = terminal.connection(config=_granted_to(switched_off))
    deleting = terminal.connection(config=_granted_to(removed))
    with admin.client() as client:
        configure_terminals(client, shared, retiring, deleting)

    with ExitStack() as stack:
        # Opened first, `kept` is re-checked first, so it has passed its check once the rest close.
        sessions = {
            label: stack.enter_context(terminal_session(actor.base_url, actor.token, target["id"]))
            for label, actor, target in [
                ("kept", kept, shared),
                ("demoted", demoted, shared),
                ("switched off", switched_off, retiring),
                ("removed", removed, deleting),
            ]
        }
        assert all(_is_live(session) for session in sessions.values())

        with admin.client() as client:
            configure_terminals(client, shared, {**retiring, "enabled": False})
            demoting = client.post(f"/api/v1/users/{demoted.id}/update", json={"role": "pending"})
        assert demoting.status_code == 200, demoting.text
        deadline = time.monotonic() + RECHECK_BOUND_SECONDS

        close_codes = {}
        for label in ("demoted", "switched off", "removed"):
            closed = close_of(sessions[label], max(deadline - time.monotonic(), 0.5))
            close_codes[label] = closed[0] if closed else "still open"
        assert close_codes == {"demoted": 4001, "switched off": 4003, "removed": 4004}, (
            "a terminal session outlived the access it was opened with (a1189a2d7)"
        )
        assert _is_live(sessions["kept"]), "a session whose access never changed was closed"
        revoked_shells = [
            shell for shell in terminal.sessions if shell.headers["x-user-id"] != kept.id
        ]
        assert len(revoked_shells) == 3
        assert all(shell.ended.wait(timeout=5) for shell in revoked_shells), (
            "the shell on the terminal server outlived the revoked session"
        )
