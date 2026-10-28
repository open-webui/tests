"""Socket-layer repairs from 0.11.1 that a connected client can see.

* `ce3c175e26`: `SocketSessionEventSink` is registered in `EVENT_SINKS`, so a
  `user.role_updated` or `user.deleted` event from any path disconnects the user's open
  Socket.IO sessions. The client reconnects and re-authenticates, so the role and permissions
  the socket layer cached for that session are read afresh instead of outliving the change.
* `5735123f5` (PR #28669): `yjs_document_update` cancelled the pending debounced note save
  before knowing whether a replacement would be scheduled, so the content-less resync update a
  client sends after rejoining dropped the edits made just before it.
* `a39126c27` (PR #28311): a tool's `__event_call__` to a tab that did not answer in time caught
  only the builtin `TimeoutError`, while python-socketio raises its own, and dropped the
  still-open session from the pool, so the next call to that tab failed at once. The instance for
  these tests gives an event call 2 seconds (`WEBSOCKET_EVENT_CALLER_TIMEOUT`).

Twin of unit/chat/test_socket_runtime.py.

Discriminates: passes on dev bbfa876af and ef67cc3fa; with `SocketSessionEventSink` dropped from
`EVENT_SINKS` the role-change and deletion tests fail, with the unconditional cancel restored ahead
of the update the resync test fails, and with `get_event_call` catching only the builtin
`TimeoutError` or evicting the session on a timeout the event-call test fails.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager

import pytest

from harness import upstream as reply
from harness.actors import admin_of, create_user
from harness.chat import ask
from harness.python_tools import python_tool
from harness.socket_client import connected

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

DISCONNECT_TIMEOUT = 10
QUIET_PERIOD = 2


@contextmanager
def watched_socket(account):
    """The account's live socket, with an event that is set once the server drops it."""
    with connected(account) as socket:
        dropped = threading.Event()
        socket.client.on("disconnect", lambda *args: dropped.set())
        yield dropped


def test_a_role_change_disconnects_the_users_live_socket(admin, make_user):
    account = make_user()
    with watched_socket(account) as dropped, admin.client() as client:
        client.post(f"/api/v1/users/{account.id}/update", json={"role": "admin"}).raise_for_status()

        assert dropped.wait(DISCONNECT_TIMEOUT), (
            "the socket kept the role it authenticated with after the admin changed it"
        )


def test_deleting_the_account_disconnects_its_live_socket(admin, make_user):
    account = make_user()
    with watched_socket(account) as dropped, admin.client() as client:
        client.delete(f"/api/v1/users/{account.id}").raise_for_status()

        assert dropped.wait(DISCONNECT_TIMEOUT), "a deleted account kept its live socket"


def test_an_update_that_keeps_the_role_leaves_the_socket_connected(admin, make_user):
    account = make_user()
    with watched_socket(account) as dropped, admin.client() as client:
        client.post(
            f"/api/v1/users/{account.id}/update", json={"name": "Renamed User"}
        ).raise_for_status()

        assert not dropped.wait(QUIET_PERIOD)


SAVE_WAIT = 5.0


def _note_markdown(client, note_id: str) -> str:
    return client.get(f"/api/v1/notes/{note_id}").json()["data"]["content"]["md"]


def _edit_then(account, followed_by: dict) -> str:
    """Edit a note over its live document, then send `followed_by`; the markdown saved after."""
    with account.client() as client:
        note = client.post(
            "/api/v1/notes/create", json={"title": "draft", "data": {"content": {"md": "old"}}}
        )
        assert note.status_code == 200, note.text
        document_id = f"note:{note.json()['id']}"
        with connected(account) as socket:
            socket.call("ydoc:document:join", {"document_id": document_id})
            edit = {"update": [1, 2, 3], "data": {"content": {"md": "edited"}}}
            socket.call("ydoc:document:update", {"document_id": document_id, **edit})
            socket.call("ydoc:document:update", {"document_id": document_id, **followed_by})

            deadline = time.monotonic() + SAVE_WAIT
            while time.monotonic() < deadline:
                markdown = _note_markdown(client, note.json()["id"])
                if markdown != "old":
                    return markdown
                time.sleep(0.2)
            return markdown


def test_a_resync_update_keeps_the_pending_note_save(make_user):
    markdown = _edit_then(make_user(), followed_by={"update": [4, 5, 6]})

    assert markdown == "edited", (
        "the content-less resync update cancelled the pending save of the edit (PR #28669)"
    )


def test_a_later_edit_replaces_the_pending_note_save(make_user):
    later = {"update": [4, 5, 6], "data": {"content": {"md": "edited again"}}}

    assert _edit_then(make_user(), followed_by=later) == "edited again"


EVENT_CALL_TIMEOUT = 2
TIMED_OUT = "Event call timed out. The browser tab may be inactive or closed."
ASK_THE_TAB = '''
import json


class Tools:
    async def ask_the_tab(self, __event_call__=None) -> str:
        """
        Ask the user's open tab for their name.
        """
        answer = await __event_call__({"type": "input", "data": {"title": "Your name?"}})
        return json.dumps(answer)
'''


@pytest.fixture(scope="module")
def short_event_calls(instance_with):
    """An instance that gives a tab 2 seconds to answer a tool's event call."""
    return instance_with({"WEBSOCKET_EVENT_CALLER_TIMEOUT": str(EVENT_CALL_TIMEOUT)})


def _tab_answering_late_once(socket) -> list[dict]:
    """Answer the first event call after it timed out and every later one at once."""
    calls: list[dict] = []

    def on_events(message: dict):
        socket.events.append(message)
        if (message.get("data") or {}).get("type") != "input":
            return None
        calls.append(message)
        if len(calls) == 1:
            time.sleep(EVENT_CALL_TIMEOUT + 1)
            return {"value": "too late"}
        return {"value": "Ada"}

    socket.client.on("events", on_events)
    return calls


def _tool_results(upstream) -> list[str]:
    return [
        entry["content"]
        for sent in upstream.chat_requests()
        for entry in sent["messages"]
        if entry["role"] == "tool"
    ]


@pytest.mark.slow
def test_an_unanswered_event_call_times_out_and_the_tab_is_still_asked_next_time(
    short_event_calls,
):
    account = create_user(short_event_calls)
    provider = short_event_calls.upstream
    with (
        python_tool(admin_of(short_event_calls), ASK_THE_TAB, name="Ask the tab") as tool_id,
        connected(account) as socket,
        account.client() as client,
    ):
        calls = _tab_answering_late_once(socket)
        options = {"tool_ids": [tool_id], "session_id": socket.client.get_sid()}
        for closing in ("first", "second"):
            provider.queue(reply.tool_call("ask_the_tab", {}), reply.text(f"{closing} done"))
            _, message = ask(client, f"who am I? ({closing})", **options)
            assert message["content"].endswith(f"{closing} done"), message

    first, second = _tool_results(provider)
    assert json.loads(first) == {"error": TIMED_OUT}, (
        f"a tab that did not answer in time was not reported as timed out: {first}"
    )
    assert len(calls) == 2, "the tab was never asked again after one call timed out"
    assert json.loads(second) == {"value": "Ada"}, (
        f"the timed-out tab's session was dropped, so the next call failed: {second}"
    )
