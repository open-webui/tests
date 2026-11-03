"""Journey: an admin tunes a function's valves and source, and users set a shared tool's valves.

An admin saves a function's valves, reads them back and the filter or pipe uses them on the very
next chat; a pipe whose valves need a value with no default installs, and answers once it is
saved. Saving new source replaces the running code on the next chat; source that does not
compile is refused, and the stored code stays switched on and keeps answering. On a shared tool,
anyone who may read the tool keeps their own user valves, while the tool's admin valves need
write access.

Since 0.13.0 a workspace tool can be switched off and on (9d1b4ef36). A switched-off tool leaves
the chat's tool picker and is no longer offered to the model, even in a chat that already had it
selected, and it does not run when the model calls it, also when it was switched off after the
reply started. Switching it on brings all of it back. Only the owner, a writer and the admin may
switch it.

Discriminates: in a backend copy, the function valves update skipping its save turned both valves
tests red; the update route not replacing the cached module while the cache also ignored changed
source turned the updated-source test red; the update storing source whose load failed, and a
refused update switching the function off, each turned the refused-source test red; the tool
admin valves routes checking read in place of write turned the reader test red, and the user
valves routes checking write in place of read turned the user valves test red. Each left the
other tests green. The three switch tests fail on v0.12.5, which has no switch, and pass on
v0.13.0. In backend copies of v0.13.0: the chat's tool loading keeping a switched-off tool turned
the withheld test red (still offered); the tool list for the picker keeping it turned the same
test red (still listed); dropping the check made when a tool runs turned the under-way test red
(the tool ran); the switch route asking for read in place of write, and dropping its owner check,
each turned the who-may-switch test red. Each left the other switch tests green.
"""

from __future__ import annotations

import textwrap
import threading
import time
import uuid

import pytest

from harness import upstream as reply
from harness.access import Shareable, attempts, cast, grant
from harness.chat import ask, send_message, wait_for_reply
from harness.plugins import installed_function
from harness.python_tools import python_tool

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]


def source(code: str) -> str:
    return textwrap.dedent(code).strip() + "\n"


TAGGING_FILTER = source(
    """
    from pydantic import BaseModel

    class Filter:
        class Valves(BaseModel):
            tag: str = "[default tag]"

        def __init__(self):
            self.valves = self.Valves()

        async def inlet(self, body):
            body["messages"][-1]["content"] += " " + self.valves.tag
            return body
    """
)

LABELLING_PIPE = source(
    """
    from pydantic import BaseModel

    class Pipe:
        class Valves(BaseModel):
            label: str = "default label"

        def __init__(self):
            self.valves = self.Valves()

        def pipe(self, body):
            return f"answered by {self.valves.label}"
    """
)


KEYED_PIPE = source(
    """
    from pydantic import BaseModel

    class Pipe:
        class Valves(BaseModel):
            api_key: str

        valves = None

        def pipe(self, body):
            return f"answered with key {self.valves.api_key}"
    """
)


def versioned_pipe(version: str) -> str:
    return source(
        f"""
        class Pipe:
            def pipe(self, body):
                return "answered by version {version}"
        """
    )


UNCOMPILABLE_PIPE = source(
    """
    class Pipe:
        def pipe(self, body)
            return "never compiled"
    """
)

VALVED_TOOL = source(
    """
    from pydantic import BaseModel

    class Tools:
        class Valves(BaseModel):
            api_key: str = ""

        class UserValves(BaseModel):
            nickname: str = ""

        def __init__(self):
            self.valves = self.Valves()

        def ping(self) -> str:
            \"\"\"Answer pong.\"\"\"
            return "pong"
    """
)


def _save_function_valves(admin, function_id: str, valves: dict) -> None:
    with admin.client() as client:
        saved = client.post(f"/api/v1/functions/id/{function_id}/valves/update", json=valves)
    assert saved.status_code == 200, saved.text


def _stored_function_valves(admin, function_id: str) -> dict:
    with admin.client() as client:
        stored = client.get(f"/api/v1/functions/id/{function_id}/valves")
    assert stored.status_code == 200, stored.text
    return stored.json()


def _update_function_source(admin, function_id: str, content: str):
    form = {
        "id": function_id,
        "name": function_id,
        "content": content,
        "meta": {"description": "installed by a regression test"},
    }
    with admin.client() as client:
        return client.post(f"/api/v1/functions/id/{function_id}/update", json=form)


def _pipe_reply(admin, pipe_id: str) -> str:
    with admin.client() as client:
        client.get("/api/models").raise_for_status()
        _, message = ask(client, "who answers?", model=pipe_id)
    return message["content"]


def test_saved_filter_valves_are_read_back_and_used_on_the_next_chat(admin, make_user, upstream):
    chatter = make_user()
    with installed_function(admin, TAGGING_FILTER, is_global=True) as filter_id:
        with chatter.client() as client:
            ask(client, "first")
        _save_function_valves(admin, filter_id, {"tag": "[saved tag]"})
        stored = _stored_function_valves(admin, filter_id)
        with chatter.client() as client:
            ask(client, "second")

    sent = [request["messages"][-1]["content"] for request in upstream.chat_requests()]
    assert stored == {"tag": "[saved tag]"}
    assert sent == ["first [default tag]", "second [saved tag]"]


def test_saved_pipe_valves_shape_the_next_reply(admin):
    with installed_function(admin, LABELLING_PIPE) as pipe_id:
        before = _pipe_reply(admin, pipe_id)
        _save_function_valves(admin, pipe_id, {"label": "the saved label"})
        stored = _stored_function_valves(admin, pipe_id)
        after = _pipe_reply(admin, pipe_id)

    assert stored == {"label": "the saved label"}
    assert (before, after) == ("answered by default label", "answered by the saved label")


@pytest.mark.regression
def test_a_pipe_whose_valves_need_a_key_installs_and_answers_once_the_key_is_saved(admin):
    """Red on dev 3dd1db147: saving a function builds its Valves from the stored values, so a
    function with a required valve and nothing stored yet is refused and cannot be installed
    (open-webui/open-webui#32145, fixed by c9f9c4b1d)."""
    with installed_function(admin, KEYED_PIPE) as pipe_id:
        _save_function_valves(admin, pipe_id, {"api_key": "the saved key"})
        reply = _pipe_reply(admin, pipe_id)

    assert reply == "answered with key the saved key"


def test_updated_source_runs_on_the_next_chat(admin):
    with installed_function(admin, versioned_pipe("one")) as pipe_id:
        before = _pipe_reply(admin, pipe_id)
        updated = _update_function_source(admin, pipe_id, versioned_pipe("two"))
        after = _pipe_reply(admin, pipe_id)

    assert updated.status_code == 200, updated.text
    assert (before, after) == ("answered by version one", "answered by version two")


def test_source_that_does_not_compile_is_refused_and_the_stored_code_keeps_answering(admin):
    with installed_function(admin, versioned_pipe("one")) as pipe_id:
        _pipe_reply(admin, pipe_id)
        refused = _update_function_source(admin, pipe_id, UNCOMPILABLE_PIPE)
        with admin.client() as client:
            stored = client.get(f"/api/v1/functions/id/{pipe_id}").json()
        after = _pipe_reply(admin, pipe_id)

    assert refused.status_code == 400
    assert (stored["content"], stored["is_active"]) == (versioned_pipe("one"), True)
    assert after == "answered by version one"


def _shared_tool(admin, reader, writer) -> str:
    tool_id = f"tool_{uuid.uuid4().hex[:8]}"
    grants = [
        grant("user", reader.id, "read"),
        grant("user", writer.id, "read"),
        grant("user", writer.id, "write"),
    ]
    with admin.client() as client:
        created = client.post(
            "/api/v1/tools/create",
            json={
                "id": tool_id,
                "name": "Valved tool",
                "content": VALVED_TOOL,
                "meta": {"description": "valved tool"},
                "access_grants": grants,
            },
        )
    assert created.status_code == 200, created.text
    return tool_id


@pytest.fixture
def valved_tool(admin, make_user):
    reader, writer, stranger = make_user(), make_user(), make_user()
    tool_id = _shared_tool(admin, reader, writer)
    yield tool_id, reader, writer, stranger
    with admin.client() as client:
        client.delete(f"/api/v1/tools/id/{tool_id}/delete")


def test_anyone_who_can_read_a_tool_keeps_their_own_user_valves(valved_tool):
    tool_id, reader, writer, stranger = valved_tool
    base = f"/api/v1/tools/id/{tool_id}/valves/user"
    stored = {}
    for account, nickname in ((reader, "reader"), (writer, "writer")):
        with account.client() as client:
            spec = client.get(f"{base}/spec")
            saved = client.post(f"{base}/update", json={"nickname": nickname})
            assert spec.status_code == 200, spec.text
            assert "nickname" in spec.json()["properties"]
            assert saved.status_code == 200, saved.text
            stored[nickname] = client.get(base).json()
    with stranger.client() as client:
        refused = client.post(f"{base}/update", json={"nickname": "stranger"})

    assert stored == {"reader": {"nickname": "reader"}, "writer": {"nickname": "writer"}}
    assert refused.status_code == 401


def test_a_reader_cannot_see_or_change_a_tools_admin_valves(admin, valved_tool):
    tool_id, reader, _, _ = valved_tool
    base = f"/api/v1/tools/id/{tool_id}/valves"
    with admin.client() as client:
        client.post(f"{base}/update", json={"api_key": "sk-admin"}).raise_for_status()
    with reader.client() as client:
        read = client.get(base)
        spec = client.get(f"{base}/spec")
        update = client.post(f"{base}/update", json={"api_key": "sk-reader"})
    with admin.client() as client:
        stored = client.get(base).json()

    assert (read.status_code, spec.status_code) == (401, 401)
    assert "sk-admin" not in read.text
    assert update.status_code in (400, 401, 403)
    assert stored == {"api_key": "sk-admin"}


def test_a_writer_reads_and_changes_a_tools_admin_valves(admin, valved_tool):
    tool_id, _, writer, _ = valved_tool
    base = f"/api/v1/tools/id/{tool_id}/valves"
    with writer.client() as client:
        spec = client.get(f"{base}/spec")
        update = client.post(f"{base}/update", json={"api_key": "sk-writer"})
        read = client.get(base)
    with admin.client() as client:
        stored = client.get(base).json()

    assert spec.status_code == 200, spec.text
    assert "api_key" in spec.json()["properties"]
    assert update.status_code == 200, update.text
    assert read.json() == stored == {"api_key": "sk-writer"}


TIDE_MARK = "high water at 12:04"
TIDE_TOOL = source(
    f"""
    class Tools:
        def tide_mark(self) -> str:
            \"\"\"Read the harbour's tide mark.\"\"\"
            return "{TIDE_MARK}"
    """
)


def _switch(actor, tool_id: str, *, to: bool) -> None:
    with actor.client() as client:
        switched = client.post(f"/api/v1/tools/id/{tool_id}/toggle")
    assert switched.status_code == 200, f"switching the tool failed: {switched.text}"
    assert switched.json()["is_active"] is to, f"the switch left the tool at {switched.json()}"


def _picker(client) -> set[str]:
    """The tool ids the chat's tool picker offers this account."""
    listed = client.get("/api/v1/tools/")
    assert listed.status_code == 200, listed.text
    return {tool["id"] for tool in listed.json()}


def _requests_answering(upstream, prompt: str) -> list[dict]:
    answering = reply.answering(prompt)
    return [request for request in upstream.chat_requests() if answering(request)]


def _offered(request: dict) -> set[str]:
    return {tool["function"]["name"] for tool in request.get("tools") or []}


def _tool_results(request: dict) -> list[str]:
    """What the tool calls answering the latest user message returned; earlier turns excluded."""
    messages = request["messages"]
    last_user = max(index for index, entry in enumerate(messages) if entry["role"] == "user")
    return [str(entry["content"]) for entry in messages[last_user:] if entry["role"] == "tool"]


def _tide_turn(client, upstream, tool_id: str, after=None):
    """One message with the tool selected, to which the model calls it whether offered or not.

    Returns the turn, the tool names the model was offered and what the tool calls returned.
    """
    prompt = f"read the tide ({uuid.uuid4().hex[:8]})"
    answering = reply.answering(prompt)
    upstream.queue(
        reply.tool_call("tide_mark", {}, match=answering), reply.text("done", match=answering)
    )
    follow_on = {"chat_id": after.chat_id, "parent_id": after.assistant_message_id} if after else {}
    turn, _ = ask(client, prompt, tool_ids=[tool_id], **follow_on)
    requests = _requests_answering(upstream, prompt)
    assert requests, f"the provider never received {prompt!r}"
    return turn, _offered(requests[0]), _tool_results(requests[-1])


def test_a_switched_off_tool_is_withheld_from_a_chat_that_had_it_until_switched_on(
    admin, make_user, upstream
):
    chatter = make_user()
    with python_tool(admin, TIDE_TOOL, name="Tides") as tool_id, chatter.client() as client:
        turn, offered_on, ran_on = _tide_turn(client, upstream, tool_id)
        _switch(admin, tool_id, to=False)
        picker_off = _picker(client)
        turn, offered_off, ran_off = _tide_turn(client, upstream, tool_id, after=turn)
        _switch(admin, tool_id, to=True)
        picker_back = _picker(client)
        _, offered_back, ran_back = _tide_turn(client, upstream, tool_id, after=turn)

    assert "tide_mark" in offered_on, f"the selected tool was not offered: {sorted(offered_on)}"
    assert any(TIDE_MARK in result for result in ran_on), f"the tool returned {ran_on}"
    assert tool_id not in picker_off, "the switched-off tool is still in the chat's tool picker"
    assert "tide_mark" not in offered_off, (
        f"the switched-off tool was still offered to the model: {sorted(offered_off)}"
    )
    assert not any(TIDE_MARK in result for result in ran_off), (
        f"the switched-off tool still ran when the model called it: {ran_off}"
    )
    assert tool_id in picker_back, "the tool switched back on is missing from the tool picker"
    assert "tide_mark" in offered_back, (
        f"the tool switched back on was not offered: {sorted(offered_back)}"
    )
    assert any(TIDE_MARK in result for result in ran_back), (
        f"the tool switched back on returned {ran_back}"
    )


def _wait_for_request(upstream, prompt: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not _requests_answering(upstream, prompt):
        assert time.monotonic() < deadline, f"the provider never received {prompt!r}"
        time.sleep(0.05)


def test_a_tool_switched_off_while_a_reply_is_under_way_does_not_run(admin, make_user, upstream):
    chatter = make_user()
    prompt = f"read the tide ({uuid.uuid4().hex[:8]})"
    answering = reply.answering(prompt)
    # the model asks for the tool only after it has been switched off
    released = threading.Event()
    upstream.queue(
        reply.tool_call("tide_mark", {}, hold=released, match=answering),
        reply.text("done", match=answering),
    )
    with python_tool(admin, TIDE_TOOL, name="Tides") as tool_id, chatter.client() as client:
        try:
            turn = send_message(client, prompt, tool_ids=[tool_id])
            _wait_for_request(upstream, prompt)
            _switch(admin, tool_id, to=False)
        finally:
            released.set()
        wait_for_reply(client, turn)

    requests = _requests_answering(upstream, prompt)
    offered, results = _offered(requests[0]), _tool_results(requests[-1])
    assert "tide_mark" in offered, f"the tool was not offered while on: {sorted(offered)}"
    assert len(requests) == 2 and results, (
        f"the reply did not go on after the call; the provider got {len(requests)} requests"
    )
    assert not any(TIDE_MARK in result for result in results), (
        f"the tool switched off mid-reply still ran: {results}"
    )


def _switchable_tool() -> dict:
    return {
        "id": f"tool_{uuid.uuid4().hex[:8]}",
        "name": "Tides",
        "content": TIDE_TOOL,
        "meta": {"description": "switched on and off"},
    }


SWITCHABLE_TOOL = Shareable(
    create_path="/api/v1/tools/create",
    create_body=_switchable_tool,
    access_path="/api/v1/tools/id/{id}/access/update",
    owner_permissions={"workspace": {"tools": True}},
)


def _is_on(client, fields: dict) -> bool:
    stored = client.get(f"/api/v1/tools/id/{fields['id']}")
    assert stored.status_code == 200, stored.text
    return stored.json()["is_active"]


def test_only_accounts_that_may_edit_a_tool_can_switch_it(admin, make_user):
    accounts = cast(SWITCHABLE_TOOL, admin, make_user)

    answered = attempts(accounts, "POST", "/api/v1/tools/id/{id}/toggle", look=_is_on)

    states = {role: (attempt.before, attempt.after) for role, attempt in answered.items()}
    assert states == {
        "owner": (True, False),
        "stranger": (True, True),
        "reader": (True, True),
        "writer": (True, False),
        "admin": (True, False),
    }, f"switching a tool on and off, as (before, after) per account: {states}"
    refused = {role: answered[role].status for role in ("stranger", "reader")}
    assert all(status in (401, 403) for status in refused.values()), f"refusals: {refused}"
