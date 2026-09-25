"""Journey: an admin tunes a function's valves and source, and users set a shared tool's valves.

An admin saves a function's valves, reads them back and the filter or pipe uses them on the very
next chat. Saving new source replaces the running code on the next chat; source that does not
compile is refused, the stored code stays as it was and the function is switched off until the
admin turns it back on, as the functions docs describe. On a shared tool, anyone who may read
the tool keeps their own user valves, while the tool's admin valves need write access.

Discriminates: in a backend copy, the function valves update skipping its save turned both valves
tests red; the update route not replacing the cached module while the cache also ignored changed
source turned the updated-source test red; the update storing source whose load failed turned the
refused-source test red; the tool admin valves routes checking read in place of write turned the
reader test red, and the user valves routes checking write in place of read turned the user valves
test red. Each left the other tests green.
"""

from __future__ import annotations

import textwrap
import uuid

import pytest

from harness.access import grant
from harness.chat import ask
from harness.plugins import installed_function

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


def test_updated_source_runs_on_the_next_chat(admin):
    with installed_function(admin, versioned_pipe("one")) as pipe_id:
        before = _pipe_reply(admin, pipe_id)
        updated = _update_function_source(admin, pipe_id, versioned_pipe("two"))
        after = _pipe_reply(admin, pipe_id)

    assert updated.status_code == 200, updated.text
    assert (before, after) == ("answered by version one", "answered by version two")


def test_source_that_does_not_compile_is_refused_and_the_stored_code_runs_once_switched_on(admin):
    with installed_function(admin, versioned_pipe("one")) as pipe_id:
        _pipe_reply(admin, pipe_id)
        refused = _update_function_source(admin, pipe_id, UNCOMPILABLE_PIPE)
        with admin.client() as client:
            stored = client.get(f"/api/v1/functions/id/{pipe_id}").json()
            # a failed load switches the function off, as documented
            client.post(f"/api/v1/functions/id/{pipe_id}/toggle").raise_for_status()
        after = _pipe_reply(admin, pipe_id)

    assert refused.status_code == 400
    assert (stored["content"], stored["is_active"]) == (versioned_pipe("one"), False)
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
