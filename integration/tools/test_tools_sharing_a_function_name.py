"""Regression: of two attached tools with a function of the same name, only the first could run.

Fix ab6976b5b (#32670). When two workspace tools, or two OpenAPI tool servers, both had a function
named `lookup`, Open WebUI kept the second one under a new name (its tool or server id in front)
but still offered it to the model as `lookup`. The model was offered the same name twice, so every
call it made ran the first tool. The second one is now offered under the name it is kept under,
and a chat with both attached can run both.

The scripted model reads the names it was offered, calls each of them in one turn, and the test
checks that both tools' answers came back to it.

Discriminates: passes on dev f36490bf2; with ab6976b5b reverted in a backend copy both tests fail
(the model is offered `lookup` twice).
"""

from __future__ import annotations

import uuid

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.listener import json_answer, text_answer
from harness.python_tools import python_tool

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

TOOL_SERVERS_CONFIG = ("/api/v1/configs/tool_servers", "/api/v1/configs/tool_servers")


def _toolkit(answer: str) -> str:
    return f'''
class Tools:
    def lookup(self) -> str:
        """Look the answer up."""
        return "{answer}"
'''


def _lookup_spec(title: str) -> dict:
    operation = {
        "operationId": "lookup",
        "summary": "Look the answer up.",
        "responses": {"200": {"description": "ok"}},
    }
    return {
        "openapi": "3.0.0",
        "info": {"title": title, "version": "1"},
        "paths": {"/lookup": {"get": operation}},
    }


def _offered_lookups(client, upstream, tool_ids: list[str]) -> list[str]:
    """The names the model was offered for the `lookup` functions, duplicates kept."""
    prompt = f"which tools are there? ({uuid.uuid4().hex[:8]})"
    answering = reply.answering(prompt)
    upstream.queue(reply.text("done", match=answering))
    ask(client, prompt, tool_ids=tool_ids)
    [request] = [request for request in upstream.chat_requests() if answering(request)]
    names = [tool["function"]["name"] for tool in request.get("tools") or []]
    return sorted(name for name in names if name.endswith("lookup"))


def _results_of_calling(client, upstream, names: list[str], tool_ids: list[str]) -> list[str]:
    """Have the model call every name in one turn; the tool results it was sent back."""
    calls = [
        {
            "id": f"call_{index}",
            "type": "function",
            "function": {"name": name, "arguments": "{}"},
        }
        for index, name in enumerate(names)
    ]
    prompt = f"look it up in both ({uuid.uuid4().hex[:8]})"
    answering = reply.answering(prompt)
    upstream.queue(
        reply.Reply(tool_calls=calls, match=answering), reply.text("done", match=answering)
    )
    ask(client, prompt, tool_ids=tool_ids)
    follow_up = [request for request in upstream.chat_requests() if answering(request)][-1]
    return sorted(
        entry["content"] for entry in reply.conversation(follow_up) if entry["role"] == "tool"
    )


def test_two_workspace_tools_with_the_same_function_both_run(admin, make_user, upstream):
    with (
        python_tool(admin, _toolkit("from the first tool"), name="First") as first_id,
        python_tool(admin, _toolkit("from the second tool"), name="Second") as second_id,
        make_user().client() as client,
    ):
        tool_ids = [first_id, second_id]
        offered = _offered_lookups(client, upstream, tool_ids)
        assert len(set(offered)) == 2, f"the model was offered {offered} for the two tools"

        results = _results_of_calling(client, upstream, offered, tool_ids)

    assert results == ["from the first tool", "from the second tool"], (
        f"calling both offered names did not run both tools: {results}"
    )


@pytest.fixture
def two_lookup_servers(admin, preserve, listener):
    """Two tool servers on the listener, each with a `lookup` operation of its own."""
    connections = []
    for server_id in ("north", "south"):
        prefix = f"/{server_id}-api"
        listener.route("GET", f"{prefix}/openapi.json", json_answer(_lookup_spec(server_id)))
        listener.route("GET", f"{prefix}/lookup", text_answer(f"from {server_id}", "text/plain"))
        connections.append(
            {
                "url": f"{listener.base_url}{prefix}",
                "path": "openapi.json",
                "auth_type": "none",
                "key": "",
                "config": {"enable": True},
                "info": {"id": server_id, "name": server_id},
            }
        )
    preserve(TOOL_SERVERS_CONFIG)
    with admin.client() as client:
        saved = client.post(TOOL_SERVERS_CONFIG[1], json={"TOOL_SERVER_CONNECTIONS": connections})
    assert saved.status_code == 200, saved.text
    return ["server:north", "server:south"]


def test_two_tool_servers_with_the_same_operation_both_run(make_user, upstream, two_lookup_servers):
    tool_ids = two_lookup_servers
    with make_user(role="admin").client() as client:
        offered = _offered_lookups(client, upstream, tool_ids)
        assert len(set(offered)) == 2, f"the model was offered {offered} for the two servers"

        results = _results_of_calling(client, upstream, offered, tool_ids)

    assert results == ["from north", "from south"], (
        f"calling both offered names did not reach both servers: {results}"
    )
