"""Dependency smoke: a workspace tool's spec, written by langchain-core.

When an admin saves a Python tool, Open WebUI turns each public method into a pydantic model and
langchain-core's `convert_to_openai_function` turns that model into the function spec the model
is offered: the method's name, the docstring's summary as its description, each parameter's JSON
type with its `:param` text, the choices of a `Literal` and which parameters are required. The
spec is stored with the tool and sent to the model whenever a chat has the tool switched on, and
the call the model makes with it runs the method.

Discriminates: passes on dev ef67cc3fa; in a backend copy whose conversion answers only the
name and description (no `parameters`), both tests fail.
"""

from __future__ import annotations

import uuid

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.upstream import conversation

pytestmark = [pytest.mark.depcheck, pytest.mark.api, pytest.mark.requires_source]

HARBOUR_TOOL = '''
from typing import Literal


class Tools:
    def book_berth(
        self,
        vessel: str,
        hours: int,
        tide: Literal["high", "low"] = "high",
        crew: list[str] = [],
    ) -> str:
        """
        Book a berth in the harbour.

        :param vessel: The vessel's name.
        :param hours: How long it stays.
        :param tide: The tide it arrives on.
        :param crew: Who is aboard.
        """
        return f"{vessel} is booked for {hours} hours on the {tide} tide"
'''


@pytest.fixture
def harbour_tool(admin):
    tool_id = f"harbour_{uuid.uuid4().hex[:8]}"
    form = {"id": tool_id, "name": "Harbour", "content": HARBOUR_TOOL, "meta": {}}
    with admin.client() as client:
        created = client.post("/api/v1/tools/create", json=form)
        assert created.status_code == 200, created.text
        yield tool_id
        client.delete(f"/api/v1/tools/id/{tool_id}/delete")


def _book_berth(spec: dict) -> None:
    """The spec of `book_berth`, as the model has to see it."""
    assert spec["name"] == "book_berth"
    # the chat path takes the docstring up to `:param` as it is, surrounding blank lines and all
    assert spec["description"].strip() == "Book a berth in the harbour."
    properties = spec["parameters"]["properties"]
    assert properties["vessel"]["type"] == "string"
    assert properties["vessel"]["description"] == "The vessel's name."
    assert properties["hours"]["type"] == "integer"
    assert properties["tide"]["enum"] == ["high", "low"]
    assert properties["crew"]["type"] == "array"
    assert properties["crew"]["items"]["type"] == "string"
    assert sorted(spec["parameters"]["required"]) == ["hours", "vessel"]


def test_a_saved_tool_carries_the_spec_of_its_method(admin, harbour_tool):
    with admin.client() as client:
        stored = client.get(f"/api/v1/tools/id/{harbour_tool}")

    assert stored.status_code == 200, stored.text
    [spec] = stored.json()["specs"]
    _book_berth(spec)


def test_the_model_is_offered_the_spec_and_its_call_runs_the_method(admin, harbour_tool, upstream):
    arguments = {"vessel": "Aurora", "hours": 6, "tide": "low"}
    upstream.queue(reply.tool_call("book_berth", arguments), reply.text("Booked."))

    with admin.client() as client:
        ask(client, "book a berth for the Aurora", tool_ids=[harbour_tool])

    first, follow_up = upstream.chat_requests()[-2:]
    offered = [tool["function"] for tool in first["tools"]]
    [offered] = [spec for spec in offered if spec["name"] == "book_berth"]
    _book_berth(offered)
    [tool_message] = [entry for entry in conversation(follow_up) if entry["role"] == "tool"]
    assert "Aurora is booked for 6 hours on the low tide" in tool_message["content"]
