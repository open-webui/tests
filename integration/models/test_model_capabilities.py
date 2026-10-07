"""Journey: what a model's capabilities, builtin tool categories and extras send to the provider.

Sent the way the web client sends a message, a chat on a preset is offered every builtin tool
category the instance has on; a preset with one category unticked (`meta.builtinTools`) is
offered the same tools without that category's. With the Memory capability off a stored memory
reaches the model neither as context nor through the memory tools. A filter and an action
attached to one preset stay with it: the model list shows the action on that preset only and a
chat on another preset goes out unfiltered. A system prompt sent by a caller other than the
browser has the person's name, date and groups filled in by the server, while the time zone and
language, which only the browser knows, stay literal as the temporal awareness docs page says.
Usage and File Context are covered in integration/chat/test_provider_payload_assembly.py and
integration/chat/test_prompt_prefix_caching.py.

Discriminates: in a backend copy whose builtin tools ignore the model's unticked categories every
category case fails; in one that ignores the Memory capability the memory test fails; in one that
runs every active filter and lists every active action on every model the scope test fails; in
one that leaves the groups out of the system prompt the server variables test fails.
"""

from __future__ import annotations

import contextlib
import re
import uuid
from datetime import datetime
from typing import Iterator

import pytest

from harness.access import make_group
from harness.chat import ask
from harness.plugins import installed_function
from harness.python_tools import EVERYONE_READS
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

# the categories a chat is offered without any further setting, and every tool of each
CATEGORIES = {
    "time": {"get_current_timestamp", "calculate_timestamp"},
    "user_input": {"ask_user"},
    "memory": {
        "search_memories",
        "list_memory_paths",
        "read_memory_path",
        "list_memories",
        "update_memory",
        "add_memory",
        "replace_memory_content",
        "delete_memory",
    },
    "chats": {"search_chats", "view_chat"},
    "notes": {"search_notes", "view_note", "write_note", "replace_note_content"},
    "knowledge": {
        "list_knowledge_bases",
        "search_knowledge_bases",
        "query_knowledge_bases",
        "grep_knowledge_files",
        "search_knowledge_files",
        "query_knowledge_files",
        "view_knowledge_file",
    },
    "tasks": {"create_tasks", "update_task"},
    "automations": {
        "create_automation",
        "update_automation",
        "list_automations",
        "toggle_automation",
        "delete_automation",
    },
    "calendar": {
        "search_calendar_events",
        "create_calendar_event",
        "update_calendar_event",
        "delete_calendar_event",
    },
}
MEMORY_ON = {"features": {"memory": True}}

GATE_FILTER = """class Filter:
    def inlet(self, body: dict, __user__=None) -> dict:
        body["messages"][-1]["content"] += " (checked at the harbour gate)"
        return body
"""

DESK_ACTION = """class Action:
    async def action(self, body: dict, __user__=None, __event_emitter__=None):
        return None
"""


@contextlib.contextmanager
def preset(owner, params: dict | None = None, **meta) -> Iterator[str]:
    """A preset on the scripted model every account may chat with, carrying `meta`."""
    model_id = f"capable-{uuid.uuid4().hex[:8]}"
    form = {
        "id": model_id,
        "base_model_id": MOCK_MODEL_ID,
        "name": model_id,
        "meta": meta,
        "params": params or {},
        "access_grants": [EVERYONE_READS],
    }
    with owner.client() as client:
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
        try:
            yield model_id
        finally:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


def sent_to_provider(actor, upstream, model_id: str, question: str, **options) -> dict:
    with actor.client() as client:
        ask(client, question, model=model_id, **options)
    return next(
        body
        for body in upstream.chat_requests()
        if question in str(body["messages"][-1]["content"])
    )


def offered(request: dict) -> set[str]:
    return {tool["function"]["name"] for tool in request.get("tools") or []}


@pytest.mark.parametrize("category", CATEGORIES)
def test_an_unticked_category_withholds_only_its_tools(make_user, upstream, category):
    owner = make_user(role="admin")
    tools = CATEGORIES[category]
    with preset(owner) as every_category, preset(owner, builtinTools={category: False}) as unticked:
        before = offered(sent_to_provider(owner, upstream, every_category, "ticked?", **MEMORY_ON))
        after = offered(sent_to_provider(owner, upstream, unticked, "unticked?", **MEMORY_ON))

    assert tools <= before, f"not offered while ticked: {sorted(tools - before)}"
    assert not tools & after, f"still offered once unticked: {sorted(tools & after)}"
    lost, gained = before - tools - after, after - before
    assert not lost and not gained, (
        f"other tools changed: lost {sorted(lost)}, gained {sorted(gained)}"
    )


def test_without_the_memory_capability_no_memory_reaches_the_model(make_user, upstream):
    owner = make_user(role="admin")
    memory = f"My boat is the Seagull, berth {uuid.uuid4().hex[:6]}"
    with owner.client() as client:
        added = client.post("/api/v1/memories/add", json={"content": memory})
    assert added.status_code == 200, added.text

    with preset(owner) as remembering, preset(owner, capabilities={"memory": False}) as forgetful:
        with_memory = sent_to_provider(owner, upstream, remembering, "my boat?", **MEMORY_ON)
        without = sent_to_provider(owner, upstream, forgetful, "my boat now?", **MEMORY_ON)

    assert memory in str(with_memory["messages"])
    assert CATEGORIES["memory"] <= offered(with_memory)
    assert memory not in str(without["messages"]), "the memory reached a model without Memory"
    assert not CATEGORIES["memory"] & offered(without), sorted(offered(without))


def test_a_filter_and_an_action_stay_with_their_model(admin, make_user, upstream):
    owner = make_user(role="admin")
    with (
        installed_function(admin, GATE_FILTER) as filter_id,
        installed_function(admin, DESK_ACTION) as action_id,
        preset(owner, filterIds=[filter_id], actionIds=[action_id]) as master,
        preset(owner) as clerk,
    ):
        with owner.client() as client:
            listed = {model["id"]: model for model in client.get("/api/models").json()["data"]}
        on_master = sent_to_provider(owner, upstream, master, "may I moor?")
        on_clerk = sent_to_provider(owner, upstream, clerk, "may I moor too?")

    actions = {
        model_id: [a["id"] for a in listed[model_id].get("actions", [])]
        for model_id in (master, clerk)
    }
    assert action_id in actions[master], actions
    assert action_id not in actions[clerk], actions
    assert on_master["messages"][-1]["content"] == "may I moor? (checked at the harbour gate)"
    assert on_clerk["messages"][-1]["content"] == "may I moor too?"


def test_the_server_fills_what_it_knows_and_leaves_the_browsers_values_literal(
    admin, make_user, upstream
):
    caller = make_user()
    staff_id = make_group(admin, [])
    crew_id = make_group(admin, [caller], parent_id=staff_id)
    with admin.client() as client:
        names = {
            client.get(f"/api/v1/groups/id/{group_id}").json()["name"]
            for group_id in (crew_id, staff_id)
        }
    system_prompt = (
        "For {{USER_NAME}} of {{USER_GROUPS}} on {{CURRENT_DATE}}, "
        "in {{CURRENT_TIMEZONE}} reading {{USER_LANGUAGE}}."
    )
    with preset(admin, params={"system": system_prompt}) as model_id:
        day_before = datetime.now().strftime("%Y-%m-%d")
        request = sent_to_provider(caller, upstream, model_id, "who am I?")

    system = request["messages"][0]["content"]
    filled = re.fullmatch(
        rf"For {re.escape(caller.name)} of (.*) on (\d{{4}}-\d{{2}}-\d{{2}}), "
        r"in \{\{CURRENT_TIMEZONE\}\} reading \{\{USER_LANGUAGE\}\}\.",
        system,
    )
    assert filled, system
    assert set(filled.group(1).split(", ")) == names, system
    assert filled.group(2) in {day_before, datetime.now().strftime("%Y-%m-%d")}, system
