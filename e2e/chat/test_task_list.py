"""Journey: the task list a model keeps while it works shows above the message box.

A model with the builtin tools plans its work with `create_tasks` and ticks a task off with
`update_task`. While the reply is still being written the chat shows the list above its message
box, with how many tasks are done and each task's text, and the list goes once the reply ends.

Runs in the main chat and in every way a side chat starts and every state it can be in but a
temporary one (`utils/chat_surface.py`): a task list is stored with its chat, so a temporary chat
has none. A side chat never shows the list (open-webui/open-webui#34015, fix PR
open-webui/open-webui#34064), so the test is red there on dev c15f28649.

Discriminates: the side chat variants fail on dev c15f28649 (#34015) and pass on the build of
#34064; the main chat variant fails in a frontend build that never shows the task list.
"""

from __future__ import annotations

import threading
import uuid

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.python_tools import EVERYONE_READS
from harness.upstream import MOCK_MODEL_ID
from utils.chat_surface import EVERYWHERE, TEMPORARY_SIDE
from utils.chat_ui import REPLY_TIMEOUT_MS, expect_reply, send, start_from_link

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

TASKS = [
    {"id": "ticket", "content": "Buy a ferry ticket"},
    {"id": "pack", "content": "Pack a rain jacket"},
]


@pytest.fixture
def planning_model(admin):
    model_id = f"planner-{uuid.uuid4().hex[:6]}"
    form = {
        "id": model_id,
        "name": f"Trip planner {model_id[-6:]}",
        "base_model_id": MOCK_MODEL_ID,
        "meta": {"capabilities": {"builtin_tools": True}},
        "params": {"function_calling": "native"},
        "access_grants": [EVERYONE_READS],
    }
    with admin.client() as client:
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
        yield model_id
        client.post("/api/v1/models/model/delete", json={"id": model_id})


# several models: each answer's tasks belong to the one chat, the list shows above its message box;
# a temporary chat keeps no task list, in the main chat either (the tasks are stored with the chat)
@pytest.mark.chat_surfaces(*(place for place in EVERYWHERE if place != TEMPORARY_SIDE))
def test_the_task_list_shows_above_the_message_box_while_the_model_works(
    chat_for, make_user, upstream, planning_model
):
    """In a side chat: open-webui/open-webui#34015."""
    chat = chat_for(make_user())
    question = f"plan my trip to the island ({uuid.uuid4().hex[:6]})"
    planned = threading.Event()
    upstream.queue(
        reply.tool_call("create_tasks", {"tasks": TASKS}, match=reply.answering(question)),
        reply.tool_call(
            "update_task",
            {"id": "ticket", "status": "completed"},
            call_id="call_2",
            match=reply.answering(question),
        ),
        reply.text("The ticket is bought.", hold=planned, match=reply.answering(question)),
    )
    start_from_link(chat, f"/?models={planning_model}")
    send(chat, question)

    task_list = chat.panel.get_by_text("1 out of 2 tasks completed")
    expect(task_list, f"the {chat.place} shows no task list while it works").to_be_visible(
        timeout=REPLY_TIMEOUT_MS
    )
    expect(chat.panel.get_by_text("2. Pack a rain jacket", exact=True)).to_be_visible()
    planned.set()
    expect_reply(chat, "The ticket is bought.")
    expect(task_list).to_have_count(0)
