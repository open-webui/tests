"""Journey: a visualization the model draws shows in the chat and works there.

Since 0.13.4 a model with Visualizations ticked under its Builtin Tools draws with
`render_visualization` (its offer and checks over HTTP are in
integration/tools/test_visualizations.py). The reply shows the drawing under its title in a frame
of its own, and once the tool accepted the call the drawing's script runs there, so a person can
work with it. A button in the drawing that asks the chat a question (`openwebui.sendMessage`)
first shows the question in a confirmation, and only a confirmed question is sent in the chat.

Discriminates: passes on the dev 485103cde build (the side chat case aside, see below). In a
backend copy whose tool refuses every drawing both tests fail (the script never runs, so the
drawing shows no tide level and its button asks nothing).

Every test runs in the main chat and in a side chat. The question test is red in the side chat on
dev 485103cde and 91f16464b: the click shows no confirmation and sends nothing, since the drawing
only asks the chat that has the page's address (open-webui/open-webui#34382). It passes on the
build of the fix 99439d150 (branch fix/visualization-question-in-side-chat).
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Iterator

import pytest
from playwright.sync_api import Locator, expect

from harness import upstream as reply
from harness.python_tools import EVERYONE_READS
from harness.upstream import MOCK_MODEL_ID
from utils.chat_surface import ChatSurface
from utils.chat_ui import conversation, expect_reply, last_reply, send, start_from_link

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

TOOL = "render_visualization"
TIDE_GAUGE = (
    '<p id="level">not drawn yet</p>'
    "<script>document.getElementById('level').textContent = 'High water at six';</script>"
)


@contextmanager
def drawing_model(owner) -> Iterator[str]:
    """A native function calling preset with Visualizations ticked under its Builtin Tools."""
    model_id = f"drawing-{uuid.uuid4().hex[:8]}"
    form = {
        "id": model_id,
        "base_model_id": MOCK_MODEL_ID,
        "name": model_id,
        "meta": {"capabilities": {}, "builtinTools": {"visualizations": True}},
        "params": {"function_calling": "native"},
        "access_grants": [EVERYONE_READS],
    }
    with owner.client() as client:
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
        try:
            yield model_id
        finally:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


@pytest.fixture
def model_id(admin) -> Iterator[str]:
    with drawing_model(admin) as created:
        yield created


def draw(chat: ChatSurface, upstream, model_id: str, arguments: dict) -> Locator:
    """Have the model draw `arguments` in a reply; the drawing under its title."""
    start_from_link(chat, f"/?model={model_id}")
    expect(chat.button(f"Selected model: {model_id}")).to_be_visible()
    question = f"draw the tide ({uuid.uuid4().hex[:8]})"
    answering = reply.answering(question)
    upstream.queue(
        reply.tool_call(TOOL, arguments, match=answering),
        reply.text("There it is.", match=answering),
    )
    send(chat, question)
    expect_reply(chat, "There it is.")
    return last_reply(chat).get_by_role("region", name=arguments["title"])


def test_a_drawing_shows_under_its_title_and_its_script_runs(
    chat_for, make_user, upstream, model_id
):
    chat = chat_for(make_user())

    drawing = draw(chat, upstream, model_id, {"title": "Harbour tide", "html": TIDE_GAUGE})

    expect(drawing, "the drawing is not shown under its title").to_be_visible()
    frame = drawing.frame_locator("iframe[title='Harbour tide']")
    expect(frame.locator("#level"), "the drawing's script did not run").to_have_text(
        "High water at six"
    )


def test_a_question_from_a_drawing_is_sent_only_once_confirmed(
    chat_for, make_user, upstream, model_id
):
    """Red in the side chat on dev 485103cde: the drawing's question shows no confirmation
    (open-webui/open-webui#34382)."""
    chat = chat_for(make_user())
    question = f"When is low water? ({uuid.uuid4().hex[:8]})"
    asker = (
        '<button id="ask" type="button">Ask about low water</button>'
        "<script>document.getElementById('ask').addEventListener('click', () => "
        f"openwebui.sendMessage('{question}'));</script>"
    )
    upstream.queue(reply.text("Just after noon.", match=reply.answering(question)))
    drawing = draw(chat, upstream, model_id, {"title": "Tide questions", "html": asker})
    button = drawing.frame_locator("iframe").get_by_role("button", name="Ask about low water")

    button.click()
    dialog = chat.page.get_by_role("dialog").filter(has_text="Confirm Prompt from Embed")
    expect(dialog).to_contain_text(question)
    dialog.get_by_role("button", name="Cancel").click()
    expect(dialog).to_be_hidden()
    expect(conversation(chat).get_by_text(question), "a cancelled question was sent").to_have_count(
        0
    )

    button.click()
    expect(dialog).to_contain_text(question)
    dialog.get_by_role("button", name="Confirm").click()

    expect_reply(chat, "Just after noon.")
    expect(conversation(chat).locator(".chat-user").last).to_contain_text(question)
