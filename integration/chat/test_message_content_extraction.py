"""Three 0.11.0 fixes to how the backend reads the text, images and characters of a stored turn.

1. Reply text missing (#26799, issue #26436). `get_content_from_message` only read a message's
   `content`, so a reply whose text lives in `output` (the structured-output path leaves
   `content` empty) came out as `None` wherever a task quoted the conversation. It now falls
   back to the text of the `output` message items.
2. Tool images (commit dd86b98). Images a tool returned were replayed inside a multimodal
   `role: tool` message, which OpenAI-compatible providers reject or ignore. The tool message
   is now text only and the images follow in a `role: user` message.
3. Unusual characters (commit 43e7eef, #27201, issue #27081). The sanitizer only looked for
   null bytes, so a lone UTF-16 surrogate reached the database and the chat failed to save and
   then to load. Surrogates are now stripped from values and keys.

The memory review fix (commit 3fe0358) needs a live reply whose text is only in `output`, which
no chat-completions provider produces, so it stays in unit/chat/test_message_content_extraction.py.

Twin of unit/chat/test_message_content_extraction.py.

Discriminates: reverting the `output` fallback in `get_content_from_message` drops the reply
from the title prompt; replaying without `flatten_tool_images` sends the image inside the tool
message; reverting 43e7eef makes saving the surrogate chat answer 500. The nearby tests pass on
all three.
"""

from __future__ import annotations

import json

import httpx
import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.chat_history import seed_chat
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

QUESTION = {"role": "user", "content": "What is the capital of France?"}
IMAGE_URL = "data:image/png;base64,iVBORw0KGgo="
LONE_SURROGATE = chr(0xD800)
LOW_SURROGATE = chr(0xDFFF)


def output_text(*texts: str) -> list[dict]:
    return [
        {"type": "message", "content": [{"type": "output_text", "text": text}]} for text in texts
    ]


@pytest.fixture
def title_generation_on(admin, preserve) -> None:
    preserve("tasks")
    with admin.client() as client:
        current = client.get("/api/v1/tasks/config").json()
        enabled = {**current, "ENABLE_TITLE_GENERATION": True}
        client.post("/api/v1/tasks/config/update", json=enabled).raise_for_status()


def title_prompt_for(user, upstream, messages: list[dict]) -> str:
    upstream.queue(reply.text('{"title": "A title"}'))
    with user.client() as client:
        response = client.post(
            "/api/v1/tasks/title/completions",
            json={"model": MOCK_MODEL_ID, "messages": messages},
        )
    assert response.status_code == 200, response.text
    return upstream.chat_requests()[-1]["messages"][-1]["content"]


def test_a_reply_stored_only_as_output_reaches_the_task_prompt(user, upstream, title_generation_on):
    structured = {"role": "assistant", "content": None, "output": output_text("It is Paris.")}

    prompt = title_prompt_for(user, upstream, [QUESTION, structured])

    assert "ASSISTANT: It is Paris." in prompt


def test_several_output_messages_join_without_blanks_or_reasoning(
    user, upstream, title_generation_on
):
    reasoning = {"type": "reasoning", "content": [{"type": "output_text", "text": "hidden"}]}
    output = [*output_text("first", "   "), reasoning, *output_text("second")]
    structured = {"role": "assistant", "content": "", "output": output}

    prompt = title_prompt_for(user, upstream, [QUESTION, structured])

    assert "ASSISTANT: first\nsecond" in prompt
    assert "hidden" not in prompt


def test_content_still_wins_over_output(user, upstream, title_generation_on):
    both = {"role": "assistant", "content": "from content", "output": output_text("from output")}

    prompt = title_prompt_for(user, upstream, [QUESTION, both])

    assert "ASSISTANT: from content" in prompt
    assert "from output" not in prompt


@pytest.mark.parametrize(
    "output",
    [None, [], "not-a-list", ["bare-string", 7], [{"type": "message", "content": None}]],
)
def test_a_reply_with_no_text_anywhere_does_not_break_the_task(
    user, upstream, title_generation_on, output
):
    empty = {"role": "assistant", "content": None, "output": output}

    prompt = title_prompt_for(user, upstream, [QUESTION, empty])

    assert "What is the capital of France?" in prompt


def tool_call_with_result(result_parts: list[dict], status: str = "completed") -> list[dict]:
    call = {
        "type": "function_call",
        "call_id": "call_chart",
        "name": "render_chart",
        "arguments": "{}",
        "status": status,
    }
    result = {
        "type": "function_call_output",
        "call_id": "call_chart",
        "output": result_parts,
        "status": "completed",
    }
    return [call, result]


def replay(user, upstream, output: list[dict]) -> list[dict]:
    """Seed a turn with `output`, send the next message, return what the provider got."""
    upstream.queue(reply.text("next answer"))
    with user.client() as client:
        chat_id, assistant_id = seed_chat(
            client,
            [QUESTION, {"role": "assistant", "content": "here it is", "output": output}],
        )
        ask(client, "and now?", chat_id=chat_id, parent_id=assistant_id)
    return upstream.chat_requests()[-1]["messages"]


CHART_RESULT = [
    {"type": "input_text", "text": "chart rendered"},
    {"type": "input_image", "image_url": IMAGE_URL},
]


def test_tool_images_follow_the_tool_message_as_a_user_message(user, upstream):
    messages = replay(
        user, upstream, [*tool_call_with_result(CHART_RESULT), *output_text("here it is")]
    )

    roles = [message["role"] for message in messages]
    assert roles == ["user", "assistant", "tool", "user", "assistant", "user"]
    assert messages[2]["content"] == "chart rendered"
    images = [part for part in messages[3]["content"] if part["type"] == "image_url"]
    assert images == [{"type": "image_url", "image_url": {"url": IMAGE_URL}}]
    assert messages[4]["content"] == "here it is"


def test_imageless_tool_output_is_replayed_unchanged(user, upstream):
    plain = [{"type": "input_text", "text": "plain"}]
    messages = replay(user, upstream, tool_call_with_result(plain))

    assert [message["role"] for message in messages] == ["user", "assistant", "tool", "user"]
    assert messages[2]["content"] == "plain"


def test_a_tool_call_awaiting_approval_is_not_replayed(user, upstream):
    messages = replay(user, upstream, tool_call_with_result(CHART_RESULT, status="pending"))

    assert [message["role"] for message in messages] == ["user", "assistant", "user"]
    assert "tool_calls" not in messages[1]


def post_escaped_json(client: httpx.Client, path: str, payload: dict) -> httpx.Response:
    # httpx encodes json= as UTF-8, which a lone surrogate cannot survive; a browser escapes it
    body = json.dumps(payload)
    return client.post(path, content=body, headers={"Content-Type": "application/json"})


def test_a_chat_with_lone_surrogates_saves_and_loads_without_them(user):
    message = {
        "id": "m1",
        "role": "user",
        "content": f"a{LONE_SURROGATE}b",
        "parentId": None,
        "childrenIds": [],
        "meta": {f"k{LONE_SURROGATE}": [f"v{LOW_SURROGATE}", {"nested": f"n{LONE_SURROGATE}"}]},
    }
    chat = {"title": "surrogates", "history": {"currentId": "m1", "messages": {"m1": message}}}
    with user.client() as client:
        created = post_escaped_json(client, "/api/v1/chats/new", {"chat": chat})
        assert created.status_code == 200, created.text
        loaded = client.get(f"/api/v1/chats/{created.json()['id']}")

    assert loaded.status_code == 200, loaded.text
    stored = loaded.json()["chat"]["history"]["messages"]["m1"]
    assert stored["content"] == "ab"
    assert stored["meta"] == {"k": ["v", {"nested": "n"}]}


def test_null_bytes_go_and_ordinary_characters_stay(user):
    emoji = chr(0x1F600)
    message = {"role": "user", "content": f"nul{chr(0)}byte, emoji {emoji} and café"}
    with user.client() as client:
        chat_id, message_id = seed_chat(client, [message])
        loaded = client.get(f"/api/v1/chats/{chat_id}")

    assert loaded.status_code == 200, loaded.text
    stored = loaded.json()["chat"]["history"]["messages"][message_id]
    assert stored["content"] == f"nulbyte, emoji {emoji} and café"
