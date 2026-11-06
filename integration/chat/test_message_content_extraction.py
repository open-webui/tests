"""0.11.0 fixes to how the backend reads the text, images and characters of a stored turn.

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

4. Memories from structured replies (commit 3fe0358, #26705, issue #26651). The background
   memory review read the reply's raw `content` and gave up when it was empty, so a reply whose
   text came only as `output` items (a finished completion from a provider that sends them) was
   never reviewed. It now reads the text the way (1) does.
5. Chained Responses turns keep the tool's image in the tool result: with
   `ENABLE_RESPONSES_API_STATEFUL` the follow-up after a tool call names the previous response
   and replays only the new items, where the Responses API takes an image inside
   `function_call_output`, so (2) must not apply there.

The sanitizer leaves values that are not text alone: numbers in a saved chat and in the chunk
metadata of a file added to a knowledge base.

Twin of unit/chat/test_message_content_extraction.py.

Discriminates: reverting the `output` fallback in `get_content_from_message` drops the reply
from the title prompt; replaying without `flatten_tool_images` sends the image inside the tool
message; reverting 43e7eef makes saving the surrogate chat answer 500. On dev ef67cc3fa,
reverting 3fe0358 fails the memory review test, flattening tool images on the chained path fails
the chained-turn test, the sanitizer stringifying non-text leaves fails the saved-chat test and
dropping its non-text guard fails the knowledge file test (HTTP 400). The nearby tests pass on
all.
"""

from __future__ import annotations

import json
import secrets
import time

import httpx
import pytest

from harness import raw_provider
from harness import responses_provider as responses_api
from harness import upstream as reply
from harness.actors import admin_of
from harness.chat import ask
from harness.chat_history import seed_chat
from harness.knowledge_bases import add_text_file, knowledge_base
from harness.mcp_server import TOOL_SERVERS, mcp_connection, serving_mcp
from harness.raw_provider import RAW_MODEL_ID
from harness.second_provider import OPENAI_CONFIG
from harness.terminal_server import read_grant
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
    return reply.conversation(upstream.chat_requests()[-1])


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


MEMORY_REVIEWER = "You are Open WebUI's private memory reviewer. Return only valid JSON."
MEMORY_REVIEW_ENV = {
    "ENABLE_MEMORIES": "true",
    "ENABLE_MEMORY_BACKGROUND_REVIEW": "true",
    "MEMORIES_REVIEW_INTERVAL_TURNS": "1",
}


def reply_with_output_only(text: str) -> dict:
    """A finished completion whose text only comes as an `output` message item."""
    return {
        "id": "structured",
        "object": "chat.completion",
        "model": RAW_MODEL_ID,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}}],
        "output": [
            {"type": "message", "role": "assistant", "content": output_text(text)[0]["content"]}
        ],
    }


def memory_reviews(raw, wait: float) -> list[dict]:
    deadline = time.monotonic() + wait
    while True:
        sent = [request.json() for request in raw.listener.requests_to("/v1/chat/completions")]
        reviews = [body for body in sent if body["messages"][0]["content"] == MEMORY_REVIEWER]
        if reviews or time.monotonic() > deadline:
            return reviews
        time.sleep(0.1)


@pytest.fixture
def structured_memory_chat(instance_with, preserve, listener):
    """A raw provider on an instance that reviews every turn for memories."""
    reviewing = instance_with(MEMORY_REVIEW_ENV)
    preserve(OPENAI_CONFIG, on=reviewing)
    reviewer = admin_of(reviewing)
    raw = raw_provider.connect(reviewer, listener)
    raw.complete(reply_with_output_only("You told me you moved to Vienna last year."))
    return reviewer, raw


@pytest.mark.slow
def test_a_reply_stored_only_as_output_is_reviewed_for_memories(structured_memory_chat):
    reviewer, raw = structured_memory_chat
    with reviewer.client() as client:
        ask(
            client,
            "where do I live now?",
            model=RAW_MODEL_ID,
            stream=False,
            features={"memory": True},
        )

    reviews = memory_reviews(raw, wait=10)
    assert len(reviews) == 1, "the memory review was skipped for a reply that came as output"
    assert "moved to Vienna last year" in reviews[0]["messages"][-1]["content"]


@pytest.mark.slow
def test_a_structured_reply_is_not_reviewed_without_the_memory_feature(structured_memory_chat):
    reviewer, raw = structured_memory_chat
    with reviewer.client() as client:
        ask(
            client,
            "where do I live now?",
            model=RAW_MODEL_ID,
            stream=False,
            features={"memory": False},
        )

    assert memory_reviews(raw, wait=1.5) == []


STATEFUL_RESPONSES_ENV = {"ENABLE_RESPONSES_API_STATEFUL": "true"}


@pytest.fixture
def stateful_camera(instance_with, preserve, listener):
    """(admin, Responses provider, MCP server id) on an instance that chains Responses turns."""
    stateful = instance_with(STATEFUL_RESPONSES_ENV)
    preserve(OPENAI_CONFIG, TOOL_SERVERS, on=stateful)
    owner = admin_of(stateful)
    server_id = f"camera_{secrets.token_hex(4)}"
    with serving_mcp(media=True) as url, owner.client() as client:
        provider = responses_api.connect_responses(client, listener)
        connection = mcp_connection(url, server_id, [read_grant(owner.id)])
        saved = client.post(TOOL_SERVERS[1], json={"TOOL_SERVER_CONNECTIONS": [connection]})
        assert saved.status_code == 200, saved.text
        yield owner, provider, server_id


@pytest.mark.slow
def test_a_chained_responses_turn_keeps_the_tool_image_in_the_tool_result(stateful_camera):
    owner, provider, server_id = stateful_camera
    provider.answer(
        responses_api.events_stream(
            *responses_api.function_call(f"{server_id}_snapshot", {}),
            responses_api.completed(response_id="resp_first"),
        ),
        responses_api.events_stream(
            *responses_api.message("I see a camera frame."),
            responses_api.completed(response_id="resp_second"),
        ),
    )
    with owner.client() as client:
        ask(
            client,
            "take a snapshot",
            model=responses_api.RESPONSES_MODEL,
            tool_ids=[f"server:mcp:{server_id}"],
        )

    follow_up = provider.sent()[-1]
    assert follow_up.get("previous_response_id") == "resp_first"
    [result] = [item for item in follow_up["input"] if item["type"] == "function_call_output"]
    assert isinstance(result["output"], list), result
    assert [part["type"] for part in result["output"]].count("input_image") == 1, result
    assert not [item for item in follow_up["input"] if item.get("role") == "user"], follow_up


def test_a_saved_chat_keeps_its_non_text_values_while_null_bytes_go(user):
    meta = {"count": 42, "ratio": 1.5, "flag": True, "missing": None, "note": f"a{chr(0)}b"}
    with user.client() as client:
        chat_id, message_id = seed_chat(client, [{"role": "user", "content": "hi", "meta": meta}])
        loaded = client.get(f"/api/v1/chats/{chat_id}")

    assert loaded.status_code == 200, loaded.text
    stored = loaded.json()["chat"]["history"]["messages"][message_id]
    assert stored["meta"] == {**meta, "note": "ab"}


def test_a_file_whose_chunks_carry_numbers_is_added_to_knowledge(admin):
    with admin.client() as client, knowledge_base(client) as knowledge_id:
        add_text_file(client, knowledge_id, "numbers.txt", "Chunk metadata holds numbers.")
