"""Regressions in how a streamed chat turn is assembled and stored, seen through the chat API.

* #27414 / #27017 (`381149ea5`): the outlet payload embedded the stored `output` by reference, so
  a filter editing it in place also edited the baseline the change check compared against; the
  edit was never persisted.
* #26687 / #26645 (`051a1f6`): reasoning that arrived after the answer text was stored below
  it, and a `reasoning_details` delta with no text opened an empty thinking block.
* #26857 / #26836 (`d3cfcd801`): the model's system prompt was captured after `params` had been
  popped. A tool whose result carries a source makes the loop rebuild the system message from
  that capture, so the next round kept only the chat's own system message.
* #26986 (`b9d7274`): selected skills were injected in set order, which varies per process.
* #27426 / #27411 (`8ab44ed`): a follow-up after a tool call that raised (a provider hanging up)
  was logged at debug and the turn ended silently. An HTTP error status takes a later branch
  (`a610d7713`), pinned here as nearby.
* #27365 / #27074 (`dd514ee20`): a plain JSON error line (no `data:` prefix) was stored with an
  unawaited upsert, so the error vanished on reload.
* #29053 / #29035 (`3749e7dc7`): a delta carrying reasoning text and `reasoning_details`, as
  OpenRouter sends, had its reasoning event dropped, so no reasoning streamed to the client.
* #29052 / #29040 (`87bed3f0b`) only showed live in the page (its e2e twin); the stored thinking
  blocks after tool rounds are pinned here as nearby.

Twin of unit/chat/test_middleware_stream_assembly.py.

Discriminates: passes on dev `bbfa876af`; each narrow test fails with its fix reverted (the outlet
deepcopy, the late-reasoning placement, the contentless-details guard, the system prompt
capture, the skill sort, the tool-call continuation's error report, the awaited upsert and the
kept reasoning delta, one mutation each).
"""

from __future__ import annotations

import contextlib
import uuid

import pytest

from harness import upstream as reply
from harness.chat import ask, send_message, wait_for_reply
from harness.plugins import installed_function
from harness.second_provider import OPENAI_CONFIG, attach, sse, tool_call_delta
from harness.socket_client import connected

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

SECOND_MODEL = "second-model"
CLOCK_CALL = tool_call_delta("get_current_timestamp", {})


def _items(message: dict, kind: str) -> list[dict]:
    return [item for item in message.get("output", []) if item["type"] == kind]


def _text(item: dict) -> str:
    return "".join(part.get("text", "") for part in item.get("content") or [])


@pytest.fixture
def owner(make_user):
    """An admin, so a workspace model or a second connection's model needs no access grant."""
    return make_user(role="admin")


@pytest.fixture
def second_provider(preserve, admin, listener):
    """The listener as one more OpenAI connection serving `SECOND_MODEL`."""
    preserve(OPENAI_CONFIG)
    with admin.client() as client:
        attach(client, listener, SECOND_MODEL)
    return listener


def _answer_with(listener, *events) -> None:
    listener.route("POST", "/v1/chat/completions", sse(*events))


# --- #27414: an outlet filter's in-place edit to the output is persisted -----------------

APPEND_TO_OUTPUT = """
class Filter:
    def outlet(self, body: dict) -> dict:
        body["messages"][-1]["output"][-1]["content"][0]["text"] += " [reviewed]"
        return body
"""

REDACT_OUTPUT = """
class Filter:
    def outlet(self, body: dict) -> dict:
        body["messages"][-1]["output"][-1]["content"][0]["text"] = "REDACTED"
        return body
"""

REWRITE_CONTENT = """
class Filter:
    def outlet(self, body: dict) -> dict:
        body["messages"][-1]["content"] = "rewritten"
        return body
"""

CHANGE_NOTHING = """
class Filter:
    def outlet(self, body: dict) -> dict:
        return body
"""


@pytest.fixture
def outlet(admin):
    """`outlet(source)` installs a global filter for the rest of the test."""
    with contextlib.ExitStack() as installed:
        yield lambda source: installed.enter_context(
            installed_function(admin, source, is_global=True)
        )


def _reply_after_the_outlet(actor, prompt: str) -> dict:
    """The stored reply once the outlet filters ran; they finish after the reply is marked done."""
    with connected(actor) as socket, actor.client() as client:
        turn = send_message(client, prompt)
        socket.wait_for(turn.chat_id, "chat:outlet")
        return wait_for_reply(client, turn)


def test_an_outlet_edit_to_the_output_is_persisted(make_user, upstream, outlet):
    outlet(APPEND_TO_OUTPUT)
    upstream.queue(reply.text("hello there"))
    message = _reply_after_the_outlet(make_user(), "hi")

    assert _text(_items(message, "message")[-1]) == "hello there [reviewed]", (
        "the outlet filter's in-place edit to the structured output was never stored, so it "
        "vanished on reload (#27414)"
    )


def test_an_outlet_redaction_of_the_output_is_persisted(make_user, upstream, outlet):
    outlet(REDACT_OUTPUT)
    upstream.queue(reply.text("secret"))
    message = _reply_after_the_outlet(make_user(), "hi")

    assert _text(_items(message, "message")[-1]) == "REDACTED", (
        "a redacting outlet filter that rewrites the output in place was dropped on reload (#27414)"
    )
    assert message["originalContent"] == "secret"


def test_an_outlet_edit_to_the_content_is_persisted(make_user, upstream, outlet):
    outlet(REWRITE_CONTENT)
    upstream.queue(reply.text("hello there"))
    message = _reply_after_the_outlet(make_user(), "hi")

    assert message["content"] == "rewritten"


def test_an_outlet_that_changes_nothing_stores_nothing_extra(make_user, upstream, outlet):
    outlet(CHANGE_NOTHING)
    upstream.queue(reply.text("hello there"))
    message = _reply_after_the_outlet(make_user(), "hi")

    assert message["content"] == "hello there"
    assert "originalContent" not in message


# --- #26645: late reasoning is stored above the answer; empty details add nothing ---------


def test_reasoning_after_the_answer_is_stored_above_it(owner, second_provider):
    _answer_with(second_provider, {"content": "the answer is 42"}, {"reasoning": "second thoughts"})
    with owner.client() as client:
        _, message = ask(client, "think late", model=SECOND_MODEL)

    kinds = [item["type"] for item in message["output"]]
    assert kinds == ["reasoning", "message"], (
        "reasoning that arrived after the answer was stored below it, so the thinking block "
        "rendered under the answer (#26645)"
    )
    assert message["output"][0]["status"] == "completed"
    assert _text(message["output"][0]) == "second thoughts"


def test_reasoning_details_without_text_open_no_thinking_block(owner, second_provider):
    empty_details = {"reasoning_details": [{"type": "reasoning.text", "index": 0}]}
    _answer_with(second_provider, empty_details, {"content": "plain answer"})
    with owner.client() as client:
        _, message = ask(client, "no thoughts", model=SECOND_MODEL)

    assert _items(message, "reasoning") == [], (
        "a reasoning_details delta with no text, summary or data opened an empty thinking "
        "block (#26645)"
    )
    assert message["content"] == "plain answer"


def test_reasoning_details_with_text_open_a_thinking_block(owner, second_provider):
    details = {"reasoning_details": [{"type": "reasoning.text", "index": 0, "text": "step one"}]}
    _answer_with(second_provider, details, {"content": "done"})
    with owner.client() as client:
        _, message = ask(client, "some thoughts", model=SECOND_MODEL)

    reasoning = _items(message, "reasoning")
    assert len(reasoning) == 1
    assert reasoning[0]["reasoning_details"][0]["text"] == "step one"


# --- #29035: reasoning sent with its details still streams to the client -----------------


def test_reasoning_sent_with_details_streams_to_the_client(owner, second_provider):
    thought = {
        "reasoning": "weighing the options",
        "reasoning_details": [{"type": "reasoning.text", "index": 0, "text": "weighing"}],
    }
    _answer_with(second_provider, thought, {"content": "42"})
    with connected(owner) as socket, owner.client() as client:
        turn, _ = ask(client, "think aloud", model=SECOND_MODEL)
        streamed = [
            event["data"].get("delta")
            for event in socket.events_of(turn.chat_id)
            if event.get("type") == "response:completion"
            and event["data"].get("type") == "response.reasoning_text.delta"
        ]

    assert "".join(streamed) == "weighing the options", (
        "a reasoning delta that also carried reasoning_details never reached the client, so "
        "nothing streamed until the answer finished (#29035)"
    )


# --- #26836: the model's system prompt survives into the round after a tool call ---------

SYSTEM_PROMPT = "You are Ada, a terse assistant."


@pytest.fixture
def model_with_system_prompt(owner):
    model_id = f"ada-{uuid.uuid4().hex[:8]}"
    form = {
        "id": model_id,
        "base_model_id": reply.MOCK_MODEL_ID,
        "name": "Ada",
        "meta": {},
        "params": {"system": SYSTEM_PROMPT},
    }
    with owner.client() as client:
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
        yield model_id
        client.post("/api/v1/models/model/delete", json={"id": model_id})


def _system_prompt(request: dict) -> str:
    return next((m["content"] for m in request["messages"] if m["role"] == "system"), "")


@pytest.fixture
def owned_file_id(owner):
    with owner.client() as client:
        uploaded = client.post(
            "/api/v1/files/?process=false",
            files={"file": ("notes.txt", b"The meeting is at noon.", "text/plain")},
        )
        assert uploaded.status_code == 200, uploaded.text
        yield uploaded.json()["id"]
        client.delete(f"/api/v1/files/{uploaded.json()['id']}")


def test_the_model_system_prompt_is_resent_after_a_tool_round(
    owner, upstream, model_with_system_prompt, owned_file_id
):
    # a result with a citation source makes the tool loop rebuild the system message
    upstream.queue(
        reply.tool_call("view_knowledge_file", {"file_id": owned_file_id}),
        reply.text("It is at noon."),
    )
    personal = [{"role": "system", "content": "Answer in English."}]
    with owner.client() as client:
        _, message = ask(
            client, "when is the meeting?", model=model_with_system_prompt, history=personal
        )

    first_round, second_round = upstream.chat_requests()[-2:]
    assert message["content"].endswith("It is at noon.")
    assert SYSTEM_PROMPT in _system_prompt(first_round)
    assert SYSTEM_PROMPT in _system_prompt(second_round), (
        "the round after the tool call lost the model's system prompt: it was captured after "
        "params had been popped, so the tool loop rebuilt the system message without it (#26836)"
    )
    assert "Answer in English." in _system_prompt(second_round)


# --- #26986: skills are injected in a stable order ---------------------------------------


@pytest.fixture
def skills(owner):
    skill_ids = [f"skill-{index}-{uuid.uuid4().hex[:6]}" for index in range(8)]
    with owner.client() as client:
        for skill_id in skill_ids:
            form = {"id": skill_id, "name": skill_id, "content": f"Instructions of {skill_id}."}
            created = client.post("/api/v1/skills/create", json=form)
            assert created.status_code == 200, created.text
        yield skill_ids
        for skill_id in skill_ids:
            client.delete(f"/api/v1/skills/id/{skill_id}/delete")


def test_selected_skills_are_injected_in_sorted_order(owner, upstream, skills):
    selected = [*reversed(skills), skills[0]]
    with owner.client() as client:
        # legacy calling injects the selected skills' full text into the system prompt
        ask(client, "use my skills", skill_ids=selected, params={"function_calling": "legacy"})

    system_prompt = _system_prompt(upstream.chat_requests()[-1])
    positions = [system_prompt.find(f'<skill name="{skill_id}">') for skill_id in skills]
    assert -1 not in positions, f"a selected skill was not injected: {system_prompt[:500]}"
    assert positions == sorted(positions), (
        "selected skills were injected in set iteration order, which changes per process and "
        "defeats the provider's prompt caching (#26986)"
    )
    assert system_prompt.count(f'<skill name="{skills[0]}">') == 1


# --- #27411: a failure after a tool call is reported and stored ---------------------------


def test_a_provider_that_hangs_up_after_a_tool_call_leaves_an_error(owner, second_provider):
    def tool_call_then_hang_up(_request):
        if len(second_provider.requests_to("/v1/chat/completions")) == 1:
            return sse(CLOCK_CALL, finish_reason="tool_calls")
        raise ConnectionResetError("the provider hangs up")

    second_provider.route("POST", "/v1/chat/completions", tool_call_then_hang_up)
    with owner.client() as client:
        _, message = ask(client, "what time is it?", model=SECOND_MODEL)

    assert message.get("error", {}).get("content") == "Open WebUI: Server Connection Error", (
        "a reply that raised while continuing after a tool call was logged at debug and the "
        "turn ended with nothing said and nothing stored (#27411)"
    )


def test_a_provider_error_after_a_tool_call_is_stored(make_user, upstream):
    upstream.queue(
        reply.tool_call("get_current_timestamp", {}),
        reply.error(500, "provider exploded after the tool"),
    )
    with make_user().client() as client:
        _, message = ask(client, "what time is it?")

    assert "provider exploded after the tool" in str(message.get("error"))


def test_a_tool_round_that_succeeds_stores_no_error(make_user, upstream):
    upstream.queue(reply.tool_call("get_current_timestamp", {}), reply.text("It is now."))
    with make_user().client() as client:
        _, message = ask(client, "what time is it?")

    assert "error" not in message
    assert message["content"].endswith("It is now.")


# --- #27074: a plain JSON error line in the stream is stored -----------------------------


def test_a_plain_json_error_line_is_stored_on_the_message(owner, second_provider):
    _answer_with(second_provider, '{"error": "rate limit exceeded"}')
    with owner.client() as client:
        _, message = ask(client, "hello", model=SECOND_MODEL)

    assert message.get("error", {}).get("content") == "rate limit exceeded", (
        "the error line's upsert was never awaited, so the provider's error left no trace on "
        "the stored message (#27074)"
    )


def test_plain_lines_without_an_error_are_ignored(owner, second_provider):
    _answer_with(second_provider, '{"choices": []}', "not json at all", {"content": "fine"})
    with owner.client() as client:
        _, message = ask(client, "hello", model=SECOND_MODEL)

    assert "error" not in message
    assert message["content"] == "fine"


# --- Nearby for #29040: each round's thinking is stored as a block of its own ------------


def test_thinking_after_a_tool_call_gets_its_own_block(make_user, upstream):
    upstream.queue(
        reply.tool_call("get_current_timestamp", {}, reasoning="I should check the clock"),
        reply.text("It is now.", reasoning="the clock says now"),
    )
    with make_user().client() as client:
        _, message = ask(client, "what time is it?")

    reasoning = _items(message, "reasoning")
    assert [_text(item) for item in reasoning] == ["I should check the clock", "the clock says now"]
    kinds = [item["type"] for item in message["output"]]
    assert kinds.index("function_call_output") < message["output"].index(reasoning[1])
    assert _text(_items(message, "message")[-1]) == "It is now."


def test_every_tool_round_keeps_its_own_thinking_block(make_user, upstream):
    rounds = [
        reply.tool_call(
            "get_current_timestamp", {}, call_id=f"call_{step}", reasoning=f"step {step}"
        )
        for step in range(3)
    ]
    upstream.queue(*rounds, reply.text("42", reasoning="done"))
    with make_user().client() as client:
        _, message = ask(client, "count the rounds")

    thoughts = [_text(item) for item in _items(message, "reasoning")]
    assert thoughts == ["step 0", "step 1", "step 2", "done"]
    assert _text(_items(message, "message")[-1]) == "42"
