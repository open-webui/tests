"""Automatic context compaction cut the conversation in the wrong place or fired at the wrong time.

Six 0.11.0 fixes in `utils/context_compaction.py` and `utils/response.py`. When a saved chat
grows past the token threshold, the next turn asks a model for a summary of the older turns and
sends the provider the system prompt (with the summary appended) plus the recent turns.

1. Turn boundaries (#27035, commits 959558f, 17e6496): the split was walked forward and then
   clamped back, so the kept part could start with an assistant reply. It now starts at the last
   user message the retention percentage allows.
2. Summary model (#26806): the configured compaction model was ignored.
3. System prompt (commits 70549c5, 1568868, 44f4f9d, #26713, #26710): the system message was
   folded into the summary and dropped from the request.
4. Continuity (#27037, commits 0c23466, f730733): the summary checkpoint was stored on the
   current message, so the next turn sliced away the whole kept window.
5. Context size (commits df94268, e8f2c12, #27031, #26752, #24410): a tool loop summed
   `prompt_tokens` over every provider call, and a chat's own threshold is now capped by the
   global one. The same commits taught the threshold to read Ollama and llama.cpp usage; the
   message table normalizes usage before compaction reads it, so that part only shows on a
   legacy chat and stays in unit/chat/test_context_compaction.py.
6. Context usage (commit 7a9928ef1, #27362): `GET /api/v1/chats/{id}` reports the tokens the next
   turn will be judged by.

Twin of unit/chat/test_context_compaction.py.

Discriminates, each fix reverted in its own copy: the old boundary walk fails the user-boundary,
checkpoint, next-turn, no-boundary, retention and summary-text tests; ignoring the compaction
model fails the configured-model case; folding the system message in fails the system-prompt
test and the empty-summary fallback; the checkpoint on the current message fails the checkpoint,
next-turn and summary-text tests; summing `prompt_tokens` fails the tool-loop test; dropping the
cap fails the capped case; ignoring the retention setting fails the keep-half case. The usage
dialect, under-threshold, compaction-off, clamp and normalization tests pass on all.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from harness import upstream as reply
from harness.chat import ChatTurn, ask
from harness.chat_history import seed_chat
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

CHAT_CONFIG = ("/api/v1/chats/config", "/api/v1/chats/config")
SYSTEM_PROMPT = "You are a careful travel agent."
SUMMARY = "SUMMARY OF THE EARLIER TURNS"
COMPACTION_MODEL = "compaction-model"


def is_summary_request(body: dict) -> bool:
    return not body.get("stream")


def summary_requests(upstream) -> list[dict]:
    return [body for body in upstream.chat_requests() if is_summary_request(body)]


def main_call(upstream) -> list[dict]:
    return [body for body in upstream.chat_requests() if body.get("stream")][-1]["messages"]


def turns(count: int, filler: int = 400) -> list[dict]:
    """`count` question and answer pairs; with the default filler each counts ~100 tokens."""
    messages = []
    for number in range(1, count + 1):
        messages.append({"role": "user", "content": f"question {number} " + "q" * filler})
        messages.append({"role": "assistant", "content": f"answer {number} " + "a" * filler})
    return [{**message, "id": str(uuid.uuid4())} for message in messages]


@pytest.fixture
def compaction(admin, preserve):
    """`compaction(**settings)` switches compaction on with those chat settings."""
    preserve(CHAT_CONFIG)

    def configure(threshold: int = 50, **settings) -> dict:
        with admin.client() as client:
            current = client.get(CHAT_CONFIG[0]).json()
            updated = client.post(
                CHAT_CONFIG[1],
                json={
                    **current,
                    "ENABLE_CONTEXT_COMPACTION": True,
                    "CONTEXT_COMPACTION_TOKEN_THRESHOLD": threshold,
                    "CONTEXT_COMPACTION_TOKEN_CAP": threshold,
                    **settings,
                },
            )
        assert updated.status_code == 200, updated.text
        return updated.json()

    return configure


def next_turn(client: httpx.Client, chat_id: str, parent_id: str, text: str, **options) -> ChatTurn:
    turn, _ = ask(
        client,
        text,
        chat_id=chat_id,
        parent_id=parent_id,
        history=[{"role": "system", "content": SYSTEM_PROMPT}],
        **options,
    )
    return turn


def stored_messages(client: httpx.Client, chat_id: str) -> dict:
    return client.get(f"/api/v1/chats/{chat_id}").json()["chat"]["history"]["messages"]


def compact_three_turns(client: httpx.Client, upstream) -> tuple[list[dict], ChatTurn]:
    """Three long turns, then a fourth question that pushes the chat over the threshold."""
    upstream.queue(reply.text(SUMMARY, match=is_summary_request), reply.text("answer 4"))
    history = turns(3)
    chat_id, last_id = seed_chat(client, history)
    return history, next_turn(client, chat_id, last_id, "question 4")


def test_the_kept_turns_start_at_a_user_message(user, upstream, compaction):
    compaction()
    with user.client() as client:
        history, _ = compact_three_turns(client, upstream)

    sent = main_call(upstream)
    assert len(summary_requests(upstream)) == 1
    assert [message["role"] for message in sent] == ["system", "user", "assistant", "user"]
    assert sent[1]["content"] == history[4]["content"]


def test_the_system_prompt_stays_first_with_the_summary(user, upstream, compaction):
    compaction()
    with user.client() as client:
        compact_three_turns(client, upstream)

    system = main_call(upstream)[0]
    assert system["role"] == "system"
    assert system["content"].startswith(SYSTEM_PROMPT)
    assert f"[CONVERSATION SUMMARY]\n{SUMMARY}" in system["content"]


def test_the_summary_is_stored_on_the_first_kept_message(user, upstream, compaction):
    compaction()
    with user.client() as client:
        history, turn = compact_three_turns(client, upstream)
        stored = stored_messages(client, turn.chat_id)

    assert stored[history[4]["id"]].get("contextSummary") == SUMMARY
    assert not stored[turn.user_message_id].get("contextSummary")


def test_the_next_turn_replays_the_summary_and_the_kept_turns(user, upstream, compaction):
    compaction()
    with user.client() as client:
        _, turn = compact_three_turns(client, upstream)
        compaction(threshold=1_000_000)
        upstream.queue(reply.text("answer 5"))
        next_turn(client, turn.chat_id, turn.assistant_message_id, "question 5")

    sent = main_call(upstream)
    assert f"[CONVERSATION SUMMARY]\n{SUMMARY}" in sent[0]["content"]
    assert [message["content"][:10] for message in sent[1:]] == [
        "question 3",
        "answer 3 a",
        "question 4",
        "answer 4",
        "question 5",
    ]


@pytest.fixture
def compaction_model(upstream, admin):
    """A second provider model the summary can be sent to."""
    upstream.models.append(COMPACTION_MODEL)
    with admin.client() as client:
        client.get("/api/models").raise_for_status()
    yield COMPACTION_MODEL
    upstream.models.remove(COMPACTION_MODEL)
    with admin.client() as client:
        client.get("/api/models").raise_for_status()


@pytest.mark.parametrize(
    "configured,expected",
    [
        pytest.param(COMPACTION_MODEL, COMPACTION_MODEL, id="configured-model"),
        pytest.param("not-a-model", MOCK_MODEL_ID, id="unknown-model-falls-back"),
    ],
)
def test_the_summary_goes_to_the_compaction_model(
    make_user, upstream, compaction, compaction_model, configured, expected
):
    compaction(CONTEXT_COMPACTION_MODEL=configured)
    # an admin account, so the extra model needs no access grant
    with make_user("admin").client() as client:
        compact_three_turns(client, upstream)

    [summary_request] = summary_requests(upstream)
    assert summary_request["model"] == expected


USAGE_DIALECTS = [
    pytest.param({"prompt_eval_count": 90_000, "eval_count": 500}, id="ollama"),
    pytest.param({"prompt_n": 90_000, "predicted_n": 500}, id="llama.cpp"),
    pytest.param({"prompt_tokens": 90_000, "completion_tokens": 500}, id="openai"),
    pytest.param({"input_tokens": 90_000, "output_tokens": 500}, id="normalized"),
]


def short_chat_with_usage(client: httpx.Client, usage: dict) -> tuple[str, str]:
    history = turns(2, filler=0)
    history[-1]["usage"] = usage
    return seed_chat(client, history)


@pytest.mark.parametrize("usage", USAGE_DIALECTS)
def test_reported_usage_counts_toward_the_threshold(user, upstream, compaction, usage):
    compaction(threshold=80_000)
    upstream.queue(reply.text(SUMMARY, match=is_summary_request), reply.text("answer 3"))
    with user.client() as client:
        chat_id, last_id = short_chat_with_usage(client, usage)
        context_usage = client.get(f"/api/v1/chats/{chat_id}").json()["context_usage"]
        next_turn(client, chat_id, last_id, "question 3")

    assert (context_usage["tokens"], context_usage["threshold"]) == (90_500, 80_000)
    assert len(summary_requests(upstream)) == 1, "the reported usage did not trigger compaction"


def test_usage_under_the_threshold_does_not_compact(user, upstream, compaction):
    compaction(threshold=80_000)
    with user.client() as client:
        chat_id, last_id = short_chat_with_usage(
            client, {"prompt_eval_count": 100, "eval_count": 10}
        )
        context_usage = client.get(f"/api/v1/chats/{chat_id}").json()["context_usage"]
        next_turn(client, chat_id, last_id, "question 3")

    assert context_usage["tokens"] == 110
    assert summary_requests(upstream) == []


def test_nothing_is_compacted_or_reported_while_compaction_is_off(user, upstream):
    with user.client() as client:
        chat_id, last_id = seed_chat(client, turns(3))
        context_usage = client.get(f"/api/v1/chats/{chat_id}").json()["context_usage"]
        next_turn(client, chat_id, last_id, "question 4")

    assert context_usage is None
    assert summary_requests(upstream) == []
    assert len(main_call(upstream)) == 8


def test_a_conversation_without_a_later_user_turn_is_sent_whole(user, upstream, compaction):
    compaction()
    replies = [
        {"role": "assistant", "content": f"part {number} " + "a" * 400} for number in (1, 2, 3)
    ]
    with user.client() as client:
        chat_id, last_id = seed_chat(client, [turns(1)[0], *replies])
        next_turn(client, chat_id, last_id, "question 2")

    assert summary_requests(upstream) == []
    assert [message["role"] for message in main_call(upstream)] == [
        "system",
        "user",
        "assistant",
        "assistant",
        "assistant",
        "user",
    ]


@pytest.mark.parametrize(
    "global_threshold,chat_threshold",
    [
        pytest.param(50, 500_000, id="chat-threshold-capped"),
        pytest.param(80_000, 50, id="lower-chat-threshold-wins"),
    ],
)
def test_a_chat_threshold_is_capped_by_the_global_one(
    user, upstream, compaction, global_threshold, chat_threshold
):
    compaction(threshold=global_threshold)
    upstream.queue(reply.text(SUMMARY, match=is_summary_request), reply.text("answer 4"))
    with user.client() as client:
        chat_id, last_id = seed_chat(client, turns(3))
        next_turn(
            client,
            chat_id,
            last_id,
            "question 4",
            params={"compact_token_threshold": chat_threshold},
        )

    assert len(summary_requests(upstream)) == 1


@pytest.mark.parametrize(
    "retention,first_kept",
    [
        pytest.param(50, "question 3", id="keep-half"),
        pytest.param(10, "question 4", id="keep-little"),
    ],
)
def test_the_retention_percentage_moves_the_boundary(
    user, upstream, compaction, retention, first_kept
):
    compaction(CONTEXT_COMPACTION_RETENTION_PERCENTAGE=retention)
    upstream.queue(reply.text(SUMMARY, match=is_summary_request), reply.text("answer 5"))
    with user.client() as client:
        chat_id, last_id = seed_chat(client, turns(4))
        next_turn(client, chat_id, last_id, "question 5")

    assert main_call(upstream)[1]["content"].startswith(first_kept)


@pytest.mark.parametrize("requested,stored", [(5, 10), (95, 50), (25, 25)])
def test_the_retention_percentage_is_clamped(compaction, requested, stored):
    saved = compaction(CONTEXT_COMPACTION_RETENTION_PERCENTAGE=requested)
    assert saved["CONTEXT_COMPACTION_RETENTION_PERCENTAGE"] == stored


@pytest.mark.parametrize(
    "reasoning,stored_prefix",
    [
        pytest.param("SUMMARY FROM REASONING", "SUMMARY FROM REASONING", id="reasoning-only"),
        pytest.param(None, "- user: question 1", id="empty-reply-falls-back"),
    ],
)
def test_a_summary_without_text_still_leaves_a_checkpoint(
    user, upstream, compaction, reasoning, stored_prefix
):
    compaction()
    summary = reply.text("", reasoning=reasoning, match=is_summary_request)
    upstream.queue(summary, reply.text("answer 4"))
    history = turns(3)
    with user.client() as client:
        chat_id, last_id = seed_chat(client, history)
        next_turn(client, chat_id, last_id, "question 4")
        stored = stored_messages(client, chat_id)

    assert stored[history[4]["id"]]["contextSummary"].startswith(stored_prefix)


def test_prompt_tokens_are_not_summed_across_a_tool_loop(user, upstream):
    first = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110, "cost": 1.5}
    second = {"prompt_tokens": 200, "completion_tokens": 20, "total_tokens": 220, "cost": 2.5}
    upstream.queue(
        reply.tool_call("get_current_timestamp", {}, usage=first),
        reply.text("It is late.", usage=second),
    )
    with user.client() as client:
        _, message = ask(client, "what time is it?")

    usage = message["usage"]
    assert (usage["prompt_tokens"], usage["completion_tokens"]) == (200, 20)
    assert (usage["total_tokens"], usage["cost"]) == (330, 4.0)


@pytest.mark.parametrize(
    "provider_usage",
    [
        pytest.param({"prompt_eval_count": 7, "eval_count": 3}, id="ollama"),
        pytest.param({"prompt_n": 7, "predicted_n": 3}, id="llama.cpp"),
        pytest.param({"prompt_tokens": 7, "completion_tokens": 3}, id="openai"),
    ],
)
def test_stored_usage_is_normalized(user, upstream, provider_usage):
    upstream.queue(reply.text("hi", usage=provider_usage))
    with user.client() as client:
        _, message = ask(client, "hello")

    usage = message["usage"]
    assert (usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]) == (7, 3, 10)
