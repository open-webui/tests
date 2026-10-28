"""Automatic context compaction cut the conversation in the wrong place or fired at the wrong time.

Six 0.11.0 fixes and two later ones in `utils/context_compaction.py` and `utils/response.py`.
When a saved chat grows past the token threshold, the next turn asks a model for a summary of the
older turns and sends the provider the system prompt (with the summary appended) plus the recent
turns.

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
   global one. The same commits taught the threshold to read Ollama, llama.cpp and OpenAI usage.
   A reply stored today has its usage normalized on the way into the message table, but the
   migration that built that table copied each older reply's usage verbatim, so an upgraded
   chat still holds the provider's own keys. That chat is made here by rewriting the usage in a
   copy of the database and booting on it.
6. Context usage (commit 7a9928ef1, #27362): `GET /api/v1/chats/{id}` reports the tokens the next
   turn will be judged by.
7. One long exchange (#32781, PR #32794): a chat over the threshold with a single earlier user turn
   was never compacted (three messages or fewer were skipped, and no later user message sat far
   enough back to keep the retention share), so the follow-up went out whole and failed against
   the model's context window. It is now cut at the follow-up. The summary prompt renders only
   each message's visible text, so tool output stored on the earlier reply reaches neither request.
8. Model threshold for the usage gauge (#32821, commit c9415a688): a model's own threshold, set
   on the workspace model or in the admin model defaults, decided when a turn compacted, but the
   model list left it out, so the chat's usage gauge measured against the global threshold and
   read a few percent right before compaction fired. The model list now carries the threshold,
   the admin default filling in where the model sets none.
9. Temporary chats (commit d5cd7f490): a temporary chat (`temporary:` or the older `local:` id)
   is never stored, so it was never compacted. It now compacts like a saved chat; the summary is
   not stored on the server but sent to the page with the compaction event, naming the message
   it belongs on, and the page sends it back on that message with the next turn.

Twin of unit/chat/test_context_compaction.py.

Discriminates, each fix reverted in its own copy: the old boundary walk fails the user-boundary,
checkpoint, next-turn, one-exchange, retention and summary-text tests; ignoring the compaction
model fails the configured-model case; folding the system message in fails the system-prompt
test and the empty-summary fallback; the checkpoint on the current message fails the checkpoint,
next-turn and summary-text tests; summing `prompt_tokens` fails the tool-loop test; dropping the
cap fails the capped case; ignoring the retention setting fails the keep-half case; reading only
`input_tokens` (dev ef67cc3fa) fails the upgraded-chat test for all three dialects. The usage
dialect, under-threshold, compaction-off, clamp and normalization tests pass on all. Both
one-exchange cases pass with the #32781 fix (0adecf863) and fail on dev 2d9852c60 before it (no
summary is made and the whole chat is sent, the tool output included). The model-threshold
cases pass on dev f844f6e89 and fail on c9415a688^ (the model list carries no threshold while
the turn still compacts at the model's). The temporary-chat cases pass on dev 226dca18c and fail
with the temporary id dropped from the compaction check (no summary is made) or with the summary
and its message dropped from the compaction event (the page has nothing to send back).
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path

import httpx
import pytest

from harness import backends
from harness import upstream as reply
from harness.chat import ChatTurn, ask, send_message
from harness.chat_history import seed_chat
from harness.prepared_data import serving, snapshot_database
from harness.socket_client import connected
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

CHAT_CONFIG = ("/api/v1/chats/config", "/api/v1/chats/config")
MODELS_CONFIG = ("/api/v1/configs/models", "/api/v1/configs/models")
SYSTEM_PROMPT = "You are a careful travel agent."
SUMMARY = "SUMMARY OF THE EARLIER TURNS"
COMPACTION_MODEL = "compaction-model"
FOLLOW_UP = "question 2"
TOOL_RESULT_ROW = "tide table row "
TOOL_ANSWER = "The harbour log is clean."


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


def one_question_in_three_long_parts() -> list[dict]:
    replies = [
        {"role": "assistant", "content": f"part {number} " + "a" * 400} for number in (1, 2, 3)
    ]
    return [turns(1)[0], *replies]


def one_question_and_a_large_tool_result() -> list[dict]:
    """A short reply whose stored output holds the 150 KB a tool returned."""
    output = [
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "read_log",
            "arguments": "{}",
            "status": "completed",
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": [{"type": "input_text", "text": TOOL_RESULT_ROW * 10_000}],
            "status": "completed",
        },
        {"type": "message", "content": [{"type": "output_text", "text": TOOL_ANSWER}]},
    ]
    return [
        {"role": "user", "content": "read me the harbour log"},
        {"role": "assistant", "content": TOOL_ANSWER, "output": output},
    ]


@pytest.mark.parametrize(
    "first_exchange",
    [
        pytest.param(one_question_in_three_long_parts, id="long-replies"),
        pytest.param(one_question_and_a_large_tool_result, id="large-tool-result"),
    ],
)
def test_one_long_exchange_is_summarized_before_the_follow_up(
    user, upstream, compaction, first_exchange
):
    compaction()
    upstream.queue(reply.text(SUMMARY, match=is_summary_request), reply.text("answer 2"))
    history = first_exchange()
    with user.client() as client:
        chat_id, last_id = seed_chat(client, history)
        next_turn(client, chat_id, last_id, FOLLOW_UP)

    summaries = summary_requests(upstream)
    sent = main_call(upstream)
    roles = [message["role"] for message in sent]
    assert len(summaries) == 1, f"no summary was made and the whole chat was sent: {roles}"
    assert roles == ["system", "user"], f"more than the follow-up was kept: {roles}"
    assert f"[CONVERSATION SUMMARY]\n{SUMMARY}" in sent[0]["content"]
    assert sent[1]["content"] == FOLLOW_UP
    summary_request = json.dumps(summaries[0])
    assert history[-1]["content"] in summary_request, "the summary left out the earlier reply"
    assert TOOL_RESULT_ROW not in summary_request, (
        f"the summary request carries the tool output ({len(summary_request)} characters)"
    )
    assert TOOL_RESULT_ROW not in json.dumps(sent), "the follow-up was sent with the tool output"


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


@pytest.fixture
def default_model_params(admin, preserve):
    """`default_model_params(params)` sets the parameters every model starts from."""
    preserve(MODELS_CONFIG)

    def configure(params: dict) -> None:
        with admin.client() as client:
            current = client.get(MODELS_CONFIG[0]).json()
            updated = client.post(
                MODELS_CONFIG[1], json={**current, "DEFAULT_MODEL_PARAMS": params}
            )
        assert updated.status_code == 200, updated.text

    return configure


@pytest.mark.parametrize(
    "model_threshold,default_threshold",
    [
        pytest.param(500, None, id="workspace-model"),
        pytest.param(None, 500, id="admin-model-defaults"),
        pytest.param(500, 900, id="workspace-model-over-admin-defaults"),
    ],
)
def test_a_users_model_list_carries_the_models_own_threshold(
    admin, make_user, upstream, compaction, default_model_params, model_threshold, default_threshold
):
    compaction(threshold=1_000_000)
    if default_threshold is not None:
        default_model_params({"compact_token_threshold": default_threshold})
    model_params = {} if model_threshold is None else {"compact_token_threshold": model_threshold}
    model_id = f"small-window-{uuid.uuid4().hex[:8]}"
    account = make_user()
    with admin.client() as client:
        created = client.post(
            "/api/v1/models/create",
            json={
                "id": model_id,
                "base_model_id": MOCK_MODEL_ID,
                "name": "Small window",
                "meta": {},
                "params": model_params,
                "access_grants": [
                    {"principal_type": "user", "principal_id": account.id, "permission": "read"}
                ],
            },
        )
        assert created.status_code == 200, created.text
    upstream.queue(reply.text(SUMMARY, match=is_summary_request), reply.text("answer 4"))
    try:
        with account.client() as client:
            listed = client.get("/api/models", params={"refresh": True})
            assert listed.status_code == 200, listed.text
            [model] = [entry for entry in listed.json()["data"] if entry["id"] == model_id]
            chat_id, last_id = seed_chat(client, turns(3), model=model_id)
            next_turn(client, chat_id, last_id, "question 4", model=model_id)
    finally:
        with admin.client() as client:
            client.post("/api/v1/models/model/delete", json={"id": model_id})

    assert len(summary_requests(upstream)) == 1, "the model's own threshold did not compact"
    listed_params = (model.get("info") or {}).get("params") or {}
    assert listed_params.get("compact_token_threshold") == 500, (
        f"the model list gives the chat page {listed_params} for the model, "
        "so its usage gauge cannot measure against the 500 tokens the model compacts at"
    )


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


UPGRADED_USAGE = {
    "ollama": {"prompt_eval_count": 90_000, "eval_count": 500},
    "llama.cpp": {"prompt_n": 90_000, "predicted_n": 500},
    "openai": {"prompt_tokens": 90_000, "completion_tokens": 500},
}


def store_usage_as_an_upgrade_left_it(database: Path, chat_ids: dict[str, str]) -> None:
    """The message table migration copied each reply's usage verbatim, provider keys and all."""
    with closing(sqlite3.connect(database)) as connection, connection:
        for dialect, chat_id in chat_ids.items():
            connection.execute(
                "UPDATE chat_message SET usage = ? WHERE chat_id = ? AND role = 'assistant'",
                (json.dumps(UPGRADED_USAGE[dialect]), chat_id),
            )


@pytest.mark.slow
@pytest.mark.skipif(
    backends.DATABASE == "postgres", reason="copies the SQLite file, none on Postgres"
)
def test_usage_an_upgrade_stored_in_the_provider_dialect_counts(
    instance, user, compaction, tmp_path
):
    compaction(threshold=80_000)
    with user.client() as client:
        chat_ids = {
            dialect: short_chat_with_usage(client, {"input_tokens": 1})[0]
            for dialect in UPGRADED_USAGE
        }
    store_usage_as_an_upgrade_left_it(snapshot_database(instance, tmp_path), chat_ids)

    with serving(tmp_path) as upgraded, upgraded.client(user.token) as client:
        counted = {
            dialect: client.get(f"/api/v1/chats/{chat_id}").json()["context_usage"]["tokens"]
            for dialect, chat_id in chat_ids.items()
        }

    assert counted == dict.fromkeys(UPGRADED_USAGE, 90_500), (
        "usage an upgrade left in a provider's own keys no longer counts toward compaction"
    )


def wait_for_replies(socket, chat_id: str, count: int, timeout: float = 30.0) -> None:
    """Until the tab has seen `count` finished replies in the chat."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        finished = [
            event
            for event in socket.events_of(chat_id)
            if event.get("type") == "chat:completion" and (event.get("data") or {}).get("done")
        ]
        if len(finished) >= count:
            return
        time.sleep(0.05)
    raise AssertionError(f"reply {count} of temporary chat {chat_id} never finished")


@pytest.mark.parametrize("prefix", ["temporary:", "local:"])
def test_a_temporary_chat_compacts_and_hands_the_summary_to_the_page(
    make_user, upstream, compaction, prefix
):
    compaction()
    person = make_user()
    history = turns(3)
    system = {"role": "system", "content": SYSTEM_PROMPT}
    upstream.queue(reply.text(SUMMARY, match=is_summary_request), reply.text("answer 4"))
    with connected(person) as socket, person.client() as client:
        chat_id = f"{prefix}{socket.client.sid}"
        send_message(
            client,
            "question 4",
            chat_id=chat_id,
            session_id=socket.client.sid,
            history=[system, *history],
        )
        compacted = socket.wait_for(chat_id, "context_compaction", done=True)
        wait_for_replies(socket, chat_id, 1)
        sent = main_call(upstream)

        # the page keeps the summary on the checkpoint message and sends it with the next turn
        history[4]["contextSummary"] = compacted["data"].get("context_summary")
        compaction(threshold=1_000_000)
        upstream.queue(reply.text("answer 5"))
        turn_4 = [
            {"role": "user", "content": "question 4"},
            {"role": "assistant", "content": "answer 4"},
        ]
        send_message(
            client,
            "question 5",
            chat_id=chat_id,
            session_id=socket.client.sid,
            history=[system, *history, *turn_4],
        )
        wait_for_replies(socket, chat_id, 2)
        stored = client.get(f"/api/v1/chats/{chat_id}")

    assert len(summary_requests(upstream)) == 1
    assert [message["role"] for message in sent] == ["system", "user", "assistant", "user"]
    assert sent[1]["content"] == history[4]["content"]
    assert f"[CONVERSATION SUMMARY]\n{SUMMARY}" in sent[0]["content"]
    assert compacted["data"].get("context_summary") == SUMMARY, (
        f"the page was not given the summary of a temporary chat: {compacted['data']}"
    )
    assert compacted["data"].get("checkpoint_message_id") == history[4]["id"]
    follow_up = main_call(upstream)
    assert f"[CONVERSATION SUMMARY]\n{SUMMARY}" in follow_up[0]["content"]
    assert [message["content"][:10] for message in follow_up[1:]] == [
        "question 3",
        "answer 3 a",
        "question 4",
        "answer 4",
        "question 5",
    ]
    assert stored.status_code != 200, "a temporary chat was stored when it compacted"
