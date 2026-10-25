"""Journey: the admin session explorer reports each stored chat as its saved messages describe it.

Chats are seeded with the shape the live pipeline leaves behind (prompts, recorded replies, a
failed turn, assistant `output` items for tool calls and reasoning), always narrowed to the
test's own preset model so the numbers are the test's own. The list names each session's owner
by id and name, aggregates every session's messages, responses, prompts, tokens, errors,
recorded responses, span and model count number for number, orders by recent and by tokens,
filters by a token range, a user-name query and the subagent flag, pages with skip and has_more,
and totals the same chats in the summary overview.
The detail routes return one session's messages, the output items of one assistant message, and
the session trace narrowed by view and charted by usage, and refuse an unknown or out-of-range id
with 404. Every session route is admin only and also refuses the admin when admin chat access is
off.

Discriminates: passes on dev 8889e8d6a; in a backend copy, with the per-session token sum reduced
to the output tokens the aggregate and summary tests fail, with the session's `user_id` read from
the chat id instead of its owner the aggregate test fails, with the start-after-end and
token-min-over-max guards dropped the validation test fails, with `require_session_access`
opened to verified users the refusal test fails, with the `hide_subagents` filter dropped the
subagent test fails, with the `view=tools` narrowing removed the trace-view test fails, with
`message_events` returning no page the events test fails and with `include_summary` adding no
overview the summary test fails. The token and `user_id` edits were run on 8889e8d6a, the rest on
dev 7b303cf49.
"""

from __future__ import annotations

import time
import uuid
from typing import Callable, Iterator

import pytest

from harness.actors import Actor, admin_of
from harness.backends import on_postgres, write_rows
from harness.chat_history import seed_chat
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

PUBLIC_READ = [{"principal_type": "user", "principal_id": "*", "permission": "read"}]


@pytest.fixture
def make_preset(admin) -> Iterator[Callable[[], str]]:
    """Presets only this test's chats name, so the session list is the test's own."""
    created: list[str] = []

    def make() -> str:
        model_id = f"session-{uuid.uuid4().hex[:8]}"
        with admin.client() as client:
            response = client.post(
                "/api/v1/models/create",
                json={
                    "id": model_id,
                    "base_model_id": MOCK_MODEL_ID,
                    "name": model_id,
                    "meta": {},
                    "params": {},
                    "access_grants": PUBLIC_READ,
                },
            )
        assert response.status_code == 200, response.text
        created.append(model_id)
        return model_id

    yield make
    with admin.client() as client:
        for model_id in created:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


@pytest.fixture
def accounts(make_user) -> Iterator[Callable[..., Actor]]:
    """Fresh accounts whose chats are deleted afterwards."""
    made: list[Actor] = []

    def make(**options) -> Actor:
        account = make_user(**options)
        made.append(account)
        return account

    yield make
    for account in made:
        with account.client() as client:
            client.delete("/api/v1/chats/")


def seed_session(owner: Actor, model_id: str, messages: list[dict]) -> str:
    """Store `messages` as one chat on `model_id`; returns the chat id."""
    with owner.client() as client:
        chat_id, _ = seed_chat(client, messages, model=model_id)
    return chat_id


def prompt(content: str, stamp: int) -> dict:
    return {"role": "user", "content": content, "timestamp": stamp}


def reply(content: str, stamp: int, **extra) -> dict:
    return {"role": "assistant", "content": content, "timestamp": stamp, **extra}


def get_json(actor: Actor, path: str, **params) -> dict:
    with actor.client() as client:
        response = client.get(f"/api/v1/analytics/{path}", params=params)
    assert response.status_code == 200, (
        f"{path} failed: HTTP {response.status_code} {response.text}"
    )
    return response.json()


def sessions(admin: Actor, **params) -> dict:
    return get_json(admin, "sessions", **params)


def ids(result: dict) -> list[str]:
    return [item["id"] for item in result["items"]]


# ---------------------------------------------------------------- per-session aggregates


def test_session_aggregates_match_the_seeded_chat(admin, accounts, make_preset):
    owner = accounts()
    preset = make_preset()
    base = int(time.time())
    chat_id = seed_session(
        owner,
        preset,
        [
            prompt("q1", base - 100),
            reply("a1", base - 90, usage={"input_tokens": 100, "output_tokens": 10}),
            prompt("q2", base - 80),
            reply("a2", base - 70, usage={"prompt_tokens": 200, "completion_tokens": 20}),
            reply("a3", base - 60, error={"content": "it failed"}),
        ],
    )

    result = sessions(admin, model_id=preset)
    assert result["has_more"] is False
    assert ids(result) == [chat_id]

    item = result["items"][0]
    # the explorer links each row to its owner's profile through these
    assert item["user_id"] == owner.id, item
    assert item["user_name"] == owner.name, item
    assert item["messages"] == 5
    assert item["responses"] == 3
    assert item["prompts"] == 2
    assert item["recorded_responses"] == 2
    assert item["tokens"] == 330, item
    assert item["errors"] == 1
    assert item["model_id"] == preset
    assert item["model_count"] == 1
    assert item["started_at"] == base - 100
    assert item["last_active"] == base - 60
    assert item["span_seconds"] == 40


# ---------------------------------------------------------------- ordering


@pytest.fixture
def three_sessions(accounts, make_preset) -> dict:
    """Three chats on one preset: A newest and lightest, C oldest and heaviest."""
    owner = accounts()
    preset = make_preset()
    base = int(time.time())
    chats = {
        "A": seed_session(
            owner, preset, [reply("a", base - 30, usage={"input_tokens": 40, "output_tokens": 10})]
        ),
        "B": seed_session(
            owner,
            preset,
            [reply("b", base - 60, usage={"input_tokens": 400, "output_tokens": 100})],
        ),
        "C": seed_session(
            owner,
            preset,
            [reply("c", base - 90, usage={"input_tokens": 4000, "output_tokens": 1000})],
        ),
    }
    return {"preset": preset, "chats": chats}


def test_sessions_order_by_recent_and_by_tokens(admin, three_sessions):
    preset, chats = three_sessions["preset"], three_sessions["chats"]

    def order(**params) -> list[str]:
        return ids(sessions(admin, model_id=preset, **params))

    newest_first = [chats["A"], chats["B"], chats["C"]]
    assert order(order_by="recent", direction="desc") == newest_first
    assert order(order_by="recent", direction="asc") == newest_first[::-1]
    assert order(order_by="tokens", direction="desc") == [chats["C"], chats["B"], chats["A"]]
    assert order(order_by="tokens", direction="asc") == [chats["A"], chats["B"], chats["C"]]


def test_sessions_page_with_skip_and_report_has_more(admin, three_sessions):
    preset, chats = three_sessions["preset"], three_sessions["chats"]

    first = sessions(admin, model_id=preset, order_by="recent", direction="desc", limit=2)
    second = sessions(admin, model_id=preset, order_by="recent", direction="desc", limit=2, skip=2)

    assert ids(first) == [chats["A"], chats["B"]]
    assert first["has_more"] is True
    assert ids(second) == [chats["C"]]
    assert second["has_more"] is False


# ---------------------------------------------------------------- filters


def test_sessions_filter_by_token_range(admin, accounts, make_preset):
    owner = accounts()
    preset = make_preset()
    base = int(time.time())
    light = seed_session(
        owner,
        preset,
        [reply("light", base - 20, usage={"input_tokens": 400, "output_tokens": 100})],
    )
    heavy = seed_session(
        owner,
        preset,
        [reply("heavy", base - 40, usage={"input_tokens": 40000, "output_tokens": 10000})],
    )

    assert ids(sessions(admin, model_id=preset, token_min=1000)) == [heavy]
    assert ids(sessions(admin, model_id=preset, token_max=1000)) == [light]
    assert set(ids(sessions(admin, model_id=preset))) == {light, heavy}


def test_sessions_filter_by_a_user_name_query(admin, accounts, make_preset):
    preset = make_preset()
    tag = uuid.uuid4().hex[:8]
    ann = accounts(name=f"Ann-{tag}")
    bob = accounts(name=f"Bob-{tag}")
    base = int(time.time())
    ann_chat = seed_session(ann, preset, [reply("from ann", base - 10)])
    seed_session(bob, preset, [reply("from bob", base - 20)])

    assert ids(sessions(admin, model_id=preset, query=f"Ann-{tag}")) == [ann_chat]
    assert len(sessions(admin, model_id=preset)["items"]) == 2


def test_hide_subagents_drops_subagent_sessions(admin, accounts, make_preset, instance):
    owner = accounts()
    preset = make_preset()
    base = int(time.time())
    normal = seed_session(owner, preset, [reply("normal", base - 10)])
    subagent = seed_session(owner, preset, [reply("spawned", base - 20)])
    meta = '{"type": "subagent"}'
    statement = (
        "UPDATE chat SET meta = CAST(:meta AS JSON) WHERE id = :id"
        if on_postgres(instance)
        else "UPDATE chat SET meta = :meta WHERE id = :id"
    )
    write_rows(instance, statement, [{"meta": meta, "id": subagent}])

    assert set(ids(sessions(admin, model_id=preset))) == {normal, subagent}
    assert ids(sessions(admin, model_id=preset, hide_subagents=True)) == [normal]


# ---------------------------------------------------------------- summary overview


def test_include_summary_totals_the_sessions(admin, accounts, make_preset):
    owner = accounts()
    preset = make_preset()
    base = int(time.time())
    seed_session(
        owner,
        preset,
        [
            prompt("q", base - 20),
            reply("low", base - 19, usage={"input_tokens": 400, "output_tokens": 100}),
        ],
    )
    seed_session(
        owner,
        preset,
        [
            prompt("q", base - 40),
            reply("high", base - 39, usage={"input_tokens": 4000, "output_tokens": 1000}),
            reply("broke", base - 38, error={"content": "it failed"}),
        ],
    )

    result = sessions(admin, model_id=preset, include_summary=True)
    overview = result["overview"]
    assert overview["sessions"] == 2
    assert overview["tokens"] == 5500
    assert overview["responses"] == 3
    assert overview["prompts"] == 2
    assert overview["errors"] == 1
    assert overview["recorded_responses"] == 2
    assert overview["complete_responses"] == 2
    assert overview["unrecorded_sessions"] == 0
    assert overview["max_tokens"] == 5000

    # TOKEN_RANGES buckets: 500 tokens in (1,1000), 5500 in (1000,10000).
    distribution = overview["token_distribution"]
    assert len(distribution) == 6
    assert distribution[1] == 1
    assert distribution[2] == 1
    assert sum(distribution) == 2

    assert "overview" not in sessions(admin, model_id=preset)


# ---------------------------------------------------------------- validation


def test_sessions_reject_an_inverted_date_or_token_range(admin):
    base = int(time.time())
    with admin.client() as client:
        bad_dates = client.get(
            "/api/v1/analytics/sessions", params={"start_date": base, "end_date": base - 100}
        )
        bad_tokens = client.get(
            "/api/v1/analytics/sessions", params={"token_min": 500, "token_max": 100}
        )
    assert bad_dates.status_code == 400, bad_dates.text
    assert bad_tokens.status_code == 400, bad_tokens.text


# ---------------------------------------------------------------- one session's messages


def test_session_messages_list_the_chat_and_404_outside_its_range(admin, accounts, make_preset):
    owner = accounts()
    preset = make_preset()
    base = int(time.time())
    chat_id = seed_session(owner, preset, [prompt("hello", base - 10), reply("hi", base - 5)])

    result = get_json(admin, f"sessions/{chat_id}")
    assert result["session"]["id"] == chat_id
    assert [item["role"] for item in result["items"]] == ["user", "assistant"]
    assert result["has_more"] is False

    with admin.client() as client:
        unknown = client.get(f"/api/v1/analytics/sessions/{uuid.uuid4()}")
        out_of_range = client.get(
            f"/api/v1/analytics/sessions/{chat_id}",
            params={"start_date": base - 100000, "end_date": base - 90000},
        )
    assert unknown.status_code == 404, unknown.text
    assert out_of_range.status_code == 404, out_of_range.text


# ---------------------------------------------------------------- events and trace


TOOL_OUTPUT = [
    {
        "type": "function_call",
        "name": "lookup",
        "call_id": "call_ok",
        "arguments": '{"q": "x"}',
        "status": "completed",
        "started_at": "1",
        "ended_at": "2",
        "duration": "1",
    },
    {
        "type": "function_call_output",
        "call_id": "call_ok",
        "output": "a result",
        "status": "completed",
    },
    {"type": "function_call", "name": "broken", "call_id": "call_err", "arguments": "{}"},
    {
        "type": "function_call_output",
        "call_id": "call_err",
        "output": "boom",
        "status": "failed",
        "error_type": "ToolError",
    },
    {"type": "reasoning", "summary": "thinking about it"},
]


@pytest.fixture
def traced_session(accounts, make_preset, admin) -> dict:
    """A chat whose one assistant message carries tool calls, a failed call and reasoning."""
    owner = accounts()
    preset = make_preset()
    base = int(time.time())
    chat_id = seed_session(
        owner,
        preset,
        [
            prompt("use a tool", base - 20),
            reply(
                "done",
                base - 10,
                usage={"input_tokens": 300, "output_tokens": 30},
                output=TOOL_OUTPUT,
            ),
        ],
    )
    page = get_json(admin, f"sessions/{chat_id}")
    assistant = next(item for item in page["items"] if item["role"] == "assistant")
    return {
        "preset": preset,
        "chat_id": chat_id,
        "message_id": assistant["id"],
        "output_count": assistant["output_count"],
    }


def test_message_events_list_the_assistant_output_items(admin, traced_session):
    chat_id, message_id = traced_session["chat_id"], traced_session["message_id"]
    assert traced_session["output_count"] == len(TOOL_OUTPUT)

    result = get_json(admin, f"sessions/{chat_id}/messages/{message_id}/events")
    assert len(result["items"]) == len(TOOL_OUTPUT)
    types = [item["type"] for item in result["items"]]
    assert "function_call" in types
    assert "reasoning" in types
    names = {item["name"] for item in result["items"] if item["name"]}
    assert {"lookup", "broken"} <= names

    with admin.client() as client:
        unknown = client.get(f"/api/v1/analytics/sessions/{chat_id}/messages/{uuid.uuid4()}/events")
    assert unknown.status_code == 404, unknown.text


def test_session_trace_narrows_by_view_and_charts_usage(admin, traced_session):
    chat_id = traced_session["chat_id"]

    full = get_json(admin, f"sessions/{chat_id}/trace")
    assert "chart" not in full
    types = [item["type"] for item in full["items"]]
    # a pseudo-item for the assistant response, its two tool calls (paired outputs fold in)
    assert types.count("response") == 1
    assert "function_call" in types
    assert "reasoning" in types

    tools = get_json(admin, f"sessions/{chat_id}/trace", view="tools")
    assert {item["type"] for item in tools["items"]} == {"function_call"}
    assert len(tools["items"]) == 2

    errors = get_json(admin, f"sessions/{chat_id}/trace", view="errors")
    assert all(item["status"] == "failed" for item in errors["items"])
    assert [item["name"] for item in errors["items"]] == ["broken"]

    charted = get_json(admin, f"sessions/{chat_id}/trace", include_chart=True)
    usage = charted["chart"]["usage"]
    assert usage, "include_chart must populate the usage bars"
    assert sum(bar["input_tokens"] or 0 for bar in usage) == 300
    assert sum(bar["output_tokens"] or 0 for bar in usage) == 30

    with admin.client() as client:
        unknown = client.get(f"/api/v1/analytics/sessions/{uuid.uuid4()}/trace")
    assert unknown.status_code == 404, unknown.text


# ---------------------------------------------------------------- access


SESSION_ROUTES = [
    "sessions",
    f"sessions/{uuid.uuid4()}",
    f"sessions/{uuid.uuid4()}/messages/{uuid.uuid4()}/events",
    f"sessions/{uuid.uuid4()}/trace",
]


@pytest.mark.parametrize("path", SESSION_ROUTES)
def test_a_plain_user_is_refused_every_session_route(admin, accounts, path):
    user = accounts()
    with user.client() as client:
        refused = client.get(f"/api/v1/analytics/{path}")
    assert refused.status_code in (401, 403), f"{path} answered {refused.status_code}"
    with admin.client() as client:
        allowed = client.get(f"/api/v1/analytics/{path}")
    assert allowed.status_code not in (401, 403), f"admin refused {path}: {allowed.status_code}"


@pytest.mark.slow
def test_the_admin_is_refused_when_admin_chat_access_is_off(instance_with):
    disabled = instance_with({"ENABLE_ADMIN_CHAT_ACCESS": "False"})
    admin = admin_of(disabled)
    with admin.client() as client:
        refused = client.get("/api/v1/analytics/sessions")
    assert refused.status_code == 403, refused.text
