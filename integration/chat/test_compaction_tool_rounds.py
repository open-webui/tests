"""A reply that ran many tool calls in a row could grow past the model's context.

Dev 67cfa917c (issue open-webui/open-webui#27599). With context compaction on and native
function calling, the context was compacted only before a reply started, so a reply that kept
calling tools sent every earlier round again on each round. Now, once the context passes the token
threshold between two tool rounds, the older rounds are summarized and the next request carries
the user's question, the summary as a system message and the recent rounds. The cut falls only
in front of an assistant message that calls tools, so every call stays with its results, and the
summary prompt now lists each call's tool name and arguments. Below the threshold, or with
compaction switched off, the whole run is sent on every round as before.

The provider's prefix cache only survives requests that append: here that holds for every pair
of requests except the one right after a summary was made.

Fix bcf540a98 (issue open-webui/open-webui#32306) closes two gaps. Between rounds the size was
judged from a character estimate and the token count the provider reported was ignored, so a run
the provider counted far over the threshold (long reasoning, dense code) was never compacted and
died on the context limit; now the count of the last request decides, sent as usage on the closing
chunk, in a trailing chunk or as llama.cpp timings. And a summary made inside a reply was dropped
when the reply finished, so the next message, a regenerate and a Continue resent the whole run;
the reply now keeps its latest summary and where it cut, the next request resumes from there and
the saved chat still shows every round.

Calling tools in legacy mode happens before the reply starts, so it has no rounds to compact and
gets no test here (the suite adds none for legacy mode).

Discriminates: in a backend copy of dev 436f06ca5 with 67cfa917c reverted, no summary is made
inside the reply and every test fails (the threshold-off cases on their control run, the same run
over the threshold, which must be compacted before the quiet run counts).
On bcf540a98~1 (29289f5ba) the provider-count tests fail (no summary is made) and the next-message
test fails (the follow-up resends all twelve pages); the under-count control and the next message
after a run that was never compacted pass on both.
With the cut allowed at any message, the pairing and crossing tests fail (a result is sent
without its call); with the tool calls left out of the summary prompt, only the summary-prompt
test fails, in its plain-prompt case.

Since dev a8b478ed7 the summary request replays the run's own request with the summary
instructions appended, so the calls reach it as they are; the plain prompt, which must name each
call, is sent when the provider refuses the replay or the summary goes to another model.
"""

from __future__ import annotations

import itertools
import json
import time
import uuid

import httpx
import pytest

from harness import upstream as reply
from harness.chat import NO_BACKGROUND_TASKS, ChatTurn, ask
from harness.prompt_caching import prefix_break
from harness.python_tools import python_tool
from harness.upstream import MOCK_MODEL_ID, Reply

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

CHAT_CONFIG = ("/api/v1/chats/config", "/api/v1/chats/config")
# two rounds of two results stay under it, a third round passes it
THRESHOLD = 2_400
ROUNDS = 6
ANSWER = "The harbour log is clean."
SUMMARIES = ("SUMMARY OF ROUNDS ONE TO THREE", "SUMMARY OF ROUNDS THREE TO FIVE")
CALL_IDS = itertools.count(1)
# the run estimates at a few thousand tokens, far under it
COUNTED_THRESHOLD = 100_000
# the provider's count per request: the third passes it, as a long reasoning trace does
COUNTED_OVER = (8_000, 30_000, 110_000, 12_000, 16_000, 20_000, 24_000)
COUNTED_UNDER = (8_000, 16_000, 24_000, 32_000, 40_000, 48_000, 56_000)
FOLLOW_UP = "and the night shift log?"

TOOL_SOURCE = '''
class Tools:
    def read_log(self, page: str) -> str:
        """Read one page of the harbour log."""
        return "entry " * 300
'''


@pytest.fixture(scope="module")
def log_tool(admin):
    with python_tool(admin, TOOL_SOURCE, name="Harbour log") as tool_id:
        yield tool_id


@pytest.fixture
def compaction(admin, preserve):
    """`compaction(enable, threshold)`: the chat settings, keeping 10% so a cut lands mid-round."""
    preserve(CHAT_CONFIG)

    def configure(enable: bool = True, threshold: int = THRESHOLD) -> None:
        with admin.client() as client:
            current = client.get(CHAT_CONFIG[0]).json()
            updated = client.post(
                CHAT_CONFIG[1],
                json={
                    **current,
                    "ENABLE_CONTEXT_COMPACTION": enable,
                    "CONTEXT_COMPACTION_TOKEN_THRESHOLD": threshold,
                    "CONTEXT_COMPACTION_TOKEN_CAP": threshold,
                    "CONTEXT_COMPACTION_RETENTION_PERCENTAGE": 10,
                },
            )
        assert updated.status_code == 200, updated.text

    return configure


def is_summary_request(body: dict) -> bool:
    return not body.get("stream")


def answering(prompt: str):
    """The chat's own requests only: the summary prompt quotes the question too."""
    asks_about = reply.answering(prompt)
    return lambda body: bool(body.get("stream")) and asks_about(body)


def page_name(number: int) -> str:
    return f"harbour-page-{number}"


def tool_round(number: int) -> Reply:
    """Round `number` reads two log pages at once."""
    calls = [
        reply.tool_call(
            "read_log", {"page": page_name(page)}, call_id=f"call_{next(CALL_IDS)}"
        ).tool_calls[0]
        for page in (2 * number - 1, 2 * number)
    ]
    return Reply(tool_calls=calls)


def counted(scripted: Reply, prompt_tokens: int, shape: str) -> Reply:
    """`scripted` carrying the provider's token count of the request it answers."""
    if shape == "llama.cpp":
        scripted.timings = {"cache_n": prompt_tokens - 500, "prompt_n": 500, "predicted_n": 40}
        return scripted
    scripted.usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 40,
        "total_tokens": prompt_tokens + 40,
    }
    scripted.usage_last = shape == "trailing-usage"
    return scripted


def long_tool_run(
    client: httpx.Client,
    upstream,
    tool_id: str,
    counts: tuple[int, ...] | None = None,
    shape: str = "usage",
) -> tuple[ChatTurn, dict]:
    prompt = f"read the whole harbour log [{uuid.uuid4().hex[:6]}]"
    rounds = [tool_round(number) for number in range(1, ROUNDS + 1)]
    scripted_replies = [*rounds, reply.text(ANSWER)]
    for number, scripted in enumerate(scripted_replies):
        scripted.match = answering(prompt)
        upstream.queue(counted(scripted, counts[number], shape) if counts else scripted)
    upstream.queue(*(reply.text(summary, match=is_summary_request) for summary in SUMMARIES))
    return ask(client, prompt, tool_ids=[tool_id], timeout=90)


def provider_rounds(upstream) -> list[tuple[dict, bool]]:
    """Each streamed request, and whether a summary was made since the request before it."""
    rounds, summarized = [], False
    for body in upstream.chat_requests():
        if is_summary_request(body):
            summarized = True
            continue
        rounds.append((body, summarized))
        summarized = False
    return rounds


def summary_requests(upstream) -> list[dict]:
    return [body for body in upstream.chat_requests() if is_summary_request(body)]


def last_chat_request(upstream) -> dict:
    return [body for body in upstream.chat_requests() if not is_summary_request(body)][-1]


def follow_up(client: httpx.Client, upstream, tool_id: str, run: ChatTurn) -> tuple[ChatTurn, dict]:
    prompt = f"{FOLLOW_UP} [{uuid.uuid4().hex[:6]}]"
    upstream.queue(reply.text("The night shift was quiet.", match=answering(prompt)))
    return ask(
        client, prompt, chat_id=run.chat_id, parent_id=run.assistant_message_id, tool_ids=[tool_id]
    )


def resend(
    client: httpx.Client, upstream, tool_id: str, run: ChatTurn, turn: ChatTurn, *, continuing: bool
) -> dict:
    """Regenerate the reply to `turn`, or continue it, as the web client sends them."""
    stored = client.get(f"/api/v1/chats/{turn.chat_id}").json()["chat"]["history"]["messages"]
    question = stored[turn.user_message_id]["content"]
    marker = f"resent {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text(marker, match=answering(question)))
    assistant_id = turn.assistant_message_id if continuing else str(uuid.uuid4())
    payload = {
        "model": MOCK_MODEL_ID,
        "messages": [{"role": "user", "content": question}],
        "stream": True,
        "chat_id": turn.chat_id,
        "parent_id": run.assistant_message_id,
        "id": assistant_id,
        "user_message": {
            **stored[turn.user_message_id],
            "childrenIds": sorted({turn.assistant_message_id, assistant_id}),
        },
        "tool_ids": [tool_id],
        "session_id": f"harness-{uuid.uuid4().hex[:8]}",
        "background_tasks": NO_BACKGROUND_TASKS,
    }
    if continuing:
        payload["assistant_message_id"] = assistant_id
    accepted = client.post("/api/chat/completions", json=payload)
    assert accepted.status_code == 200, accepted.text
    # a continued reply is already marked done, so wait for the new text
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        stored = client.get(f"/api/v1/chats/{turn.chat_id}").json()["chat"]["history"]["messages"]
        message = stored.get(assistant_id, {})
        if message.get("done") and marker in str(message.get("content")):
            return last_chat_request(upstream)
        time.sleep(0.1)
    raise AssertionError(f"the resent reply never finished: {stored.get(assistant_id)}")


def resumed_from_the_last_summary(body: dict) -> list[str]:
    """Why a request misses the run's last summary or resends rounds before it."""
    messages, problems = body["messages"], []
    pages = called_pages(messages)
    if pages != [page_name(page) for page in range(9, 13)]:
        problems.append(f"{len(pages)} pages resent ({len(json.dumps(messages))} chars): {pages}")
    system = [entry["content"] for entry in messages if entry["role"] == "system"]
    if f"[CONVERSATION SUMMARY]\n{SUMMARIES[1]}" not in system:
        problems.append(f"the run's last summary is missing: {system}")
    if SUMMARIES[0] in json.dumps(messages):
        problems.append("the run's first summary came back")
    if not str(messages[1]["content"]).startswith("read the whole harbour log"):
        problems.append("the run's question is missing")
    return problems + unpaired(messages)


def called_pages(messages: list[dict]) -> list[str]:
    return [
        json.loads(call["function"]["arguments"])["page"]
        for message in messages
        for call in message.get("tool_calls") or []
    ]


def unpaired(messages: list[dict]) -> list[str]:
    """Calls sent without their result and results sent without their call."""
    problems, open_calls = [], set()
    for message in messages:
        if message["role"] == "tool":
            if message.get("tool_call_id") not in open_calls:
                problems.append(f"result {message.get('tool_call_id')} without its call")
            open_calls.discard(message.get("tool_call_id"))
            continue
        problems += [f"call {call_id} without its result" for call_id in sorted(open_calls)]
        open_calls = {call["id"] for call in message.get("tool_calls") or []}
    return problems + [f"call {call_id} without its result" for call_id in sorted(open_calls)]


def test_a_long_tool_run_is_compacted_between_rounds(user, upstream, compaction, log_tool):
    compaction()
    with user.client() as client:
        turn, message = long_tool_run(client, upstream, log_tool)

    rounds = provider_rounds(upstream)
    assert len(rounds) == ROUNDS + 1
    assert [summarized for _, summarized in rounds] == [
        False,
        False,
        False,
        True,
        False,
        True,
        False,
    ], "the reply was not compacted after its third and fifth rounds"
    fourth = rounds[3][0]["messages"]
    assert [entry["role"] for entry in fourth] == ["system", "user", "assistant", "tool", "tool"]
    assert fourth[0]["content"] == f"[CONVERSATION SUMMARY]\n{SUMMARIES[0]}"
    assert fourth[1]["content"].startswith("read the whole harbour log")
    assert called_pages(fourth) == [page_name(5), page_name(6)]
    last = rounds[-1][0]["messages"]
    assert last[0]["content"] == f"[CONVERSATION SUMMARY]\n{SUMMARIES[1]}"
    assert called_pages(last) == [page_name(page) for page in range(9, 13)]

    assert ANSWER in message["content"]
    calls = [item for item in message["output"] if item["type"] == "function_call"]
    assert len(calls) == 2 * ROUNDS, "the stored reply lost rounds the request left out"


def test_each_tool_call_stays_with_its_result(user, upstream, compaction, log_tool):
    compaction()
    with user.client() as client:
        long_tool_run(client, upstream, log_tool)

    rounds = provider_rounds(upstream)
    assert any(summarized for _, summarized in rounds), "the reply was never compacted"
    for number, (body, _) in enumerate(rounds, start=1):
        assert unpaired(body["messages"]) == [], f"request {number}"


def is_replayed_summary_request(body: dict) -> bool:
    """The run's request replayed with the summary instructions; the plain prompt is one message."""
    return is_summary_request(body) and len(body["messages"]) > 1


@pytest.mark.parametrize("replay_refused", [False, True], ids=["replayed", "plain-prompt"])
def test_the_summary_request_names_the_tools_and_arguments(
    user, upstream, compaction, log_tool, replay_refused
):
    compaction()
    if replay_refused:
        upstream.queue(
            *(
                reply.error(400, "context length exceeded", match=is_replayed_summary_request)
                for _ in SUMMARIES
            )
        )
    with user.client() as client:
        long_tool_run(client, upstream, log_tool)

    answered = [
        json.dumps(body)
        for body in summary_requests(upstream)
        if not (replay_refused and is_replayed_summary_request(body))
    ]
    assert len(answered) == len(SUMMARIES), f"{len(answered)} summary requests were answered"
    first, second = answered
    assert "read_log" in first
    for page in range(1, 7):
        assert page_name(page) in first, f"page {page} is missing from the summary prompt"
    assert SUMMARIES[0] in second, "the second summary was not given the first one"
    assert page_name(9) in second


def test_between_compactions_requests_only_append(user, upstream, compaction, log_tool):
    compaction()
    with user.client() as client:
        long_tool_run(client, upstream, log_tool)

    rounds = provider_rounds(upstream)
    appended_after_a_summary = False
    for number in range(1, len(rounds)):
        (earlier, earlier_summarized), (later, summarized) = rounds[number - 1], rounds[number]
        if summarized:
            continue
        broken = prefix_break(earlier, later)
        assert broken is None, f"request {number + 1} of {len(rounds)}: {broken}"
        appended_after_a_summary |= earlier_summarized
    assert appended_after_a_summary, "no request followed a compacted one without a summary"


@pytest.mark.parametrize(
    "enable,threshold",
    [
        pytest.param(True, 1_000_000, id="under-threshold"),
        pytest.param(False, THRESHOLD, id="switched-off"),
    ],
)
def test_without_passing_an_active_threshold_the_run_is_sent_whole(
    user, upstream, compaction, log_tool, enable, threshold
):
    compaction()
    with user.client() as client:
        long_tool_run(client, upstream, log_tool)
        assert summary_requests(upstream), (
            "control: the same run over the threshold was not compacted"
        )
        upstream.reset()
        compaction(enable=enable, threshold=threshold)
        turn, message = long_tool_run(client, upstream, log_tool)
        follow_up(client, upstream, log_tool, turn)

    assert summary_requests(upstream) == []
    requests = [body for body, _ in provider_rounds(upstream)]
    for number, (earlier, later) in enumerate(zip(requests, requests[1:]), start=2):
        assert prefix_break(earlier, later) is None, f"request {number}"
    assert called_pages(requests[-2]["messages"]) == [page_name(page) for page in range(1, 13)]
    assert ANSWER in message["content"]
    assert called_pages(requests[-1]["messages"]) == [page_name(page) for page in range(1, 13)], (
        "the next message lost rounds of a run that was never compacted"
    )


@pytest.mark.parametrize("shape", ["usage", "trailing-usage", "llama.cpp"])
def test_a_provider_count_over_the_threshold_compacts_short_rounds(
    user, upstream, compaction, log_tool, shape
):
    compaction(threshold=COUNTED_THRESHOLD)
    with user.client() as client:
        _, message = long_tool_run(client, upstream, log_tool, COUNTED_OVER, shape)

    rounds = provider_rounds(upstream)
    assert [summarized for _, summarized in rounds] == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    ], f"the provider counted {COUNTED_OVER[2]} tokens after round three and nothing was compacted"
    fourth = rounds[3][0]["messages"]
    assert fourth[0]["content"] == f"[CONVERSATION SUMMARY]\n{SUMMARIES[0]}"
    assert page_name(1) not in json.dumps(fourth)
    assert unpaired(fourth) == []
    assert ANSWER in message["content"]


def test_a_provider_count_under_the_threshold_keeps_the_run_whole(
    user, upstream, compaction, log_tool
):
    compaction(threshold=COUNTED_THRESHOLD)
    with user.client() as client:
        long_tool_run(client, upstream, log_tool, COUNTED_UNDER)

    assert summary_requests(upstream) == []
    last = last_chat_request(upstream)["messages"]
    assert called_pages(last) == [page_name(page) for page in range(1, 13)]


def test_the_next_message_resumes_from_the_runs_last_summary(user, upstream, compaction, log_tool):
    compaction()
    with user.client() as client:
        run, _ = long_tool_run(client, upstream, log_tool)
        assert len(summary_requests(upstream)) == 2, "control: the run was not compacted twice"
        upstream.reset()
        turn, answer = follow_up(client, upstream, log_tool, run)
        sent = {"next message": last_chat_request(upstream)}
        sent["regenerate"] = resend(client, upstream, log_tool, run, turn, continuing=False)
        sent["continue"] = resend(client, upstream, log_tool, run, turn, continuing=True)
        stored = client.get(f"/api/v1/chats/{run.chat_id}").json()["chat"]["history"]["messages"]

    assert answer["content"] == "The night shift was quiet."
    for action, body in sent.items():
        assert resumed_from_the_last_summary(body) == [], action
    shown = stored[run.assistant_message_id]
    assert ANSWER in shown["content"]
    for kind in ("function_call", "function_call_output"):
        items = [item for item in shown["output"] if item["type"] == kind]
        assert len(items) == 2 * ROUNDS, f"the saved run lost {kind} items"
