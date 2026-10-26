"""A reply that ran many tool calls in a row could grow past the model's context.

Dev 67cfa917c (issue open-webui/open-webui#27599). With context compaction on and native
function calling, the context was compacted only before a reply started. Now, once the context
passes the token threshold between two tool rounds, the older rounds are summarized: the chat
shows the Context compacted notice mid-reply, the next request carries the question, the summary
and the recent rounds with every call next to its results, and the reply still finishes. The
stored reply keeps every round and, since dev f465a4be1, each Context compacted notice between the
rounds it summarized, so a reload shows the answer and the next turn is answered.
Between two compactions every request only appends, as the provider's prefix cache needs.
Twin of integration/chat/test_compaction_tool_rounds.py.

Discriminates: passes on dev 436f06ca5; with 67cfa917c reverted in a backend copy the long run is
never compacted, no Context compacted notice appears and the test fails.
"""

from __future__ import annotations

import itertools
import uuid

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.prompt_caching import prefix_break
from harness.python_tools import python_tool
from harness.upstream import Reply
from utils.cached_chat import turn_on_tool
from utils.chat_ui import conversation, expect_reply, send

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

CHAT_CONFIG = ("/api/v1/chats/config", "/api/v1/chats/config")
THRESHOLD = 2_400
ROUNDS = 6
ANSWER = "The harbour log is clean."
SUMMARIES = ("SUMMARY OF ROUNDS ONE TO THREE", "SUMMARY OF ROUNDS THREE TO FIVE")
CALL_IDS = itertools.count(1)

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


def tool_round(number: int) -> Reply:
    calls = [
        reply.tool_call(
            "read_log", {"page": f"harbour-page-{page}"}, call_id=f"call_{next(CALL_IDS)}"
        ).tool_calls[0]
        for page in (2 * number - 1, 2 * number)
    ]
    return Reply(tool_calls=calls)


def start_long_tool_run(page, upstream) -> None:
    prompt = f"read the whole harbour log [{uuid.uuid4().hex[:6]}]"
    for scripted in [*(tool_round(number) for number in range(1, ROUNDS + 1)), reply.text(ANSWER)]:
        scripted.match = answering(prompt)
        upstream.queue(scripted)
    upstream.queue(*(reply.text(summary, match=is_summary_request) for summary in SUMMARIES))
    turn_on_tool(page, "Harbour log")
    send(page, prompt)


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


def unpaired(messages: list[dict]) -> list[str]:
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


def assert_append_only_between_summaries(rounds: list[tuple[dict, bool]]) -> None:
    for number in range(1, len(rounds)):
        (earlier, _), (later, summarized) = rounds[number - 1], rounds[number]
        if not summarized:
            broken = prefix_break(earlier, later)
            assert broken is None, f"request {number + 1} of {len(rounds)}: {broken}"


def test_a_long_tool_run_is_compacted_mid_reply_and_the_chat_carries_on(
    page_for, make_user, upstream, compaction, log_tool
):
    compaction()
    page = page_for(make_user())
    start_long_tool_run(page, upstream)
    expect(page.get_by_text("Context compacted").first).to_be_visible(timeout=30_000)
    expect_reply(page, ANSWER)

    rounds = provider_rounds(upstream)
    assert [summarized for _, summarized in rounds] == [
        False,
        False,
        False,
        True,
        False,
        True,
        False,
    ], "the reply was not compacted after its third and fifth rounds"
    after_summary = rounds[3][0]["messages"]
    assert after_summary[0]["content"] == f"[CONVERSATION SUMMARY]\n{SUMMARIES[0]}"
    assert [entry["role"] for entry in after_summary[1:]] == ["user", "assistant", "tool", "tool"]
    for number, (body, _) in enumerate(rounds, start=1):
        assert unpaired(body["messages"]) == [], f"request {number}"
    assert_append_only_between_summaries(rounds)

    page.reload()
    expect_reply(page, ANSWER)
    # consecutive calls are grouped under one line that counts them, and each compaction stays
    # in the reply between the rounds it summarized: three rounds, two, then the last one
    chat = conversation(page)
    for calls in (6, 4, 2):
        expect(chat.get_by_text(f"Explored {calls} read_log")).to_be_visible()
    expect(chat.get_by_role("button", name="Context compacted")).to_have_count(2)

    follow_up = f"which page was longest? [{uuid.uuid4().hex[:6]}]"
    upstream.queue(reply.text("They were all the same.", match=answering(follow_up)))
    send(page, follow_up)
    expect_reply(page, "They were all the same.")
    sent = provider_rounds(upstream)[-1][0]["messages"]
    assert sent[-1]["content"] == follow_up
    assert unpaired(sent) == []

    page.reload()
    expect_reply(page, "They were all the same.")
    expect(conversation(page).get_by_text(ANSWER)).to_be_visible()
