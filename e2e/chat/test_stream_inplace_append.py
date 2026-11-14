"""A reply assembled piece by piece looks the same in the chat page whether pieces extend in place.

`ENABLE_CHAT_RESPONSE_STREAM_INPLACE_APPEND` only changes how the server appends each streamed
piece to the text it already holds. A person sees no difference: the reply grows on screen, is
complete when it ends and is the same after a reload. Every test below runs on an instance with
the toggle on, and the five that depend on how pieces are appended (merged deltas, think tags,
a tool call in pieces, Stop, a long reply) on it off as well, asserting the same literal text,
thinking and tool call for both. The other tests run on the toggle on only; the toggle only
changes the append, so what a person sees there does not depend on its value. The paths are a
slowly streamed reply with accents, emoji, CJK and a code fence split across pieces, deltas
merged by a chunk size, reasoning deltas and `<think>` tags, a tool call whose arguments arrive
in pieces, Stop, regenerate, continue, edit and resend, two models answering at once, a
temporary chat and a long reply. A second page opened mid-stream, thoughts growing live, a model
answering a channel member, a timer and a background sub-agent replying into the chat, a stream
filter rewriting a word, a pipe yielding pieces, an Action run on the reply and merged deltas
with thoughts and a tool call are seen live and after a reload as well.

The open page does not follow a fired timer or a sub-agent report on dev (the chat's stored
current message still points at the earlier reply), so those two tests read the follow-up after
a reload.

`test_a_background_subagent_report_and_its_follow_up_show_after_a_reload` and
`test_a_timer_fires_and_its_whole_follow_up_shows_after_a_reload` are red on dev 62f70a844: since
de73bb830 a chat request whose reply message is already stored in the chat, the way automations,
sub-agents and timers prepare their reply, is refused with 409 and the reply is never written
(open-webui/open-webui#32066, fixed by 641c82df1).

`test_reasoning_deltas_fill_a_thinking_block_above_the_answer` and
`test_a_long_reply_over_many_pieces_is_shown_in_full` are red now and then on dev 93fc3fcb7: since
de73bb830 the reply in a new chat sometimes stays blank until a reload although the server saved
it whole (open-webui/open-webui#32091, fixed by 958ad3673). The first passes six of six on
de73bb830^ and fails two of six on de73bb830 and on 93fc3fcb7.

Every test but the channel one runs in the main chat and in a side chat (`utils/chat_surface.py`),
on the instance with the toggle; the channel test is about a channel page. On dev c15f28649 three
tests are red in a side chat, all green on dev e791cbdde (fix PR open-webui/open-webui#34064): the
split code fence (code shown without its editor) and Regenerate, Continue and Edit
(open-webui/open-webui#34014), and two models at once (no Compare, open-webui/open-webui#34030).

Discriminates: passes on dev 176d31d1d with the toggle off and on; with the in-place branch
appending a marker before each piece every append-in-place case turns red and the append-copies
cases (the five two-valued tests) stay green, and with the copying branch marked the five
append-copies cases turn red.
"""

from __future__ import annotations

import dataclasses
import json
import re
import threading
import time

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.actors import Actor, admin_of, create_user
from harness.channel_chat import enable_channels
from harness.channel_quotes import group_channel, post_message
from harness.plugins import installed_function
from harness.second_provider import OPENAI_CONFIG, attach, sse
from harness.upstream import MOCK_MODEL_ID
from utils.chat_surface import MAIN_CHAT, ChatOpener, ChatSurface
from utils.chat_ui import (
    REPLY_TIMEOUT_MS,
    Chat,
    chat_input,
    conversation,
    expect_reply,
    last_reply,
    replies,
    send,
    start_on_default_models,
    stop_button,
    turn_temporary,
)

pytestmark = [
    pytest.mark.journey,
    pytest.mark.slow,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
]

INPLACE_APPEND = "ENABLE_CHAT_RESPONSE_STREAM_INPLACE_APPEND"
SECOND_MODEL = "second-model"
THOUGHTS = re.compile(r"^(Thinking|Thought)")


@pytest.fixture
def streaming(request, instance_with):
    """A scratch instance with the toggle on, or the value `on_both_toggle_values` picks."""
    return instance_with({INPLACE_APPEND: getattr(request, "param", "true")})


on_both_toggle_values = pytest.mark.parametrize(
    "streaming", ["false", "true"], ids=["append-copies", "append-in-place"], indirect=True
)


@pytest.fixture
def account(streaming) -> Actor:
    if not streaming.serves_frontend:
        pytest.skip("the checkout has no built frontend")
    return create_user(streaming, role="admin")


@pytest.fixture
def streaming_chat_for(chat_for, streaming, preserve) -> ChatOpener:
    """`chat_for` on the scratch instance with the toggle, side chats allowed there."""
    admin = admin_of(streaming)
    if chat_for.place != MAIN_CHAT:
        preserve("permissions", on=streaming)
        with admin.client() as client:
            permissions = client.get("/api/v1/users/default/permissions").json()
            permissions["chat"].update(side_chat=True)
            saved = client.post("/api/v1/users/default/permissions", json=permissions)
        assert saved.status_code == 200, saved.text
    return dataclasses.replace(chat_for, upstream=streaming.upstream, admin=admin)


@pytest.fixture
def chat(streaming_chat_for, account) -> ChatSurface:
    return streaming_chat_for(account)


@pytest.fixture
def second_provider(streaming, preserve, listener):
    """The listener as one more OpenAI connection serving `SECOND_MODEL`."""
    preserve(OPENAI_CONFIG, on=streaming)
    with admin_of(streaming).client() as client:
        attach(client, listener, SECOND_MODEL)
    return listener


def choose_models(account: Actor, *model_ids: str, **params) -> None:
    ui = {"models": list(model_ids), **({"params": params} if params else {})}
    with account.client() as client:
        saved = client.post("/api/v1/users/user/settings/update", json={"ui": ui})
    saved.raise_for_status()


def chat_id_of(page: Page) -> str:
    expect(page).to_have_url(re.compile(r"/c/[0-9a-f-]+$"))
    return page.url.rsplit("/", 1)[-1]


def stored_messages(account: Actor, chat_id: str, role: str) -> list[dict]:
    with account.client() as client:
        chat = client.get(f"/api/v1/chats/{chat_id}")
    chat.raise_for_status()
    messages = chat.json()["chat"]["history"]["messages"].values()
    return sorted((m for m in messages if m["role"] == role), key=lambda m: m["timestamp"])


def stored_replies(account: Actor, chat_id: str, count: int, timeout_ms: int = 30_000) -> list[str]:
    """The saved assistant texts once `count` replies are done."""
    deadline = time.monotonic() + timeout_ms / 1000
    while True:
        found = [m for m in stored_messages(account, chat_id, "assistant") if m.get("done")]
        if len(found) >= count or time.monotonic() > deadline:
            return [message["content"] for message in found]
        time.sleep(0.2)


def thought_button(chat: Chat) -> Locator:
    return last_reply(chat).get_by_role("button", name=THOUGHTS)


def explored_button(chat: Chat) -> Locator:
    """The collapsed group holding a tool call and the thoughts that came with it."""
    return last_reply(chat).get_by_text("Explored")


def expect_whole_reply(chat: Chat, text: str, thought: bool = False) -> None:
    """The last reply reads exactly `text`, ignoring the Ask and Explain buttons after it."""
    opening = r"Thought for [^\n]*?\s+" if thought else ""
    shown = re.compile(rf"^\s*{opening}{re.escape(text)}(\s+Ask\s+Explain)?\s*$")
    expect(last_reply(chat)).to_have_text(shown, timeout=REPLY_TIMEOUT_MS)


def reload_and_expect(chat: ChatSurface, text: str) -> ChatSurface:
    chat = chat.reload()
    expect_reply(chat, text)
    return chat


def test_a_slow_reply_grows_on_screen_and_is_whole_after_a_reload(chat, account, streaming):
    pieces = [f"piece-{index} " for index in range(10)]
    streaming.upstream.queue(reply.text(pieces, chunk_delay=0.4, match=reply.answering("grow")))
    send(chat, "grow")

    expect(last_reply(chat)).to_contain_text("piece-3", timeout=REPLY_TIMEOUT_MS)
    assert "piece-9" not in last_reply(chat).inner_text(), "the reply had already ended"
    expect_reply(chat, "piece-9")
    expect_whole_reply(chat, "".join(pieces).strip())

    assert stored_replies(account, chat_id_of(chat.page), 1) == ["".join(pieces).strip()]
    chat = reload_and_expect(chat, "piece-9")
    expect_whole_reply(chat, "".join(pieces).strip())


def test_accents_emoji_and_cjk_split_across_pieces_stay_whole(chat, account, streaming):
    pieces = [
        "cafe",
        "\u0301 au lait ",
        "\U0001f469",
        "\u200d",
        "\U0001f4bb at work ",
        "日本",
        "語のテスト",
        ".",
    ]
    whole = "".join(pieces)
    streaming.upstream.queue(reply.text(pieces, chunk_delay=0.05, match=reply.answering("unicode")))
    send(chat, "unicode")

    expect_whole_reply(chat, whole)
    assert stored_replies(account, chat_id_of(chat.page), 1) == [whole]

    chat = reload_and_expect(chat, "日本語のテスト.")
    expect_whole_reply(chat, whole)


def test_a_code_fence_split_across_pieces_renders_one_block_with_the_code(chat, account, streaming):
    pieces = [
        "Here it is:\n\n``",
        "`python\nprint('a",
        "b')\nvalue = [1, 2",
        "]\n``",
        "`\n\nThat is all.",
    ]
    streaming.upstream.queue(reply.text(pieces, chunk_delay=0.1, match=reply.answering("code")))
    send(chat, "code")

    expect_reply(chat, "That is all.")
    lines = last_reply(chat).locator(".cm-line")
    expect(lines).to_have_text(["print('ab')", "value = [1, 2]"])
    assert stored_replies(account, chat_id_of(chat.page), 1) == ["".join(pieces)]

    chat = reload_and_expect(chat, "That is all.")
    expect(last_reply(chat).locator(".cm-line")).to_have_text(["print('ab')", "value = [1, 2]"])


@on_both_toggle_values
def test_deltas_merged_by_a_chunk_size_give_the_same_reply(streaming_chat_for, account, streaming):
    choose_models(account, MOCK_MODEL_ID, stream_delta_chunk_size=3)
    pieces = [f"chunk-{index} " for index in range(12)]
    streaming.upstream.queue(reply.text(pieces, chunk_delay=0.05, match=reply.answering("merge")))
    chat = streaming_chat_for(account)
    start_on_default_models(chat)
    send(chat, "merge")

    expect_whole_reply(chat, "".join(pieces).strip())
    assert stored_replies(account, chat_id_of(chat.page), 1) == ["".join(pieces).strip()]
    chat = reload_and_expect(chat, "chunk-11")
    expect_whole_reply(chat, "".join(pieces).strip())


def test_reasoning_deltas_fill_a_thinking_block_above_the_answer(
    account, streaming_chat_for, streaming, second_provider
):
    thoughts = [{"reasoning_content": part} for part in ("weighing ", "the two ", "options")]
    answer = [{"content": part} for part in ("The answer ", "is ", "forty-two.")]
    second_provider.route("POST", "/v1/chat/completions", sse(*thoughts, *answer))
    choose_models(account, SECOND_MODEL)
    chat = streaming_chat_for(account)
    start_on_default_models(chat)
    send(chat, "think first")

    expect_reply(chat, "forty-two.")
    expect(last_reply(chat)).to_contain_text("The answer is forty-two.")
    expect(thought_button(chat)).to_have_count(1)
    thought_button(chat).click()
    expect(last_reply(chat)).to_contain_text("weighing the two options")

    chat = reload_and_expect(chat, "The answer is forty-two.")
    thought_button(chat).click()
    expect(last_reply(chat)).to_contain_text("weighing the two options")


@on_both_toggle_values
def test_think_tags_split_across_pieces_fill_a_thinking_block(chat, account, streaming):
    pieces = ["<thi", "nk>weighing ", "the two ", "options</th", "ink>The answer ", "is forty-two."]
    streaming.upstream.queue(reply.text(pieces, chunk_delay=0.05, match=reply.answering("tags")))
    send(chat, "tags")

    expect_reply(chat, "is forty-two.")
    expect(last_reply(chat)).to_contain_text("The answer is forty-two.")
    expect(thought_button(chat)).to_have_count(1)
    thought_button(chat).click()
    expect(last_reply(chat)).to_contain_text("weighing the two options")

    chat = reload_and_expect(chat, "The answer is forty-two.")
    thought_button(chat).click()
    expect(last_reply(chat)).to_contain_text("weighing the two options")
    assert "<think>" not in last_reply(chat).inner_text()


@on_both_toggle_values
def test_tool_call_arguments_arriving_in_pieces_reach_the_tool_and_the_answer_follows(
    account, streaming_chat_for, streaming, second_provider
):
    arguments = ['{"days_', 'ago": 3, ', '"weeks_ago"', ": 1}"]
    header = {"index": 0, "id": "call_1", "type": "function"}
    opening = {
        "tool_calls": [{**header, "function": {"name": "calculate_timestamp", "arguments": ""}}]
    }
    parts = [{"tool_calls": [{"index": 0, "function": {"arguments": part}}]} for part in arguments]
    rounds = []

    def answer(_request):
        rounds.append(1)
        if len(rounds) == 1:
            return sse(opening, *parts, finish_reason="tool_calls")
        return sse({"content": "Ten days "}, {"content": "ago it was."})

    second_provider.route("POST", "/v1/chat/completions", answer)
    choose_models(account, SECOND_MODEL)
    chat = streaming_chat_for(account)
    start_on_default_models(chat)
    send(chat, "what was ten days ago?")

    expect_reply(chat, "Ten days ago it was.")
    sent = [r.json() for r in second_provider.requests_to("/v1/chat/completions")]
    assert len(sent) == 2
    called = [m for m in reply.conversation(sent[1]) if m.get("tool_calls")]
    function = called[0]["tool_calls"][0]["function"]
    assert function["name"] == "calculate_timestamp"
    assert json.loads(function["arguments"]) == {"days_ago": 3, "weeks_ago": 1}
    assert any(
        m["role"] == "tool" and "calculated_timestamp" in m["content"] for m in sent[1]["messages"]
    )

    chat = reload_and_expect(chat, "Ten days ago it was.")
    expect(last_reply(chat)).to_contain_text("calculate_timestamp")


@on_both_toggle_values
def test_stopping_a_reply_keeps_the_text_streamed_so_far(chat, account, streaming):
    pieces = [f"part-{index} " for index in range(30)]
    streaming.upstream.queue(reply.text(pieces, chunk_delay=0.3, match=reply.answering("stop")))
    send(chat, "stop")
    expect(last_reply(chat)).to_contain_text("part-3", timeout=REPLY_TIMEOUT_MS)

    stop_button(chat).click()
    expect(stop_button(chat)).to_have_count(0)
    kept = last_reply(chat).inner_text()
    assert "part-3" in kept
    assert "part-29" not in kept
    prefix = "".join(pieces).strip()
    assert prefix.startswith(kept.strip())

    stored = stored_replies(account, chat_id_of(chat.page), 1)
    assert [text.strip() for text in stored] == [kept.strip()]
    chat = reload_and_expect(chat, "part-3")
    assert last_reply(chat).inner_text().strip() == kept.strip()


def test_regenerate_continue_and_edit_give_each_reply_its_text(chat, account, streaming):
    streaming.upstream.queue(
        reply.text(["first ", "try ", "here"], match=reply.answering("name a colour")),
        reply.text(["second ", "try ", "here"], match=reply.answering("name a colour")),
        reply.text([" and ", "then ", "more."], match=reply.answering("name a colour")),
        reply.text(["reply ", "to ", "the rewrite"], match=reply.answering("rewritten")),
    )
    send(chat, "name a colour")
    expect_reply(chat, "first try here")

    last_reply(chat).hover()
    conversation(chat).get_by_role("button", name="Regenerate").last.click()
    chat.page.get_by_text("Try Again").click()
    expect_whole_reply(chat, "second try here")
    expect(conversation(chat).get_by_text("2/2")).to_be_visible()

    last_reply(chat).hover()
    conversation(chat).get_by_role("button", name="Continue Response").click()
    expect_whole_reply(chat, "second try here and then more.")
    resumed_from = streaming.upstream.chat_requests()[-1]["messages"][-1]
    assert (resumed_from["role"], resumed_from["content"]) == ("assistant", "second try here")

    question = conversation(chat).locator(".chat-user").last
    question.hover()
    question.get_by_role("button", name="Edit").click()
    question.locator("textarea").fill("rewritten question")
    question.get_by_role("button", name="Send").click()
    expect_whole_reply(chat, "reply to the rewrite")

    expected = ["first try here", "second try here and then more.", "reply to the rewrite"]
    stored = stored_replies(account, chat_id_of(chat.page), 3)
    assert stored == expected
    chat = reload_and_expect(chat, "reply to the rewrite")
    _previous_question(chat).click()
    expect_whole_reply(chat, "second try here and then more.")


def _previous_question(chat: Chat) -> Locator:
    question = conversation(chat).locator(".chat-user").last
    question.hover()
    return question.get_by_role("button", name="Previous message")


def test_two_models_answer_at_once_each_with_its_own_text(
    account, streaming_chat_for, streaming, second_provider
):
    def answer_second(_request):
        return sse({"content": "second "}, {"content": "model "}, {"content": "says hi"})

    second_provider.route("POST", "/v1/chat/completions", answer_second)
    streaming.upstream.queue(
        reply.text(["first ", "model ", "says hi"], chunk_delay=0.1, match=reply.answering("both"))
    )
    choose_models(account, MOCK_MODEL_ID, SECOND_MODEL)
    chat = streaming_chat_for(account)
    start_on_default_models(chat)
    send(chat, "both")

    expect(replies(chat)).to_have_count(2, timeout=REPLY_TIMEOUT_MS)
    expect(replies(chat).nth(0)).to_contain_text("first model says hi", timeout=REPLY_TIMEOUT_MS)
    expect(replies(chat).nth(1)).to_contain_text("second model says hi", timeout=REPLY_TIMEOUT_MS)
    stored = stored_replies(account, chat_id_of(chat.page), 2)
    assert sorted(stored) == ["first model says hi", "second model says hi"]

    chat = chat.reload()
    expect(replies(chat).nth(0)).to_contain_text("first model says hi")
    expect(replies(chat).nth(1)).to_contain_text("second model says hi")


def test_a_temporary_chat_reply_with_merged_deltas_streams_in_and_is_never_saved(
    streaming_chat_for, account, streaming
):
    pieces = [f"temp-{index} " for index in range(8)]
    choose_models(account, MOCK_MODEL_ID, stream_delta_chunk_size=3)
    chat = streaming_chat_for(account)
    start_on_default_models(chat)
    streaming.upstream.queue(reply.text(pieces, chunk_delay=0.3, match=reply.answering("secretly")))
    turn_temporary(chat)
    send(chat, "secretly")

    expect(last_reply(chat)).to_contain_text("temp-2", timeout=REPLY_TIMEOUT_MS)
    assert "temp-7" not in last_reply(chat).inner_text(), "the reply had already ended"
    expect_whole_reply(chat, "".join(pieces).strip())
    with account.client() as client:
        listed = client.get("/api/v1/chats/", params={"page": 1})
    assert [chat for chat in listed.json() if "secretly" in chat["title"]] == []


@on_both_toggle_values
def test_a_long_reply_over_many_pieces_is_shown_in_full(chat, account, streaming):
    pieces = [
        f"Paragraph {index:04d}: " + "lorem ipsum dolor " * 5 + "\n\n" for index in range(400)
    ]
    whole = "".join(pieces)
    assert len(whole) > 40_000
    streaming.upstream.queue(reply.text(pieces, match=reply.answering("long story")))
    send(chat, "long story")

    expect_reply(chat, "Paragraph 0399:")
    expect(last_reply(chat).locator("p")).to_have_count(400)
    assert stored_replies(account, chat_id_of(chat.page), 1) == [whole.strip()]

    chat = reload_and_expect(chat, "Paragraph 0399:")
    expect(last_reply(chat).locator("p")).to_have_count(400)
    expect(last_reply(chat).locator("p").first).to_contain_text("Paragraph 0000:")


# --- what a person sees live: a second page, growing thoughts, channels, Functions ---------------


def slow_thoughts(
    prompt: str, thoughts: list[str], answer: list[str], delay: float = 0.5, **options
):
    """A reply that thinks in `<think>` tags over many pieces, then answers over more pieces."""
    pieces = ["<think>", *thoughts, "</think>", *answer]
    return reply.text(pieces, chunk_delay=delay, match=reply.answering(prompt), **options)


def test_a_chat_opened_on_a_second_page_mid_stream_shows_the_text_so_far_then_all_of_it(
    chat, account, page_for, streaming
):
    thoughts = [f"idea-{index} " for index in range(6)]
    answer = [f"word-{index} " for index in range(8)]
    # the reply pauses after word-1, so it cannot end before a slow second page has loaded
    rest_of_reply = threading.Event()
    held_after = len(thoughts) + 5  # the role, <think>, the thoughts, </think>, word-0, word-1
    try:
        streaming.upstream.queue(
            slow_thoughts("both pages", thoughts, answer, hold=rest_of_reply, hold_after=held_after)
        )
        send(chat, "both pages")
        expect(last_reply(chat)).to_contain_text("Thinking", timeout=REPLY_TIMEOUT_MS)

        second = page_for(account, path=f"/c/{chat_id_of(chat.page)}")
        expect(thought_button(second)).to_have_count(1, timeout=REPLY_TIMEOUT_MS)
        expect(last_reply(second)).to_contain_text("word-1", timeout=REPLY_TIMEOUT_MS)
        expect(stop_button(second)).to_be_visible()
    finally:
        rest_of_reply.set()

    whole = "".join(answer).strip()
    expect_whole_reply(second, whole, thought=True)
    thought_button(second).click()
    expect(last_reply(second)).to_contain_text("".join(thoughts).strip())
    expect_whole_reply(chat, whole, thought=True)
    assert stored_replies(account, chat_id_of(chat.page), 1)[0].endswith(whole)


def test_thoughts_grow_while_the_block_still_reads_thinking_then_it_closes_and_answers(
    chat, account, streaming
):
    thoughts = [f"idea-{index} " for index in range(8)]
    answer = ["Done ", "thinking, ", "the answer ", "is ", "seven."]
    streaming.upstream.queue(slow_thoughts("grow thoughts", thoughts, answer))
    send(chat, "grow thoughts")

    expect(thought_button(chat)).to_contain_text("Thinking", timeout=REPLY_TIMEOUT_MS)
    thought_button(chat).click()
    expect(last_reply(chat)).to_contain_text("idea-1", timeout=REPLY_TIMEOUT_MS)
    assert "idea-7" not in last_reply(chat).inner_text(), "the thinking had already ended"
    expect(last_reply(chat)).to_contain_text("idea-7", timeout=REPLY_TIMEOUT_MS)

    expect(thought_button(chat)).to_contain_text("Thought for", timeout=REPLY_TIMEOUT_MS)
    expect_reply(chat, "the answer is seven.")
    assert stored_replies(account, chat_id_of(chat.page), 1)[0].endswith("the answer is seven.")

    chat = reload_and_expect(chat, "the answer is seven.")
    expect(thought_button(chat)).to_contain_text("Thought for")
    thought_button(chat).click()
    expect(last_reply(chat)).to_contain_text("".join(thoughts).strip())


@pytest.fixture
def channel_member(streaming, account, preserve, page_for):
    """A member's page open on a group channel with the owner; channels answer inline."""
    preserve("admin_config", on=streaming)
    with account.client() as client:
        enable_channels(client, reply_mode="channel")
    member = create_user(streaming)
    channel_id = group_channel(account, member)
    page = page_for(member, path=f"/channels/{channel_id}")
    expect(chat_input(page)).to_be_visible()
    return page, channel_id


def test_a_model_mentioned_in_a_channel_answers_the_member_whole_and_after_a_reload(
    channel_member, account, streaming
):
    page, channel_id = channel_member
    pieces = ["Friday ", "works ", "for ", "everyone ", "in ", "the ", "team."]
    streaming.upstream.queue(
        reply.text(pieces, chunk_delay=0.2, match=reply.answering("Which day"))
    )
    post_message(account, channel_id, f"<@M:{MOCK_MODEL_ID}|{MOCK_MODEL_ID}> Which day?")

    answer = "Friday works for everyone in the team."
    expect(page.get_by_text(answer)).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    page.reload()
    expect(page.get_by_text(answer)).to_be_visible(timeout=REPLY_TIMEOUT_MS)


SUBAGENTS = ("/api/v1/configs/subagents", "/api/v1/configs/subagents")
FIRE_TIMEOUT_MS = 40_000


@pytest.fixture
def subagents_on(streaming, preserve):
    preserve(SUBAGENTS, on=streaming)
    with admin_of(streaming).client() as client:
        current = client.get(SUBAGENTS[0]).json()
        settings = {**current, "ENABLE_SUBAGENTS": True, "SUBAGENTS_BACKGROUND_ENABLED": True}
        client.post(SUBAGENTS[1], json=settings).raise_for_status()


def test_a_timer_fires_and_its_whole_follow_up_shows_after_a_reload(
    subagents_on, chat, account, streaming
):
    pieces = [f"steeped-{index} " for index in range(8)]
    streaming.upstream.queue(
        reply.tool_call(
            "timer", {"prompt": "The tea has steeped", "at": "3s"}, match=reply.answering("remind")
        ),
        reply.text("Timer set.", match=reply.answering("remind")),
        reply.text(pieces, chunk_delay=0.3, match=reply.answering("The tea has steeped")),
    )
    send(chat, "remind me shortly")
    expect_reply(chat, "Timer set.")

    whole = "".join(pieces).strip()
    assert stored_replies(account, chat_id_of(chat.page), 2, FIRE_TIMEOUT_MS)[1] == whole
    chat = reload_and_expect(chat, "steeped-7")
    expect_whole_reply(chat, whole)


def test_a_background_subagent_report_and_its_follow_up_show_after_a_reload(
    subagents_on, chat, account, streaming
):
    findings = ["The ledger ", "is ", "balanced ", "for ", "March."]
    pieces = [f"checked-{index} " for index in range(8)]
    call = reply.tool_call(
        "delegate_task",
        {"task": "check the ledger", "background": True},
        match=reply.answering("hand over"),
    )

    def report(body: dict) -> bool:
        users = [entry for entry in body.get("messages", []) if entry.get("role") == "user"]
        return "[ASYNC SUBAGENT COMPLETE" in str(users[-1].get("content")) if users else False

    streaming.upstream.queue(
        call,
        reply.text(findings, delay=2.0, chunk_delay=0.1, match=reply.answering("check the ledger")),
        reply.text("It is being checked.", match=reply.answering("hand over")),
        reply.text(pieces, chunk_delay=0.3, match=report),
    )
    send(chat, "hand over the books")
    expect_reply(chat, "It is being checked.")

    whole = "".join(pieces).strip()
    assert stored_replies(account, chat_id_of(chat.page), 2, FIRE_TIMEOUT_MS)[-1] == whole
    chat = reload_and_expect(chat, "checked-7")
    expect(conversation(chat)).to_contain_text("".join(pieces).strip())


REDACTING_STREAM = """
class Filter:
    def stream(self, event):
        for choice in event.get("choices", []):
            content = choice.get("delta", {}).get("content")
            if isinstance(content, str):
                choice["delta"]["content"] = content.replace("swordfish", "[redacted]")
        return event
"""


def test_a_stream_filter_rewriting_a_word_shows_the_rewrite_live_and_after_a_reload(
    chat, account, streaming
):
    pieces = ["The password ", "is ", "swordfish", " so ", "keep ", "it ", "quiet ", "please."]
    streaming.upstream.queue(reply.text(pieces, chunk_delay=0.3, match=reply.answering("secret")))
    with installed_function(admin_of(streaming), REDACTING_STREAM, is_global=True):
        send(chat, "the secret?")
        expect(last_reply(chat)).to_contain_text("[redacted]", timeout=REPLY_TIMEOUT_MS)
        assert "please." not in last_reply(chat).inner_text(), "the reply had already ended"
        expect_whole_reply(chat, "The password is [redacted] so keep it quiet please.")
        assert "swordfish" not in conversation(chat).inner_text()

    saved = ["The password is [redacted] so keep it quiet please."]
    assert stored_replies(account, chat_id_of(chat.page), 1) == saved
    chat = reload_and_expect(chat, "quiet please.")
    expect_whole_reply(chat, saved[0])


SLOW_PIPE = """
import time


WORDS = ["Roses ", "are ", "red, ", "violets ", "are ", "blue, ", "sugar ", "is sweet."]


class Pipe:
    def pipe(self, body):
        for piece in WORDS:
            time.sleep(0.3)
            yield piece
"""


def test_a_pipe_answering_with_a_generator_of_pieces_fills_the_page_and_stays_after_a_reload(
    account, streaming_chat_for, streaming
):
    with installed_function(admin_of(streaming), SLOW_PIPE) as pipe_id:
        with account.client() as client:
            client.get("/api/models", params={"refresh": "true"}).raise_for_status()
        choose_models(account, pipe_id)
        chat = streaming_chat_for(account)
        start_on_default_models(chat)
        send(chat, "recite")

        expect(last_reply(chat)).to_contain_text("red,", timeout=REPLY_TIMEOUT_MS)
        assert "sweet." not in last_reply(chat).inner_text(), "the reply had already ended"
        whole = "Roses are red, violets are blue, sugar is sweet."
        expect_whole_reply(chat, whole)
        assert stored_replies(account, chat_id_of(chat.page), 1) == [whole]

        chat = reload_and_expect(chat, "is sweet.")
        expect_whole_reply(chat, whole)


LENGTH_ACTION = """
class Action:
    async def action(self, body, __event_emitter__=None):
        reply = body["messages"][-1]["content"]
        note = {"type": "info", "content": f"The reply has {len(reply)} characters"}
        await __event_emitter__({"type": "notification", "data": note})
"""


def test_an_action_run_on_a_streamed_reply_reports_the_length_of_the_whole_text(
    chat, account, streaming
):
    pieces = ["Pack ", "the ", "tent, ", "the ", "stove."]
    streaming.upstream.queue(reply.text(pieces, chunk_delay=0.1, match=reply.answering("pack")))
    with installed_function(admin_of(streaming), LENGTH_ACTION, is_global=True) as action_id:
        chat = chat.reload()
        send(chat, "what do I pack?")
        expect_whole_reply(chat, "Pack the tent, the stove.")

        last_reply(chat).hover()
        conversation(chat).get_by_role("button", name=action_id).last.click()
        expect(chat.page.get_by_text("The reply has 25 characters")).to_be_visible()


def test_merged_deltas_with_thoughts_and_a_tool_call_show_the_same_reply_live_and_reloaded(
    account, streaming_chat_for, streaming, second_provider
):
    thoughts = [{"reasoning_content": part} for part in ("weighing ", "the two ", "options")]
    header = {"index": 0, "id": "call_1", "type": "function"}
    opening = {
        "tool_calls": [{**header, "function": {"name": "calculate_timestamp", "arguments": ""}}]
    }
    arguments = ['{"days_', 'ago": 3, ', '"weeks_ago"', ": 1}"]
    parts = [{"tool_calls": [{"index": 0, "function": {"arguments": part}}]} for part in arguments]
    rounds = []

    def answer(_request):
        rounds.append(1)
        if len(rounds) == 1:
            return sse(opening, *parts, finish_reason="tool_calls")
        return sse(*thoughts, {"content": "Ten days "}, {"content": "ago "}, {"content": "it was."})

    second_provider.route("POST", "/v1/chat/completions", answer)
    choose_models(account, SECOND_MODEL, stream_delta_chunk_size=3)
    chat = streaming_chat_for(account)
    start_on_default_models(chat)
    send(chat, "what was ten days ago?")

    expect_reply(chat, "Ten days ago it was.")
    expect(last_reply(chat)).to_contain_text("calculate_timestamp")
    explored_button(chat).click()
    thought_button(chat).click()
    expect(last_reply(chat)).to_contain_text("weighing the two options")
    sent = [r.json() for r in second_provider.requests_to("/v1/chat/completions")]
    called = [m for m in reply.conversation(sent[1]) if m.get("tool_calls")]
    assert json.loads(called[0]["tool_calls"][0]["function"]["arguments"]) == {
        "days_ago": 3,
        "weeks_ago": 1,
    }

    chat = reload_and_expect(chat, "Ten days ago it was.")
    expect(last_reply(chat)).to_contain_text("calculate_timestamp")
    explored_button(chat).click()
    thought_button(chat).click()
    expect(last_reply(chat)).to_contain_text("weighing the two options")
