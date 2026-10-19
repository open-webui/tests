"""Regression: context compaction ignored token weight in two places.

Two defects in open-webui 0.12.0's `compact_messages_for_request`:

1. The pre-compaction guard `... or len(messages) <= 3` skipped compaction for
   any chat with three or fewer messages regardless of size. Dual-write storage
   keeps a whole tool loop in one assistant row's `output`, so a three-message
   chat can carry megabytes of history; such a chat never compacted and 400s
   on every send against the model's context wall.

2. `_find_compaction_boundary` chose the boundary by message count alone, so a
   disproportionately heavy recent message survived in the kept tail. The tail
   was returned without a re-check, and the request hit the model wall anyway
   (one 400, self-healing only on the next send).

Fix under test: the guard skips only small chats whose stored estimate is
actually under the threshold, and the boundary deepens (dropping whole turns,
least aggressively) until the kept tail fits under the threshold.

Discriminates: passes on the fix branch; on v0.12.0 (15818d88) the
huge-content tests observe `compacted=False` (guard skip) or the count-based
boundary (heavy row kept).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression

THRESHOLD = 100_000


@pytest.fixture(scope="module")
def compaction(owui_module):
    """`open_webui.utils.context_compaction`."""
    return owui_module("open_webui.utils.context_compaction")


@pytest.fixture()
def enabled_compaction_config(compaction, monkeypatch):
    """Compaction enabled, 100k threshold, 40% retention, no prompt override."""

    async def _load_config():
        return {
            'enable': True,
            'token_threshold': THRESHOLD,
            'token_cap': THRESHOLD,
            'retention_percentage': 40,
            'prompt_template': '',
        }

    monkeypatch.setattr(compaction, '_load_config', _load_config)


@pytest.fixture()
def fake_summary(compaction, monkeypatch):
    """Record the (compacted, kept) split instead of calling a model."""
    calls = []

    async def _generate_summary(request, user, model_id, models, compacted, recent, previous, template):
        calls.append((compacted, recent))
        return 'SUMMARY'

    monkeypatch.setattr(compaction, '_generate_summary', _generate_summary)
    return calls


def user_message(message_id, content='hello world'):
    return {'id': message_id, 'role': 'user', 'content': content}


def assistant_message(message_id, content='pong', output=None, usage=None):
    return {'id': message_id, 'role': 'assistant', 'content': content, 'output': output, 'usage': usage}


def tool_loop_blob(items=50, chars_per_item=20_000):
    """A tool-loop `output` array heavy enough to blow past a 100k-token estimate."""
    return [
        {'type': 'function_call_output', 'call_id': f'call_{i}', 'output': 'x' * chars_per_item}
        for i in range(items)
    ]


@pytest.mark.asyncio
async def test_three_messages_with_huge_content_now_compacts(compaction, enabled_compaction_config, fake_summary):
    # The incident shape: a tool loop persisted in one assistant row.
    messages = [user_message('u1'), assistant_message('a1', output=tool_loop_blob()), user_message('u2')]

    result, summary, compacted = await compaction.compact_messages_for_request(
        None, None, messages, {}, 'model', {}
    )

    assert compacted is True
    assert summary == 'SUMMARY'
    assert [message['id'] for message in result] == ['u2']
    (dropped, kept), = fake_summary
    assert [message['id'] for message in dropped] == ['u1', 'a1']
    assert [message['id'] for message in kept] == ['u2']


@pytest.mark.asyncio
async def test_three_trivial_messages_still_skip(compaction, enabled_compaction_config):
    # The guard's original intent: a fresh three-message chat is not worth an LLM call.
    messages = [user_message('u1'), assistant_message('a1'), user_message('u2')]

    result, summary, compacted = await compaction.compact_messages_for_request(
        None, None, messages, {}, 'model', {}
    )

    assert compacted is False
    assert summary is None
    assert result == messages


@pytest.mark.asyncio
async def test_many_small_messages_under_threshold_still_skip(compaction, enabled_compaction_config):
    messages = [
        user_message(f'u{i // 2}') if i % 2 == 0 else assistant_message(f'a{i // 2}') for i in range(10)
    ]

    _, summary, compacted = await compaction.compact_messages_for_request(None, None, messages, {}, 'model', {})

    assert compacted is False
    assert summary is None


@pytest.mark.asyncio
async def test_heavy_recent_message_compacts_instead_of_shipping_doomed_request(
    compaction, enabled_compaction_config, fake_summary
):
    # Blob in the most recent assistant row: the count-based boundary kept it.
    messages = [
        user_message('u1'),
        assistant_message('a1'),
        user_message('u2'),
        assistant_message('a2', output=tool_loop_blob()),
        user_message('u3'),
    ]

    result, summary, compacted = await compaction.compact_messages_for_request(
        None, None, messages, {}, 'model', {}
    )

    assert compacted is True
    assert summary == 'SUMMARY'
    assert [message['id'] for message in result] == ['u3']
    (dropped, kept), = fake_summary
    assert [message['id'] for message in dropped] == ['u1', 'a1', 'u2', 'a2']


@pytest.mark.asyncio
async def test_small_chat_with_high_recorded_usage_but_small_stored_context_skips(
    compaction, enabled_compaction_config
):
    # The threshold tripped on recorded usage, but the stored messages are small:
    # compaction could drop at most one message, so the guard still skips.
    messages = [
        user_message('u1'),
        assistant_message('a1', usage={'prompt_tokens': 150_000}),
        user_message('u2'),
    ]

    _, summary, compacted = await compaction.compact_messages_for_request(None, None, messages, {}, 'model', {})

    assert compacted is False
    assert summary is None


def test_count_based_boundary_unchanged_when_tail_fits(compaction):
    messages = [
        user_message(f'u{i // 2}') if i % 2 == 0 else assistant_message(f'a{i // 2}') for i in range(10)
    ]

    # N=10, retention 40: keep 4, target 6, user boundaries [2, 4, 6, 8] -> 6.
    assert compaction._find_compaction_boundary(messages, 40, THRESHOLD) == 6
    assert compaction._find_compaction_boundary(messages, 40) == 6


def test_boundary_deepens_to_exclude_a_heavy_middle_row(compaction):
    # The incident shape: three messages, blob in the middle row. The count-based
    # boundary is 0 (nothing droppable while keeping the last two messages).
    messages = [user_message('u1'), assistant_message('a1', output=tool_loop_blob()), user_message('u2')]

    assert compaction._find_compaction_boundary(messages, 40) == 0
    assert compaction._find_compaction_boundary(messages, 40, THRESHOLD) == 2


def test_boundary_deepens_past_a_heavy_recent_row(compaction):
    messages = [
        user_message('u1'),
        assistant_message('a1'),
        user_message('u2'),
        assistant_message('a2', output=tool_loop_blob()),
        user_message('u3'),
    ]

    # Count-based pick is 2 (keeps the blob); deepening lands on 4.
    assert compaction._find_compaction_boundary(messages, 40, THRESHOLD) == 4


def test_boundary_stops_at_the_shallowest_fitting_boundary(compaction):
    messages = [
        user_message('u1'),
        assistant_message('a1'),
        user_message('u2'),
        assistant_message('a2'),
        user_message('u3'),
        assistant_message('a3'),
        user_message('u4'),
        assistant_message('a4', output=tool_loop_blob()),
        user_message('u5'),
        assistant_message('a5'),
        user_message('u6'),
    ]

    # Count-based pick is 6 (keeps the blob); 8 already fits, so it must not
    # deepen all the way to 10.
    assert compaction._find_compaction_boundary(messages, 40, THRESHOLD) == 8


def test_boundary_never_drops_past_the_last_user_boundary(compaction):
    # Blob after the last user message: even the deepest boundary keeps it.
    messages = [user_message('u1'), assistant_message('a1'), user_message('u2'), assistant_message('a2', output=tool_loop_blob())]

    boundary = compaction._find_compaction_boundary(messages, 40, THRESHOLD)

    assert boundary == 2
    assert messages[boundary:]  # the kept tail is never empty


def test_boundary_with_tool_batches_deepens_over_call_boundaries(compaction):
    # In-flight tool loop: boundaries fall after assistant tool-call messages.
    messages = [
        user_message('u1'),
        {'id': 'a1', 'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'c1'}]},
        {'id': 't1', 'role': 'tool', 'content': 'ok'},
        user_message('u2'),
        {'id': 'a2', 'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'c2'}], 'output': tool_loop_blob()},
        {'id': 't2', 'role': 'tool', 'content': 'ok'},
        user_message('u3'),
        {'id': 'a3', 'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'c3'}]},
        {'id': 't3', 'role': 'tool', 'content': 'ok'},
    ]

    # Count-based pick is 4 (keeps the blob); deepening lands on 7, just
    # before the last call/result batch, keeping that batch intact.
    assert compaction._find_compaction_boundary(messages, 40, THRESHOLD) == 7


def test_boundary_without_threshold_keeps_legacy_behavior(compaction):
    messages = [user_message('u1'), assistant_message('a1', output=tool_loop_blob()), user_message('u2')]

    assert compaction._find_compaction_boundary(messages, 40, None) == 0
