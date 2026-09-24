"""Provider usage dialects in a chat whose messages predate the message table.

Commits df94268 and e8f2c12 (#27031, #26752, #24410) in `utils/context_compaction.py`. The
compaction threshold and the context-usage readout only read `input_tokens` from a stored usage
block, so a chat whose last reply carried Ollama (`prompt_eval_count`), llama.cpp (`prompt_n`)
or OpenAI (`prompt_tokens`) counts never reached the threshold. Every message written today goes
through the message table, which normalizes usage on the way in, so over HTTP the dialects arrive
already translated; only a chat whose history still lives in the legacy JSON (no message rows)
shows the raw dialects, and a fresh instance has no way to make one. The rest of the compaction
fixes are pinned in integration/chat/test_context_compaction.py.

Discriminates: passes on dev bbfa876af; reading only `input_tokens` again fails the Ollama,
llama.cpp and OpenAI cases and passes the normalized one.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.regression

COMPACTION_ON = {
    "chat.context_compaction.enable": True,
    "chat.context_compaction.token_threshold": 80_000,
}


@pytest.fixture(scope="session")
def compaction(owui_module):
    owui_module("open_webui.config")  # runs the migrations, so the message table exists
    return owui_module("open_webui.utils.context_compaction")


@pytest.fixture(scope="session")
def chat_model(owui_module):
    return owui_module("open_webui.models.chats").ChatModel


def legacy_chat(chat_model, usage: dict):
    """A chat known only by its embedded history: nothing for it in the message table."""
    question = {"id": "q1", "role": "user", "content": "hello", "parentId": None}
    answer = {"id": "a1", "role": "assistant", "content": "hi", "parentId": "q1", "usage": usage}
    history = {"currentId": "a1", "messages": {"q1": question, "a1": answer}}
    return chat_model(
        id=f"legacy-{uuid.uuid4()}",
        user_id="legacy-user",
        title="legacy",
        chat={"history": history},
        created_at=0,
        updated_at=0,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage",
    [
        pytest.param({"prompt_eval_count": 90_000, "eval_count": 500}, id="ollama"),
        pytest.param({"prompt_n": 90_000, "predicted_n": 500}, id="llama.cpp"),
        pytest.param({"prompt_tokens": 90_000, "completion_tokens": 500}, id="openai"),
        pytest.param({"input_tokens": 90_000, "output_tokens": 500}, id="normalized"),
    ],
)
async def test_every_usage_dialect_counts_toward_the_threshold(compaction, chat_model, usage):
    with patch.object(compaction.Config, "get_many", AsyncMock(return_value=COMPACTION_ON)):
        context_usage = await compaction.get_chat_context_usage(
            chat=legacy_chat(chat_model, usage), model_id="m"
        )

    assert context_usage["tokens"] == 90_500
    assert context_usage["tokens"] > context_usage["threshold"]
