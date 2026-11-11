"""Regression: the context usage in the Status panel read against the wrong amount.

1. Inline images (fix 95dd3321a, PR #31915, issue #31913). The chat page estimates the context
   usage it shows in the Status panel itself, and counted an image kept in the chat record as a
   base64 data URI as text: a few hundred kilobytes of picture read as a hundred thousand
   tokens, far past the compaction threshold of a chat that is nearly empty. The encoded data is
   now left out of the estimate, here as on the server. Twin of
   integration/chat/test_compaction_inline_images.py.
2. Model threshold (fix c9415a688, issue #32821). A model's own compaction threshold decides when
   its chats compact, but the model list left it out, so the panel measured against the chat's
   own or the global threshold. The model list now carries it and the model's value wins over
   the chat's. Twin of the model-threshold cases in integration/chat/test_context_compaction.py.

Discriminates: the inline-image test passes on the dev b859124f9 build and fails on a build with
95dd3321a reverted (the panel reads "500% 100k/20k"). The model-threshold test passes on dev
f844f6e89 and fails on c9415a688^ (the panel reads against the chat's 900).

Both tests run in the main chat and in a side chat, a saved chat dragged into a tab. On dev
c15f28649 both side chat variants are red and green on head e791cbdde (fix PR
open-webui/open-webui#34064): `/status` did nothing in a side chat, so the Status panel never
opened (open-webui/open-webui#34018).
"""

from __future__ import annotations

import base64
import re
import uuid

import pytest
from playwright.sync_api import expect

from harness.chat_history import seed_chat
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import chat_input, expect_reply

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

CHAT_CONFIG = ("/api/v1/chats/config", "/api/v1/chats/config")
THRESHOLD = 20_000
# about 400 KB of base64, a hundred thousand tokens if counted as text
INLINE_IMAGE = "data:image/png;base64," + base64.b64encode(bytes(300_000)).decode()


@pytest.fixture
def compaction(admin, preserve):
    preserve(CHAT_CONFIG)
    with admin.client() as client:
        current = client.get(CHAT_CONFIG[0]).json()
        updated = client.post(
            CHAT_CONFIG[1],
            json={
                **current,
                "ENABLE_CONTEXT_COMPACTION": True,
                "CONTEXT_COMPACTION_TOKEN_THRESHOLD": THRESHOLD,
                "CONTEXT_COMPACTION_TOKEN_CAP": THRESHOLD,
            },
        )
    assert updated.status_code == 200, updated.text


def test_the_status_panel_does_not_count_an_inline_image_as_text(chat_for, make_user, compaction):
    """In a side chat: open-webui/open-webui#34018."""
    account = make_user()
    image = {"type": "image", "id": str(uuid.uuid4()), "url": INLINE_IMAGE}
    with account.client() as client:
        chat_id, _ = seed_chat(
            client,
            [
                {"role": "user", "content": "what is in this picture?", "files": [image]},
                {"role": "assistant", "content": "a blank square"},
            ],
        )
    chat = chat_for(account, chat_id=chat_id)
    expect_reply(chat, "a blank square")

    expect(status_panel_usage(chat)).to_have_text(re.compile(r"^\d% \d+/20k$"))


def status_panel_usage(chat):
    chat_input(chat).click()
    chat.page.keyboard.type("/status")
    chat.page.keyboard.press("Enter")
    return chat.page.get_by_text("Context usage").locator("xpath=following-sibling::span")


def test_the_status_panel_measures_against_the_models_own_threshold(
    chat_for, admin, make_user, compaction
):
    """In a side chat: open-webui/open-webui#34018."""
    account = make_user()
    model_id = f"small-window-{uuid.uuid4().hex[:8]}"
    with admin.client() as client:
        created = client.post(
            "/api/v1/models/create",
            json={
                "id": model_id,
                "base_model_id": MOCK_MODEL_ID,
                "name": "Small window",
                "meta": {},
                "params": {"compact_token_threshold": 500},
                "access_grants": [
                    {"principal_type": "user", "principal_id": account.id, "permission": "read"}
                ],
            },
        )
        assert created.status_code == 200, created.text
    try:
        with account.client() as client:
            chat_id, _ = seed_chat(
                client,
                [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi there"},
                ],
                model=model_id,
            )
            saved = client.post(
                f"/api/v1/chats/{chat_id}",
                json={"chat": {"params": {"compact_token_threshold": 900}}},
            )
            assert saved.status_code == 200, saved.text
        chat = chat_for(account, chat_id=chat_id)
        expect_reply(chat, "hi there")

        expect(status_panel_usage(chat)).to_have_text(re.compile(r"^\d+% \d+/500$"))
    finally:
        with admin.client() as client:
            client.post("/api/v1/models/model/delete", json={"id": model_id})
