"""Regression: a chat opened in a tab beside another chat took over system prompts not its own.

A saved chat without a system prompt of its own follows the one in Settings > General. Sent from
while dragged into a tab, it kept a copy of the Settings prompt as its own, so a later change in
Settings no longer reached it (open-webui/open-webui#34028). A new chat opened in a tab from a
pinned model is a normal new chat; it was sent the system prompt of the chat beside it and the
side conversation instruction, and kept both (open-webui/open-webui#34029). Fix PR
open-webui/open-webui#34064.

The saved chat test runs in the main chat, as the twin that shows the plain behaviour, and dragged
into a tab beside a new chat and beside a saved one (`utils/chat_surface.py`). The new chat test
opens the tab from a pinned model beside a chat whose Controls hold a system prompt. The New Chat
tab shortcut opens a side chat, which is a side conversation by design, so it is left out.

Discriminates: the tab variants fail on dev c15f28649 (the Settings prompt is stored with the
dragged chat; the new tab is sent the neighbour's prompt and the side instruction) and pass on dev
e791cbdde, which holds the fix; the main chat variant passes on both.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.chat_history import seed_chat
from harness.upstream import MOCK_MODEL_ID
from utils.chat_surface import (
    DRAGGED_TAB,
    MAIN_CHAT,
    SIDE_CHAT,
    ChatSurface,
    main_chat,
    open_pinned_model_tab,
)
from utils.chat_ui import expect_reply, send

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

FIRST_PROMPT = "Answer only with the word APPLE."
LATER_PROMPT = "Answer only with the word BANANA."


def _save_settings_prompt(account, prompt: str) -> None:
    with account.client() as client:
        saved = client.post("/api/v1/users/user/settings/update", json={"ui": {"system": prompt}})
    assert saved.status_code == 200, saved.text


def _system_prompts_for(chat: ChatSurface | Page, upstream, question: str) -> list[str]:
    upstream.queue(reply.text("Noted.", match=reply.answering(question)))
    send(chat, question)
    expect_reply(chat, "Noted.")
    request = [body for body in upstream.chat_requests() if reply.answering(question)(body)][-1]
    return [entry["content"] for entry in request["messages"] if entry["role"] == "system"]


def _stored_system_prompt(account, chat_id: str):
    with account.client() as client:
        stored = client.get(f"/api/v1/chats/{chat_id}")
    assert stored.status_code == 200, stored.text
    return (stored.json()["chat"].get("params") or {}).get("system")


# a saved chat opens in the main view or is dragged into a tab, the only places it can be opened
@pytest.mark.chat_surfaces(MAIN_CHAT, SIDE_CHAT, DRAGGED_TAB)
def test_a_chat_sent_from_in_a_tab_keeps_following_the_settings_system_prompt(
    chat_for, page_for, make_user, upstream
):
    """In a tab: open-webui/open-webui#34028."""
    account = make_user()
    _save_settings_prompt(account, FIRST_PROMPT)
    with account.client() as client:
        chat_id, _ = seed_chat(
            client,
            [
                {"role": "user", "content": "Which fruit is on the table?"},
                {"role": "assistant", "content": "APPLE"},
            ],
        )
    chat = chat_for(account, chat_id=chat_id)
    sent = _system_prompts_for(chat, upstream, f"hello ({uuid.uuid4().hex[:8]})")
    assert any(FIRST_PROMPT in prompt for prompt in sent), (
        f"the Settings prompt was not sent: {sent}"
    )

    stored = _stored_system_prompt(account, chat_id)
    assert not stored, f"sending from the {chat.place} stored the prompt {stored!r} with the chat"
    _save_settings_prompt(account, LATER_PROMPT)
    reopened = main_chat(page_for(account, path=f"/c/{chat_id}"))
    later = _system_prompts_for(reopened, upstream, f"hello again ({uuid.uuid4().hex[:8]})")
    assert later == [LATER_PROMPT], f"after a message in the {chat.place} the chat sent {later}"


def test_a_new_chat_opened_from_a_pinned_model_starts_like_any_new_chat(
    page_for, make_user, admin, upstream, preserve
):
    """open-webui/open-webui#34029."""
    preserve("permissions")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["chat"].update(side_chat=True)
        allowed = client.post("/api/v1/users/default/permissions", json=permissions)
    assert allowed.status_code == 200, allowed.text
    account = make_user()
    with account.client() as client:
        pinned = client.post(
            "/api/v1/users/user/settings/update", json={"ui": {"pinnedModels": [MOCK_MODEL_ID]}}
        )
        assert pinned.status_code == 200, pinned.text
        neighbour_id, _ = seed_chat(
            client,
            [
                {"role": "user", "content": "Which fruit is on the table?"},
                {"role": "assistant", "content": "APPLE"},
            ],
        )
        given = client.post(
            f"/api/v1/chats/{neighbour_id}", json={"chat": {"params": {"system": FIRST_PROMPT}}}
        )
    assert given.status_code == 200, given.text
    page = page_for(account, path=f"/c/{neighbour_id}")
    expect(main_chat(page).last_reply).to_contain_text("APPLE")

    tab = open_pinned_model_tab(page, MOCK_MODEL_ID)
    sent = _system_prompts_for(tab, upstream, f"hello ({uuid.uuid4().hex[:8]})")

    assert sent == [], f"the new chat in a tab was sent the system prompts {sent}"
    expect(page).not_to_have_url(re.compile(neighbour_id))
    new_chat_id = page.url.rsplit("/", 1)[-1]
    stored = _stored_system_prompt(account, new_chat_id)
    assert not stored, f"the new chat in a tab kept the system prompt {stored!r}"
