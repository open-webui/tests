"""Regression: a deactivated account kept its live WebSocket access.

open-webui 0.11.0 fix `f517cc717` (PR #27537): the WebSocket entry points accepted the token of
an account moved to `pending` while every HTTP route refused it. The web client ignores the
socket's answer, so the regression leaves no mark in the page and is pinned over the socket in
integration/security/test_websocket_role_gate.py. This pins the HTTP half the fix aligned the
sockets with: the open page of an account moved to `pending` is locked behind the activation
screen on its next load.

Twin of unit/security/test_websocket_role_gate.py.

Discriminates: nothing on its own; nearby behaviour that passes on dev bbfa876af and with the
role check removed from `get_verified_user_by_token` alike.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

from utils.chat_ui import chat_input

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

PAGE_TIMEOUT_MS = 30_000


def test_an_account_moved_to_pending_gets_the_activation_screen(page_for, make_user, admin):
    account = make_user()
    page = page_for(account)
    expect(chat_input(page)).to_be_visible(timeout=PAGE_TIMEOUT_MS)

    with admin.client() as client:
        demoted = client.post(f"/api/v1/users/{account.id}/update", json={"role": "pending"})
    assert demoted.status_code == 200, demoted.text
    page.reload()

    expect(page.get_by_text("Account Activation Pending")).to_be_visible(timeout=PAGE_TIMEOUT_MS)
    expect(chat_input(page)).to_be_hidden()
