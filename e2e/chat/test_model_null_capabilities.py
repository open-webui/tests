"""Regression: chatting with a model whose `meta.capabilities` is null showed an error, no reply.

open-webui 0.10.2, issue #26412, fix `0016266c0`: the web client sends the memory feature with
every chat while memories are enabled, and `model_allows_memory` read `capabilities` with
`.get('capabilities', {})`, which returns None for a stored null, so the next `.get` raised and
the chat ended in `'NoneType' object has no attribute 'get'`. The fix reads it with `or {}`.

Twin of unit/chat/test_model_null_capabilities.py.

Discriminates: passes on bbfa876af, fails with 0016266c0 reverted (the reply never appears).
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import expect_reply, send

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

MODEL_NAME = "Report writer"


@pytest.fixture
def account(admin, make_user):
    """A fresh user who alone may read a preset with `capabilities: null`."""
    account = make_user()
    model_id = f"capabilities-{uuid.uuid4().hex[:8]}"
    form = {
        "id": model_id,
        "base_model_id": MOCK_MODEL_ID,
        "name": MODEL_NAME,
        "meta": {"capabilities": None},
        "params": {},
        "access_grants": [
            {"principal_type": "user", "principal_id": account.id, "permission": "read"}
        ],
    }
    with admin.client() as client:
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
    yield account
    with admin.client() as client:
        client.post("/api/v1/models/model/delete", json={"id": model_id})


def test_a_chat_with_the_model_gets_its_reply(page_for, account, upstream):
    upstream.queue(reply.text("the report is ready"))
    page = page_for(account)
    page.goto("/")

    page.get_by_role("button", name=re.compile("^Selected model")).first.click()
    page.get_by_role("option", name=f"Select {MODEL_NAME} model").click()
    expect(page.get_by_role("button", name=f"Selected model: {MODEL_NAME}").first).to_be_visible()
    send(page, "is the report ready?")

    expect_reply(page, "the report is ready")
    assert upstream.chat_requests()[-1]["model"] == MOCK_MODEL_ID
