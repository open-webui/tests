"""Journey: follow-up suggestions are skipped for an account deactivated while its reply ran.

Since dd42f4e91 (#33801) the work Open WebUI does after a reply (follow-up suggestions, title and
tags) runs only while the account that sent the message is still active. An admin who sets an
account to pending while one of its replies is still streaming stops the task model from being
asked for that account's follow-ups. An account that stays active still gets them.

The same commit skips channel replies to a model mention for a deactivated account; that check
runs right after the post is answered, with no step a test can hold, so it has no test here.

Discriminates: passes on dev f659c0d12; with dd42f4e91 reverted in a backend copy the task model
is asked for the deactivated account's follow-ups. The active account's case passes on both.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from harness import upstream as reply
from harness.chat import ChatTurn, send_message, wait_for_reply

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

PROMPT = "Which river runs through Graz?"
ANSWER = "The Mur runs through Graz."
# the JSON key the default follow-up prompt asks for
FOLLOW_UP_TASK = '"follow_ups"'
FOLLOW_UPS = ["How long is the Mur?", "Can you raft on it?"]
TASKS_CONFIG = ("/api/v1/tasks/config", "/api/v1/tasks/config/update")
# the follow-up request goes out right after the reply is stored
QUIET_PERIOD = 3.0
ARRIVAL_TIMEOUT = 30.0


@pytest.fixture
def follow_ups_on(admin, preserve):
    preserve(TASKS_CONFIG)
    with admin.client() as client:
        current = client.get(TASKS_CONFIG[0]).json()
        saved = client.post(TASKS_CONFIG[1], json={**current, "ENABLE_FOLLOW_UP_GENERATION": True})
    assert saved.status_code == 200, saved.text


def _follow_up_requests(upstream, account_question: str) -> list[dict]:
    return [
        body
        for body in upstream.chat_requests()
        if reply.answering(FOLLOW_UP_TASK)(body) and account_question in str(body)
    ]


def _set_role(admin, account, role: str) -> None:
    with admin.client() as client:
        updated = client.post(f"/api/v1/users/{account.id}/update", json={"role": role})
    assert updated.status_code == 200, updated.text


def _ask_with_follow_ups(account, upstream, question: str, while_held=None) -> ChatTurn:
    """Ask `question` with follow-ups on; `while_held()` runs while the reply is held back."""
    release = threading.Event()
    upstream.queue(
        reply.text(
            json.dumps({"follow_ups": FOLLOW_UPS}),
            match=lambda body: reply.answering(FOLLOW_UP_TASK)(body) and question in str(body),
        ),
        reply.text(ANSWER, hold=release, match=reply.answering(question)),
    )
    try:
        with account.client() as client:
            turn = send_message(client, question, background_tasks={"follow_up_generation": True})
        if while_held:
            while_held()
    finally:
        release.set()
    return turn


def test_an_account_deactivated_during_its_reply_gets_no_follow_ups(
    admin, make_user, upstream, follow_ups_on
):
    account = make_user()
    question = f"{PROMPT} ({account.id[:8]})"

    turn = _ask_with_follow_ups(
        account, upstream, question, while_held=lambda: _set_role(admin, account, "pending")
    )
    with admin.client() as client:
        stored = wait_for_reply(client, turn)
    time.sleep(QUIET_PERIOD)  # the follow-up request would have gone out by now

    assert ANSWER in str(stored), f"the reply itself was not stored: {stored}"
    asked = _follow_up_requests(upstream, question)
    assert asked == [], (
        f"the task model was asked {len(asked)} time(s) for the follow-ups of an account "
        "deactivated while its reply ran"
    )


def test_an_active_account_still_gets_follow_ups(make_user, upstream, follow_ups_on):
    account = make_user()
    question = f"{PROMPT} ({account.id[:8]})"

    turn = _ask_with_follow_ups(account, upstream, question)
    with account.client() as client:
        wait_for_reply(client, turn)
        deadline = time.monotonic() + ARRIVAL_TIMEOUT
        while time.monotonic() < deadline:
            stored = client.get(f"/api/v1/chats/{turn.chat_id}").json()
            message = stored["chat"]["history"]["messages"][turn.assistant_message_id]
            if message.get("followUps"):
                break
            time.sleep(0.2)

    assert len(_follow_up_requests(upstream, question)) == 1
    assert message.get("followUps") == FOLLOW_UPS, f"no follow-ups were stored: {message}"
