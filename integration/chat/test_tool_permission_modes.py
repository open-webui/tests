"""Journey: the tool permission modes an admin offers and the server enforces for each account.

With the admin's Tool Permissions on, each chat runs its tool calls in one of three modes: Full
access runs them, Ask for approval pauses before every call until the account allows or denies
it, and Auto (0.13.1, switched on with Auto permissions) has a reviewer model rate each call and
pauses only those rated at or above the admin's threshold. The reviewer is reached at
`<base URL>/decisions` with the admin's key and model. Auto never fails open: a reviewer error,
or Auto picked while the admin has it off, asks instead. A paused call runs only once its own
account allows it; a denied call never runs, and the model is told. The sidebar list marks a chat
whose reply waits for approval.

Discriminates: passes on dev f659c0d12. Each mutation in a backend copy turns exactly one test
red: the threshold compared with `>` (threshold test), a reviewer error that lets the call run
(reviewer-error test), an Auto picked while off resolved to Full access (Auto-off test), no Auto
default (default test), the admin's prompt ignored (default test), no Auto review after a denial
(deny-then-review test), the resolve owner check removed (stranger test), the already-resolved
check removed (late-allow test), Auto announced while Tool Permissions is off or unset settings
not kept on save (settings test), and a chat never marked as waiting (risky-call test).
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import httpx
import pytest

from harness import upstream as reply
from harness.chat import ChatTurn, send_message, wait_for_reply
from harness.listener import ReceivedRequest, json_answer, text_answer
from harness.python_tools import python_tool

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

CHAT_CONFIG = ("/api/v1/chats/config", "/api/v1/chats/config")
WAIT = 20.0
REVIEWER_KEY = "sk-reviewer-test"
REVIEWER_MODEL = "reviewer-model"
REVIEW_PROMPT = "Ask before anything marked risky."
REJECTED = "Error: tool call rejected by user."

NOTICE_TOOL = '''
class Tools:
    def post_notice(self, text: str) -> str:
        """Post a notice on the board.

        :param text: The notice to post
        """
        with open({ledger!r}, "a") as ledger:
            ledger.write(text + "\\n")
        return f"posted {{text}}"
'''


@pytest.fixture
def chat_settings(admin, preserve):
    """`chat_settings(**changes)` updates the admin chat settings for the rest of the test."""
    preserve(CHAT_CONFIG)

    def change(**changes) -> None:
        with admin.client() as client:
            current = client.get(CHAT_CONFIG[0]).json()
            saved = client.post(CHAT_CONFIG[1], json={**current, **changes})
            assert saved.status_code == 200, f"saving {changes} failed: {saved.text}"

    return change


@pytest.fixture
def ledger(tmp_path: Path) -> Path:
    """The file the notice tool appends to on the server each time it really runs."""
    path = tmp_path / "notices.txt"
    path.touch()
    return path


@pytest.fixture
def notice_tool(admin, ledger) -> str:
    with python_tool(admin, NOTICE_TOOL.format(ledger=str(ledger)), name="Notice board") as tool_id:
        yield tool_id


def risk_by_text(request: ReceivedRequest):
    """The reviewer rates a call risky when its notice text says so."""
    evidence = json.loads(request.json()["input"])
    probability = 0.9 if "risky" in evidence["arguments"]["text"] else 0.1
    return json_answer(
        {
            "answers": [
                {"type": "predicate", "name": "requires_confirmation", "probability": probability}
            ]
        }
    )


@pytest.fixture
def auto_permissions(chat_settings, listener):
    """Tool Permissions with Auto on, reviewed by the local listener; yields the listener."""
    listener.route("POST", "/v1/decisions", risk_by_text)
    chat_settings(
        ENABLE_TOOL_PERMISSIONS=True,
        ENABLE_TOOL_PERMISSIONS_AUTO=True,
        TOOL_PERMISSIONS_DECISIONS_API_BASE_URL=f"{listener.base_url}/v1",
        TOOL_PERMISSIONS_DECISIONS_API_KEY=REVIEWER_KEY,
        TOOL_PERMISSIONS_DECISIONS_MODEL=REVIEWER_MODEL,
        TOOL_PERMISSIONS_DECISIONS_THRESHOLD=0.5,
        TOOL_PERMISSIONS_DECISIONS_PROMPT=REVIEW_PROMPT,
    )
    return listener


def notice_call(text: str, call_id: str = "call_1") -> dict:
    arguments = json.dumps({"text": text})
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "post_notice", "arguments": arguments},
    }


def queue_notices(upstream, question: str, *texts: str, answer: str = "Done.") -> None:
    """The model posts each of `texts` in one turn, then answers `answer`."""
    calls = [notice_call(text, f"call_{number}") for number, text in enumerate(texts, start=1)]
    upstream.queue(
        reply.Reply(tool_calls=calls, match=reply.answering(question)),
        reply.text(answer, match=reply.answering(question)),
    )


def unique(prompt: str) -> str:
    return f"{prompt} {uuid.uuid4().hex[:6]}"


def posted(ledger: Path) -> list[str]:
    return ledger.read_text().splitlines()


def statuses(client: httpx.Client, turn: ChatTurn) -> dict[str, str]:
    chat = client.get(f"/api/v1/chats/{turn.chat_id}").json()["chat"]
    output = chat["history"]["messages"][turn.assistant_message_id].get("output") or []
    return {
        item.get("call_id"): item.get("status")
        for item in output
        if item.get("type") == "function_call"
    }


def wait_until_pending(client: httpx.Client, turn: ChatTurn, call_id: str = "call_1") -> None:
    deadline = time.monotonic() + WAIT
    seen: dict = {}
    while time.monotonic() < deadline:
        seen = statuses(client, turn)
        if seen.get(call_id) == "pending":
            return
        time.sleep(0.2)
    raise AssertionError(f"{call_id} never waited for approval; the calls stood at {seen}")


def resolve(client: httpx.Client, turn: ChatTurn, action: str, call_id: str = "call_1"):
    return client.post(
        f"/api/v1/chats/{turn.chat_id}/messages/{turn.assistant_message_id}/resolve",
        json={"call_id": call_id, "action": action},
    )


def tool_results(upstream, question: str) -> dict[str, str]:
    final = [body for body in upstream.chat_requests() if reply.answering(question)(body)][-1]
    return {
        entry["tool_call_id"]: str(entry["content"])
        for entry in reply.conversation(final)
        if entry["role"] == "tool"
    }


def waiting_in_list(client: httpx.Client, chat_id: str) -> bool:
    listed = client.get("/api/v1/chats/")
    listed.raise_for_status()
    [entry] = [chat for chat in listed.json() if chat["id"] == chat_id]
    return entry["pending_approval"]


def test_auto_is_the_default_and_runs_a_call_the_reviewer_finds_safe(
    auto_permissions, notice_tool, ledger, make_user, upstream
):
    question = unique("post that the ferry is on time")
    queue_notices(upstream, question, "ferry on time", answer="Posted it.")
    with make_user().client() as client:
        turn = send_message(client, question, tool_ids=[notice_tool])
        message = wait_for_reply(client, turn)

    assert message["content"].endswith("Posted it."), message["content"]
    assert posted(ledger) == ["ferry on time"], "a call rated safe did not run without asking"
    [review] = [entry for entry in auto_permissions.received if entry.path == "/v1/decisions"]
    assert review.headers.get("Authorization") == f"Bearer {REVIEWER_KEY}"
    body = review.json()
    assert body["model"] == REVIEWER_MODEL
    [rating] = body["questions"]
    assert rating["instructions"] == REVIEW_PROMPT, "the admin's review prompt was not used"
    evidence = json.loads(body["input"])
    assert evidence["tool"]["name"] == "post_notice"
    assert evidence["arguments"] == {"text": "ferry on time"}
    assert question in evidence["user_request"], f"the reviewer never saw the request: {evidence}"


def test_auto_pauses_a_risky_call_until_its_account_allows_it(
    auto_permissions, notice_tool, ledger, make_user, upstream
):
    question = unique("post that the harbour is closed")
    queue_notices(upstream, question, "risky: harbour closed", answer="Posted the closure.")
    with make_user().client() as client:
        turn = send_message(
            client, question, tool_ids=[notice_tool], params={"tool_approval_mode": "auto"}
        )
        wait_until_pending(client, turn)
        assert posted(ledger) == [], "a call rated risky ran before anyone allowed it"
        assert waiting_in_list(client, turn.chat_id), (
            "the chat list does not show the reply waiting"
        )

        allowed = resolve(client, turn, "approve")
        assert allowed.status_code == 200, allowed.text
        message = wait_for_reply(client, turn)
        assert not waiting_in_list(client, turn.chat_id), (
            "the chat still shows as waiting after Allow"
        )

    assert message["content"].endswith("Posted the closure."), message["content"]
    assert posted(ledger) == ["risky: harbour closed"]


def test_auto_asks_when_the_rating_is_exactly_the_threshold(
    auto_permissions, notice_tool, ledger, make_user, upstream
):
    auto_permissions.route(
        "POST",
        "/v1/decisions",
        json_answer(
            {
                "answers": [
                    {"type": "predicate", "name": "requires_confirmation", "probability": 0.5}
                ]
            }
        ),
    )
    question = unique("post the tide table")
    queue_notices(upstream, question, "tide table")
    with make_user().client() as client:
        turn = send_message(
            client, question, tool_ids=[notice_tool], params={"tool_approval_mode": "auto"}
        )
        wait_until_pending(client, turn)

    assert posted(ledger) == [], "a call rated at the threshold ran without asking"


def test_auto_asks_when_the_reviewer_fails_and_hides_its_answer(
    auto_permissions, notice_tool, ledger, make_user, upstream
):
    auto_permissions.route(
        "POST", "/v1/decisions", text_answer("REVIEWER-INTERNAL-TRACE", "text/plain", status=500)
    )
    question = unique("post the weather")
    queue_notices(upstream, question, "sunny")
    with make_user().client() as client:
        turn = send_message(
            client, question, tool_ids=[notice_tool], params={"tool_approval_mode": "auto"}
        )
        wait_until_pending(client, turn)
        stored = client.get(f"/api/v1/chats/{turn.chat_id}").text

    assert posted(ledger) == [], "a call ran although the reviewer could not rate it"
    assert "REVIEWER-INTERNAL-TRACE" not in stored, "the reviewer's error body reached the chat"


def test_after_a_denied_call_auto_reviews_and_runs_the_next_one(
    auto_permissions, notice_tool, ledger, make_user, upstream
):
    question = unique("post both notices")
    queue_notices(upstream, question, "risky: close the pier", "pier open", answer="Posted one.")
    with make_user().client() as client:
        turn = send_message(
            client, question, tool_ids=[notice_tool], params={"tool_approval_mode": "auto"}
        )
        wait_until_pending(client, turn, "call_1")
        assert posted(ledger) == []

        denied = resolve(client, turn, "reject", "call_1")
        assert denied.status_code == 200, denied.text
        message = wait_for_reply(client, turn)

    assert message["content"].endswith("Posted one."), message["content"]
    assert posted(ledger) == ["pier open"], "the safe call after a denial did not run on its own"
    results = tool_results(upstream, question)
    assert results["call_1"] == REJECTED
    assert results["call_2"] == "posted pier open"


def test_auto_picked_while_the_admin_has_it_off_still_asks(
    chat_settings, listener, notice_tool, ledger, make_user, upstream
):
    listener.route("POST", "/v1/decisions", risk_by_text)
    chat_settings(
        ENABLE_TOOL_PERMISSIONS=True,
        ENABLE_TOOL_PERMISSIONS_AUTO=False,
        TOOL_PERMISSIONS_DECISIONS_API_BASE_URL=f"{listener.base_url}/v1",
    )
    question = unique("post the timetable")
    queue_notices(upstream, question, "timetable")
    with make_user().client() as client:
        turn = send_message(
            client, question, tool_ids=[notice_tool], params={"tool_approval_mode": "auto"}
        )
        wait_until_pending(client, turn)

    assert posted(ledger) == [], "Auto picked while switched off ran the call without asking"
    assert listener.received == [], "the reviewer was asked although Auto is off"


def test_with_tool_permissions_off_calls_run_whatever_the_chat_asks(
    chat_settings, notice_tool, ledger, make_user, upstream
):
    chat_settings(ENABLE_TOOL_PERMISSIONS=False)
    question = unique("post the menu")
    queue_notices(upstream, question, "menu", answer="Posted the menu.")
    with make_user().client() as client:
        turn = send_message(
            client, question, tool_ids=[notice_tool], params={"tool_approval_mode": "ask"}
        )
        message = wait_for_reply(client, turn)

    assert message["content"].endswith("Posted the menu."), message["content"]
    assert posted(ledger) == ["menu"]


def test_an_unknown_permission_mode_is_refused(chat_settings, make_user, upstream):
    chat_settings(ENABLE_TOOL_PERMISSIONS=True)
    before = len(upstream.chat_requests())
    with make_user().client() as client:
        with pytest.raises(AssertionError, match="HTTP 422"):
            send_message(client, unique("hello"), params={"tool_approval_mode": "everything"})
    assert len(upstream.chat_requests()) == before, "the model was called for a refused mode"


def test_a_denied_call_never_runs_even_when_allowed_afterwards(
    chat_settings, notice_tool, ledger, make_user, upstream
):
    chat_settings(ENABLE_TOOL_PERMISSIONS=True)
    question = unique("post the warning")
    queue_notices(upstream, question, "storm warning", answer="I was not allowed to post it.")
    with make_user().client() as client:
        turn = send_message(
            client, question, tool_ids=[notice_tool], params={"tool_approval_mode": "ask"}
        )
        wait_until_pending(client, turn)
        denied = resolve(client, turn, "reject")
        assert denied.status_code == 200, denied.text
        message = wait_for_reply(client, turn)

        late_allow = resolve(client, turn, "approve")

    assert late_allow.status_code == 409, f"a denied call could be allowed later: {late_allow.text}"
    assert message["content"].endswith("I was not allowed to post it."), message["content"]
    assert tool_results(upstream, question) == {"call_1": REJECTED}
    assert posted(ledger) == [], "a denied call ran on the server"


def test_another_account_cannot_allow_or_deny_a_paused_call(
    chat_settings, notice_tool, ledger, make_user, upstream
):
    chat_settings(ENABLE_TOOL_PERMISSIONS=True)
    question = unique("post the lost and found")
    queue_notices(upstream, question, "lost umbrella", answer="Posted.")
    owner, stranger = make_user(), make_user()
    with owner.client() as client, stranger.client() as other:
        turn = send_message(
            client, question, tool_ids=[notice_tool], params={"tool_approval_mode": "ask"}
        )
        wait_until_pending(client, turn)

        for action in ("approve", "reject"):
            attempt = resolve(other, turn, action)
            assert attempt.status_code in (401, 403, 404), (
                f"another account could {action} the call: HTTP {attempt.status_code}"
            )
        assert statuses(client, turn) == {"call_1": "pending"}
        assert posted(ledger) == []

        allowed = resolve(client, turn, "approve")
        assert allowed.status_code == 200, allowed.text
        wait_for_reply(client, turn)

    assert posted(ledger) == ["lost umbrella"]


def test_the_auto_settings_are_kept_checked_and_announced(
    chat_settings, admin, make_user, listener
):
    base_url = f"{listener.base_url}/v1"
    chat_settings(
        ENABLE_TOOL_PERMISSIONS=True,
        ENABLE_TOOL_PERMISSIONS_AUTO=True,
        TOOL_PERMISSIONS_DECISIONS_API_BASE_URL=base_url + "/",
        TOOL_PERMISSIONS_DECISIONS_API_KEY=REVIEWER_KEY,
        TOOL_PERMISSIONS_DECISIONS_MODEL=REVIEWER_MODEL,
        TOOL_PERMISSIONS_DECISIONS_THRESHOLD=0.25,
        TOOL_PERMISSIONS_DECISIONS_PROMPT=REVIEW_PROMPT,
    )
    with admin.client() as client:
        stored = client.get(CHAT_CONFIG[0]).json()
        assert stored["ENABLE_TOOL_PERMISSIONS_AUTO"] is True
        assert stored["TOOL_PERMISSIONS_DECISIONS_API_BASE_URL"] == base_url
        assert stored["TOOL_PERMISSIONS_DECISIONS_MODEL"] == REVIEWER_MODEL
        assert stored["TOOL_PERMISSIONS_DECISIONS_THRESHOLD"] == 0.25
        assert stored["TOOL_PERMISSIONS_DECISIONS_PROMPT"] == REVIEW_PROMPT
        assert stored["DEFAULT_TOOL_PERMISSIONS_DECISIONS_PROMPT"].strip()

        # A form from before Auto existed leaves the Auto settings as they were.
        older_form = {
            key: value
            for key, value in stored.items()
            if not key.startswith(("TOOL_PERMISSIONS_", "ENABLE_TOOL_PERMISSIONS_AUTO", "DEFAULT_"))
        }
        assert client.post(CHAT_CONFIG[1], json=older_form).status_code == 200
        kept = client.get(CHAT_CONFIG[0]).json()
        assert kept["ENABLE_TOOL_PERMISSIONS_AUTO"] is True
        assert kept["TOOL_PERMISSIONS_DECISIONS_MODEL"] == REVIEWER_MODEL
        assert kept["TOOL_PERMISSIONS_DECISIONS_THRESHOLD"] == 0.25

        for bad in (
            {"TOOL_PERMISSIONS_DECISIONS_THRESHOLD": 1.5},
            {"TOOL_PERMISSIONS_DECISIONS_API_BASE_URL": "ftp://reviewer.local/v1"},
            {"TOOL_PERMISSIONS_DECISIONS_MODEL": "  "},
        ):
            refused = client.post(CHAT_CONFIG[1], json={**kept, **bad})
            assert refused.status_code in (400, 422), f"{bad} was saved: HTTP {refused.status_code}"
        assert client.get(CHAT_CONFIG[0]).json() == kept, "a refused save changed the settings"

    account = make_user()
    with account.client() as client:
        features = client.get("/api/config").json()["features"]
        assert features["enable_tool_permissions_auto"] is True
        assert client.get(CHAT_CONFIG[0]).status_code in (401, 403), "a user read the reviewer key"

    chat_settings(ENABLE_TOOL_PERMISSIONS=False)
    with account.client() as client:
        features = client.get("/api/config").json()["features"]
    assert features["enable_tool_permissions_auto"] is False, (
        "Auto is offered while Tool Permissions is off"
    )
