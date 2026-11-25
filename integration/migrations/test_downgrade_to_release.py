"""Journey: chats made on the current version open whole on v0.13.4 after a downgrade.

The manual migration guide (troubleshooting/manual-database-migration, "Downgrade (Rollback)")
tells an operator to stop Open WebUI and run `alembic downgrade <revision>` to go back to an
earlier version. What a user expects after going back to v0.13.4 this way: every chat is where
it was, in the chat list or its folder, and opens on the release with every message: every
version of a regenerated reply, the saved text of an edited reply, a tool call with its output,
a delegation with the sub-agent's report and the sub-agent's own chat, each on the message the
user was last on. Each data set starts empty on the checkout (SQLite, and an embedded
Postgres), gets these chats the way the web client sends and saves them, is taken down to
v0.13.4's last revision with the manual command, and is then served by the v0.13.4 backend
itself, which is what the release's web client reads. One more message on the release must
continue the chat on its branch.

Discriminates: passes on dev f80a4f782 on SQLite and Postgres (40 tests). With the downgrade in a
backend copy leaving out each chat's last stored message, 16 of the 20 SQLite tests fail.
"""

from __future__ import annotations

import contextlib
import copy
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import httpx
import pytest

from harness import backends
from harness import upstream as reply
from harness.chat import NO_BACKGROUND_TASKS, ChatTurn, send_message, wait_for_reply
from harness.prepared_data import manual_alembic, serving
from harness.release_backend import release_backend
from harness.subagents import delegated, report_of
from harness.tool_calls import without_hidden_calls
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.slow, pytest.mark.api, pytest.mark.requires_source]

RELEASE = "v0.13.4"
RELEASE_REVISION = "9f6c2a8d1b43"  # the last migration v0.13.4 ships
ADMIN = {"name": "Ada Admin", "email": "admin@example.com", "password": "admin-pass-1"}
CHATS = ("tooled", "edited", "regenerated", "foldered", "delegating", "subagent")

TOOL_SOURCE = '''"""
title: Weather
"""


class Tools:
    def get_weather(self, city: str) -> str:
        """Report the weather in a city."""
        return f"Sunny in {city}"
'''

FOLLOW_UP_ON_RELEASE = ("Is the Prater open late?", "Until midnight in summer.")
SETTINGS = {
    "WEBUI_AUTH": "true",
    "BYPASS_MODEL_ACCESS_CONTROL": "true",
    "ENABLE_TITLE_GENERATION": "false",
    "ENABLE_TAGS_GENERATION": "false",
    "ENABLE_FOLLOW_UP_GENERATION": "false",
    "ENABLE_SUBAGENTS": "true",
}


@dataclass
class Made:
    """A chat as it was made on the checkout, and what its user must see of it."""

    id: str
    thread: list[tuple[str, str]]  # role and text of each shown message, first to last
    facts: dict  # message ids and texts a test checks by name
    before: dict | None = None  # the chat as the checkout serves it, before the downgrade
    after: dict | str | None = None  # the chat as the release serves it, or why it did not open


@dataclass
class Steps:
    chats: dict[str, Made]
    folder_id: str
    listed_before: dict[str, set[str]]  # chat list and folder -> chat ids, on the checkout
    listed_after: dict[str, set[str] | str]  # the same on the release
    release_version: str  # what the downgraded instance reports it runs
    continued: dict | str  # the regenerated chat after one more message on the release


# --- what the web client does -----------------------------------------------------------------


def _history(client: httpx.Client, chat_id: str) -> dict:
    opened = client.get(f"/api/v1/chats/{chat_id}")
    assert opened.status_code == 200, f"the chat did not open: {opened.text}"
    return opened.json()["chat"]["history"]


def _path(messages: dict, message_id: str | None) -> list[dict]:
    shown, seen = [], set()
    while message_id in messages and message_id not in seen:
        seen.add(message_id)
        shown.append(messages[message_id])
        message_id = messages[message_id].get("parentId")
    return shown[::-1]


def _output_text(message: dict) -> str:
    return "".join(
        part.get("text", "")
        for item in message.get("output") or []
        if item.get("type") == "message"
        for part in item.get("content") or []
    )


def _shown_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or _output_text(message)


def _thread(history: dict) -> list[tuple[str, str]]:
    shown = _path(history["messages"], history.get("currentId"))
    return [(message["role"], _shown_text(message)) for message in shown]


def _wait_for(client: httpx.Client, chat_id: str, done: Callable[[dict], bool]) -> dict:
    deadline = time.monotonic() + 90
    history: dict = {}
    while time.monotonic() < deadline:
        history = _history(client, chat_id)
        if done(history):
            return history
        time.sleep(0.2)
    raise AssertionError(f"the chat never reached the expected state: {history}")


def _regenerate(client: httpx.Client, chat_id: str, question: dict) -> str:
    """Ask the model again for the question, as Regenerate does."""
    reply_id = str(uuid.uuid4())
    asked = {**question, "childrenIds": [*question["childrenIds"], reply_id]}
    sent = client.post(
        "/api/chat/completions",
        json={
            "model": MOCK_MODEL_ID,
            "stream": True,
            "chat_id": chat_id,
            "parent_id": question["parentId"],
            "id": reply_id,
            "user_message": asked,
            "session_id": f"harness-{uuid.uuid4().hex[:8]}",
            "background_tasks": NO_BACKGROUND_TASKS,
        },
    )
    assert sent.status_code == 200, f"regenerating was refused: {sent.text}"
    wait_for_reply(client, ChatTurn(chat_id, question["id"], reply_id), timeout=90)
    return reply_id


def _save(client: httpx.Client, chat_id: str, history: dict) -> None:
    """Save the whole conversation the page holds, as the reply editor does."""
    shown = _path(history["messages"], history["currentId"])
    saved = client.post(
        f"/api/v1/chats/{chat_id}", json={"chat": {"history": history, "messages": shown}}
    )
    assert saved.status_code == 200, f"saving the chat failed: {saved.text}"


def _with_output_text(output: list[dict], text: str) -> list[dict]:
    """The reply's output with its last message reading `text`, as the reply editor saves it."""
    last = [index for index, item in enumerate(output) if item.get("type") == "message"][-1]
    return [
        {**item, "content": [{"type": "output_text", "text": text}]} if index == last else item
        for index, item in enumerate(output)
    ]


def _tag() -> str:
    return uuid.uuid4().hex[:8]


# --- the chats, made on the checkout ----------------------------------------------------------


def _tooled(client: httpx.Client, upstream: reply.MockUpstream) -> Made:
    prompt, answer = f"What is the weather in Graz? {_tag()}", "Leave the umbrella at home."
    asks = reply.answering(prompt)
    upstream.queue(
        reply.tool_call("get_weather", {"city": "Graz"}, match=asks),
        reply.text(answer, match=asks),
    )
    turn = send_message(client, prompt, tool_ids=["weather_tool"])
    wait_for_reply(client, turn)
    return Made(
        turn.chat_id,
        [("user", prompt), ("assistant", answer)],
        {"reply": turn.assistant_message_id},
    )


def _edited(client: httpx.Client, upstream: reply.MockUpstream) -> Made:
    prompt, edited = f"Name a castle near Vienna. {_tag()}", "Burg Kreuzenstein, north of Vienna."
    upstream.queue(reply.text("Burg Kreuzenstein.", match=reply.answering(prompt)))
    turn = send_message(client, prompt)
    wait_for_reply(client, turn)
    history = copy.deepcopy(_history(client, turn.chat_id))
    message = history["messages"][turn.assistant_message_id]
    message.update(
        originalContent=message["content"],
        content=edited,
        output=_with_output_text(message["output"], edited),
    )
    _save(client, turn.chat_id, history)
    facts = {"reply": turn.assistant_message_id, "edited": edited}
    return Made(turn.chat_id, [("user", prompt), ("assistant", edited)], facts)


def _regenerated(client: httpx.Client, upstream: reply.MockUpstream) -> Made:
    prompt, follow_up = f"Suggest a day trip from Vienna. {_tag()}", f"And by night? {_tag()}"
    versions = ["Bratislava by boat.", "Melk Abbey by train.", "The Wachau by bike."]
    asks = reply.answering(prompt)
    upstream.queue(reply.text(versions[0], match=asks))
    turn = send_message(client, prompt)
    wait_for_reply(client, turn)
    reply_ids = [turn.assistant_message_id]
    for text in versions[1:]:
        upstream.queue(reply.text(text, match=asks))
        question = _history(client, turn.chat_id)["messages"][turn.user_message_id]
        reply_ids.append(_regenerate(client, turn.chat_id, question))
    upstream.queue(reply.text("The Prater wheel.", match=reply.answering(follow_up)))
    earlier = [{"role": "user", "content": prompt}, {"role": "assistant", "content": versions[2]}]
    next_turn = send_message(
        client, follow_up, chat_id=turn.chat_id, parent_id=reply_ids[2], history=earlier
    )
    wait_for_reply(client, next_turn)
    thread = [
        ("user", prompt),
        ("assistant", versions[2]),
        ("user", follow_up),
        ("assistant", "The Prater wheel."),
    ]
    facts = {"question": turn.user_message_id, "versions": reply_ids, "texts": versions}
    return Made(turn.chat_id, thread, facts)


def _foldered(client: httpx.Client, upstream: reply.MockUpstream, folder_id: str) -> Made:
    prompt, answer = f"Where do trains to Graz leave from? {_tag()}", "From Wien Hauptbahnhof."
    upstream.queue(reply.text(answer, match=reply.answering(prompt)))
    turn = send_message(client, prompt, folder_id=folder_id)
    wait_for_reply(client, turn)
    return Made(turn.chat_id, [("user", prompt), ("assistant", answer)], {})


def _delegated(client: httpx.Client, upstream: reply.MockUpstream) -> tuple[Made, Made]:
    prompt, task = f"Plan my museum day. {_tag()}", f"List three museums near the Ring. {_tag()}"
    found = "Albertina, Leopold Museum and Kunsthistorisches Museum."
    follow_up = "Start at the Albertina, then walk to the Leopold Museum."
    upstream.queue(
        reply.tool_call("delegate_task", {"task": task}, match=reply.answering(prompt)),
        reply.text(found, match=delegated(task)),
        reply.text("Handed over.", match=reply.answering(prompt)),
        reply.text(follow_up, match=report_of(task)),
    )
    turn = send_message(client, prompt)
    stored = wait_for_reply(client, turn)
    [handle] = [
        json.loads(item["output"][0]["text"])
        for item in without_hidden_calls(stored["output"])
        if item["type"] == "function_call_output"
    ]

    def followed_up(history: dict) -> bool:
        current = history["messages"].get(history.get("currentId"), {})
        return current.get("done") and _shown_text(current) == follow_up

    history = _wait_for(client, turn.chat_id, followed_up)
    [report] = [
        message
        for message in history["messages"].values()
        if (message.get("meta") or {}).get("type") == "subagent"
    ]
    assert found in report["content"], f"the report lacks the sub-agent's answer: {report}"
    thread = [
        ("user", prompt),
        ("assistant", "Handed over."),
        ("user", report["content"]),
        ("assistant", follow_up),
    ]
    facts = {"reply": turn.assistant_message_id, "report": report["id"]}
    delegating = Made(turn.chat_id, thread, facts)
    subagent = Made(handle["subagent_chat_id"], [("user", task), ("assistant", found)], {})
    return delegating, subagent


# --- the journey ------------------------------------------------------------------------------


def _listed(client: httpx.Client, folder_id: str) -> dict[str, set[str] | str]:
    listed: dict[str, set[str] | str] = {}
    for where, path in (
        ("chat list", "/api/v1/chats/"),
        ("folder", f"/api/v1/chats/folder/{folder_id}/list"),
    ):
        response = client.get(path)
        listed[where] = (
            {chat["id"] for chat in response.json()}
            if response.status_code == 200
            else f"HTTP {response.status_code}: {response.text[:300]}"
        )
    return listed


def _opened(client: httpx.Client, chat_id: str) -> dict | str:
    response = client.get(f"/api/v1/chats/{chat_id}")
    if response.status_code != 200:
        return f"HTTP {response.status_code}: {response.text[:300]}"
    return response.json()


def _make_chats(base_url: str, upstream: reply.MockUpstream) -> tuple[dict[str, Made], str]:
    signup = httpx.post(f"{base_url}/api/v1/auths/signup", json=ADMIN, timeout=60)
    assert signup.status_code == 200, f"signing up failed: {signup.text}"
    headers = {"Authorization": f"Bearer {signup.json()['token']}"}
    with httpx.Client(base_url=base_url, headers=headers, timeout=120) as client:
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
        created = client.post(
            "/api/v1/tools/create",
            json={
                "id": "weather_tool",
                "name": "Weather",
                "content": TOOL_SOURCE,
                "meta": {"description": "Weather"},
            },
        )
        assert created.status_code == 200, f"creating the tool failed: {created.text}"
        folder = client.post("/api/v1/folders/", json={"name": "Trips"})
        assert folder.status_code == 200, f"creating the folder failed: {folder.text}"
        folder_id = folder.json()["id"]
        delegating, subagent = _delegated(client, upstream)
        chats = {
            "tooled": _tooled(client, upstream),
            "edited": _edited(client, upstream),
            "regenerated": _regenerated(client, upstream),
            "foldered": _foldered(client, upstream, folder_id),
            "delegating": delegating,
            "subagent": subagent,
        }
        for made in chats.values():
            made.before = _opened(client, made.id)
    return chats, folder_id


def _continue_on_release(
    client: httpx.Client, upstream: reply.MockUpstream, made: Made
) -> dict | str:
    """One more message on the branch the chat is on, sent as the v0.13 web client sends it."""
    if not isinstance(made.after, dict):
        return f"the chat did not open: {made.after}"
    question, answer = FOLLOW_UP_ON_RELEASE
    upstream.queue(reply.text(answer, match=reply.answering(question)))
    user_id, reply_id = str(uuid.uuid4()), str(uuid.uuid4())
    user_message = {
        "id": user_id,
        "parentId": made.after["chat"]["history"]["currentId"],
        "childrenIds": [reply_id],
        "role": "user",
        "content": question,
        "models": [MOCK_MODEL_ID],
        "timestamp": int(time.time()),
    }
    sent = client.post(
        "/api/chat/completions",
        json={
            "stream": True,
            "model": MOCK_MODEL_ID,
            "params": {},
            "features": {},
            "variables": {},
            "session_id": f"harness-{uuid.uuid4().hex[:8]}",
            "id": reply_id,
            "message_ids": [{"model_id": MOCK_MODEL_ID, "message_id": reply_id, "modelIdx": 0}],
            "parent_id": user_message["parentId"],
            "user_message": user_message,
            "background_tasks": NO_BACKGROUND_TASKS,
            "chat_id": made.id,
        },
    )
    if sent.status_code != 200:
        return f"sending failed: HTTP {sent.status_code}: {sent.text[:300]}"
    try:
        wait_for_reply(client, ChatTurn(made.id, user_id, reply_id))
    except AssertionError as error:
        return f"the reply never finished: {error}"
    return _opened(client, made.id)


def _sign_in(base_url: str) -> httpx.Client:
    signed_in = httpx.post(
        f"{base_url}/api/v1/auths/signin",
        json={"email": ADMIN["email"], "password": ADMIN["password"]},
        timeout=60,
    )
    assert signed_in.status_code == 200, f"signing in on {RELEASE} failed: {signed_in.text}"
    headers = {"Authorization": f"Bearer {signed_in.json()['token']}"}
    return httpx.Client(base_url=base_url, headers=headers, timeout=120)


@pytest.fixture(
    scope="module",
    params=[
        pytest.param("sqlite", id="sqlite"),
        pytest.param("postgres", id="postgres", marks=pytest.mark.requires_postgres),
    ],
)
def steps(request, tmp_path_factory) -> Iterator[Steps]:
    root: Path = tmp_path_factory.mktemp(f"downgrade-{request.param}")
    release = release_backend(RELEASE, root)
    data_dir = root / "data"
    data_dir.mkdir()
    with contextlib.ExitStack() as stack:
        upstream, shutdown = reply.serve()
        stack.callback(shutdown)
        settings = {
            **SETTINGS,
            "ENABLE_OPENAI_API": "true",
            "OPENAI_API_BASE_URLS": upstream.base_url,
            "OPENAI_API_KEYS": "sk-mock",
        }
        if request.param == "postgres":
            settings["DATABASE_URL"] = stack.enter_context(backends.postgres_database())
        with serving(data_dir, settings) as checkout:
            chats, folder_id = _make_chats(checkout.base_url, upstream)
            with _sign_in(checkout.base_url) as client:
                listed_before = _listed(client, folder_id)

        downgraded = manual_alembic(
            data_dir, "downgrade", RELEASE_REVISION, database_url=settings.get("DATABASE_URL")
        )
        assert downgraded.returncode == 0, (
            f"alembic downgrade {RELEASE_REVISION} failed:\n{downgraded.stderr[-3000:]}"
        )

        with serving(data_dir, {**settings, **release.settings}, backend=release.backend) as old:
            with _sign_in(old.base_url) as client:
                listed_after = _listed(client, folder_id)
                for made in chats.values():
                    made.after = _opened(client, made.id)
                release_version = client.get("/api/version").json().get("version")
                continued = _continue_on_release(client, upstream, chats["regenerated"])
        yield Steps(chats, folder_id, listed_before, listed_after, release_version, continued)


# --- what the user sees on the release --------------------------------------------------------


def _opened_history(steps: Steps, name: str) -> dict:
    made = steps.chats[name]
    assert isinstance(made.before, dict), f"the {name} chat did not open before the downgrade"
    assert isinstance(made.after, dict), (
        f"the {name} chat does not open on {RELEASE} after the downgrade: {made.after}"
    )
    return made.after["chat"]["history"]


def _visible(message: dict) -> dict:
    """What the web client shows of a message and how it hangs in the conversation."""
    return {
        "role": message.get("role"),
        "text": _shown_text(message),
        "parentId": message.get("parentId"),
        "childrenIds": message.get("childrenIds"),
        "model": message.get("model"),
        "timestamp": message.get("timestamp"),
        "output": message.get("output"),
    }


def _unchanged_by_a_reply(message: dict) -> dict:
    """What a later message must leave alone: all but the replies hanging below."""
    return {key: value for key, value in _visible(message).items() if key != "childrenIds"}


def test_the_downgraded_data_is_served_by_the_release(steps):
    assert steps.release_version == RELEASE.lstrip("v")


def test_every_chat_is_listed_where_it_was(steps):
    chats = steps.chats
    expected = {chats[name].id for name in ("tooled", "edited", "regenerated", "delegating")}
    assert steps.listed_before["chat list"] == expected, (
        f"the chat list before the downgrade: {steps.listed_before['chat list']}"
    )
    assert steps.listed_before["folder"] == {chats["foldered"].id}
    assert steps.listed_after == steps.listed_before, (
        f"after the downgrade {RELEASE} lists {steps.listed_after}, before it {steps.listed_before}"
    )


@pytest.mark.parametrize("name", CHATS)
def test_the_chat_opens_on_the_message_the_user_was_on(steps, name):
    history = _opened_history(steps, name)
    assert _thread(history) == steps.chats[name].thread, (
        f"the {name} chat shows {_thread(history)} on {RELEASE}"
    )
    before = steps.chats[name].before["chat"]["history"]
    assert history.get("currentId") == before.get("currentId")


@pytest.mark.parametrize("name", CHATS)
def test_every_message_of_the_chat_is_intact(steps, name):
    history = _opened_history(steps, name)
    before = steps.chats[name].before["chat"]["history"]["messages"]
    after = history["messages"]
    assert set(after) == set(before), (
        f"the {name} chat lost {set(before) - set(after)} and gained {set(after) - set(before)}"
    )
    changed = {
        message_id: {"before": _visible(message), "after": _visible(after[message_id])}
        for message_id, message in before.items()
        if _visible(message) != _visible(after[message_id])
    }
    assert not changed, f"messages of the {name} chat changed on {RELEASE}: {changed}"


def test_the_tool_call_and_its_output_are_shown(steps):
    made = steps.chats["tooled"]
    message = _opened_history(steps, "tooled")["messages"][made.facts["reply"]]
    items = without_hidden_calls(message.get("output") or [])
    calls = [item for item in items if item.get("type") == "function_call"]
    outputs = json.dumps([item for item in items if item.get("type") == "function_call_output"])
    assert [call.get("name") for call in calls] == ["get_weather"], f"tool calls: {calls}"
    assert "Sunny in Graz" in outputs, f"the tool's output is gone: {items}"


def test_the_edited_reply_reads_the_edit(steps):
    made = steps.chats["edited"]
    message = _opened_history(steps, "edited")["messages"][made.facts["reply"]]
    assert message.get("content") == made.facts["edited"], (
        f"the reply reads {message.get('content')!r}"
    )
    assert _output_text(message) == made.facts["edited"], (
        f"the reply's shown text reads {_output_text(message)!r}"
    )


def test_every_version_of_the_regenerated_reply_is_offered(steps):
    made = steps.chats["regenerated"]
    messages = _opened_history(steps, "regenerated")["messages"]
    versions = messages[made.facts["question"]].get("childrenIds")
    assert versions == made.facts["versions"], f"the question offers {versions}"
    texts = [_shown_text(messages.get(version, {})) for version in made.facts["versions"]]
    assert texts == made.facts["texts"], f"the versions read {texts}"


def test_the_folder_chat_is_still_in_its_folder(steps):
    after = steps.chats["foldered"].after
    assert isinstance(after, dict), f"the folder chat does not open: {after}"
    assert after.get("folder_id") == steps.folder_id, f"the chat is in {after.get('folder_id')}"


def test_the_delegation_keeps_its_call_and_the_report_after_it(steps):
    made = steps.chats["delegating"]
    messages = _opened_history(steps, "delegating")["messages"]
    calls = [
        item.get("name")
        for item in without_hidden_calls(messages[made.facts["reply"]].get("output") or [])
        if item.get("type") == "function_call"
    ]
    assert calls == ["delegate_task"], f"the delegating reply shows the calls {calls}"
    report = messages.get(made.facts["report"], {})
    assert report.get("parentId") == made.facts["reply"], f"the report hangs at {report}"


def test_the_chat_takes_one_more_message_on_the_release(steps):
    made = steps.chats["regenerated"]
    assert isinstance(steps.continued, dict), f"the chat cannot be continued: {steps.continued}"
    history = steps.continued["chat"]["history"]
    expected = [
        *made.thread,
        ("user", FOLLOW_UP_ON_RELEASE[0]),
        ("assistant", FOLLOW_UP_ON_RELEASE[1]),
    ]
    assert _thread(history) == expected, f"the chat reads {_thread(history)} after one more message"
    before = made.before["chat"]["history"]["messages"]
    changed = {
        message_id: history["messages"].get(message_id)
        for message_id, message in before.items()
        if _unchanged_by_a_reply(message)
        != _unchanged_by_a_reply(history["messages"].get(message_id, {}))
    }
    assert not changed, f"the next message on {RELEASE} lost or changed messages: {changed}"
