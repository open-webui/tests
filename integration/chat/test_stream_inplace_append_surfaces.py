"""Streamed replies through Functions, the API passthrough and other surfaces, on both modes.

Whatever assembles a streamed reply from many small pieces (the socket chat handler, the API
passthrough that rebuilds the reply for an outlet filter) either copies the text on every piece
or extends it in place, depending on `ENABLE_CHAT_RESPONSE_STREAM_INPLACE_APPEND`. Every test
runs on an instance of each kind and expects the same literal outcome: a global filter rewriting
each chunk, an outlet filter editing the reply and reporting what it saw, a pipe yielding text
pieces or chunks and SSE lines with reasoning, an action run on the finished reply, an API client
without a socket session (with and without reasoning), a temporary chat and its follow-up, a
channel where a model answers and what the members receive, a sub-agent in the foreground and
one in the background, a timer, an automation that streams into a chat and an Anthropic client
of the messages endpoint whose reply an outlet filter audits.

`test_a_background_subagent_reports_its_whole_reply_and_the_chat_continues`,
`test_a_subagent_reply_of_many_pieces_comes_back_whole_to_the_parent_chat`,
`test_a_timer_run_saves_its_reply_of_many_pieces_in_the_chat` and
`test_an_automation_run_streams_its_reply_into_a_new_chat` are red on dev 62f70a844: since de73bb830
a chat request whose reply message is already stored in the chat, the way automations, sub-agents
and timers prepare their reply, is refused with 409 and the reply is never written
(open-webui/open-webui#32066).

Discriminates: in a backend copy whose in-place branch adds a bar before each piece every
`append-in-place` case goes red and every `append-copies` case stays green; with the bar in the
copying branch it is the other way round.
"""

from __future__ import annotations

import json
import textwrap
import time
import uuid

import httpx
import pytest

from harness import raw_provider
from harness import upstream as reply
from harness.actors import admin_of, create_user
from harness.channel_quotes import enable_channels, group_channel, post_message
from harness.chat import ask, send_message
from harness.plugins import installed_function
from harness.raw_provider import RAW_MODEL_ID, chunk, sse
from harness.second_provider import OPENAI_CONFIG
from harness.socket_client import connected
from harness.upstream import MOCK_MODEL_ID

pytestmark = [
    pytest.mark.journey,
    pytest.mark.slow,
    pytest.mark.api,
    pytest.mark.requires_source,
]

INPLACE_APPEND = "ENABLE_CHAT_RESPONSE_STREAM_INPLACE_APPEND"


@pytest.fixture(params=["false", "true"], ids=["append-copies", "append-in-place"])
def streaming(request, instance_with):
    """A scratch instance with the toggle off, then on; every test below runs on both."""
    return instance_with({INPLACE_APPEND: request.param})


@pytest.fixture
def instance_admin(streaming):
    return admin_of(streaming)


@pytest.fixture
def person(streaming):
    return create_user(streaming)


@pytest.fixture
def make_member(streaming):
    return lambda: create_user(streaming)


def source(code: str) -> str:
    return textwrap.dedent(code).strip() + "\n"


def wait_for(condition, what: str, timeout: float = 30.0):
    """The first truthy value of `condition()`, polled until the deadline."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if found := condition():
            return found
        time.sleep(0.2)
    raise AssertionError(f"timed out waiting for {what}")


def reasoning_of(message: dict) -> str:
    return "".join(
        part["text"]
        for item in message["output"]
        if item["type"] == "reasoning"
        for part in item.get("content") or []
    )


def stored_message(client: httpx.Client, chat_id: str, message_id: str) -> dict:
    stored = client.get(f"/api/v1/chats/{chat_id}")
    stored.raise_for_status()
    return stored.json()["chat"]["history"]["messages"][message_id]


def direct_stream(client: httpx.Client, model: str = MOCK_MODEL_ID) -> list[dict]:
    """A streamed chat without a socket session; returns its decoded events."""
    request = {"model": model, "stream": True, "messages": [{"role": "user", "content": "hi"}]}
    response = client.post("/api/chat/completions", json=request)
    assert response.status_code == 200, response.text
    return [
        json.loads(line.removeprefix("data:"))
        for line in response.text.splitlines()
        if line.startswith("data:") and line != "data: [DONE]"
    ]


def streamed(events: list[dict], field: str = "content") -> str:
    return "".join(
        choice["delta"].get(field) or "" for event in events for choice in event.get("choices", [])
    )


def audit_source(url: str, body: str) -> str:
    """A global outlet filter that posts what it saw to `url`; `body` edits the messages."""
    return source(
        f"""
        import json
        import urllib.request


        class Filter:
            async def outlet(self, body):
                last = body["messages"][-1]
                seen = {{"content": last.get("content"), "output": last.get("output")}}
                request = urllib.request.Request("{url}", data=json.dumps(seen).encode())
                urllib.request.urlopen(request, timeout=5)
        {textwrap.indent(textwrap.dedent(body).strip(), " " * 8)}
                return body
        """
    )


# --- Functions that touch the stream -----------------------------------------------------------

REDACTING_STREAM = source(
    """
    class Filter:
        def stream(self, event):
            for choice in event.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if isinstance(content, str):
                    choice["delta"]["content"] = content.replace("swordfish", "[redacted]")
            return event
    """
)


def test_a_stream_filter_redacts_a_word_in_the_pieces_of_a_saved_reply(
    instance_admin, person, streaming
):
    streaming.upstream.queue(
        reply.text(["The password is ", "swordfish", " so ", "keep it quiet."])
    )
    with (
        installed_function(instance_admin, REDACTING_STREAM, is_global=True),
        person.client() as client,
    ):
        _, message = ask(client, "what is the password?")

    assert message["content"] == "The password is [redacted] so keep it quiet."


def edit_of(message: dict) -> dict | None:
    return message if message["content"].endswith("[checked]") else None


def test_an_outlet_filter_sees_the_whole_reply_and_its_edit_is_saved(
    instance_admin, person, streaming, listener
):
    listener.route("POST", "/audit", lambda _: (200, {}, b""))
    outlet = audit_source(
        f"{listener.base_url}/audit", 'body["messages"][-1]["content"] += " [checked]"'
    )
    streaming.upstream.queue(reply.text(["Tea is ", "ready ", "in five ", "minutes."]))
    with installed_function(instance_admin, outlet, is_global=True), person.client() as client:
        turn, _ = ask(client, "when is the tea ready?")
        edited = wait_for(
            lambda: edit_of(stored_message(client, turn.chat_id, turn.assistant_message_id)),
            "the outlet edit",
        )

    assert edited["content"] == "Tea is ready in five minutes. [checked]"
    assert edited["originalContent"] == "Tea is ready in five minutes."
    [audit] = listener.requests_to("/audit")
    assert audit.json()["content"] == "Tea is ready in five minutes."


TEXT_PIPE = source(
    """
    class Pipe:
        def pipe(self, body):
            for piece in ["Roses ", "are ", "red, ", "violets ", "are blue."]:
                yield piece
    """
)


def test_a_pipe_generator_of_text_pieces_makes_one_saved_reply(instance_admin):
    with (
        installed_function(instance_admin, TEXT_PIPE) as pipe_id,
        instance_admin.client() as client,
    ):
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
        _, message = ask(client, "recite", model=pipe_id)

    assert message["content"] == "Roses are red, violets are blue."


CHUNK_PIPE = source(
    """
    import json


    def piece(**delta):
        return {"choices": [{"index": 0, "delta": delta}]}


    class Pipe:
        def pipe(self, body):
            yield piece(role="assistant", reasoning_content="The user wants ")
            yield piece(reasoning_content="a greeting. ")
            yield piece(content="Good ")
            yield "data: " + json.dumps(piece(content="morning, "))
            yield piece(content="and welcome.")
    """
)


def test_a_pipe_yielding_chunks_and_sse_lines_with_reasoning_keeps_both_texts(instance_admin):
    with (
        installed_function(instance_admin, CHUNK_PIPE) as pipe_id,
        instance_admin.client() as client,
    ):
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
        _, message = ask(client, "greet me", model=pipe_id)

    assert message["content"] == "Good morning, and welcome."
    assert reasoning_of(message) == "The user wants a greeting. "


ECHO_ACTION = source(
    """
    import json
    import urllib.request


    class Action:
        async def action(self, body, __user__=None):
            reply = body["messages"][-1]["content"]
            request = urllib.request.Request("{url}", data=json.dumps({{"reply": reply}}).encode())
            urllib.request.urlopen(request, timeout=5)
            return {{"length": len(reply)}}
    """
)


def test_an_action_run_on_a_streamed_reply_sees_all_of_its_text(
    instance_admin, streaming, listener
):
    listener.route("POST", "/action", lambda _: (200, {}, b""))
    streaming.upstream.queue(reply.text(["Pack ", "the ", "tent, ", "the ", "stove."]))
    action = ECHO_ACTION.format(url=f"{listener.base_url}/action")
    with installed_function(instance_admin, action) as action_id, instance_admin.client() as client:
        turn, message = ask(client, "what do I pack?")
        history = [
            {"id": turn.user_message_id, "role": "user", "content": "what do I pack?"},
            {"id": turn.assistant_message_id, "role": "assistant", "content": message["content"]},
        ]
        acted = client.post(
            f"/api/chat/actions/{action_id}",
            json={
                "model": MOCK_MODEL_ID,
                "chat_id": turn.chat_id,
                "id": turn.assistant_message_id,
                "session_id": "harness",
                "messages": history,
            },
        )

    assert message["content"] == "Pack the tent, the stove."
    assert acted.status_code == 200, acted.text
    assert acted.json() == {"length": 25}
    assert [call.json() for call in listener.requests_to("/action")] == [
        {"reply": "Pack the tent, the stove."}
    ]


# --- the API passthrough ------------------------------------------------------------------------


def test_an_api_client_gets_the_stream_and_the_outlet_filter_sees_the_rebuilt_reply(
    instance_admin, person, streaming, listener
):
    listener.route("POST", "/audit", lambda _: (200, {}, b""))
    outlet = audit_source(f"{listener.base_url}/audit", "pass")
    streaming.upstream.queue(reply.text(["Trains ", "leave ", "at ", "noon."]))
    with installed_function(instance_admin, outlet, is_global=True), person.client() as client:
        events = direct_stream(client)
        wait_for(lambda: listener.requests_to("/audit"), "the outlet filter's audit")

    assert streamed(events) == "Trains leave at noon."
    [audit] = listener.requests_to("/audit")
    assert audit.json()["content"] == "Trains leave at noon."


@pytest.fixture
def raw(instance_admin, preserve, listener, streaming) -> raw_provider.RawProvider:
    preserve(OPENAI_CONFIG, on=streaming)
    return raw_provider.connect(instance_admin, listener)


def test_an_api_client_with_reasoning_deltas_gets_them_and_the_outlet_sees_the_reply(
    instance_admin, listener, raw
):
    listener.route("POST", "/audit", lambda _: (200, {}, b""))
    outlet = audit_source(f"{listener.base_url}/audit", "pass")
    raw.stream(
        sse(
            chunk({"role": "assistant", "reasoning_content": "Rivers run "}),
            chunk({"reasoning_content": "to the sea. "}),
            chunk({"content": "Water "}),
            chunk({"content": "flows "}),
            chunk({"content": "downhill."}),
            chunk({}, "stop"),
        )
    )
    with (
        installed_function(instance_admin, outlet, is_global=True),
        instance_admin.client() as client,
    ):
        events = direct_stream(client, RAW_MODEL_ID)
        wait_for(lambda: listener.requests_to("/audit"), "the outlet filter's audit")

    assert streamed(events, "reasoning_content") == "Rivers run to the sea. "
    assert streamed(events) == "Water flows downhill."
    [audit] = listener.requests_to("/audit")
    seen = audit.json()
    assert seen["content"] == "Water flows downhill."
    assert reasoning_of(seen) == "Rivers run to the sea. "


# --- temporary chats ----------------------------------------------------------------------------


def output_text(output: list[dict], kind: str) -> str:
    return "".join(
        part["text"]
        for item in output
        if item["type"] == kind
        for part in item.get("content") or []
    )


@pytest.mark.parametrize("prefix", ["temporary:", "local:"])
def test_a_temporary_chat_streams_over_the_socket_and_its_next_turn_carries_the_reply(
    person, streaming, prefix
):
    upstream = streaming.upstream
    upstream.queue(
        reply.text(
            ["The deepest lake ", "in Carinthia ", "is the Millstatter See."],
            reasoning="Carinthia has many lakes. ",
            match=reply.answering("Which lake is deepest?"),
        ),
        reply.text("Roughly 141 metres.", match=reply.answering("How deep is it?")),
    )
    with connected(person) as socket, person.client() as client:
        chat_id = f"{prefix}{socket.client.sid}"
        send_message(
            client, "Which lake is deepest?", chat_id=chat_id, session_id=socket.client.sid
        )
        done = socket.wait_for(chat_id, "chat:completion", done=True)
        deltas = [
            event["data"]["delta"]
            for event in socket.events_of(chat_id)
            if event["data"].get("type") == "response.output_text.delta"
        ]
        answer = output_text(done["data"]["output"], "message")
        send_message(
            client,
            "How deep is it?",
            chat_id=chat_id,
            session_id=socket.client.sid,
            history=[
                {"role": "user", "content": "Which lake is deepest?"},
                {"role": "assistant", "content": answer},
            ],
        )
        wait_for(lambda: len(upstream.chat_requests()) >= 2, "the follow-up request")

    assert "".join(deltas) == "The deepest lake in Carinthia is the Millstatter See."
    assert answer == "The deepest lake in Carinthia is the Millstatter See."
    assert output_text(done["data"]["output"], "reasoning") == "Carinthia has many lakes. "
    follow_up = [r for r in upstream.chat_requests() if reply.answering("How deep is it?")(r)][0]
    assert [(m["role"], m["content"]) for m in follow_up["messages"]] == [
        ("user", "Which lake is deepest?"),
        ("assistant", "The deepest lake in Carinthia is the Millstatter See."),
        ("user", "How deep is it?"),
    ]
    assert client_has_no_chat(person, chat_id)


def client_has_no_chat(person, chat_id: str) -> bool:
    with person.client() as client:
        return client.get(f"/api/v1/chats/{chat_id}").status_code != 200


# --- a model answering in a channel -------------------------------------------------------------


@pytest.fixture
def channels_on(instance_admin, preserve, streaming):
    preserve("admin_config", on=streaming)
    enable_channels(instance_admin)


def channel_updates(session, channel_id: str) -> list[dict]:
    """The `message:update` payloads that reached this tab for the channel."""
    updates: list[dict] = []
    session.client.on(
        "events:channel",
        lambda event: (
            updates.append(event["data"]["data"])
            if event.get("channel_id") == channel_id and event["data"]["type"] == "message:update"
            else None
        ),
    )
    return updates


def test_a_model_answering_in_a_channel_saves_the_reply_and_members_receive_it(
    channels_on, streaming, make_member
):
    owner, member = make_member(), make_member()
    channel_id = group_channel(owner, member)
    streaming.upstream.queue(
        reply.text(["Friday ", "works ", "for ", "everyone."], match=reply.answering("Which day?"))
    )
    with connected(member) as tab:
        tab.call("user-join", {"auth": {"token": member.token}})
        updates = channel_updates(tab, channel_id)
        post_message(owner, channel_id, f"<@M:{MOCK_MODEL_ID}|{MOCK_MODEL_ID}> Which day?")
        answer = wait_for(lambda: finished_model_message(member, channel_id), "the answer")
        wait_for(lambda: [u for u in updates if u["meta"].get("done")], "the finished update")

    assert answer["content"] == "Friday works for everyone."
    finished = [update for update in updates if update["meta"].get("done")]
    assert {update["content"] for update in finished} == {"Friday works for everyone."}
    assert updates[-1]["id"] == answer["id"]


def finished_model_message(actor, channel_id: str) -> dict | None:
    with actor.client() as client:
        listed = client.get(f"/api/v1/channels/{channel_id}/messages").json()
        threads = [
            reply_message
            for message in listed
            for reply_message in client.get(
                f"/api/v1/channels/{channel_id}/messages/{message['id']}/thread"
            ).json()
        ]
    done = [
        message
        for message in [*listed, *threads]
        if (message.get("meta") or {}).get("model_id") and message["meta"].get("done")
    ]
    return done[0] if done else None


# --- sub-agents, timers and automations ---------------------------------------------------------

SUBAGENTS = ("/api/v1/configs/subagents", "/api/v1/configs/subagents")
TIMER_PROMPT = "The tea has steeped."
SCHEDULE = "DTSTART:20990101T090000\nRRULE:FREQ=DAILY"


@pytest.fixture
def subagents_on(instance_admin, preserve, streaming):
    preserve(SUBAGENTS, on=streaming)
    with instance_admin.client() as client:
        current = client.get(SUBAGENTS[0]).json()
        client.post(SUBAGENTS[1], json={**current, "ENABLE_SUBAGENTS": True}).raise_for_status()


def test_a_subagent_reply_of_many_pieces_comes_back_whole_to_the_parent_chat(
    subagents_on, person, streaming
):
    upstream = streaming.upstream
    upstream.queue(
        reply.tool_call(
            "delegate_task", {"task": "check the ledger"}, match=reply.answering("hand over")
        ),
        reply.text(
            ["The ledger ", "is ", "balanced ", "for March."],
            match=reply.answering("check the ledger"),
        ),
        reply.text(["Handed ", "over."], match=reply.answering("hand over")),
    )
    with person.client() as client:
        _, message = ask(client, "hand over the books")

    follow_up = upstream.chat_requests()[-1]
    assert follow_up["messages"][-1]["role"] == "tool"
    assert follow_up["messages"][-1]["content"] == "The ledger is balanced for March."
    assert message["content"].endswith("Handed over.")


def timer_reply(client, chat_id: str) -> dict:
    """The reply to the timer's prompt once it is done."""

    def finished():
        messages = client.get(f"/api/v1/chats/{chat_id}").json()["chat"]["history"]["messages"]
        prompts = [
            m for m in messages.values() if m["role"] == "user" and m["content"] == TIMER_PROMPT
        ]
        if prompts and prompts[0].get("childrenIds"):
            answer = messages.get(prompts[0]["childrenIds"][0], {})
            return answer if answer.get("done") else None

    return wait_for(finished, "the timer's reply", timeout=45.0)


def test_a_timer_run_saves_its_reply_of_many_pieces_in_the_chat(subagents_on, person, streaming):
    upstream = streaming.upstream
    upstream.queue(
        reply.tool_call("timer", {"prompt": TIMER_PROMPT, "at": "3s"}),
        reply.text("Timer set."),
        reply.text(["Pour ", "it ", "now, ", "please."], match=reply.answering(TIMER_PROMPT)),
    )
    with person.client() as client:
        turn, message = ask(client, "tell me when the tea is ready")
        fired = timer_reply(client, turn.chat_id)

    assert message["content"].endswith("Timer set.")
    assert fired["content"] == "Pour it now, please."


@pytest.fixture
def scheduler(streaming):
    account = create_user(streaming, role="admin")
    yield account
    with account.client() as client:
        for automation in client.get("/api/v1/automations/list").json().get("items", []):
            client.delete(f"/api/v1/automations/{automation['id']}/delete")


def run_automation(client, automation: dict) -> dict:
    client.post(f"/api/v1/automations/{automation['id']}/run").raise_for_status()

    def recorded():
        runs = client.get(f"/api/v1/automations/{automation['id']}/runs").json()
        return runs[0] if runs else None

    return wait_for(recorded, "the automation's run")


def test_an_automation_run_streams_its_reply_into_a_new_chat(scheduler, streaming):
    prompt = f"Summarise the day, batch {uuid.uuid4().hex[:6]}."
    streaming.upstream.queue(
        reply.text(["All ", "quiet ", "on ", "the ", "front."], match=reply.answering(prompt))
    )
    with scheduler.client() as client:
        created = client.post(
            "/api/v1/automations/create",
            json={
                "name": "digest",
                "is_active": True,
                "data": {"prompt": prompt, "model_id": MOCK_MODEL_ID, "rrule": SCHEDULE},
            },
        )
        created.raise_for_status()
        run = run_automation(client, created.json())
        messages = client.get(f"/api/v1/chats/{run['chat_id']}").json()["chat"]["history"][
            "messages"
        ]

    assert run["status"] == "success", run
    answers = [m for m in messages.values() if m["role"] == "assistant"]
    assert [m["content"] for m in answers] == ["All quiet on the front."]


def test_an_automation_run_streams_its_reply_into_a_channel(scheduler, channels_on, streaming):
    prompt = f"Post the schedule, batch {uuid.uuid4().hex[:6]}."
    channel_id = group_channel(scheduler)
    streaming.upstream.queue(
        reply.text(["Standup ", "at ", "nine ", "sharp."], match=reply.answering(prompt))
    )
    with scheduler.client() as client:
        created = client.post(
            "/api/v1/automations/create",
            json={
                "name": "standup",
                "is_active": True,
                "data": {
                    "prompt": prompt,
                    "model_id": MOCK_MODEL_ID,
                    "rrule": SCHEDULE,
                    "target": {"type": "channel", "channel_id": channel_id},
                },
            },
        )
        created.raise_for_status()
        run = run_automation(client, created.json())
        answer = wait_for(lambda: finished_model_message(scheduler, channel_id), "the answer")

    assert run["status"] == "success", run
    assert answer["content"] == "Standup at nine sharp."


# --- an Anthropic client through the messages endpoint ------------------------------------------


def anthropic_stream(client: httpx.Client, model: str) -> list[dict]:
    """A streamed request to the Anthropic-compatible endpoint; returns its decoded events."""
    request = {
        "model": model,
        "max_tokens": 64,
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    }
    response = client.post("/api/v1/messages", json=request)
    assert response.status_code == 200, response.text
    return [
        json.loads(line.removeprefix("data:"))
        for line in response.text.splitlines()
        if line.startswith("data:")
    ]


def test_an_anthropic_client_gets_the_stream_and_the_outlet_sees_the_whole_reply(
    instance_admin, listener, raw
):
    listener.route("POST", "/audit", lambda _: (200, {}, b""))
    outlet = audit_source(f"{listener.base_url}/audit", "pass")
    raw.stream(
        sse(
            chunk({"role": "assistant", "content": "Snow "}),
            chunk({"content": "falls "}),
            chunk({"content": "softly."}),
            chunk({}, "stop"),
        )
    )
    with (
        installed_function(instance_admin, outlet, is_global=True),
        instance_admin.client() as client,
    ):
        events = anthropic_stream(client, RAW_MODEL_ID)
        wait_for(lambda: listener.requests_to("/audit"), "the outlet filter's audit")

    text = "".join(
        event["delta"]["text"]
        for event in events
        if event.get("type") == "content_block_delta" and event["delta"].get("type") == "text_delta"
    )
    assert text == "Snow falls softly."
    [audit] = listener.requests_to("/audit")
    assert audit.json()["content"] == "Snow falls softly."


# --- a background sub-agent reporting back ------------------------------------------------------


@pytest.fixture
def background_subagents_on(instance_admin, preserve, streaming):
    preserve(SUBAGENTS, on=streaming)
    with instance_admin.client() as client:
        current = client.get(SUBAGENTS[0]).json()
        switched = {**current, "ENABLE_SUBAGENTS": True, "SUBAGENTS_BACKGROUND_ENABLED": True}
        client.post(SUBAGENTS[1], json=switched).raise_for_status()


def carries_the_report(body: dict) -> bool:
    users = [entry for entry in body.get("messages", []) if entry.get("role") == "user"]
    return bool(users) and "[ASYNC SUBAGENT COMPLETE" in str(users[-1].get("content"))


def test_a_background_subagent_reports_its_whole_reply_and_the_chat_continues(
    background_subagents_on, person, streaming
):
    task = f"count the crates {uuid.uuid4().hex[:6]}"
    streaming.upstream.queue(
        reply.tool_call(
            "delegate_task",
            {"task": task, "background": True},
            match=reply.answering("send someone"),
        ),
        reply.text(["Forty ", "crates ", "in ", "bay three."], match=reply.answering(task)),
        reply.text(["Sent ", "a helper."], match=reply.answering("send someone")),
        reply.text(["The ", "count ", "is in."], match=carries_the_report),
    )
    with person.client() as client:
        turn, message = ask(client, "send someone to the warehouse")

        def continued():
            messages = client.get(f"/api/v1/chats/{turn.chat_id}").json()["chat"]["history"]
            reports = [
                entry
                for entry in messages["messages"].values()
                if (entry.get("meta") or {}).get("type") == "subagent"
            ]
            replies = [
                entry
                for entry in messages["messages"].values()
                if reports and entry.get("parentId") == reports[0]["id"] and entry.get("done")
            ]
            return (reports[0], replies[0]) if replies else None

        report, follow_up = wait_for(continued, "the reply after the report", timeout=60.0)

    assert message["content"].endswith("Sent a helper.")
    assert "Forty crates in bay three." in report["content"]
    assert follow_up["content"] == "The count is in."
    sent_back = [body for body in streaming.upstream.chat_requests() if carries_the_report(body)]
    assert "Forty crates in bay three." in str(sent_back[-1]["messages"][-1]["content"])
