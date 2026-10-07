"""Every socket delivery reaches the same tabs across instances with room channels on and off.

PR open-webui/open-webui#28818 (0.11.5): with `WEBSOCKET_MANAGER=redis`, an emit to one room
(every live chat event goes to the account's `user:{id}` room, a channel's to `channel:{id}`, a
note's to `note:{id}` and `doc_note:{id}`, an event call to the tab's own sid) is published on a
Redis channel of that room's own, so an instance with nobody in the room drops it by channel name
instead of decoding it. `WEBSOCKET_REDIS_ROOM_CHANNELS=false` puts every emit back on the shared
channel. Acknowledgements, disconnects and room changes for a tab on another instance stay on the
shared channel either way. An instance with the switch off listens on the shared channel only, so
room emits from an instance with the switch on do not reach it: the fleet has to run one mode.

Four instances share one database and one real Redis, two with the switch on (one by default) and
two with it off. Each delivery runs on both pairs: an account acts on one instance of the pair and
its tabs, or the tabs of the people it shares with, listen on the other (and on the same one), and
the tabs get the same events in both modes, while a tab that should get nothing gets nothing. That
covers a chat's live stream and a tool's events, a tool's question to the user's tab and its
answer, a direct connection answered by the browser, a pushed message event, a read receipt, an
admin change that drops the account's socket, a tab closing and coming back, a second tab closing
and a dozen accounts streaming at once; a channel's messages, edits, reactions, thread replies,
deletions and typing, a direct message, a member removed, a reader's grant revoked, a deleted
channel and a model answering in a channel; a note's live edits, cursors, joins and leaves, its
edits followed through `join-note` and a collaborator removed; a model shown as in use; and a
Redis that drops every subscription or restarts. A test client subscribed to every Redis channel
records which channel carried a chat's events. That an instance with nobody in the room skips the
message is only visible as CPU and is not pinned.

Finding, red on purpose: an instance whose Redis account may use only the pub/sub channels Open
WebUI needed before (`socketio` and `open-webui:*`) neither hears nor reaches the other instances
with the switch on, since Redis refuses the pattern subscription `socketio#*` and every publish on
a room channel (NOPERM) and the listener retries every second; with the switch off it works (fix
PR open-webui/open-webui#31617 open).

Also red on dev b859124f9, in some of its four cases per run: the direct connection answered by the
browser (open-webui/open-webui#31953). Since 24e30d1cb the socket router checks the tab's session
token again for every event the tab sends, so the reply's pieces can overtake each other while those
checks run and the stored reply comes back scrambled or empty. The same change leaves the events of
a tab opened after a Redis restart unanswered on an instance with the switch off, which turns the
last Redis loss case red on a Postgres and Redis run. Both pass on dev 015dbc861 and on b859124f9
with that check taken back out of the router.

Discriminates: passes on dev 176d31d1d apart from that finding (red with the switch on only). In a
backend copy: the room listener dropping room channel messages turns every switched-on case that
crosses instances red and leaves every switched-off one green; the stock manager deaf to other
instances does the reverse; a remote emit handed to every tab of the instance turns the "gets
nothing" assertions red in the mode of the manager it was put in only; the room manager ignoring
the shared channel turns the switched-on event call, dropped socket and subscription drop cases
red; the pattern subscription not renewed on a reconnect turns both Redis loss cases red when on;
a room's channel forgotten when one of several local members leaves turns the remaining tab, the
removed member, the revoked reader and the removed collaborator cases red when on; the switch
ignored (the room manager always) turns the shared channel and the mixed fleet cases red;
skipping `close_room` turns the deleted channel case red and usage not recorded the in-use case,
in both modes.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import threading
import time
from typing import Iterator

import pytest
import redis
import socketio

from harness import backends
from harness import upstream as reply
from harness.actors import Actor, admin_of, create_user
from harness.channel_chat import ADMIN_CONFIG, model_reply
from harness.channel_quotes import enable_channels, group_channel, post_message
from harness.chat import ask, send_message, wait_for_reply
from harness.direct_connection import answering, chunk_line, direct_model
from harness.instance import LaunchedInstance
from harness.python_tools import python_tool
from harness.socket_client import SocketSession, connected, note_text
from harness.upstream import MOCK_MODEL_ID

pytestmark = [
    pytest.mark.journey,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

SHARED_CHANNEL = "socketio"
PIECES = ["Wien ", "liegt ", "an der ", "Donau."]
ANSWER = "".join(PIECES)
# how long a tab keeps listening after the sender's own tab saw the stream end
GRACE_SECONDS = 2.0
ARRIVAL_TIMEOUT = 15.0
QUIET_PERIOD = 2.0
# the instance an account acts on and the one its other tab listens on, per switch value
PAIRS = {"on": ("on-a", "on-b"), "off": ("off-a", "off-b")}
MODES = list(PAIRS)


@pytest.fixture(scope="module")
def shared_redis() -> Iterator[backends.RedisProcess]:
    server = backends.RedisProcess()
    server.start()
    yield server
    server.close()


@pytest.fixture(scope="module")
def fleet(shared_redis, instance_with) -> dict[str, LaunchedInstance]:
    """Four instances on one database and one Redis, by name: on-a, on-b, off-a and off-b."""
    shared = {"REDIS_URL": shared_redis.url, "WEBSOCKET_MANAGER": "redis"}
    first = instance_with({**shared, "WEBUI_NAME": "on-a"})  # the switch at its default
    joined = {**shared, "DATABASE_URL": first.database_url}
    instances = {
        "on-a": first,
        "on-b": instance_with(
            {**joined, "WEBUI_NAME": "on-b", "WEBSOCKET_REDIS_ROOM_CHANNELS": "true"}
        ),
        "off-a": instance_with(
            {**joined, "WEBUI_NAME": "off-a", "WEBSOCKET_REDIS_ROOM_CHANNELS": "false"}
        ),
        "off-b": instance_with(
            {**joined, "WEBUI_NAME": "off-b", "WEBSOCKET_REDIS_ROOM_CHANNELS": "false"}
        ),
    }
    # an instance listens on Redis only once a socket has connected; event calls need that
    for instance in instances.values():
        with connected(admin_of(instance)):
            pass
    return instances


@pytest.fixture
def provider(fleet):
    """The provider every instance asks: the connection settings live in the shared database."""
    fleet["on-a"].upstream.reset()
    return fleet["on-a"].upstream


def pair(fleet: dict, mode: str) -> tuple[LaunchedInstance, LaunchedInstance]:
    near, far = PAIRS[mode]
    return fleet[near], fleet[far]


def signed_in_on(account: Actor, instance: LaunchedInstance) -> Actor:
    """The same account on another instance: they share the secret key and the database."""
    return dataclasses.replace(account, base_url=instance.base_url)


@contextlib.contextmanager
def watching_redis(url: str) -> Iterator[list[tuple[str, bytes]]]:
    """Every (channel, message) published while the block runs, filled in when it ends."""
    client = redis.Redis.from_url(url)
    subscription = client.pubsub(ignore_subscribe_messages=True)
    subscription.psubscribe("*")
    subscription.get_message(timeout=1.0)  # the psubscribe confirmation
    published: list[tuple[str, bytes]] = []
    try:
        yield published
    finally:
        while message := subscription.get_message(timeout=0.5):
            published.append((message["channel"].decode(), message["data"]))
        subscription.close()
        client.close()


@dataclasses.dataclass
class Streamed:
    chat_id: str
    user_id: str
    receiver_events: list[dict]
    sender_events: list[dict]
    published: list[tuple[str, bytes]]

    def channels(self) -> set[str]:
        """The Redis channels that carried an emit about this chat."""
        return {channel for channel, data in self.published if self.chat_id.encode() in data}


def stream_across(fleet: dict, shared_redis, sender: str, receiver: str) -> Streamed:
    """Send a chat to `sender` while the account has a tab open on each of the two instances."""
    sending, receiving = fleet[sender], fleet[receiver]
    account = create_user(sending)
    prompt = f"Where is Vienna? ({sender} to {receiver})"
    fleet["on-a"].upstream.queue(reply.text(PIECES, match=reply.answering(prompt)))
    with (
        watching_redis(shared_redis.url) as published,
        connected(signed_in_on(account, receiving)) as far_tab,
        connected(account) as near_tab,
        account.client() as client,
    ):
        turn = send_message(client, prompt)
        wait_for_reply(client, turn)
        near_tab.wait_for(turn.chat_id, "chat:completion", done=True)
        wait_for_end(far_tab, turn.chat_id)
    return Streamed(
        chat_id=turn.chat_id,
        user_id=account.id,
        receiver_events=far_tab.events_of(turn.chat_id),
        sender_events=near_tab.events_of(turn.chat_id),
        published=published,
    )


def wait_for_end(tab: SocketSession, chat_id: str) -> None:
    deadline = time.monotonic() + GRACE_SECONDS
    while time.monotonic() < deadline and not _finished(tab.events_of(chat_id)):
        time.sleep(0.05)


def _finished(events: list[dict]) -> bool:
    return any(_is_done(event) for event in events)


def _is_done(event: dict) -> bool:
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    return event.get("type") == "chat:completion" and data.get("done") is True


def _streamed_text(events: list[dict]) -> str:
    """The reply as its live text deltas spelled it out."""
    payloads = [event.get("data") for event in events]
    return "".join(
        payload["delta"]
        for payload in payloads
        if isinstance(payload, dict) and isinstance(payload.get("delta"), str)
    )


def _event_types(events: list[dict]) -> set[str]:
    return {event.get("type") for event in events}


@pytest.mark.parametrize(("sender", "receiver"), [("on-a", "on-b"), ("off-a", "off-b")])
def test_a_tab_on_another_instance_sees_the_chat_stream_live(fleet, shared_redis, sender, receiver):
    streamed = stream_across(fleet, shared_redis, sender, receiver)

    assert streamed.receiver_events, (
        f"a tab on {receiver} saw nothing of a chat streamed on {sender}; the sender's own tab "
        f"saw {sorted(_event_types(streamed.sender_events))}"
    )
    assert _streamed_text(streamed.receiver_events) == ANSWER
    assert _finished(streamed.receiver_events), "the tab never learned the reply had finished"
    assert "chat:title" in _event_types(streamed.receiver_events)
    assert streamed.receiver_events == streamed.sender_events, (
        "the tab on the other instance saw different events from the one on the sender"
    )


def test_room_emits_travel_on_the_room_channel_only(fleet, shared_redis):
    streamed = stream_across(fleet, shared_redis, "on-a", "on-b")

    channels = streamed.channels()
    assert channels, "nothing about the chat was published on Redis at all"
    assert SHARED_CHANNEL not in channels, (
        "the chat's events were published on the shared channel, which every instance decodes"
    )
    assert all(channel.endswith(f"user:{streamed.user_id}") for channel in channels), (
        f"the chat's events went to {sorted(channels)}, not the account's room channel"
    )


def test_with_room_channels_off_every_emit_travels_on_the_shared_channel(fleet, shared_redis):
    streamed = stream_across(fleet, shared_redis, "off-a", "off-b")

    assert streamed.channels() == {SHARED_CHANNEL}


def test_a_switched_on_instance_still_receives_from_a_switched_off_one(fleet, shared_redis):
    streamed = stream_across(fleet, shared_redis, "off-a", "on-b")

    assert _streamed_text(streamed.receiver_events) == ANSWER
    assert _finished(streamed.receiver_events)


def test_a_switched_off_instance_does_not_receive_room_emits(fleet, shared_redis):
    """The Changed note of 0.11.5: an instance on the old delivery misses an updated one's emits."""
    streamed = stream_across(fleet, shared_redis, "on-a", "off-b")

    assert _finished(streamed.sender_events), "the sender's own tab did not see the stream"
    assert streamed.receiver_events == [], (
        "a tab on a switched-off instance received room emits from a switched-on one; "
        "the release notes say it does not"
    )


ASK_THE_TAB = '''
import json


class Tools:
    async def ask_the_tab(self, __event_call__=None) -> str:
        """
        Ask the user's open tab for their name.
        """
        answer = await __event_call__({"type": "input", "data": {"title": "Your name?"}})
        return json.dumps(answer)
'''


def _answer_inputs(tab: SocketSession, value: str) -> list[dict]:
    """Answer every input the server asks this tab for with `value`; returns what it asked."""
    asked: list[dict] = []

    def on_events(message: dict):
        tab.events.append(message)
        if (message.get("data") or {}).get("type") != "input":
            return None
        asked.append(message)
        return {"value": value}

    tab.client.on("events", on_events)
    return asked


def _tool_results(provider) -> list[str]:
    return [
        entry["content"]
        for sent in provider.chat_requests()
        for entry in sent["messages"]
        if entry["role"] == "tool"
    ]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("tab_on", ["same instance", "other instance"])
def test_a_tool_asks_the_users_tab_and_gets_its_answer(fleet, provider, mode, tab_on):
    near, far = pair(fleet, mode)
    account = create_user(near)
    listening_on = far if tab_on == "other instance" else near
    prompt = f"who am I? ({mode}, {tab_on})"
    provider.queue(
        reply.tool_call("ask_the_tab", {}, match=reply.answering(prompt)),
        reply.text("Nice to meet you."),
    )
    with (
        python_tool(admin_of(near), ASK_THE_TAB, name="Ask the tab") as tool_id,
        connected(signed_in_on(account, listening_on)) as tab,
        account.client() as client,
    ):
        asked = _answer_inputs(tab, "Ada")
        options = {"tool_ids": [tool_id], "session_id": tab.client.get_sid()}
        _, message = ask(client, prompt, **options)

    assert len(asked) == 1, f"the tab was asked {len(asked)} times"
    assert _tool_results(provider) == [json.dumps({"value": "Ada"})], (
        "the tab's answer did not come back to the tool"
    )
    assert message["content"].endswith("Nice to meet you."), message


TELLS_THE_USER = '''
class Tools:
    async def look_it_up(self, __event_emitter__=None) -> str:
        """
        Look something up and keep the user posted.
        """
        emit = __event_emitter__
        await emit({"type": "status", "data": {"description": "Looking", "done": False}})
        await emit({"type": "notification", "data": {"type": "info", "content": "Found it"}})
        await emit(
            {
                "type": "citation",
                "data": {
                    "source": {"name": "Atlas"},
                    "document": ["Vienna is on the Danube."],
                    "metadata": [{"source": "Atlas"}],
                },
            }
        )
        await emit({"type": "status", "data": {"description": "Done", "done": True}})
        return "Vienna is on the Danube."
'''


def _of_type(events: list[dict], event_type: str) -> list[dict]:
    return [event for event in events if event.get("type") == event_type]


@pytest.mark.parametrize("mode", MODES)
def test_a_tools_events_reach_the_accounts_tabs_on_every_instance(fleet, provider, mode):
    near, far = pair(fleet, mode)
    account = create_user(near)
    prompt = f"where is Vienna? ({mode}, with a tool)"
    provider.queue(
        reply.tool_call("look_it_up", {}, match=reply.answering(prompt)),
        reply.text("On the Danube."),
    )
    with (
        python_tool(admin_of(near), TELLS_THE_USER, name="Look it up") as tool_id,
        connected(signed_in_on(account, far)) as far_tab,
        connected(account) as near_tab,
        account.client() as client,
    ):
        turn, _ = ask(client, prompt, tool_ids=[tool_id])
        near_tab.wait_for(turn.chat_id, "chat:completion", done=True)
        wait_for_end(far_tab, turn.chat_id)

    far_events, near_events = far_tab.events_of(turn.chat_id), near_tab.events_of(turn.chat_id)
    statuses = [event["data"]["description"] for event in _of_type(far_events, "status")]
    assert statuses[:1] == ["Looking"] and "Done" in statuses, statuses
    [notification] = _of_type(far_events, "notification")
    assert notification["data"]["content"] == "Found it"
    [citation] = _of_type(far_events, "citation")
    assert citation["data"]["source"] == {"name": "Atlas"}
    assert far_events == near_events, "the tab on the other instance saw different events"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("tab_on", ["same instance", "other instance"])
def test_a_direct_connection_is_answered_by_the_users_tab(fleet, mode, tab_on):
    near, far = pair(fleet, mode)
    account = create_user(near)
    listening_on = far if tab_on == "other instance" else near
    model = "my-own-model"
    with answering(signed_in_on(account, listening_on)) as tab, account.client() as client:
        tab.stream(chunk_line({"content": "from "}), chunk_line({"content": "my browser"}))
        turn = send_message(
            client,
            "hello",
            model=model,
            model_item=direct_model(model),
            session_id=tab.session_id,
        )
        message = wait_for_reply(client, turn)

    assert len(tab.requests) == 1, "the tab was not asked to run the completion"
    assert message["content"] == "from my browser", f"reply scrambled (#31953): {message}"


def _stored_chat(owner: Actor) -> tuple[str, str]:
    """A stored chat with one finished assistant message; its id and the message's id."""
    message_id = f"m-{time.monotonic_ns()}"
    message = {
        "id": message_id,
        "parentId": None,
        "childrenIds": [],
        "role": "assistant",
        "content": "the owner's reply",
        "done": True,
    }
    chat = {
        "title": "owned",
        "history": {"currentId": message_id, "messages": {message_id: message}},
    }
    with owner.client() as client:
        created = client.post("/api/v1/chats/new", json={"chat": chat})
    assert created.status_code == 200, created.text
    return created.json()["id"], message_id


def _quietly_waits_for(tab: SocketSession, chat_id: str, event_type: str) -> dict | None:
    try:
        return tab.wait_for(chat_id, event_type, timeout=QUIET_PERIOD)
    except AssertionError:
        return None


@pytest.mark.parametrize("mode", MODES)
def test_a_pushed_message_event_reaches_only_the_owners_tabs(fleet, mode):
    near, far = pair(fleet, mode)
    owner, stranger = create_user(near), create_user(near)
    chat_id, message_id = _stored_chat(owner)
    event = {"type": "status", "data": {"description": "pushed"}}
    with (
        connected(signed_in_on(owner, far)) as owners_far_tab,
        connected(owner) as owners_near_tab,
        connected(signed_in_on(stranger, far)) as strangers_tab,
        owner.client() as client,
    ):
        pushed = client.post(f"/api/v1/chats/{chat_id}/messages/{message_id}/event", json=event)
        assert pushed.status_code == 200, pushed.text
        far_status = owners_far_tab.wait_for(chat_id, "status", timeout=ARRIVAL_TIMEOUT)
        near_status = owners_near_tab.wait_for(chat_id, "status", timeout=ARRIVAL_TIMEOUT)
        leaked = _quietly_waits_for(strangers_tab, chat_id, "status")

    assert far_status == near_status
    assert far_status["data"] == {"description": "pushed"}
    assert leaked is None, "the owner's chat event reached another account's tab"


@pytest.mark.parametrize("mode", MODES)
def test_reading_a_chat_in_one_tab_updates_the_accounts_tabs_on_the_other(fleet, mode):
    near, far = pair(fleet, mode)
    account = create_user(near)
    chat_id, _ = _stored_chat(account)
    with (
        connected(signed_in_on(account, far)) as far_tab,
        connected(account) as near_tab,
    ):
        near_tab.call("events:chat", {"chat_id": chat_id, "data": {"type": "last_read_at"}})
        far_update = far_tab.wait_for(chat_id, "chat:list", timeout=ARRIVAL_TIMEOUT)
        near_update = near_tab.wait_for(chat_id, "chat:list", timeout=ARRIVAL_TIMEOUT)

    assert far_update == near_update
    assert far_update["data"]["chat_id"] == chat_id


@contextlib.contextmanager
def watched_socket(account: Actor) -> Iterator[threading.Event]:
    """The account's live socket, with an event that is set once the server drops it."""
    with connected(account) as socket:
        dropped = threading.Event()
        socket.client.on("disconnect", lambda *args: dropped.set())
        yield dropped


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("change", ["role change", "deletion"])
def test_an_admin_change_on_one_instance_drops_the_accounts_socket_on_the_other(
    fleet, mode, change
):
    near, far = pair(fleet, mode)
    account, bystander = create_user(near), create_user(near)
    with (
        watched_socket(signed_in_on(account, far)) as dropped,
        watched_socket(signed_in_on(bystander, far)) as bystander_dropped,
        near.client() as admin_client,
    ):
        if change == "role change":
            changed = admin_client.post(
                f"/api/v1/users/{account.id}/update", json={"role": "admin"}
            )
        else:
            changed = admin_client.delete(f"/api/v1/users/{account.id}")
        assert changed.status_code == 200, changed.text

        assert dropped.wait(ARRIVAL_TIMEOUT), f"the socket outlived the account's {change}"
        assert not bystander_dropped.wait(QUIET_PERIOD), "another account's socket was dropped"


def _stream_to(account: Actor, provider, prompt: str) -> str:
    """Stream a reply to `prompt` on the account's own instance; returns the chat id."""
    provider.queue(reply.text(PIECES, match=reply.answering(prompt)))
    with account.client() as client:
        turn = send_message(client, prompt)
        wait_for_reply(client, turn)
    return turn.chat_id


@pytest.mark.parametrize("mode", MODES)
def test_a_tab_hears_nothing_while_away_and_everything_once_back(fleet, provider, mode):
    near, far = pair(fleet, mode)
    account = create_user(near)
    elsewhere = signed_in_on(account, far)

    with connected(elsewhere) as first_visit:
        before = _stream_to(account, provider, f"before ({mode})")
        wait_for_end(first_visit, before)
    during = _stream_to(account, provider, f"while away ({mode})")
    with connected(elsewhere) as second_visit:
        after = _stream_to(account, provider, f"after ({mode})")
        wait_for_end(second_visit, after)

    assert _streamed_text(first_visit.events_of(before)) == ANSWER
    assert first_visit.events_of(during) == [], "a closed tab still received events"
    assert second_visit.events_of(during) == [], "a reconnected tab got events from before it"
    assert _streamed_text(second_visit.events_of(after)) == ANSWER, (
        "a tab that reconnected to the other instance no longer hears the account's chats"
    )


@pytest.mark.parametrize("mode", MODES)
def test_a_tab_keeps_hearing_the_account_after_its_other_tab_there_closes(fleet, provider, mode):
    near, far = pair(fleet, mode)
    account = create_user(near)
    elsewhere = signed_in_on(account, far)

    with connected(elsewhere) as staying_tab:
        with connected(elsewhere):
            pass
        chat_id = _stream_to(account, provider, f"after one tab left ({mode})")
        wait_for_end(staying_tab, chat_id)

    assert _streamed_text(staying_tab.events_of(chat_id)) == ANSWER, (
        "the account's remaining tab on the other instance stopped hearing its chats once "
        "another of its tabs there closed"
    )


ACCOUNTS_AT_ONCE = 12


@pytest.mark.parametrize("mode", MODES)
def test_many_accounts_streaming_at_once_each_hear_only_their_own_chat(fleet, provider, mode):
    near, far = pair(fleet, mode)
    accounts = [create_user(near) for _ in range(ACCOUNTS_AT_ONCE)]
    chats: dict[str, str] = {}
    with contextlib.ExitStack() as stack:
        tabs = {
            account.id: stack.enter_context(connected(signed_in_on(account, far)))
            for account in accounts
        }
        clients = {account.id: stack.enter_context(account.client()) for account in accounts}
        turns = {}
        for index, account in enumerate(accounts):
            prompt = f"crowd {mode} #{index}."
            provider.queue(
                reply.text([f"only for {index} ", "and nobody else"], match=reply.answering(prompt))
            )
            turns[account.id] = send_message(clients[account.id], prompt)
        for account in accounts:
            turn = turns[account.id]
            wait_for_reply(clients[account.id], turn)
            chats[account.id] = turn.chat_id
            wait_for_end(tabs[account.id], turn.chat_id)

    for index, account in enumerate(accounts):
        tab = tabs[account.id]
        heard_chats = {entry.get("chat_id") for entry in tab.events}
        assert heard_chats == {chats[account.id]}, (
            f"account {index}'s tab heard chats {heard_chats}, not only its own"
        )
        assert (
            _streamed_text(tab.events_of(chats[account.id])) == f"only for {index} and nobody else"
        )


# the pub/sub channels Open WebUI uses with room channels off: python-socketio's and its own
LOCKED_USER = "owui-locked"
LOCKED_RIGHTS = ["on", ">locked-password", "~*", "+@all", "&socketio", "&open-webui:*"]


@pytest.fixture(scope="module")
def locked(fleet, shared_redis, instance_with) -> dict[str, LaunchedInstance]:
    """Per switch value, one more instance whose Redis account may use only the channels above."""
    client = redis.Redis(port=shared_redis.port)
    try:
        client.execute_command("ACL", "SETUSER", LOCKED_USER, *LOCKED_RIGHTS)
    finally:
        client.close()
    url = f"redis://{LOCKED_USER}:locked-password@127.0.0.1:{shared_redis.port}/0"
    shared = {
        "REDIS_URL": url,
        "WEBSOCKET_MANAGER": "redis",
        "DATABASE_URL": fleet["on-a"].database_url,
    }
    return {
        mode: instance_with(
            {**shared, "WEBUI_NAME": f"{mode}-locked", "WEBSOCKET_REDIS_ROOM_CHANNELS": switch}
        )
        for mode, switch in (("on", "true"), ("off", "false"))
    }


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("direction", ["to the limited instance", "from the limited instance"])
def test_an_instance_on_a_redis_account_limited_to_its_channels_is_heard(
    fleet, locked, provider, mode, direction
):
    """Finding: with room channels on, such an instance hears no other instance and none hears it.

    Redis matches a PSUBSCRIBE pattern literally against the account's channel rights, so
    `socketio#*` is refused (NOPERM) and the instance's listener retries forever, and a PUBLISH
    on `socketio#/#<room>` is refused as well. With the switch off the same account works.
    """
    near, _ = pair(fleet, mode)
    limited = locked[mode]
    acting_on, listening_on = (near, limited) if direction.startswith("to") else (limited, near)
    account = create_user(acting_on)

    with connected(signed_in_on(account, listening_on)) as tab:
        chat_id = _stream_to(account, provider, f"through a limited account ({mode}, {direction})")
        wait_for_end(tab, chat_id)

    assert _streamed_text(tab.events_of(chat_id)) == ANSWER, (
        f"a chat streamed {direction} never reached the account's tab: its Redis account may use "
        "the channels socketio and open-webui:*, which is all room channels off needs, but room "
        "channels on also need the pattern socketio#* (refused with NOPERM)"
    )


DOCUMENT_EVENTS = ("ydoc:awareness:update", "ydoc:user:joined", "ydoc:user:left", "events:note")


def _listening(tab: SocketSession) -> dict[str, list]:
    """Everything the note events send to the tab, by event name, filled in as it arrives."""
    seen = {event: [] for event in DOCUMENT_EVENTS}
    for event, arrived in seen.items():
        tab.client.on(event, arrived.append)
    return seen


def _heard_nothing(tab: SocketSession, seen: dict[str, list]) -> bool:
    return not (tab.document_updates or tab.document_states or any(seen.values()))


def _wait_until(condition, timeout: float = ARRIVAL_TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return condition()


def _grant(account: Actor, permission: str) -> dict:
    return {"principal_type": "user", "principal_id": account.id, "permission": permission}


def _write_grants(account: Actor) -> list[dict]:
    """What the share dialog stores for an editor: a read grant and a write grant."""
    return [_grant(account, "read"), _grant(account, "write")]


def _shared_note(owner: Actor, *grants: dict) -> str:
    note = {"title": f"shared {time.monotonic_ns()}", "data": {"content": {"md": ""}}}
    with owner.client() as client:
        created = client.post("/api/v1/notes/create", json=note)
        assert created.status_code == 200, created.text
        note_id = created.json()["id"]
        shared = client.post(
            f"/api/v1/notes/{note_id}/access/update", json={"access_grants": list(grants)}
        )
    assert shared.status_code == 200, shared.text
    return note_id


def _follow_note(tab: SocketSession, account: Actor, note_id: str) -> None:
    """Open the note the way the note page does: follow its edits and its live document."""
    tab.call("join-note", {"auth": {"token": account.token}, "note_id": note_id})
    tab.join_note(note_id)


def _retitle(owner: Actor, note_id: str, title: str, markdown: str | None = None) -> None:
    body = {"title": title}
    if markdown is not None:
        body["data"] = {"content": {"md": markdown}}
    with owner.client() as client:
        updated = client.post(f"/api/v1/notes/{note_id}/update", json=body)
    assert updated.status_code == 200, updated.text


def _saved_markdown(owner: Actor, note_id: str, expected: str) -> str | None:
    deadline = time.monotonic() + ARRIVAL_TIMEOUT
    with owner.client() as client:
        while True:
            note = client.get(f"/api/v1/notes/{note_id}").json()
            markdown = ((note.get("data") or {}).get("content") or {}).get("md")
            if markdown == expected or time.monotonic() > deadline:
                return markdown
            time.sleep(0.2)


@pytest.mark.parametrize("mode", MODES)
def test_a_note_edited_on_one_instance_reaches_the_tabs_that_joined_it_on_the_other(fleet, mode):
    near, far = pair(fleet, mode)
    owner, writer, reader, stranger = (create_user(near) for _ in range(4))
    note_id = _shared_note(owner, *_write_grants(writer), _grant(reader, "read"))
    document_id = f"note:{note_id}"
    with (
        connected(signed_in_on(writer, far)) as writers_tab,
        connected(signed_in_on(writer, far)) as late_tab,
        connected(signed_in_on(reader, far)) as idle_readers_tab,
        connected(signed_in_on(stranger, far)) as strangers_tab,
        connected(owner) as owners_second_tab,
        connected(owner) as owners_tab,
    ):
        writer_heard, late_heard = _listening(writers_tab), _listening(late_tab)
        idle_heard, stranger_heard = _listening(idle_readers_tab), _listening(strangers_tab)
        second_heard = _listening(owners_second_tab)
        writers_tab.join_note(note_id)
        strangers_tab.join_note(note_id)
        owners_second_tab.join_note(note_id)
        owners_tab.call(
            "ydoc:document:join",
            {"document_id": document_id, "user_id": "owners-tab", "user_name": "Owner"},
        )
        owners_tab.edit_note(note_id, "edited on near")
        owners_tab.call("ydoc:awareness:update", {"document_id": document_id, "update": [1, 2, 3]})
        late_tab.join_note(note_id)
        late_document = late_tab.note_state(note_id)
        owners_tab.call("ydoc:document:leave", {"document_id": document_id})

        edit = writers_tab.note_update(note_id)
        assert _wait_until(lambda: writer_heard["ydoc:awareness:update"]), "no cursor update"
        assert _wait_until(lambda: writer_heard["ydoc:user:left"]), "no leave event"
        time.sleep(QUIET_PERIOD)
        saved = _saved_markdown(owner, note_id, "edited on near")

    assert edit["document_id"] == document_id
    assert note_text(edit["update"]) == "<paragraph>edited on near</paragraph>"
    [cursor] = writer_heard["ydoc:awareness:update"]
    assert (cursor["document_id"], cursor["user_id"], cursor["update"]) == (
        document_id,
        owner.id,
        [1, 2, 3],
    )
    joined = {event["user_id"]: event for event in writer_heard["ydoc:user:joined"]}
    assert joined["owners-tab"]["user_name"] == "Owner"
    [left] = writer_heard["ydoc:user:left"]
    assert (left["document_id"], left["user_id"]) == (document_id, owner.id)
    assert late_document == "<paragraph>edited on near</paragraph>"
    assert late_heard["ydoc:user:left"] == [left], "the late tab missed the leave event"
    assert owners_second_tab.document_updates == writers_tab.document_updates
    assert second_heard["ydoc:awareness:update"] == [cursor]
    assert _heard_nothing(idle_readers_tab, idle_heard), (
        "a tab that never joined the document heard it"
    )
    assert _heard_nothing(strangers_tab, stranger_heard), "an account without access heard it"
    assert saved == "edited on near"


@pytest.mark.parametrize("mode", MODES)
def test_a_notes_http_edit_reaches_the_tabs_that_followed_it_on_the_other_instance(fleet, mode):
    near, far = pair(fleet, mode)
    owner, writer, stranger = (create_user(near) for _ in range(3))
    note_id = _shared_note(owner, *_write_grants(writer))
    with (
        connected(signed_in_on(writer, far)) as writers_tab,
        connected(signed_in_on(stranger, far)) as strangers_tab,
        connected(owner) as owners_tab,
    ):
        writer_heard, stranger_heard = _listening(writers_tab), _listening(strangers_tab)
        owner_heard = _listening(owners_tab)
        _follow_note(writers_tab, writer, note_id)
        _follow_note(strangers_tab, stranger, note_id)
        _follow_note(owners_tab, owner, note_id)
        _retitle(owner, note_id, "Fahrplan", markdown="Zug nach Graz um 8:15")
        assert _wait_until(lambda: writer_heard["events:note"]), "the edit never arrived"
        assert _wait_until(lambda: owner_heard["events:note"])
        time.sleep(QUIET_PERIOD)

    [seen] = writer_heard["events:note"]
    assert (seen["id"], seen["title"]) == (note_id, "Fahrplan")
    assert seen["data"] == {"content": {"md": "Zug nach Graz um 8:15"}}
    assert owner_heard["events:note"] == [seen]
    assert _heard_nothing(strangers_tab, stranger_heard), "an account without access heard it"


@pytest.mark.parametrize("mode", MODES)
def test_a_collaborator_removed_on_one_instance_stops_hearing_the_note_on_the_other(fleet, mode):
    near, far = pair(fleet, mode)
    owner, removed, kept = (create_user(near) for _ in range(3))
    note_id = _shared_note(owner, *_write_grants(removed), *_write_grants(kept))
    with (
        connected(signed_in_on(removed, far)) as removed_tab,
        connected(signed_in_on(kept, far)) as kept_tab,
        connected(owner) as owners_tab,
    ):
        removed_heard, kept_heard = _listening(removed_tab), _listening(kept_tab)
        for tab, account in [(removed_tab, removed), (kept_tab, kept), (owners_tab, owner)]:
            _follow_note(tab, account, note_id)
        _retitle(owner, note_id, "while shared")
        assert _wait_until(lambda: removed_heard["events:note"] and kept_heard["events:note"])

        with owner.client() as client:
            revoked = client.post(
                f"/api/v1/notes/{note_id}/access/update",
                json={"access_grants": [*_write_grants(kept)]},
            )
        assert revoked.status_code == 200, revoked.text
        _retitle(owner, note_id, "after the removal")
        owners_tab.edit_note(note_id, "after the removal")
        assert _wait_until(lambda: kept_tab.document_updates), "the kept tab lost the edit"
        assert _wait_until(lambda: len(kept_heard["events:note"]) == 2)
        time.sleep(QUIET_PERIOD)

    titles = [event["title"] for event in kept_heard["events:note"]]
    assert titles == ["while shared", "after the removal"]
    assert [event["title"] for event in removed_heard["events:note"]] == ["while shared"], (
        "a removed collaborator's tab on the other instance still gets the note's edits"
    )
    assert removed_tab.document_updates == [], "a removed collaborator still gets the live edits"


@pytest.mark.parametrize("mode", MODES)
def test_a_model_a_tab_uses_on_the_other_instance_shows_as_in_use_until_it_closes(fleet, mode):
    near, far = pair(fleet, mode)
    account = create_user(near)
    model = f"llama-{mode}-{time.monotonic_ns()}"
    with connected(signed_in_on(account, far)) as far_tab, account.client() as client:
        far_tab.call("usage", {"model": model})
        in_use = _wait_until(lambda: model in client.get("/api/usage").json()["model_ids"])

    with account.client() as client:
        released = _wait_until(lambda: model not in client.get("/api/usage").json()["model_ids"])

    assert in_use, "the model a tab on the other instance streams from is not shown as in use"
    assert released, "the model stayed in use after its tab disconnected"


@pytest.fixture(scope="module")
def channels_on(fleet) -> Iterator[None]:
    """Channels enabled for the whole fleet (one shared database); the old setting comes back."""
    admin = admin_of(fleet["on-a"])
    with admin.client() as client:
        previous = client.get(ADMIN_CONFIG)
        previous.raise_for_status()
    enable_channels(admin)
    yield
    with admin.client() as client:
        client.post(ADMIN_CONFIG, json=previous.json()).raise_for_status()


@contextlib.contextmanager
def joined(account: Actor) -> Iterator[SocketSession]:
    """A tab that has joined the rooms of its account's channels, as the web client does."""
    with connected(account) as tab:
        _join_channels(tab, account)
        yield tab


def _recording(tab: SocketSession) -> list[dict]:
    """Every `events:channel` message the tab receives from now on."""
    received: list[dict] = []
    tab.client.on("events:channel", received.append)
    return received


def _types(received: list[dict], channel_id: str) -> list[str]:
    return [
        event["data"]["type"] for event in list(received) if event.get("channel_id") == channel_id
    ]


def _wait_for_types(received: list[dict], channel_id: str, count: int) -> list[str]:
    deadline = time.monotonic() + ARRIVAL_TIMEOUT
    while time.monotonic() < deadline and len(_types(received, channel_id)) < count:
        time.sleep(0.05)
    return _types(received, channel_id)


def _join_channels(tab: SocketSession, account: Actor) -> None:
    """What the web client sends once it learns of a new channel."""
    tab.call("join-channels", {"auth": {"token": account.token}})


def _wait_for_announcement(received: list[dict]) -> None:
    """The `channel:created` event a new channel sends to its members' user rooms."""
    deadline = time.monotonic() + ARRIVAL_TIMEOUT
    while time.monotonic() < deadline and not received:
        time.sleep(0.05)
    assert received, "the recipient's tab was never told about the new direct message"


def _post_via(actor: Actor, channel_id: str, path: str, method: str = "POST", **body):
    with actor.client() as client:
        answered = client.request(
            method, f"/api/v1/channels/{channel_id}/{path}", json=body or None
        )
    assert answered.status_code == 200, answered.text
    return answered.json()


@pytest.mark.parametrize("mode", MODES)
def test_every_change_to_a_channel_message_reaches_the_members_tabs(fleet, channels_on, mode):
    near, far = pair(fleet, mode)
    author, member, stranger = create_user(near), create_user(near), create_user(near)
    channel_id = group_channel(author, member)
    with (
        joined(signed_in_on(member, far)) as far_tab,
        joined(member) as near_tab,
        joined(signed_in_on(stranger, far)) as strangers_tab,
    ):
        far_events, near_events = _recording(far_tab), _recording(near_tab)
        strangers_events = _recording(strangers_tab)
        message_id = post_message(author, channel_id, "lunch at noon?")
        _post_via(
            author, channel_id, f"messages/{message_id}/update", content="lunch at half past noon?"
        )
        for action in ("add", "remove"):
            _post_via(author, channel_id, f"messages/{message_id}/reactions/{action}", name="heart")
        post_message(member, channel_id, "works for me", parent_id=message_id)
        _post_via(author, channel_id, f"messages/{message_id}/delete", method="DELETE")
        expected = [
            "message",
            "message:update",
            "message:reaction:add",
            "message:reaction:remove",
            "message",
            "message:reply",
            "message:delete",
        ]
        far_types = _wait_for_types(far_events, channel_id, len(expected))
        near_types = _wait_for_types(near_events, channel_id, len(expected))
        time.sleep(QUIET_PERIOD)

    assert far_types == expected
    assert near_types == expected
    assert [event["data"]["data"]["content"] for event in far_events[:2]] == [
        "lunch at noon?",
        "lunch at half past noon?",
    ]
    assert far_events == near_events, "the tab on the other instance saw different events"
    assert strangers_events == [], "someone outside the channel got its events"


@pytest.mark.parametrize("mode", MODES)
def test_a_members_typing_reaches_the_other_members_tabs(fleet, channels_on, mode):
    near, far = pair(fleet, mode)
    typist, member, stranger = create_user(near), create_user(near), create_user(near)
    channel_id = group_channel(typist, member)
    typing = {"channel_id": channel_id, "data": {"type": "typing", "data": {"typing": True}}}
    with (
        joined(signed_in_on(member, far)) as far_tab,
        joined(signed_in_on(stranger, far)) as strangers_tab,
        joined(typist) as typists_tab,
    ):
        far_events, strangers_events = _recording(far_tab), _recording(strangers_tab)
        typists_tab.call("events:channel", typing)
        heard = _wait_for_types(far_events, channel_id, 1)
        time.sleep(QUIET_PERIOD)

    assert heard == ["typing"]
    assert far_events[0]["user"]["id"] == typist.id
    assert far_events[0]["data"]["data"] == {"typing": True}
    assert strangers_events == [], "someone outside the channel saw the typing"


@pytest.mark.parametrize("mode", MODES)
def test_a_direct_message_reaches_the_recipients_tab_once_it_joins_the_channel(
    fleet, channels_on, mode
):
    near, far = pair(fleet, mode)
    sender, recipient, stranger = create_user(near), create_user(near), create_user(near)
    with (
        connected(signed_in_on(recipient, far)) as recipients_tab,
        joined(signed_in_on(stranger, far)) as strangers_tab,
        joined(sender) as senders_tab,
    ):
        recipients_events, strangers_events = _recording(recipients_tab), _recording(strangers_tab)
        senders_events = _recording(senders_tab)
        with sender.client() as client:
            opened = client.get(f"/api/v1/channels/users/{recipient.id}")
        assert opened.status_code == 200, opened.text
        channel_id = opened.json()["id"]
        _wait_for_announcement(recipients_events)
        post_message(sender, channel_id, "before you joined")
        time.sleep(QUIET_PERIOD)
        before_joining = _types(recipients_events, channel_id)
        _join_channels(recipients_tab, recipient)
        post_message(sender, channel_id, "after you joined")
        after_joining = _wait_for_types(recipients_events, channel_id, 1)
        _wait_for_types(senders_events, channel_id, 2)
        time.sleep(QUIET_PERIOD)

    assert [event["data"]["type"] for event in recipients_events[:1]] == ["channel:created"]
    assert before_joining == [], "a tab that had not joined the channel got its messages"
    assert after_joining == ["message"]
    assert recipients_events[-1]["data"]["data"]["content"] == "after you joined"
    assert _types(senders_events, channel_id) == ["message", "message"]
    assert strangers_events == [], "a third account's tab heard about the direct message"


@pytest.mark.parametrize("mode", MODES)
def test_a_member_removed_on_one_instance_stops_getting_messages_on_the_other(
    fleet, channels_on, mode
):
    near, far = pair(fleet, mode)
    owner, leaving, staying = create_user(near), create_user(near), create_user(near)
    channel_id = group_channel(owner, leaving, staying)
    typing = {"channel_id": channel_id, "data": {"type": "typing", "data": {"typing": True}}}
    with (
        joined(signed_in_on(leaving, far)) as leaving_tab,
        joined(signed_in_on(staying, far)) as staying_tab,
    ):
        leaving_events, staying_events = _recording(leaving_tab), _recording(staying_tab)
        post_message(owner, channel_id, "everyone is in")
        assert _wait_for_types(leaving_events, channel_id, 1) == ["message"]
        assert _wait_for_types(staying_events, channel_id, 1) == ["message"]

        _post_via(owner, channel_id, "update/members/remove", user_ids=[leaving.id])
        post_message(owner, channel_id, "after the removal")
        after = _wait_for_types(staying_events, channel_id, 2)
        leaving_tab.call("events:channel", typing)  # ignored while it is out of the room
        time.sleep(QUIET_PERIOD)

    assert after == ["message", "message"]
    assert _types(leaving_events, channel_id) == ["message"], "a removed member still got messages"
    assert _types(staying_events, channel_id) == ["message", "message"], (
        "the removed member's own typing reached the channel"
    )


@pytest.mark.parametrize("mode", MODES)
def test_a_reader_whose_grant_is_revoked_on_one_instance_stops_getting_messages_on_the_other(
    fleet, channels_on, mode
):
    near, far = pair(fleet, mode)
    admin = admin_of(near)
    revoked, kept = create_user(near), create_user(near)
    with admin.client() as client:
        created = client.post(
            "/api/v1/channels/create",
            json={
                "name": f"news-{time.monotonic_ns()}",
                "type": None,
                "access_grants": [_grant(revoked, "read"), _grant(kept, "read")],
            },
        )
    assert created.status_code == 200, created.text
    channel_id, name = created.json()["id"], created.json()["name"]
    with (
        joined(signed_in_on(revoked, far)) as revoked_tab,
        joined(signed_in_on(kept, far)) as kept_tab,
    ):
        revoked_events, kept_events = _recording(revoked_tab), _recording(kept_tab)
        post_message(admin, channel_id, "before the change")
        assert _wait_for_types(revoked_events, channel_id, 1) == ["message"]
        assert _wait_for_types(kept_events, channel_id, 1) == ["message"]

        _post_via(admin, channel_id, "update", name=name, access_grants=[_grant(kept, "read")])
        post_message(admin, channel_id, "after the change")
        after = _wait_for_types(kept_events, channel_id, 2)
        time.sleep(QUIET_PERIOD)

    assert after == ["message", "message"]
    assert _types(revoked_events, channel_id) == ["message"], (
        "a reader whose grant was taken away still got the channel's messages"
    )


@pytest.mark.parametrize("mode", MODES)
def test_a_channel_deleted_on_one_instance_leaves_its_members_tabs_on_the_other(
    fleet, channels_on, mode
):
    near, far = pair(fleet, mode)
    owner, member, listener = create_user(near), create_user(near), create_user(near)
    channel_id = group_channel(owner, member, listener)
    typing = {"channel_id": channel_id, "data": {"type": "typing", "data": {"typing": True}}}
    with (
        joined(signed_in_on(member, far)) as member_tab,
        joined(signed_in_on(listener, far)) as listener_tab,
    ):
        member_events, listener_events = _recording(member_tab), _recording(listener_tab)
        post_message(owner, channel_id, "still here")
        assert _wait_for_types(member_events, channel_id, 1) == ["message"]
        assert _wait_for_types(listener_events, channel_id, 1) == ["message"]

        _post_via(owner, channel_id, "delete", method="DELETE")
        time.sleep(QUIET_PERIOD)
        member_tab.call("events:channel", typing)  # ignored once the room is closed here
        time.sleep(QUIET_PERIOD)

    assert set(_types(member_events, channel_id)) <= {"message", "channel:deleted"}
    assert _types(listener_events, channel_id) == _types(member_events, channel_id)
    assert "typing" not in _types(listener_events, channel_id), (
        "a member's tab on the other instance was still in the room of the deleted channel"
    )


@pytest.mark.parametrize("mode", MODES)
def test_a_models_streamed_answer_in_a_channel_reaches_the_members_tab(
    fleet, provider, channels_on, mode
):
    near, far = pair(fleet, mode)
    owner, member, stranger = create_user(near), create_user(near), create_user(near)
    channel_id = group_channel(owner, member)
    question = f"where is Vienna? ({mode}, in a channel)"
    provider.queue(reply.text(PIECES, match=reply.answering(question)))
    with (
        joined(signed_in_on(member, far)) as far_tab,
        joined(signed_in_on(stranger, far)) as strangers_tab,
        owner.client() as client,
    ):
        far_events, strangers_events = _recording(far_tab), _recording(strangers_tab)
        post_message(owner, channel_id, f"<@M:{MOCK_MODEL_ID}|{MOCK_MODEL_ID}> {question}")
        answer = model_reply(client, channel_id)
        deadline = time.monotonic() + ARRIVAL_TIMEOUT
        while time.monotonic() < deadline and not _model_finished(far_events):
            time.sleep(0.05)
        time.sleep(QUIET_PERIOD)

    assert answer["content"] == ANSWER
    updates = [
        event["data"]["data"]
        for event in far_events
        if event["data"]["type"] == "message:update"
        and (event["data"]["data"].get("meta") or {}).get("model_id")
    ]
    assert updates, f"the member's tab saw no streamed update: {_types(far_events, channel_id)}"
    assert updates[-1]["content"] == ANSWER
    assert (updates[-1]["meta"] or {}).get("done") is True
    assert strangers_events == [], "someone outside the channel saw the model's answer"


def _model_finished(received: list[dict]) -> bool:
    return any(
        event["data"]["type"] == "message:update"
        and ((event["data"]["data"].get("meta") or {}).get("done") is True)
        for event in list(received)
    )


def _listeners(url: str) -> tuple[int, int]:
    """The fleet's pub/sub clients on a channel, and those on a pattern (the room channels).

    Only the default account's: the limited instances' account does not outlive a restart.
    """
    client = redis.Redis.from_url(url, decode_responses=True)
    try:
        entries = [
            entry for entry in client.client_list(_type="pubsub") if entry["user"] == "default"
        ]
    except redis.exceptions.RedisError:
        return 0, 0
    finally:
        client.close()
    on_a_channel = sum(1 for entry in entries if entry["sub"] != "0")
    on_a_pattern = sum(1 for entry in entries if entry["psub"] != "0")
    return on_a_channel, on_a_pattern


def _subscriptions(url: str) -> list[tuple[int, int]]:
    """Each pub/sub client's count of channels and of patterns."""
    client = redis.Redis.from_url(url, decode_responses=True)
    try:
        entries = client.client_list(_type="pubsub")
        return sorted((int(entry["sub"]), int(entry["psub"])) for entry in entries)
    finally:
        client.close()


def _listen_again(url: str, before: tuple[int, int]) -> bool:
    return all(now >= then for now, then in zip(_listeners(url), before))


# Open WebUI's own Redis clients fail the first command on each connection Redis closed, and a
# listener waits a second before it subscribes again, so a person's first try after the loss may
# be lost in either mode; the tab must be heard again within a few tries
ATTEMPTS_AFTER_A_LOSS = 10


def _heard_again(account: Actor, tab: SocketSession, provider, label: str) -> bool:
    for attempt in range(ATTEMPTS_AFTER_A_LOSS):
        try:
            chat_id = _stream_to(account, provider, f"after the {label} (try {attempt})")
        except AssertionError:
            time.sleep(1)
            continue
        wait_for_end(tab, chat_id)
        if _streamed_text(tab.events_of(chat_id)) == ANSWER:
            return True
    return False


def _answered_again(account: Actor, tool_id: str, tab: SocketSession, provider, label: str) -> bool:
    for attempt in range(ATTEMPTS_AFTER_A_LOSS):
        prompt = f"who am I, after the {label}? (try {attempt})"
        provider.queue(
            reply.tool_call("ask_the_tab", {}, match=reply.answering(prompt)),
            reply.text("Welcome back."),
        )
        try:
            with account.client() as client:
                ask(client, prompt, tool_ids=[tool_id], session_id=tab.client.get_sid())
        except AssertionError:
            time.sleep(1)
            continue
        if json.dumps({"value": "Ada"}) in _tool_results(provider):
            return True
    return False


@contextlib.contextmanager
def registered_tab(account: Actor) -> Iterator[SocketSession]:
    """A tab the server really put in the account's room: a connect right after an earlier loss
    can fail on one of those broken connections and leave the tab outside it."""
    chat_id, _ = _stored_chat(account)
    for _ in range(ATTEMPTS_AFTER_A_LOSS):
        with connected(account) as tab:
            try:
                tab.call("events:chat", {"chat_id": chat_id, "data": {"type": "last_read_at"}})
            except socketio.exceptions.TimeoutError:
                continue
            if _quietly_waits_for(tab, chat_id, "chat:list") is not None:
                yield tab
                return
    raise AssertionError("no tab of the account got into its room")


def _lose_redis(server: backends.RedisProcess, loss: str) -> None:
    if loss == "restart":
        server.restart(outage=3.0)
        return
    # every pub/sub connection cut, as a network blip or a proxy's idle timeout does
    client = redis.Redis(port=server.port)
    try:
        client.client_kill_filter(_type="pubsub", skipme=True)
    finally:
        client.close()


# the restart last: the instances' pooled connections break with it, in both modes
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("loss", ["subscription drop", "restart"])
def test_tabs_open_through_a_redis_loss_hear_the_account_again(
    fleet, shared_redis, provider, loss, mode
):
    near, far = pair(fleet, mode)
    account = create_user(near)
    with (
        python_tool(admin_of(near), ASK_THE_TAB, name="Ask the tab") as tool_id,
        registered_tab(signed_in_on(account, far)) as tab,
        registered_tab(account),
    ):
        _answer_inputs(tab, "Ada")
        # an instance listens once its first socket connects, so both of the pair listen by now
        listening_before = _listeners(shared_redis.url)
        _lose_redis(shared_redis, loss)
        assert backends.wait_until(
            lambda: _listen_again(shared_redis.url, listening_before), timeout=60
        ), (
            f"after the Redis {loss} the instances did not all listen again: "
            f"(on a channel, on a pattern) went from {listening_before} "
            f"to {_listeners(shared_redis.url)}"
        )
        heard = _heard_again(account, tab, provider, loss)
        answered = _answered_again(account, tool_id, tab, provider, loss)

    assert heard, f"a tab open through the Redis {loss} no longer hears the other instance"
    assert answered, (
        f"after the Redis {loss} the tab's answer to a tool no longer comes back across instances; "
        f"pub/sub clients (channels, patterns): {_subscriptions(shared_redis.url)}"
    )
