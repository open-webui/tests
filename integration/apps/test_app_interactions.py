"""Journey: mentioning an app, pressing its buttons and getting work it finishes later.

A mention is handed to the app with the actor and a signed execution token; the app only
confirms it received it (`{"received": true}`) and says whatever it has to say as an ordinary
channel message, as late as it likes. When a mention cannot be delivered, Open WebUI posts why
as the app's own "App error:" reply to it, for everyone in the channel; only when the app cannot
post there (inactive, outside the channel, no write scope) is the asker alone told over their
socket (`channel:app`). The reason is the app's own error text, or Open WebUI's when the app was
never asked, with the app's API key, its connection token and the execution token replaced by
"[redacted]".
A button press is still answered within 2.5 seconds: a private answer reaches only the presser,
a channel answer or an update becomes an app message everyone sees, an acknowledgement says
nothing, and a download or an error goes back to the presser; an app that fails keeps its own
status and error text, redacted the same way. Later work rechecks the asker's access with the
same execution token before it posts.

Discriminates: on dev c7193760b with oix 0.8, mutation `app_channel_reply_handler` sending the
delivery error to every channel member turns test_a_failed_delivery_is_told_only_to_the_asker
red; `send_app_mention` accepting any answer as a receipt turns the ack, receipt-with-extras and
private-answer cases of test_an_answer_that_is_not_a_receipt_is_a_failed_delivery red;
`authorize_app_execution` skipping the channel access check turns
test_background_work_loses_access_when_the_asker_leaves red. Retargeted for dev b704d0613, where a
failed delivery becomes a public app reply and the app's own error text is passed on: there,
`app_channel_reply_handler` sending its private fallback to every channel member turns
test_a_failure_the_app_cannot_post_is_told_only_to_the_asker red; it never posting the public
reply turns test_a_user_without_access_to_the_app_is_refused,
test_a_failed_delivery_is_told_to_the_whole_channel and every case of
test_the_channel_learns_why_a_mention_was_not_delivered red; `send_app_invocation` passing the
app's error text on unredacted turns test_an_app_error_shows_none_of_its_credentials red; and the
message list flagging app data instead of carrying it turns
test_a_channel_message_with_buttons_stores_them red.
"""

from __future__ import annotations

import json
import time

import pytest

from harness.apps import (
    ALL_SCOPES,
    channel_messages,
    connect,
    install_raw_app,
    installed,
    invite,
    mention_text,
    new_channel,
    post,
    reply_events,
    serving_raw_app,
    serving_reference_app,
    wait_for_message,
)
from harness.socket_client import connected

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

INVALID_RECEIPT = "App returned an invalid delivery receipt"
NOT_CONFIRMED = (
    "\nDelivery was not confirmed. The app may have accepted the message; check before retrying."
)
WRITE_SCOPE = "channels:messages:write"


@pytest.fixture
def channel(apps_admin, installed_app) -> str:
    with apps_admin.client() as client:
        channel_id = new_channel(client)
        assert invite(client, channel_id, installed_app["id"]).status_code == 200
    return channel_id


def _mention(actor, channel_id, app, text, **form):
    with actor.client() as client:
        return post(client, channel_id, mention_text(app, text), **form)


def _press(actor, app_id, message_id, action_id):
    with actor.client() as client:
        return client.post(
            f"/api/v1/apps/{app_id}/interactions",
            json={"message_id": message_id, "action_id": action_id},
        )


def _replies_to(socket, message_id) -> list:
    return [event for event in socket.app_events if event["message_id"] == message_id]


def app_error(detail: str) -> str:
    """The reply an app posts, in its name, for a mention it could not take."""
    return f"App error:\n\n```text\n{detail}\n```"


def _posted_error(actor, channel_id, app, mention, detail) -> dict:
    with actor.client() as client:
        error = wait_for_message(client, channel_id, app_error(detail))
    assert error["user"]["id"] == app["id"]
    assert error["reply_to_id"] == mention["id"]
    return error


def _first_reply(actor, channel_id, message) -> dict:
    deadline = time.monotonic() + 20
    with actor.client() as client:
        while time.monotonic() < deadline:
            listed = channel_messages(client, channel_id)
            replies = [reply for reply in listed if reply.get("reply_to_id") == message["id"]]
            if replies:
                return replies[0]
            time.sleep(0.2)
    raise AssertionError(f"nothing replied to {message['content']!r}")


def _contents(actor, channel_id) -> list:
    with actor.client() as client:
        return [message["content"] for message in channel_messages(client, channel_id)]


def _tools_message(actor, channel_id) -> dict:
    with actor.client() as client:
        post(client, channel_id, "!tools")
        return wait_for_message(client, channel_id, "Tools")


def test_a_mention_is_answered_in_the_channel(apps_admin, installed_app, reference_app, channel):
    posted = _mention(apps_admin, channel, installed_app, "hi there")
    with apps_admin.client() as client:
        answer = wait_for_message(client, channel, f"pong: {posted['content']}")
    assert answer["user"]["id"] == installed_app["id"]
    assert answer["reply_to_id"] == posted["id"]
    seen = reference_app.wait_for("mention", app_id=installed_app["id"], channel_id=channel)
    assert seen["actor"] == {"id": apps_admin.id, "name": apps_admin.name, "is_admin": True}
    assert seen["data"]["message_id"] == posted["id"]


def test_a_delivered_mention_sends_the_asker_nothing_privately(
    apps_admin, installed_app, reference_app, channel
):
    with connected(apps_admin) as socket:
        posted = _mention(apps_admin, channel, installed_app, "quietly")
        reference_app.wait_for("mention", app_id=installed_app["id"], channel_id=channel)
        with apps_admin.client() as client:
            wait_for_message(client, channel, f"pong: {posted['content']}")
        time.sleep(1)
        assert _replies_to(socket, posted["id"]) == []


def test_a_mention_answer_is_seen_by_every_member(apps_admin, make_apps_user, installed_app):
    asker, bystander = make_apps_user(), make_apps_user()
    with apps_admin.client() as client:
        channel_id = new_channel(client, type="group", user_ids=[asker.id, bystander.id])
        invite(client, channel_id, installed_app["id"])
    posted = _mention(asker, channel_id, installed_app, "for everyone")
    with bystander.client() as client:
        answer = wait_for_message(client, channel_id, f"pong: {posted['content']}")
    assert answer["reply_to_id"] == posted["id"]


def test_a_public_answer_becomes_a_channel_message(apps_admin, installed_app, channel):
    _mention(apps_admin, channel, installed_app, "public")
    with apps_admin.client() as client:
        answer = wait_for_message(client, channel, f"public answer for {apps_admin.id}")
    assert answer["user"]["id"] == installed_app["id"]
    assert answer["reply_to_id"] is None


def test_a_mention_in_a_thread_is_answered_in_that_thread(apps_admin, installed_app, channel):
    with apps_admin.client() as client:
        root = post(client, channel, "thread root")
    posted = _mention(apps_admin, channel, installed_app, "in thread", parent_id=root["id"])
    with apps_admin.client() as client:
        answer = wait_for_message(
            client, channel, f"pong: {posted['content']}", parent_id=root["id"]
        )
    assert answer["parent_id"] == root["id"]


def test_a_handler_that_crashes_after_the_receipt_shows_nothing(
    apps_admin, installed_app, reference_app, channel
):
    with connected(apps_admin) as socket:
        posted = _mention(apps_admin, channel, installed_app, "crash")
        reference_app.wait_for("mention", app_id=installed_app["id"], channel_id=channel)
        time.sleep(1.5)
        assert _replies_to(socket, posted["id"]) == [], "the app had already confirmed receipt"
    with apps_admin.client() as client:
        contents = [message["content"] for message in channel_messages(client, channel)]
    assert contents == [posted["content"]]
    assert "crashed on purpose" in reference_app.log()


def test_the_app_rechecks_the_execution_token(apps_admin, make_apps_user, installed_app):
    member = make_apps_user()
    with apps_admin.client() as client:
        channel_id = new_channel(client, type="group", user_ids=[member.id])
        invite(client, channel_id, installed_app["id"])
    _mention(member, channel_id, installed_app, "authorize")
    with member.client() as client:
        wait_for_message(client, channel_id, f"authorized {member.id} admin=False")


def test_mentioning_two_apps_asks_both(apps_admin, reference_app):
    with serving_reference_app(name="Other App") as other:
        with apps_admin.client() as client:
            first = installed(client, reference_app)
            second = installed(client, other)
            channel_id = new_channel(client)
            invite(client, channel_id, first["id"])
            invite(client, channel_id, second["id"])
            text = f"{mention_text(first, '')} {mention_text(second, 'both of you')}"
            posted = post(client, channel_id, text)
        reference_app.wait_for("mention", app_id=first["id"], channel_id=channel_id)
        other.wait_for("mention", app_id=second["id"], channel_id=channel_id)
        with apps_admin.client() as client:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                answers = [
                    message
                    for message in channel_messages(client, channel_id)
                    if message["content"] == f"pong: {posted['content']}"
                ]
                if len(answers) == 2:
                    break
                time.sleep(0.2)
    assert sorted(answer["user"]["id"] for answer in answers) == sorted([first["id"], second["id"]])


def test_a_mention_of_an_app_outside_the_channel_is_refused(
    apps_admin, installed_app, reference_app
):
    with apps_admin.client() as client:
        channel_id = new_channel(client)
    with connected(apps_admin) as socket:
        posted = _mention(apps_admin, channel_id, installed_app, "are you here?")
        event = reply_events(socket, message_id=posted["id"])
    assert event["reply"] == {"type": "error", "content": "Invite the app into this channel first"}
    assert event["app_id"] == installed_app["id"]
    assert event["app_name"] == installed_app["name"]
    assert event["channel_id"] == channel_id
    assert reference_app.records("mention", app_id=installed_app["id"]) == []
    assert _contents(apps_admin, channel_id) == [posted["content"]], "the app cannot post here"


def test_a_user_without_access_to_the_app_is_refused(apps_admin, make_apps_user, reference_app):
    member = make_apps_user()
    with apps_admin.client() as client:
        restricted = installed(client, reference_app, access_grants=[])
        channel_id = new_channel(client, type="group", user_ids=[member.id])
        invite(client, channel_id, restricted["id"])
    with connected(member) as socket:
        posted = _mention(member, channel_id, restricted, "let me in")
        error = _posted_error(apps_admin, channel_id, restricted, posted, "App invocation denied")
        time.sleep(1)
        assert _replies_to(socket, posted["id"]) == [], "the channel was told, so not privately"
    assert reference_app.records("mention", app_id=restricted["id"]) == []
    assert _contents(member, channel_id) == [error["content"], posted["content"]]


def test_a_mention_of_a_disabled_app_is_refused(apps_admin, installed_app, reference_app, channel):
    with apps_admin.client() as client:
        client.patch(f"/api/v1/apps/{installed_app['id']}", json={"is_active": False})
    with connected(apps_admin) as socket:
        posted = _mention(apps_admin, channel, installed_app, "wake up")
        event = reply_events(socket, message_id=posted["id"])
    assert event["reply"] == {"type": "error", "content": "App is inactive"}
    assert reference_app.records("mention", app_id=installed_app["id"]) == []
    assert _contents(apps_admin, channel) == [posted["content"]], "an inactive app posts nothing"


def test_a_failure_the_app_cannot_post_is_told_only_to_the_asker(
    apps_admin, make_apps_user, reference_app
):
    asker = make_apps_user()
    read_only = [scope for scope in ALL_SCOPES if scope != WRITE_SCOPE]
    with apps_admin.client() as client:
        app = installed(client, reference_app, scopes=read_only, access_grants=[])
        channel_id = new_channel(client, type="group", user_ids=[asker.id])
        invite(client, channel_id, app["id"])
    with connected(asker) as asker_socket, connected(apps_admin) as bystander_socket:
        posted = _mention(asker, channel_id, app, "only I should know")
        event = reply_events(asker_socket, message_id=posted["id"])
        time.sleep(1)
        assert _replies_to(bystander_socket, posted["id"]) == []
    assert event["reply"] == {"type": "error", "content": "App invocation denied"}
    assert _contents(apps_admin, channel_id) == [posted["content"]]


def test_a_failed_delivery_is_told_to_the_whole_channel(apps_admin, make_apps_user, installed_app):
    asker = make_apps_user()
    with apps_admin.client() as client:
        channel_id = new_channel(client, type="group", user_ids=[asker.id])
        invite(client, channel_id, installed_app["id"])
        client.patch(f"/api/v1/apps/{installed_app['id']}", json={"access_grants": []})
    with apps_admin.client() as client:
        root = post(client, channel_id, "thread root")
    with connected(asker) as asker_socket, connected(apps_admin) as bystander_socket:
        posted = _mention(
            asker, channel_id, installed_app, "everyone should know", parent_id=root["id"]
        )
        with apps_admin.client() as client:
            error = wait_for_message(
                client, channel_id, app_error("App invocation denied"), parent_id=root["id"]
            )
        time.sleep(1)
        assert _replies_to(asker_socket, posted["id"]) == []
        assert _replies_to(bystander_socket, posted["id"]) == []
    assert error["user"]["id"] == installed_app["id"]
    assert error["reply_to_id"] == posted["id"]


def test_a_mention_only_app_hears_mentions_but_not_other_messages(apps_admin, reference_app):
    with apps_admin.client() as client:
        quiet = installed(client, reference_app, scopes=["channels:mentions:read"])
        channel_id = new_channel(client)
        invite(client, channel_id, quiet["id"])
        post(client, channel_id, "an ordinary message")
    with connected(apps_admin) as socket:
        posted = _mention(apps_admin, channel_id, quiet, "only mentions")
        reference_app.wait_for("mention", app_id=quiet["id"], channel_id=channel_id)
        time.sleep(1)
        assert _replies_to(socket, posted["id"]) == []
    assert reference_app.records("message.created", app_id=quiet["id"]) == []


def test_an_app_without_write_scope_cannot_post_publicly(apps_admin, reference_app):
    read_only = ["channels:mentions:read", "channels:messages:read", "channels:read"]
    with apps_admin.client() as client:
        app = installed(client, reference_app, scopes=read_only)
        channel_id = new_channel(client)
        invite(client, channel_id, app["id"])
        post(client, channel_id, "!send not allowed")
    refused = reference_app.wait_for("api_error", app_id=app["id"], channel_id=channel_id)
    assert refused["status"] == 403
    posted = _mention(apps_admin, channel_id, app, "public")
    refused = reference_app.wait_for(
        "api_error", app_id=app["id"], type="channel.mention", channel_id=channel_id
    )
    assert refused["status"] == 403
    assert refused["method"] == "create_message"
    with apps_admin.client() as client:
        contents = [message["content"] for message in channel_messages(client, channel_id)]
    assert contents == [posted["content"], "!send not allowed"]


def test_an_app_without_channel_scope_cannot_list_channels(apps_admin, reference_app):
    scopes = ["channels:messages:read", "channels:messages:write"]
    with apps_admin.client() as client:
        app = installed(client, reference_app, scopes=scopes)
        channel_id = new_channel(client)
        invite(client, channel_id, app["id"])
        post(client, channel_id, "!channels")
    refused = reference_app.wait_for("api_error", app_id=app["id"], channel_id=channel_id)
    assert refused["status"] == 403
    assert refused["method"] == "list_channels"


def test_a_channel_message_with_buttons_stores_them(apps_admin, installed_app, channel):
    _mention(apps_admin, channel, installed_app, "buttons")
    with apps_admin.client() as client:
        message = wait_for_message(client, channel, "Choose")
        data = client.get(f"/api/v1/channels/{channel}/messages/{message['id']}/data").json()
    labels = [action["label"] for action in data["app"]["actions"]]
    assert labels == ["Approve", "Who am I", "Reject", "Noted"]
    assert message["data"] == data, "the list carries the buttons without a second request"


def test_pressing_a_button_updates_the_message_for_everyone(apps_admin, installed_app, channel):
    _mention(apps_admin, channel, installed_app, "buttons")
    with apps_admin.client() as client:
        message = wait_for_message(client, channel, "Choose")
    pressed = _press(apps_admin, installed_app["id"], message["id"], "approve")
    assert pressed.status_code == 200, pressed.text
    assert pressed.json()["type"] == "update"
    with apps_admin.client() as client:
        wait_for_message(client, channel, f"Approved by {apps_admin.id}")


def test_a_button_without_its_own_handler_reaches_the_generic_one(
    apps_admin, installed_app, reference_app, channel
):
    with apps_admin.client() as client:
        post(client, channel, "!buttons")
        message = wait_for_message(client, channel, "Pick one")
    pressed = _press(apps_admin, installed_app["id"], message["id"], "reject")
    assert pressed.status_code == 200, pressed.text
    assert pressed.json()["content"] == 'unhandled reject {"reason": "unhandled"}'
    seen = reference_app.wait_for("action.generic", app_id=installed_app["id"])
    assert seen["data"]["context"] == {"reason": "unhandled"}


def test_a_button_answer_names_whoever_pressed_it(apps_admin, make_apps_user, installed_app):
    member = make_apps_user()
    with apps_admin.client() as client:
        channel_id = new_channel(client, type="group", user_ids=[member.id])
        invite(client, channel_id, installed_app["id"])
        post(client, channel_id, "!buttons")
        message = wait_for_message(client, channel_id, "Pick one")
    by_member = _press(member, installed_app["id"], message["id"], "whoami")
    by_admin = _press(apps_admin, installed_app["id"], message["id"], "whoami")
    assert by_member.json()["content"] == f"you are {member.id} admin=False"
    assert by_admin.json()["content"] == f"you are {apps_admin.id} admin=True"


def test_a_button_can_be_acknowledged_silently(apps_admin, installed_app, reference_app, channel):
    with apps_admin.client() as client:
        post(client, channel, "!buttons")
        message = wait_for_message(client, channel, "Pick one")
    pressed = _press(apps_admin, installed_app["id"], message["id"], "noted")
    assert pressed.status_code == 200, pressed.text
    assert pressed.json() == {"type": "ack"}
    reference_app.wait_for("action.noted", app_id=installed_app["id"], channel_id=channel)
    with apps_admin.client() as client:
        contents = [message["content"] for message in channel_messages(client, channel)]
    assert contents == ["Pick one", "!buttons"], "an acknowledgement posts nothing"


def test_a_download_answer_carries_the_file(apps_admin, installed_app, channel):
    message = _tools_message(apps_admin, channel)
    pressed = _press(apps_admin, installed_app["id"], message["id"], "report")
    assert pressed.status_code == 200, pressed.text
    assert pressed.json()["type"] == "download"
    assert pressed.json()["filename"] == "weekly-report.md"
    assert pressed.json()["title"] == "Weekly report"
    assert pressed.json()["content"] == "# Report"


def test_an_error_answer_reaches_the_presser(apps_admin, installed_app, channel):
    message = _tools_message(apps_admin, channel)
    pressed = _press(apps_admin, installed_app["id"], message["id"], "fail")
    assert pressed.status_code == 200, pressed.text
    assert pressed.json()["type"] == "error"
    assert pressed.json()["content"] == "It broke"


def test_a_button_that_outlasts_its_budget_times_out(apps_admin, installed_app, channel):
    message = _tools_message(apps_admin, channel)
    pressed = _press(apps_admin, installed_app["id"], message["id"], "slow")
    assert pressed.status_code == 504, pressed.text


def test_a_button_handler_that_crashes_is_a_failed_answer(apps_admin, installed_app, channel):
    message = _tools_message(apps_admin, channel)
    pressed = _press(apps_admin, installed_app["id"], message["id"], "crash")
    assert pressed.status_code == 500, pressed.text
    assert pressed.json()["detail"] == "App handler failed", "the app's own error text"


def test_a_link_button_cannot_be_pressed_through_the_app(
    apps_admin, installed_app, reference_app, channel
):
    message = _tools_message(apps_admin, channel)
    for action_id in ("Docs", "https://docs.openwebui.com/"):
        pressed = _press(apps_admin, installed_app["id"], message["id"], action_id)
        assert pressed.status_code == 400, pressed.text
        assert pressed.json()["detail"] == "Unknown action"
    with apps_admin.client() as client:
        pressed = client.post(
            f"/api/v1/apps/{installed_app['id']}/interactions",
            json={"message_id": message["id"], "action_id": None},
        )
    assert pressed.status_code == 422
    assert reference_app.records("action.generic", app_id=installed_app["id"]) == []


def test_an_unknown_button_is_refused(apps_admin, installed_app, channel):
    with apps_admin.client() as client:
        post(client, channel, "!buttons")
        message = wait_for_message(client, channel, "Pick one")
    pressed = _press(apps_admin, installed_app["id"], message["id"], "self-destruct")
    assert pressed.status_code == 400
    assert pressed.json()["detail"] == "Unknown action"


def test_a_button_on_a_human_message_is_not_found(apps_admin, installed_app, channel):
    with apps_admin.client() as client:
        human = post(client, channel, "no buttons here")
    pressed = _press(apps_admin, installed_app["id"], human["id"], "approve")
    assert pressed.status_code == 404


def test_someone_outside_the_channel_cannot_press_its_buttons(
    apps_admin, make_apps_user, installed_app
):
    outsider = make_apps_user()
    with apps_admin.client() as client:
        channel_id = new_channel(client, type="group")
        invite(client, channel_id, installed_app["id"])
        post(client, channel_id, "!buttons")
        message = wait_for_message(client, channel_id, "Pick one")
    pressed = _press(outsider, installed_app["id"], message["id"], "whoami")
    assert pressed.status_code == 403


def test_a_disabled_app_takes_no_button_presses(apps_admin, installed_app, channel):
    with apps_admin.client() as client:
        post(client, channel, "!buttons")
        message = wait_for_message(client, channel, "Pick one")
        client.patch(f"/api/v1/apps/{installed_app['id']}", json={"is_active": False})
    pressed = _press(apps_admin, installed_app["id"], message["id"], "whoami")
    assert pressed.status_code == 403
    assert pressed.json()["detail"] == "App is inactive"


def test_background_work_publishes_its_result_later(
    apps_admin, installed_app, reference_app, channel
):
    posted = _mention(apps_admin, channel, installed_app, "later")
    with apps_admin.client() as client:
        result = wait_for_message(client, channel, f"Job finished for {apps_admin.name}")
    assert result["reply_to_id"] == posted["id"]
    assert result["user"]["id"] == installed_app["id"]
    reference_app.wait_for("later.finished", app_id=installed_app["id"], channel_id=channel)


def test_background_work_loses_access_when_the_asker_leaves(
    apps_admin, make_apps_user, installed_app, reference_app
):
    member = make_apps_user()
    with apps_admin.client() as client:
        channel_id = new_channel(client, type="group", user_ids=[member.id])
        invite(client, channel_id, installed_app["id"])
    _mention(member, channel_id, installed_app, "later")
    # the app has confirmed receipt and is working; only now does the asker leave
    reference_app.wait_for("mention", app_id=installed_app["id"], channel_id=channel_id)
    with apps_admin.client() as client:
        removed = client.post(
            f"/api/v1/channels/{channel_id}/update/members/remove", json={"user_ids": [member.id]}
        )
    assert removed.status_code == 200, removed.text
    refused = reference_app.wait_for(
        "later.refused", app_id=installed_app["id"], channel_id=channel_id
    )
    assert refused["status"] == 403
    with apps_admin.client() as client:
        contents = [message["content"] for message in channel_messages(client, channel_id)]
    assert not any(content.startswith("Job finished") for content in contents)


@pytest.fixture
def raw(apps_admin, channel):
    with serving_raw_app() as raw:
        with apps_admin.client() as client:
            raw.installed = install_raw_app(client, raw)
            assert connect(client, raw.installed["id"]).status_code == 200
            raw.wait_until_ready()
            invite(client, channel, raw.installed["id"])
        yield raw


@pytest.mark.parametrize(
    "answer",
    [
        {"status": 200, "data": {"type": "ack"}},
        {"status": 200, "data": {"received": True, "content": "sneaky"}},
        {"status": 200, "data": {"type": "message", "content": "an old-style private answer"}},
    ],
    ids=["ack", "receipt-with-extras", "private-answer"],
)
def test_an_answer_that_is_not_a_receipt_is_a_failed_delivery(apps_admin, raw, channel, answer):
    raw.mention_answer = answer
    posted = _mention(apps_admin, channel, raw.installed, "anything")
    error = _posted_error(apps_admin, channel, raw.installed, posted, INVALID_RECEIPT)
    contents = _contents(apps_admin, channel)
    assert contents == [error["content"], posted["content"]], "nothing the app answered is posted"


@pytest.mark.parametrize(
    ("answer", "detail"),
    [
        ({"status": 503, "error": "App is at capacity"}, "App is at capacity"),
        (
            {"status": 504, "error": "Event acceptance timed out"},
            "Event acceptance timed out" + NOT_CONFIRMED,
        ),
        ({"status": 403, "error": "Event installation mismatch"}, "Event installation mismatch"),
        ({"status": 500, "error": "boom"}, "boom"),
    ],
    ids=["busy", "timed-out", "denied", "server-error"],
)
def test_the_channel_learns_why_a_mention_was_not_delivered(
    apps_admin, raw, channel, answer, detail
):
    raw.mention_answer = answer
    with connected(apps_admin) as socket:
        posted = _mention(apps_admin, channel, raw.installed, "anything")
        error = _posted_error(apps_admin, channel, raw.installed, posted, detail)
        time.sleep(1)
        assert _replies_to(socket, posted["id"]) == [], "the channel was told, so not privately"
    assert _contents(apps_admin, channel) == [error["content"], posted["content"]]


@pytest.mark.parametrize("secret", ["api-key", "connection-token", "execution-token"])
@pytest.mark.parametrize("where", ["mention", "button"])
def test_an_app_error_shows_none_of_its_credentials(apps_admin, raw, channel, where, secret):
    created = raw.request(
        "create_message",
        channel_id=channel,
        content="Pick",
        actions=[{"action_id": "go", "label": "Go"}],
    )
    assert created["status"] == 200, created
    echoed = []

    def failing(event):
        value = {
            "api-key": raw.api_key,
            "connection-token": raw.connection_token,
            "execution-token": event["execution_token"],
        }[secret]
        echoed.append(value)
        return {"status": 500, "error": f"refused {value} here"}

    raw.answer_with = failing
    if where == "mention":
        posted = _mention(apps_admin, channel, raw.installed, "anything")
        shown = _first_reply(apps_admin, channel, posted)["content"]
        with apps_admin.client() as client:
            everything = json.dumps(channel_messages(client, channel))
    else:
        pressed = _press(apps_admin, raw.installed["id"], created["data"]["id"], "go")
        assert pressed.status_code == 500, pressed.text
        shown = pressed.json()["detail"]
        everything = pressed.text
    assert echoed and echoed[0], "the app never saw the credential it was to echo"
    assert echoed[0] not in everything
    assert "refused [redacted] here" in shown


def test_an_ack_with_content_is_refused(apps_admin, raw, channel):
    """Open WebUI checks the reply shape itself, whatever an app sends."""
    created = raw.request(
        "create_message",
        channel_id=channel,
        content="Pick",
        actions=[{"action_id": "go", "label": "Go"}],
    )
    assert created["status"] == 200, created
    raw.event_answer = {"type": "ack", "content": "sneaky"}
    pressed = _press(apps_admin, raw.installed["id"], created["data"]["id"], "go")
    assert pressed.status_code == 502
    assert pressed.json()["detail"] == "App returned an invalid response"


def test_leaving_the_channel_revokes_interactions(apps_admin, make_apps_user, installed_app):
    member = make_apps_user()
    with apps_admin.client() as client:
        channel_id = new_channel(client, type="group", user_ids=[member.id])
        invite(client, channel_id, installed_app["id"])
        post(client, channel_id, "!buttons")
        message = wait_for_message(client, channel_id, "Pick one")
        assert _press(member, installed_app["id"], message["id"], "whoami").status_code == 200
        removed = client.post(
            f"/api/v1/channels/{channel_id}/update/members/remove", json={"user_ids": [member.id]}
        )
    assert removed.status_code == 200, removed.text
    assert _press(member, installed_app["id"], message["id"], "whoami").status_code == 403


def test_losing_access_to_the_app_revokes_interactions(apps_admin, make_apps_user, installed_app):
    member = make_apps_user()
    with apps_admin.client() as client:
        channel_id = new_channel(client, type="group", user_ids=[member.id])
        invite(client, channel_id, installed_app["id"])
        post(client, channel_id, "!buttons")
        message = wait_for_message(client, channel_id, "Pick one")
        client.patch(f"/api/v1/apps/{installed_app['id']}", json={"access_grants": []})
    pressed = _press(member, installed_app["id"], message["id"], "whoami")
    assert pressed.status_code == 403
    assert pressed.json()["detail"] == "App invocation denied"
