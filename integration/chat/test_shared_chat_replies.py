"""Journey: sharing a chat with "Allow replies" and answering in someone else's chat.

The owner shares a chat with the share dialog's two calls: the link is made with a share mode,
then the access save names who may open it and whether they get the live chat ("continue") or
a frozen copy (null, "Clone only"). A user with a grant in continue mode opens the live chat and
keeps chatting in it; the owner sees each message with its author. Anyone else who can open the
link, and everyone in Clone only mode, gets the snapshot and cannot send. The owner's private
settings (params, variables, message meta) never reach a member, a member's turn runs without
the owner's chat variables, going back to Clone only freezes a fresh snapshot, and cloning a
shared chat gives the reader a copy without the owner's folder, variables or settings.

Discriminates: passes on dev b5a20423e; in a backend copy, making the `share_mode == 'continue'`
branch of `get_accessible_chat_by_id` never match fails the reply, open visitors, settings,
variables, freeze and allow-replies clone tests, dropping that share mode check there fails the
clone only test and the freeze test, `get_shared_chat_by_id` calling a reader live without the
write access check fails the open visitors test, `shared_chat_response` skipping the stripping of
params and variables fails the settings test and both clone tests, the same function keeping
`meta` on other people's messages fails the settings test, dropping the non-owner
`chat_variables = {}` override in `chat_completion` fails the variables test, `set_share_mode`
skipping the re-snapshot fails the freeze test, and the clone route storing `chat.variables` and
`chat.folder_id` fails the allow-replies clone test.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from harness import upstream as reply
from harness.access import grant, make_group
from harness.chat import ask, send_message
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

PRIVATE_SETTINGS = ("params", "tool_servers", "tool_ids", "filter_ids", "variables")


def _answered(owner, upstream, question: str, **options) -> tuple[str, str]:
    """The owner asks `question`; returns the chat id and the reply's id."""
    upstream.queue(reply.text(f"answer to {question}", match=reply.answering(question)))
    with owner.client() as client:
        turn, stored = ask(client, question, **options)
    assert stored["content"] == f"answer to {question}"
    return turn.chat_id, turn.assistant_message_id


def _save_access(owner, chat_id: str, grants: list[dict], share_mode: str | None) -> None:
    with owner.client() as client:
        saved = client.post(
            f"/api/v1/chats/shared/{chat_id}/access/update",
            json={"access_grants": grants, "share_mode": share_mode},
        )
    assert saved.status_code == 200, saved.text


def _share(owner, chat_id: str, grants: list[dict], share_mode: str | None = None) -> str:
    """Share as the dialog does: make the link with no mode, then save the access and the mode."""
    with owner.client() as client:
        shared = client.post(f"/api/v1/chats/{chat_id}/share", json={"share_mode": None})
    assert shared.status_code == 200, shared.text
    _save_access(owner, chat_id, grants, share_mode)
    return shared.json()["share_id"]


def _read_grants(*members) -> list[dict]:
    return [grant("user", member.id, "read") for member in members]


def _open_share(account, share_id: str) -> httpx.Response:
    with account.client() as client:
        return client.get(f"/api/v1/chats/share/{share_id}")


def _stored_chat(account, chat_id: str) -> dict:
    with account.client() as client:
        stored = client.get(f"/api/v1/chats/{chat_id}")
    assert stored.status_code == 200, stored.text
    return stored.json()


def _messages(chat: dict) -> dict:
    return chat["chat"]["history"]["messages"]


def _try_to_send(account, chat_id: str, parent_id: str, content: str) -> str:
    """The reason the chat endpoint gives, or "" when it took the message."""
    with account.client() as client:
        try:
            send_message(client, content, chat_id=chat_id, parent_id=parent_id)
        except AssertionError as refused:
            return str(refused)
    return ""


def _provider_text(upstream, question: str) -> str:
    request = next(filter(reply.answering(question), upstream.chat_requests()))
    return "\n".join(str(entry["content"]) for entry in request["messages"])


def test_a_user_granted_allow_replies_answers_in_the_owners_chat(make_user, upstream):
    owner, member = make_user(), make_user()
    chat_id, owner_reply_id = _answered(owner, upstream, "what is the capital of Norway?")
    share_id = _share(owner, chat_id, _read_grants(member), "continue")

    opened = _open_share(member, share_id)
    assert opened.status_code == 200, opened.text
    assert opened.json()["id"] == chat_id
    assert opened.json()["chat"]["share_mode"] == "continue"

    question = f"and the capital of Sweden? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Stockholm.", match=reply.answering(question)))
    with member.client() as client:
        turn, member_reply = ask(client, question, chat_id=chat_id, parent_id=owner_reply_id)

    sent = _provider_text(upstream, question)
    assert "what is the capital of Norway?" in sent
    assert "answer to what is the capital of Norway?" in sent
    assert member_reply["content"] == "Stockholm."
    seen_by_owner = _stored_chat(owner, chat_id)
    messages = _messages(seen_by_owner)
    member_message = messages[turn.user_message_id]
    assert member_message["user_id"] == member.id
    assert member_message["user"]["name"] == member.name
    first_message = next(
        message
        for message in messages.values()
        if message["role"] == "user" and message["parentId"] is None
    )
    assert first_message["user"]["name"] == owner.name
    assert member_message["childrenIds"] == [turn.assistant_message_id]
    assert seen_by_owner["chat"]["history"]["currentId"] == turn.assistant_message_id


def test_clone_only_gives_the_user_a_snapshot_and_no_way_to_reply(make_user, upstream):
    owner, member = make_user(), make_user()
    chat_id, owner_reply_id = _answered(owner, upstream, "what is the capital of Norway?")
    share_id = _share(owner, chat_id, _read_grants(member), None)

    opened = _open_share(member, share_id)
    refused = _try_to_send(member, chat_id, owner_reply_id, "can I reply here?")
    with member.client() as client:
        direct = client.get(f"/api/v1/chats/{chat_id}")

    assert opened.status_code == 200, opened.text
    assert opened.json()["chat"]["share_mode"] is None
    assert opened.json()["id"] == share_id != chat_id
    assert "answer to what is the capital of Norway?" in str(opened.json()["chat"])
    assert "HTTP 404" in refused, refused
    assert direct.status_code in (401, 403, 404), direct.text
    assert "can I reply here?" not in str(_messages(_stored_chat(owner, chat_id)))


def test_open_visitors_get_the_snapshot_while_a_granted_member_gets_the_live_chat(
    instance, admin, make_user, upstream
):
    owner, member, stranger = make_user(), make_user(), make_user()
    make_group(admin, [owner], {"sharing": {"open_chats": True}})
    chat_id, owner_reply_id = _answered(owner, upstream, "before sharing")
    grants = [*_read_grants(member), grant("anyone", "*", "read")]
    share_id = _share(owner, chat_id, grants, "continue")
    with owner.client() as client:
        stored_grants = client.get(f"/api/v1/chats/shared/{chat_id}/access").json()
    assert "anyone" in {stored["principal_type"] for stored in stored_grants}
    later_question = f"after sharing {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("later answer", match=reply.answering(later_question)))
    with owner.client() as client:
        ask(client, later_question, chat_id=chat_id, parent_id=owner_reply_id)

    signed_out = httpx.get(f"{instance.base_url}/api/v1/chats/share/{share_id}", timeout=60.0)
    seen = {"signed out": signed_out, "stranger": _open_share(stranger, share_id)}
    refused = _try_to_send(stranger, chat_id, owner_reply_id, "let me in")
    live = _open_share(member, share_id)

    for who, response in seen.items():
        assert response.status_code == 200, f"{who}: {response.text}"
        assert response.json()["chat"]["share_mode"] is None, who
        assert response.json()["id"] != chat_id, who
        assert "before sharing" in str(response.json()["chat"]), who
        assert later_question not in str(response.json()["chat"]), who
    assert refused != ""
    assert live.json()["id"] == chat_id
    assert live.json()["chat"]["share_mode"] == "continue"
    assert later_question in str(live.json()["chat"])


def test_the_owners_private_settings_stay_with_the_owner(make_user, upstream):
    owner, member = make_user(), make_user()
    chat_id, owner_reply_id = _answered(owner, upstream, "plan my week")
    params, variables = {"system": "Answer like a pirate."}, {"codename": "Bluebird"}
    with owner.client() as client:
        saved_controls = client.post(f"/api/v1/chats/{chat_id}", json={"chat": {"params": params}})
        saved_variables = client.post(
            f"/api/v1/chats/{chat_id}", json={"chat": {}, "variables": variables}
        )
    assert saved_controls.status_code == 200, saved_controls.text
    assert saved_variables.status_code == 200, saved_variables.text
    share_id = _share(owner, chat_id, _read_grants(member), "continue")

    question = f"anything to add? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Nothing more.", match=reply.answering(question)))
    with member.client() as client:
        turn, _ = ask(client, question, chat_id=chat_id, parent_id=owner_reply_id)

    views = {
        "chat": _stored_chat(member, chat_id),
        "share": _open_share(member, share_id).json(),
    }
    for where, view in views.items():
        for key in PRIVATE_SETTINGS:
            assert key not in view["chat"], f"{where} shows {key} to the member"
        assert view["variables"] == {}, where
        assert "meta" not in _messages(view)[owner_reply_id], where
        assert "meta" in _messages(view)[turn.assistant_message_id], where
    seen_by_owner = _stored_chat(owner, chat_id)
    assert seen_by_owner["chat"]["params"] == params
    assert seen_by_owner["variables"] == variables
    assert "meta" in _messages(seen_by_owner)[owner_reply_id]


def test_a_members_turn_runs_without_the_owners_chat_variables(admin, make_user, upstream):
    owner, member = make_user(), make_user()
    model_id = f"codename-{uuid.uuid4().hex[:6]}"
    form = {
        "id": model_id,
        "name": model_id,
        "base_model_id": MOCK_MODEL_ID,
        "meta": {},
        "params": {"system": "Codename: {{chat.variables.codename | text}}."},
    }
    with admin.client() as client:
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
        client.get("/api/models").raise_for_status()  # registers it for chat
        published = client.post(
            "/api/v1/models/model/access/update",
            json={**form, "access_grants": [grant("user", "*", "read")]},
        )
    assert published.status_code == 200, published.text
    try:
        first = f"first question {uuid.uuid4().hex[:6]}"
        chat_id, owner_reply_id = _answered(owner, upstream, first, model=model_id)
        with owner.client() as client:
            client.post(
                f"/api/v1/chats/{chat_id}", json={"chat": {}, "variables": {"codename": "Bluebird"}}
            ).raise_for_status()
        _share(owner, chat_id, _read_grants(member), "continue")
        owner_question = f"owner again {uuid.uuid4().hex[:6]}"
        member_question = f"member now {uuid.uuid4().hex[:6]}"
        upstream.queue(
            reply.text("owner done", match=reply.answering(owner_question)),
            reply.text("member done", match=reply.answering(member_question)),
        )
        with owner.client() as client:
            _, owner_second = ask(
                client, owner_question, chat_id=chat_id, parent_id=owner_reply_id, model=model_id
            )
        with member.client() as client:
            ask(
                client,
                member_question,
                chat_id=chat_id,
                parent_id=owner_second["id"],
                model=model_id,
            )

        assert "Codename: Bluebird." in _provider_text(upstream, owner_question)
        assert "Codename: Bluebird." not in _provider_text(upstream, member_question)
        assert "Codename:" in _provider_text(upstream, member_question)
        assert _stored_chat(owner, chat_id)["variables"] == {"codename": "Bluebird"}
    finally:
        with admin.client() as client:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


def test_going_back_to_clone_only_freezes_a_fresh_snapshot(make_user, upstream):
    owner, member = make_user(), make_user()
    chat_id, owner_reply_id = _answered(owner, upstream, "first question")
    grants = _read_grants(member)
    share_id = _share(owner, chat_id, grants, "continue")
    member_question = f"member question {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("member answer", match=reply.answering(member_question)))
    with member.client() as client:
        _, member_reply = ask(client, member_question, chat_id=chat_id, parent_id=owner_reply_id)

    _save_access(owner, chat_id, grants, None)
    frozen = _open_share(member, share_id).json()
    later_question = f"owner afterwards {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("later answer", match=reply.answering(later_question)))
    with owner.client() as client:
        ask(client, later_question, chat_id=chat_id, parent_id=member_reply["id"])
    still_frozen = _open_share(member, share_id).json()
    refused = _try_to_send(member, chat_id, member_reply["id"], "one more from me")

    assert frozen["chat"]["share_mode"] is None
    assert frozen["id"] == share_id != chat_id
    assert member_question in str(frozen["chat"])
    assert later_question not in str(still_frozen["chat"])
    assert still_frozen["chat"]["share_mode"] is None
    assert refused != ""
    assert later_question in str(_stored_chat(owner, chat_id)["chat"])


@pytest.mark.parametrize("share_mode", [None, "continue"], ids=["clone-only", "allow-replies"])
def test_cloning_a_shared_chat_leaves_the_owners_folder_variables_and_settings_behind(
    make_user, upstream, share_mode
):
    owner, member = make_user(), make_user()
    chat_id, owner_reply_id = _answered(owner, upstream, "plan my week")
    with owner.client() as client:
        folder = client.post("/api/v1/folders/", json={"name": f"folder {uuid.uuid4().hex[:6]}"})
        assert folder.status_code == 200, folder.text
        folder_id = folder.json()["id"]
        client.post(
            f"/api/v1/chats/{chat_id}/folder", json={"folder_id": folder_id}
        ).raise_for_status()
        client.post(
            f"/api/v1/chats/{chat_id}", json={"chat": {"params": {"system": "Be brief."}}}
        ).raise_for_status()
        client.post(
            f"/api/v1/chats/{chat_id}", json={"chat": {}, "variables": {"codename": "Bluebird"}}
        ).raise_for_status()
    share_id = _share(owner, chat_id, _read_grants(member), share_mode)
    member_question = f"member question {uuid.uuid4().hex[:6]}"
    if share_mode == "continue":
        upstream.queue(reply.text("member answer", match=reply.answering(member_question)))
        with member.client() as client:
            ask(client, member_question, chat_id=chat_id, parent_id=owner_reply_id)

    with member.client() as client:
        cloned = client.post(f"/api/v1/chats/{share_id}/clone/shared")
        assert cloned.status_code == 200, cloned.text
        clone = cloned.json()
        stored_clone = client.get(f"/api/v1/chats/{clone['id']}").json()
    with owner.client() as client:
        folder_chats = client.get(f"/api/v1/chats/folder/{folder_id}").json()

    assert clone["user_id"] == member.id
    assert clone["id"] not in (chat_id, share_id)
    assert stored_clone["folder_id"] is None
    assert stored_clone["variables"] == {}
    assert "params" not in stored_clone["chat"]
    assert "answer to plan my week" in str(stored_clone["chat"])
    assert (member_question in str(stored_clone["chat"])) == (share_mode == "continue")
    assert [chat["id"] for chat in folder_chats] == [chat_id]
