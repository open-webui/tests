"""Regression: permission and feature switches an admin set were ignored on a second code path.

open-webui 0.11.1 and 0.11.2 fixes, each pinned where a user or the provider sees it:

* `80d2f4154` (#27716) and `8fc5ffe26` (#27609): `SharingPermissions.public_tools` and
  `public_notes` defaulted to True, so a permissions save that did not carry them (an instance
  upgraded from before they existed) granted everyone public tool and note sharing, and
  `open_chats` had no field at all, so enabling open sharing reverted on save.
* `d9e23b90c` (#28366): sending the first message of a chat with a `folder_id` filed the new chat
  into that folder with no write check, so any account could drop chats into another's folder.
* `7d392bedc` (#28631): the `channel:` completion path only checked that the target message was in
  the channel, so a member could point the model at another member's message and overwrite it.
* `934802e18` (#27668): the legacy features block injected stored memories for any request whose
  client-supplied `features.memory` was set, with no `features.memories` permission check.
* `e17dfae72` / `646a568ae` (#27759, #27669): the legacy function-calling path generated images
  with image generation switched off and ran the web search step with web search switched off.
  The search route now refuses on its own, so that step shows in the reply as a search that
  failed on a permission error; the test reads the reply's steps as well as the search engine.
* `e92638dc7` (PR #32991): the per-kind sharing switches (models, knowledge,
  prompts, tools, skills, notes, folders) only hid the Access button, so a user with sharing off
  could still create an item shared with a group or share an existing one over the API. The
  server now drops such grants on create, and the access route refuses them outright with a 403
  and leaves the stored grants as they were.

The router-level half of `ad8c79f686` (#27766, a partial settings save serialized every default)
is no longer visible on its own: the settings store now patches `ui` field by field, so a
defaulted empty `ui` changes nothing. The partial-save test pins what users see and goes red only
with both layers reverted.

Twin of unit/config/test_permission_gates.py.

Discriminates: passes on bbfa876af; reverting `80d2f4154` or `8fc5ffe26` fails the sharing tests,
`d9e23b90c` the foreign and missing folder tests, `7d392bedc` the foreign-message test,
`934802e18` the barred memory test and `646a568ae` with `e17dfae72` the switched-off image and web
search tests; the partial-save test fails with `ad8c79f686` and the store's field-level `ui`
patching (98a920168) both reverted. The positive tests, image editing with generation off among
them, and the all-off round trip pass on both. The 13 "without sharing" tests fail on dev
2ea6bf436 (the group grant is stored) and pass with `e92638dc7`; the group-allowed, admin and
private tests pass on both. On dev 302b8cfd5 the access route refuses the share with a 403; letting
it through there fails all 7 "cannot share" tests.
"""

from __future__ import annotations

import uuid

import pytest

from harness.access import grant, make_group
from harness.channel_chat import serve_openai_images
from harness.channel_quotes import enable_channels, group_channel, post_message
from harness.chat import ask, send_message
from harness.chat_history import seed_chat
from harness.image_engines import IMAGES_CONFIG, PNG_BASE64, save_image_settings
from harness.upstream import MOCK_MODEL_ID
from harness.web_retrieval import RETRIEVAL_CONFIG, save_web_settings, serve_search_results

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

PERMISSIONS = "/api/v1/users/default/permissions"
SETTINGS = "/api/v1/users/user/settings"
STORED_MEMORY = "keeps a pet iguana called Doris"
LEGACY = {"function_calling": "legacy"}


@pytest.fixture
def permissions(admin, preserve) -> dict:
    preserve("permissions")
    with admin.client() as client:
        current = client.get(PERMISSIONS)
    assert current.status_code == 200, current.text
    return current.json()


def _save_permissions(admin, permissions: dict) -> dict:
    """Save the default permissions as the admin's modal does; returns what reads back."""
    with admin.client() as client:
        saved = client.post(PERMISSIONS, json=permissions)
        assert saved.status_code == 200, saved.text
        stored = client.get(PERMISSIONS)
    assert stored.status_code == 200, stored.text
    return stored.json()


def _session_permissions(actor) -> dict:
    with actor.client() as client:
        session = client.get("/api/v1/auths/")
    assert session.status_code == 200, session.text
    return session.json()["permissions"]


# sharing permissions: a save persists exactly what the admin set


@pytest.mark.parametrize("flag", ["public_tools", "public_notes"])
def test_a_save_without_a_public_sharing_flag_does_not_grant_it(
    admin, permissions, make_user, flag
):
    sharing = {key: value for key, value in permissions["sharing"].items() if key != flag}

    stored = _save_permissions(admin, {**permissions, "sharing": sharing})

    assert stored["sharing"][flag] is False, (
        f"a permissions save without sharing.{flag} granted it to everyone, because the schema "
        "defaulted it to True (#27716)"
    )
    assert _session_permissions(make_user())["sharing"][flag] is False


def test_enabled_open_sharing_survives_a_save(admin, permissions, make_user):
    sharing = {**permissions["sharing"], "open_chats": True}

    stored = _save_permissions(admin, {**permissions, "sharing": sharing})

    assert stored["sharing"].get("open_chats") is True, (
        "the admin enabled open sharing and the save dropped it, because the schema had no "
        "open_chats field (#27609)"
    )
    assert _session_permissions(make_user())["sharing"]["open_chats"] is True


def test_every_sharing_flag_saved_off_stays_off(admin, permissions):
    sharing = {key: False for key in permissions["sharing"]}

    stored = _save_permissions(admin, {**permissions, "sharing": sharing})

    assert stored["sharing"] == sharing


# sharing permissions: the server holds each kind's sharing switch, not only the Access button

SHARED_KINDS = ["models", "knowledge", "prompts", "tools", "skills", "notes", "folders"]
TOOL_SOURCE = 'class Tools:\n    def ping(self) -> str:\n        return "pong"\n'


def _create_body(kind: str, tag: str) -> tuple[str, dict]:
    bodies = {
        "models": (
            "/api/v1/models/create",
            {
                "id": f"shared-{tag}",
                "base_model_id": MOCK_MODEL_ID,
                "name": tag,
                "meta": {},
                "params": {},
            },
        ),
        "knowledge": ("/api/v1/knowledge/create", {"name": tag, "description": tag}),
        "prompts": (
            "/api/v1/prompts/create",
            {"command": f"shared{tag}", "name": tag, "content": "hi"},
        ),
        "tools": (
            "/api/v1/tools/create",
            {
                "id": f"shared_{tag}",
                "name": tag,
                "content": TOOL_SOURCE,
                "meta": {"description": tag},
            },
        ),
        "skills": (
            "/api/v1/skills/create",
            {"id": f"shared-{tag}", "name": tag, "description": tag, "content": "hi", "meta": {}},
        ),
        "notes": ("/api/v1/notes/create", {"title": tag, "data": {"content": {"md": "hi"}}}),
        "folders": ("/api/v1/folders/", {"name": tag}),
    }
    return bodies[kind]


def _create_item(actor, kind: str, grants: list[dict] | None = None) -> str:
    path, body = _create_body(kind, uuid.uuid4().hex[:8])
    if grants is not None:
        body["access_grants"] = grants
    with actor.client() as client:
        created = client.post(path, json=body)
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _share_item(actor, kind: str, item_id: str, grants: list[dict], status: int = 200) -> None:
    """Save the grants the way the item's Access dialog does."""
    paths = {
        "models": "/api/v1/models/model/access/update",
        "knowledge": f"/api/v1/knowledge/{item_id}/access/update",
        "prompts": f"/api/v1/prompts/id/{item_id}/access/update",
        "tools": f"/api/v1/tools/id/{item_id}/access/update",
        "skills": f"/api/v1/skills/id/{item_id}/access/update",
        "notes": f"/api/v1/notes/{item_id}/access/update",
        "folders": f"/api/v1/folders/{item_id}/access/update",
    }
    body = {"access_grants": grants}
    if kind == "models":
        body |= {"id": item_id, "name": item_id}
    with actor.client() as client:
        saved = client.post(paths[kind], json=body)
    assert saved.status_code == status, saved.text


def _stored_grants(admin, kind: str, item_id: str) -> list[tuple]:
    paths = {
        "models": f"/api/v1/models/model?id={item_id}",
        "knowledge": f"/api/v1/knowledge/{item_id}",
        "prompts": f"/api/v1/prompts/id/{item_id}",
        "tools": f"/api/v1/tools/id/{item_id}",
        "skills": f"/api/v1/skills/id/{item_id}",
        "notes": f"/api/v1/notes/{item_id}",
        "folders": f"/api/v1/folders/{item_id}",
    }
    with admin.client() as client:
        stored = client.get(paths[kind])
    assert stored.status_code == 200, stored.text
    return [
        (entry["principal_type"], entry["principal_id"], entry["permission"])
        for entry in stored.json()["access_grants"]
    ]


@pytest.fixture
def sharing_switched_off(admin, permissions) -> dict:
    """Every kind's sharing off for users, while user and group grants stay allowed."""
    sharing = {**permissions["sharing"], **{kind: False for kind in SHARED_KINDS}}
    workspace = {**permissions["workspace"], **{kind: True for kind in SHARED_KINDS[:5]}}
    features = {**permissions["features"], "folders": True, "notes": True}
    allowed = {**permissions["access_grants"], "allow_users": True, "allow_groups": True}
    return _save_permissions(
        admin,
        {
            **permissions,
            "sharing": sharing,
            "workspace": workspace,
            "features": features,
            "access_grants": allowed,
        },
    )


def _shared_with(admin, make_user) -> tuple[list[dict], tuple]:
    group_id = make_group(admin, [make_user()])
    return [grant("group", group_id, "read")], ("group", group_id, "read")


@pytest.mark.parametrize("kind", [kind for kind in SHARED_KINDS if kind != "folders"])
def test_a_user_without_sharing_cannot_create_a_shared_item(
    admin, make_user, sharing_switched_off, kind
):
    grants, group_grant = _shared_with(admin, make_user)

    item_id = _create_item(make_user(), kind, grants)

    stored = _stored_grants(admin, kind, item_id)
    assert group_grant not in stored, (
        f"a user with {kind} sharing switched off created an item shared with a group, because "
        f"only the interface checked the switch: {stored}"
    )


@pytest.mark.parametrize("kind", SHARED_KINDS)
def test_a_user_without_sharing_cannot_share_their_item(
    admin, make_user, sharing_switched_off, kind
):
    owner = make_user()
    grants, group_grant = _shared_with(admin, make_user)
    item_id = _create_item(owner, kind)
    before = _stored_grants(admin, kind, item_id)

    _share_item(owner, kind, item_id, grants, status=403)

    stored = _stored_grants(admin, kind, item_id)
    assert stored == before and group_grant not in stored, (
        f"a user with {kind} sharing switched off shared their item with a group, because only "
        f"the interface checked the switch: {stored}"
    )


@pytest.mark.parametrize("kind", SHARED_KINDS)
def test_a_user_whose_group_allows_sharing_can_share(admin, make_user, sharing_switched_off, kind):
    owner = make_user()
    make_group(admin, [owner], {"sharing": {kind: True}})
    grants, group_grant = _shared_with(admin, make_user)
    item_id = _create_item(owner, kind)

    _share_item(owner, kind, item_id, grants)

    assert group_grant in _stored_grants(admin, kind, item_id)


@pytest.mark.parametrize("kind", SHARED_KINDS)
def test_an_admin_shares_with_user_sharing_switched_off(
    admin, make_user, sharing_switched_off, kind
):
    grants, group_grant = _shared_with(admin, make_user)
    item_id = _create_item(admin, kind)

    _share_item(admin, kind, item_id, grants)

    assert group_grant in _stored_grants(admin, kind, item_id)


@pytest.mark.parametrize("kind", SHARED_KINDS)
def test_a_user_without_sharing_keeps_private_items(admin, make_user, sharing_switched_off, kind):
    owner = make_user()
    item_id = _create_item(owner, kind, [] if kind != "folders" else None)

    _share_item(owner, kind, item_id, [])

    assert _stored_grants(admin, kind, item_id) == []


# a partial settings save keeps what the user did not touch


def test_saving_one_setting_keeps_the_others(make_user):
    with make_user().client() as client:
        client.post(f"{SETTINGS}/update", json={"ui": {"widescreenMode": True}}).raise_for_status()
        # what the Shortcuts page saves: keybindings alone
        client.post(f"{SETTINGS}/update", json={"keybindings": {}}).raise_for_status()
        stored = client.get(SETTINGS)

    assert stored.status_code == 200, stored.text
    assert stored.json()["ui"].get("widescreenMode") is True, (
        "saving one setting wrote defaults over the interface settings the user had not touched "
        f"(#27766): {stored.json()}"
    )


# the first message of a chat may only file it into a folder the sender can write


def _create_folder(actor) -> str:
    with actor.client() as client:
        created = client.post("/api/v1/folders/", json={"name": "Private"})
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _own_chat_ids(client) -> list[str]:
    listed = client.get("/api/v1/chats/")
    assert listed.status_code == 200, listed.text
    return [chat["id"] for chat in listed.json()]


def test_a_new_chat_cannot_be_filed_into_another_users_folder(make_user, upstream):
    folder_id = _create_folder(make_user())

    with make_user().client() as client:
        with pytest.raises(AssertionError, match="HTTP 404"):
            send_message(client, "hi", folder_id=folder_id)
        created = _own_chat_ids(client)

    assert created == [], (
        f"a chat was filed into another user's folder with no write check (#28366): {created}"
    )


def test_a_new_chat_cannot_be_filed_into_a_folder_that_does_not_exist(make_user, upstream):
    with make_user().client() as client:
        with pytest.raises(AssertionError, match="HTTP 404"):
            send_message(client, "hi", folder_id="no-such-folder")


def test_a_new_chat_is_filed_into_the_senders_own_folder(make_user, upstream):
    owner = make_user()
    folder_id = _create_folder(owner)

    with owner.client() as client:
        turn = send_message(client, "hi", folder_id=folder_id)
        stored = client.get(f"/api/v1/chats/{turn.chat_id}")

    assert stored.json()["folder_id"] == folder_id


# the channel completion path only writes into the caller's own message


@pytest.fixture
def channel_members(admin, preserve, make_user):
    """A group channel with two members; returns (author, editor, channel id)."""
    preserve("admin_config")
    enable_channels(admin)
    author, editor = make_user(), make_user()
    return author, editor, group_channel(author, editor)


def _complete_into(actor, channel_id: str, message_id: str):
    with actor.client() as client:
        return client.post(
            "/api/chat/completions",
            json={
                "model": MOCK_MODEL_ID,
                "messages": [{"role": "user", "content": "rewrite this"}],
                "chat_id": f"channel:{channel_id}",
                "id": message_id,
            },
        )


def test_a_member_cannot_point_the_model_at_another_members_message(channel_members, upstream):
    author, editor, channel_id = channel_members
    message_id = post_message(author, channel_id, "the author's own words")

    refused = _complete_into(editor, channel_id, message_id)

    assert refused.status_code == 403, (
        "write access on a channel let a member aim the model at another member's message and "
        f"overwrite it (#28631): HTTP {refused.status_code} {refused.text}"
    )


def test_a_message_from_another_channel_is_refused(channel_members, upstream):
    author, editor, channel_id = channel_members
    elsewhere = group_channel(editor, author)
    message_id = post_message(editor, elsewhere, "posted in the other channel")

    assert _complete_into(editor, channel_id, message_id).status_code == 403


@pytest.mark.parametrize("caller", ["author", "admin"])
def test_the_author_and_an_admin_may_still_complete_into_the_message(
    channel_members, admin, upstream, caller
):
    author, _, channel_id = channel_members
    message_id = post_message(author, channel_id, "please rewrite me")

    accepted = _complete_into(author if caller == "author" else admin, channel_id, message_id)

    assert accepted.status_code == 200, accepted.text


# the legacy features block honours the memories permission


@pytest.fixture
def account_with_a_memory(admin, make_user, preserve, upstream):
    account = make_user()
    with account.client() as client:
        added = client.post("/api/v1/memories/add", json={"content": STORED_MEMORY, "type": "user"})
    assert added.status_code == 200, added.text
    return account


def _memory_reached_the_provider(account, upstream) -> bool:
    with account.client() as client:
        ask(client, "what do you remember about me?", features={"memory": True})
    return STORED_MEMORY in str(upstream.chat_requests()[-1]["messages"])


def test_a_user_barred_from_memories_gets_none_in_the_prompt(
    account_with_a_memory, admin, permissions, upstream
):
    _save_permissions(
        admin, {**permissions, "features": {**permissions["features"], "memories": False}}
    )

    assert not _memory_reached_the_provider(account_with_a_memory, upstream), (
        "stored memories were injected for a user denied features.memories, just because the "
        "client set the memory flag (#27668)"
    )


def test_a_permitted_user_still_gets_their_memories(account_with_a_memory, upstream):
    assert _memory_reached_the_provider(account_with_a_memory, upstream)


# the legacy function-calling path honours the image generation and web search switches


def _draw_in_chat(admin, listener, preserve, generation_enabled: bool) -> list:
    preserve(IMAGES_CONFIG)
    settings = {**serve_openai_images(listener), "ENABLE_IMAGE_GENERATION": generation_enabled}
    with admin.client() as client:
        save_image_settings(client, **settings)
        ask(client, "draw me a cat", features={"image_generation": True}, params=LEGACY)
    return listener.requests_to("/images/generations")


def test_switched_off_image_generation_draws_nothing(admin, listener, preserve, upstream):
    drawn = _draw_in_chat(admin, listener, preserve, generation_enabled=False)

    assert drawn == [], (
        "a chat generated an image while image generation was switched off, through the legacy "
        "feature path that bypasses the /images routes (#27759)"
    )


def test_switched_on_image_generation_still_draws(admin, listener, preserve, upstream):
    assert len(_draw_in_chat(admin, listener, preserve, generation_enabled=True)) == 1


def test_editing_an_attached_image_still_runs_with_generation_off(
    admin, listener, preserve, upstream
):
    preserve(IMAGES_CONFIG)
    picture = {"type": "image", "url": f"data:image/png;base64,{PNG_BASE64}"}
    with admin.client() as client:
        save_image_settings(
            client, **{**serve_openai_images(listener), "ENABLE_IMAGE_GENERATION": False}
        )
        chat_id, last_id = seed_chat(
            client,
            [
                {"role": "user", "content": "here is my cat", "files": [picture]},
                {"role": "assistant", "content": "What a cat."},
            ],
        )
        ask(
            client,
            "make it blue",
            chat_id=chat_id,
            parent_id=last_id,
            features={"image_generation": True},
            params=LEGACY,
        )

    assert listener.requests_to("/images/edits"), "editing has its own switch and must still run"
    assert listener.requests_to("/images/generations") == []


def _search_in_chat(admin, listener, preserve, search_enabled: bool) -> tuple[list, list]:
    """What the search engine was sent, and the web search steps the reply shows."""
    preserve(RETRIEVAL_CONFIG)
    settings = {
        **serve_search_results(listener, ["https://example.com/weather"]),
        "ENABLE_WEB_SEARCH": search_enabled,
        "BYPASS_WEB_SEARCH_WEB_LOADER": True,
    }
    with admin.client() as client:
        save_web_settings(client, **settings)
        _, answer = ask(
            client, "what is the weather?", features={"web_search": True}, params=LEGACY
        )
    steps = [
        status
        for status in answer.get("statusHistory") or []
        if status.get("action", "").startswith("web_search")
    ]
    return listener.requests_to("/search"), steps


def test_switched_off_web_search_searches_nothing(admin, listener, preserve, upstream):
    searched, steps = _search_in_chat(admin, listener, preserve, search_enabled=False)

    assert steps == [], (
        "a chat ran the web search step while web search was switched off, because the legacy "
        f"branch never read web.search.enable (#27669): {steps}"
    )
    assert searched == []


def test_switched_on_web_search_still_searches(admin, listener, preserve, upstream):
    searched, steps = _search_in_chat(admin, listener, preserve, search_enabled=True)

    assert searched
    assert steps
