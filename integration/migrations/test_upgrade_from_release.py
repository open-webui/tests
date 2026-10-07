"""Journey: data written by earlier releases survives starting the checkout on it.

Each data set under `upgrade_data/` was made by booting a release (the last patch of the three
most recent minor lines) and filling it through its own API: four accounts, a group with a
permission of its own, a chat with a regenerated reply, a tool call and an attached file, a chat
the server streamed itself, one it answered from a knowledge base with its citation, folders,
a shared chat, a note, a knowledge base with a file, a model preset, prompts, a tool, a filter
function with valves, memories, a channel with a thread and a reaction, feedback and settings
changed from their defaults, a default model among them. Its manifest records what was made and
for whom. The checkout boots on a copy of it (`harness.upgraded_release`), running every migration
since that release, and each account signs in with its old password and finds its data intact,
while access grants still apply to the right accounts. The Postgres data sets restore a `pg_dump`
into an embedded server first. `scripts/seed_upgrade_data.py` regenerates the data sets; the
browser twin, `e2e/migrations/test_upgrade_from_release.py`, opens the same data in the app.

Discriminates: passes on dev ac00d40e3 for all six data sets; with the copy step of
`3ff2c63645b8` (config reshape) skipping the `ui.` keys in a copy of it, both v0.9.6 sets fail
the settings test (default user role back to `pending`); with `b0018471bbbe` no longer adding
`user.variables`, the v0.9.6 and v0.10.2 sets fail to boot (`no such column`) while the v0.11.4
sets still pass. Every old group is top-level after the upgrade and can then be nested under a new
group, whose grants its members inherit; with `b8e4f0a3c752` not adding the `parent_group_id`
column in a copy of it, the two group tests error at setup on every set (the admin cannot sign
in).
"""

from __future__ import annotations

from typing import Iterator

import httpx
import pytest

from harness.access import grant
from harness.knowledge_bases import knowledge_base
from harness.upgraded_release import Upgraded, data_set_params, upgraded_release

pytestmark = [
    pytest.mark.journey,
    pytest.mark.slow,
    pytest.mark.api,
    pytest.mark.requires_source,
]


@pytest.fixture(scope="module", params=data_set_params())
def upgraded(request, tmp_path_factory) -> Iterator[Upgraded]:
    name = request.param
    with upgraded_release(name, tmp_path_factory.mktemp(name)) as release:
        yield release


def _refused(response: httpx.Response) -> bool:
    return response.status_code in (401, 403, 404)


def test_every_account_signs_in_with_its_old_password(upgraded):
    for who, account in upgraded.manifest["accounts"].items():
        signed_in = upgraded.sign_in(who)
        assert signed_in.status_code == 200, f"{who} cannot sign in: {signed_in.text}"
        session = signed_in.json()
        assert (session["id"], session["role"]) == (account["id"], account["role"])
        assert (session["name"], session["email"]) == (account["name"], account["email"])


def test_changed_settings_are_kept(upgraded):
    expected = upgraded.manifest["settings"]
    config = upgraded.get("admin", "/api/v1/auths/admin/config").json()
    for key, value in expected["admin"].items():
        assert config.get(key) == value, f"admin setting {key} is {config.get(key)!r}"

    permissions = upgraded.get("admin", "/api/v1/users/default/permissions").json()
    for dotted, value in expected["permissions"].items():
        section, key = dotted.split(".")
        assert permissions[section][key] == value, f"default permission {dotted} changed"

    banners = upgraded.get("alice", "/api/v1/configs/banners").json()
    assert [{key: banner.get(key) for key in expected["banner"]} for banner in banners] == [
        expected["banner"]
    ]

    alice_settings = upgraded.get("alice", "/api/v1/users/user/settings").json()
    assert alice_settings["ui"] == expected["alice_ui"]


def test_the_group_keeps_its_members(upgraded):
    group = upgraded.manifest["group"]
    with upgraded.client("admin") as client:
        members = client.post(f"/api/v1/groups/id/{group['id']}/users")
    assert members.status_code == 200, members.text
    accounts = upgraded.manifest["accounts"]
    assert {member["id"] for member in members.json()} == {
        accounts[who]["id"] for who in group["members"]
    }


def test_the_old_group_is_top_level_and_lists_as_a_direct_membership(upgraded):
    group = upgraded.manifest["group"]
    with upgraded.client("admin") as client:
        stored = client.get(f"/api/v1/groups/id/{group['id']}")
    assert stored.status_code == 200, stored.text
    assert stored.json()["parent_group_id"] is None
    assert stored.json()["member_count"] == len(group["members"])
    for who in group["members"]:
        with upgraded.client(who) as client:
            marked = client.get("/api/v1/users/groups", params={"include_inherited": "true"}).json()
        assert [(entry["id"], entry["membership_type"]) for entry in marked] == [
            (group["id"], "direct")
        ]


def test_the_old_group_can_be_nested_and_its_members_inherit_from_the_new_parent(upgraded):
    group, handbook = upgraded.manifest["group"], upgraded.manifest["knowledge"]
    accounts = upgraded.manifest["accounts"]
    with upgraded.client("admin") as admin:
        parent = admin.post(
            "/api/v1/groups/create", json={"name": "Institute", "description": "the new parent"}
        )
        assert parent.status_code == 200, parent.text
        parent_id = parent.json()["id"]
        admin.post(
            f"/api/v1/groups/id/{parent_id}/users/add", json={"user_ids": [accounts["carol"]["id"]]}
        ).raise_for_status()
        shared_with_parent = [grant("group", parent_id, "read")]
        with knowledge_base(admin, "Institute notes", shared_with_parent) as notes_id:
            nested = admin.post(
                f"/api/v1/groups/id/{group['id']}/update",
                json={"name": group["name"], "description": "", "parent_group_id": parent_id},
            )
            assert nested.status_code == 200, nested.text
            assert nested.json()["parent_group_id"] == parent_id

            for who in group["members"]:
                opened = upgraded.get(who, f"/api/v1/knowledge/{notes_id}")
                assert opened.status_code == 200, f"{who} does not inherit from the parent group"
                kept = upgraded.get(who, f"/api/v1/knowledge/{handbook['id']}")
                assert kept.status_code == 200, f"{who} lost the grant the old group held"
            assert upgraded.get("carol", f"/api/v1/knowledge/{notes_id}").status_code == 200
            assert _refused(upgraded.get("carol", f"/api/v1/knowledge/{handbook['id']}")), (
                "the parent's member reads what only the subgroup holds"
            )
            members = admin.get(
                f"/api/v1/groups/id/{parent_id}/members", params={"membership": "inherited"}
            ).json()
            assert {item["id"] for item in members["items"]} == {
                accounts[who]["id"] for who in group["members"]
            }

        assert admin.delete(f"/api/v1/groups/id/{parent_id}/delete").json() is True
        lifted = admin.get(f"/api/v1/groups/id/{group['id']}").json()
        assert lifted["parent_group_id"] is None


def _messages(chat: dict) -> dict:
    return {
        message_id: {
            "parent": message.get("parentId"),
            "children": message.get("childrenIds", []),
            "role": message["role"],
            "content": message["content"],
        }
        for message_id, message in chat["chat"]["history"]["messages"].items()
    }


@pytest.mark.parametrize("which", ["branched", "live", "cited", "archived"])
def test_each_chat_opens_with_every_message_and_branch(upgraded, which):
    expected = upgraded.manifest["chats"][which]
    opened = upgraded.get("alice", f"/api/v1/chats/{expected['id']}")
    assert opened.status_code == 200, opened.text
    chat = opened.json()
    assert chat["title"] == expected["title"]
    if "messages" in expected:
        assert _messages(chat) == expected["messages"]
        assert chat["chat"]["history"]["currentId"] == expected["current_id"], (
            "the chat no longer opens on the message the user was last on"
        )


def test_chats_keep_their_folder_pin_tag_and_archive(upgraded):
    chats = upgraded.manifest["chats"]
    folders = upgraded.manifest["folders"]
    branched_id = chats["branched"]["id"]

    listed = upgraded.get("alice", "/api/v1/folders/").json()
    by_id = {folder["id"]: folder for folder in listed}
    assert by_id[folders["parent"]["id"]]["name"] == folders["parent"]["name"]
    assert by_id[folders["child"]["id"]]["name"] == folders["child"]["name"]
    assert by_id[folders["child"]["id"]]["parent_id"] == folders["parent"]["id"]

    opened = upgraded.get("alice", f"/api/v1/chats/{branched_id}").json()
    assert opened["folder_id"] == folders["child"]["id"]
    pinned = upgraded.get("alice", "/api/v1/chats/pinned").json()
    assert branched_id in {chat["id"] for chat in pinned}
    tags = upgraded.get("alice", f"/api/v1/chats/{branched_id}/tags").json()
    assert "travel" in {tag["name"] for tag in tags}
    archived = upgraded.get("alice", "/api/v1/chats/archived").json()
    assert chats["archived"]["id"] in {chat["id"] for chat in archived}

    listed_chats = upgraded.get("alice", "/api/v1/chats/list").json()
    assert chats["live"]["id"] in {chat["id"] for chat in listed_chats}


def test_the_file_attached_to_a_chat_still_downloads(upgraded):
    attached = upgraded.manifest["chat_file"]
    content = upgraded.get("alice", f"/api/v1/files/{attached['id']}/content")
    assert content.status_code == 200, content.text
    assert content.text == attached["text"]
    assert _refused(upgraded.get("carol", f"/api/v1/files/{attached['id']}/content"))


def test_the_shared_chat_opens_only_for_the_account_it_was_shared_with(upgraded):
    branched = upgraded.manifest["chats"]["branched"]
    shared = upgraded.get("bob", f"/api/v1/chats/share/{branched['share_id']}")
    assert shared.status_code == 200, shared.text
    assert shared.json()["title"] == branched["title"]
    assert _refused(upgraded.get("carol", f"/api/v1/chats/share/{branched['share_id']}"))


def test_the_note_keeps_its_text_and_its_writer(upgraded):
    note = upgraded.manifest["note"]
    for who in ("alice", "bob"):
        opened = upgraded.get(who, f"/api/v1/notes/{note['id']}")
        assert opened.status_code == 200, f"{who}: {opened.text}"
        body = opened.json()
        assert body["title"] == note["title"]
        assert body["data"]["content"]["md"] == note["md"]
    assert upgraded.get("bob", f"/api/v1/notes/{note['id']}").json()["write_access"] is True
    assert _refused(upgraded.get("carol", f"/api/v1/notes/{note['id']}"))


def test_the_knowledge_base_keeps_its_file_and_answers_searches(upgraded):
    knowledge = upgraded.manifest["knowledge"]
    for who in knowledge["readers"]:
        opened = upgraded.get(who, f"/api/v1/knowledge/{knowledge['id']}")
        assert opened.status_code == 200, f"{who}: {opened.text}"
        assert opened.json()["name"] == knowledge["name"]
        files = upgraded.get(who, f"/api/v1/knowledge/{knowledge['id']}/files").json()
        assert knowledge["file_id"] in {item["id"] for item in files["items"]}
    assert _refused(upgraded.get("carol", f"/api/v1/knowledge/{knowledge['id']}"))

    with upgraded.client("bob") as client:
        found = client.post(
            "/api/v1/retrieval/query/collection",
            json={"collection_names": [knowledge["id"]], "query": "Where is the key?", "k": 4},
        )
    assert found.status_code == 200, found.text
    documents = [text for batch in found.json()["documents"] for text in batch]
    assert knowledge["text"] in documents


def _prompt_commands(upgraded: Upgraded, who: str) -> dict[str, dict]:
    listed = upgraded.get(who, "/api/v1/prompts/list").json()
    return {prompt["command"].lstrip("/"): prompt for prompt in listed["items"]}


def test_prompts_keep_their_content_and_readers(upgraded):
    for command, prompt in upgraded.manifest["prompts"].items():
        visible_to = {prompt["owner"], *prompt["readers"]}
        for who in ("alice", "bob", "carol"):
            listed = _prompt_commands(upgraded, who)
            if who in visible_to:
                assert command in listed, f"{who} lost the prompt {command}"
                opened = upgraded.get(who, f"/api/v1/prompts/id/{listed[command]['id']}")
                assert opened.status_code == 200, opened.text
                assert opened.json()["content"] == prompt["content"]
                assert opened.json()["name"] == prompt["name"]
            else:
                assert command not in listed, f"{who} can now see the prompt {command}"


def test_the_tool_still_loads_for_its_readers_only(upgraded):
    tool = upgraded.manifest["tool"]
    source = upgraded.get("admin", f"/api/v1/tools/id/{tool['id']}")
    assert source.status_code == 200, source.text
    assert "def get_weather" in source.json()["content"]
    spec = upgraded.get("admin", f"/api/v1/tools/id/{tool['id']}/valves/spec")
    assert spec.status_code == 200, f"the tool no longer loads: {spec.text}"
    assert "units" in spec.json()["properties"]
    for who in tool["readers"]:
        listed = upgraded.get(who, "/api/v1/tools/").json()
        assert tool["id"] in {item["id"] for item in listed}, f"{who} lost the tool"
    assert tool["id"] not in {item["id"] for item in upgraded.get("carol", "/api/v1/tools/").json()}


def test_the_function_stays_active_global_and_keeps_its_valves(upgraded):
    function = upgraded.manifest["function"]
    stored = upgraded.get("admin", f"/api/v1/functions/id/{function['id']}").json()
    assert (stored["name"], stored["is_active"], stored["is_global"]) == (
        function["name"],
        True,
        True,
    )
    valves = upgraded.get("admin", f"/api/v1/functions/id/{function['id']}/valves").json()
    assert valves == function["valves"]
    spec = upgraded.get("admin", f"/api/v1/functions/id/{function['id']}/valves/spec")
    assert spec.status_code == 200, f"the function no longer loads: {spec.text}"
    assert "suffix" in spec.json()["properties"]


def test_the_model_preset_keeps_its_prompt_and_its_reader(upgraded):
    model = upgraded.manifest["model"]
    for who in ("admin", *model["readers"]):
        opened = upgraded.get(who, "/api/v1/models/model", params={"id": model["id"]})
        assert opened.status_code == 200, f"{who}: {opened.text}"
        assert opened.json()["name"] == model["name"]
    # readers without write access are shown the preset without its parameters
    stored = upgraded.get("admin", "/api/v1/models/model", params={"id": model["id"]}).json()
    assert stored["params"]["system"] == model["system"]
    for who in ("alice", "carol"):
        opened = upgraded.get(who, "/api/v1/models/model", params={"id": model["id"]})
        assert _refused(opened), f"{who} can now open the preset shared with bob only"


def test_memories_are_kept_for_their_owner(upgraded):
    memories = upgraded.manifest["memories"]
    listed = upgraded.get(memories["owner"], "/api/v1/memories/").json()
    assert sorted(memory["content"] for memory in listed) == sorted(memories["contents"])
    assert upgraded.get("carol", "/api/v1/memories/").json() == []


def test_the_channel_keeps_its_thread_and_reaction(upgraded):
    channel = upgraded.manifest["channel"]
    for who in channel["readers"]:
        listed = upgraded.get(who, "/api/v1/channels/").json()
        assert channel["id"] in {item["id"] for item in listed}, f"{who} lost the channel"
    assert _refused(upgraded.get("carol", f"/api/v1/channels/{channel['id']}"))

    messages = upgraded.get("bob", f"/api/v1/channels/{channel['id']}/messages").json()
    top = {message["id"]: message for message in messages}
    assert top[channel["message"]["id"]]["content"] == channel["message"]["content"]
    reactions = {reaction["name"] for reaction in top[channel["message"]["id"]]["reactions"]}
    assert channel["reaction"] in reactions

    thread_path = f"/api/v1/channels/{channel['id']}/messages/{channel['message']['id']}/thread"
    thread = upgraded.get("alice", thread_path).json()
    assert channel["reply"]["content"] in {message["content"] for message in thread}


def test_feedback_is_kept(upgraded):
    feedback = upgraded.manifest["feedback"]
    stored = upgraded.get("alice", f"/api/v1/evaluations/feedback/{feedback['id']}")
    assert stored.status_code == 200, stored.text
    data = stored.json()["data"]
    assert (data["rating"], data["reason"]) == (feedback["rating"], feedback["reason"])
