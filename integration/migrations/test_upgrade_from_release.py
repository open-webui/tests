"""Journey: data written by earlier releases survives starting the checkout on it.

Each data set under `upgrade_data/` was made by booting a release (the last patch of the three
most recent minor lines) and filling it through its own API: four accounts, a group, a chat with
a regenerated reply, a tool call and an attached file, a chat the server streamed itself, folders,
a shared chat, a note, a knowledge base with a file, a model preset, prompts, a tool, a filter
function with valves, memories, a channel with a thread and a reaction, feedback and settings
changed from their defaults. Its manifest records what was made and for whom. The checkout boots
on a copy of it, running every migration since that release, and each account signs in with its
old password and finds its data intact, while access grants still apply to the right accounts.
The Postgres data sets restore a `pg_dump` into an embedded server first.
`scripts/seed_upgrade_data.py` regenerates the data sets.

Discriminates: passes on dev ac00d40e3 for all six data sets; with the copy step of
`3ff2c63645b8` (config reshape) skipping the `ui.` keys in a copy of it, both v0.9.6 sets fail
the settings test (default user role back to `pending`); with `b0018471bbbe` no longer adding
`user.variables`, the v0.9.6 and v0.10.2 sets fail to boot (`no such column`) while the v0.11.4
sets still pass.
"""

from __future__ import annotations

import contextlib
import json
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import httpx
import pytest
import sqlalchemy

from harness import upstream as upstream_module
from harness.prepared_data import RunningBackend, restored_postgres, serving

pytestmark = [
    pytest.mark.journey,
    pytest.mark.slow,
    pytest.mark.api,
    pytest.mark.requires_source,
]

DATA_SETS = Path(__file__).parent / "upgrade_data"


def _data_set_params() -> list:
    params = []
    for manifest in sorted(DATA_SETS.glob("*.json")):
        name = manifest.stem
        marks = [pytest.mark.requires_postgres] if name.endswith("-postgres") else []
        params.append(pytest.param(name, marks=marks, id=name))
    return params


@dataclass
class Upgraded:
    backend: RunningBackend
    manifest: dict
    tokens: dict[str, str] = field(default_factory=dict)

    def sign_in(self, who: str) -> httpx.Response:
        account = self.manifest["accounts"][who]
        return httpx.post(
            f"{self.backend.base_url}/api/v1/auths/signin",
            json={"email": account["email"], "password": account["password"]},
            timeout=60.0,
        )

    def client(self, who: str) -> httpx.Client:
        if who not in self.tokens:
            signed_in = self.sign_in(who)
            assert signed_in.status_code == 200, f"{who} cannot sign in: {signed_in.text}"
            self.tokens[who] = signed_in.json()["token"]
        return self.backend.client(self.tokens[who])

    def get(self, who: str, path: str, **options) -> httpx.Response:
        with self.client(who) as client:
            return client.get(path, **options)


@pytest.fixture(scope="module", params=_data_set_params())
def upgraded(request, tmp_path_factory) -> Iterator[Upgraded]:
    name = request.param
    manifest = json.loads((DATA_SETS / f"{name}.json").read_text(encoding="utf-8"))
    root = tmp_path_factory.mktemp(name)
    data_dir = root / "data"
    with tarfile.open(DATA_SETS / f"{name}.tar.gz") as archive:
        archive.extractall(data_dir, filter="data")
    with contextlib.ExitStack() as stack:
        provider, shutdown = upstream_module.serve()
        stack.callback(shutdown)
        settings = {
            "WEBUI_AUTH": "true",
            "RAG_EMBEDDING_ENGINE": "openai",
            "RAG_OPENAI_API_BASE_URL": provider.base_url,
            "RAG_OPENAI_API_KEY": "sk-mock",
        }
        if manifest["engine"] == "postgres":
            dump = data_dir / "webui.sql"
            database_url = stack.enter_context(restored_postgres(dump, root))
            settings["DATABASE_URL"] = database_url
            _run_sql(database_url, _RELOCATE_UPLOADS, manifest["data_dir"], str(data_dir))
        else:
            database_url = f"sqlite:///{data_dir / 'webui.db'}"
            _run_sql(database_url, _RELOCATE_UPLOADS, manifest["data_dir"], str(data_dir))
        backend = stack.enter_context(serving(data_dir, settings))
        upgraded = Upgraded(backend, manifest)
        _point_embeddings_at(upgraded, provider.base_url)
        yield upgraded


# uploads are stored by absolute path; a real upgrade keeps its data directory where it was
_RELOCATE_UPLOADS = "UPDATE file SET path = REPLACE(path, :old, :new)"


def _run_sql(database_url: str, statement: str, old: str, new: str) -> None:
    engine = sqlalchemy.create_engine(database_url)
    try:
        with engine.begin() as connection:
            connection.execute(sqlalchemy.text(statement), {"old": old, "new": new})
    finally:
        engine.dispose()


def _point_embeddings_at(upgraded: Upgraded, base_url: str) -> None:
    """The embedding endpoint saved at seeding time is gone; the admin points it at this one."""
    with upgraded.client("admin") as client:
        current = client.get("/api/v1/retrieval/embedding")
        current.raise_for_status()
        form = {
            **current.json(),
            "RAG_EMBEDDING_ENGINE": "openai",
            "openai_config": {"url": base_url, "key": "sk-mock"},
        }
        client.post("/api/v1/retrieval/embedding/update", json=form).raise_for_status()


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


@pytest.mark.parametrize("which", ["branched", "live", "archived"])
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
