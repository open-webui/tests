"""Journey: chats, completions, tool calls and sources read the same with the JSON codec on or off.

`ENABLE_ORJSON` swaps the codec behind request bodies, stored chat JSON, provider payloads, tool
results and socket packets. Two instances share one database, one per value. A chat sent the way
the web client sends it, with accents, CJK, right-to-left text, emoji and a PDF line separator in
the prompt and in a streamed reply with reasoning and usage, is stored and sent to the provider
identically, and a long pasted reply is stored whole. The OpenAI-compatible completions route
answers the same JSON and the same SSE frames, and `/api/models` lists the same presets. A chat
saved through one instance (history with metadata, files, sources) is read, updated, pinned,
archived, tagged, cloned, shared, exported and imported through the other and comes back equal
whichever value wrote it. A workspace tool returning mixed text, numbers, booleans and nulls, a
builtin tool and a tool emitting a citation leave the same tool messages and sources, a file
attached to the chat is cited the same, and a tab receives the same final message over the socket.
Since 6cfd6987e a chat read back names the author of each user message, and a cloned or imported
chat names its account on every message, so the expected history carries those fields.

Discriminates: passes on dev 62f70a844. In backend copies of dev 176d31d1d, the orjson codec
decoding its output as Latin-1 turns the orjson side of every test red except the completions
stream, which orjson request parsing that mangles non-ASCII turns red; the stdlib codec writing
mojibake turns the stdlib side red. No mutation of one value turns a case red that runs only on the
other.
"""

from __future__ import annotations

import dataclasses
import json
import uuid

import httpx
import pytest

from harness import upstream as reply
from harness.actors import Actor, admin_of, create_user
from harness.chat import ask, send_message, wait_for_reply
from harness.json_codecs import (
    ALL_MIXED,
    CODECS,
    CROSSINGS,
    LONG_TEXT,
    MIXED_TEXT,
    codec_pair,
    nested,
)
from harness.python_tools import EVERYONE_READS, python_tool
from harness.socket_client import connected
from harness.upstream import MOCK_MODEL_ID

pytestmark = [
    pytest.mark.journey,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

PIECES = list(MIXED_TEXT.values())
REASONING = f"Erst nachdenken: {MIXED_TEXT['arabic']} {MIXED_TEXT['emoji']}"
USAGE = {"prompt_tokens": 12, "completion_tokens": 34, "total_tokens": 46}
LONG_PIECES = [f"{section}\n\n" for section in LONG_TEXT.split("\n\n")]
TAGS = [MIXED_TEXT["chinese"], "Übersicht", MIXED_TEXT["hebrew"], "plain"]
TAG_IDS = [tag.replace(" ", "_").lower() for tag in TAGS]

TOOL_SOURCE = f'''
class Tools:
    async def quarterly_report(self, region: str, quarter: int, __event_emitter__=None) -> dict:
        """Report the quarter's numbers for a region."""
        await __event_emitter__(
            {{
                "type": "citation",
                "data": {{
                    "document": [{ALL_MIXED!r}],
                    "metadata": [{{"source": {MIXED_TEXT["windows_path"]!r}, "score": 0.75}}],
                    "source": {{"name": {MIXED_TEXT["german"]!r}, "url": "https://example.com/ü"}},
                }},
            }}
        )
        return {{
            "region": region,
            "quarter": quarter,
            "summary": {MIXED_TEXT["markdown"]!r},
            "note": {MIXED_TEXT["pdf_paste"]!r},
            "labels": [{MIXED_TEXT["chinese"]!r}, {MIXED_TEXT["emoji"]!r}],
            "revenue": 1234567.5,
            "units": 42,
            "growth": -0.075,
            "audited": True,
            "restated": False,
            "comment": None,
            "empty": {{}},
        }}
'''
TOOL_ARGUMENTS = {"region": MIXED_TEXT["german"], "quarter": 3}
TOOL_RESULT = {
    "region": MIXED_TEXT["german"],
    "quarter": 3,
    "summary": MIXED_TEXT["markdown"],
    "note": MIXED_TEXT["pdf_paste"],
    "labels": [MIXED_TEXT["chinese"], MIXED_TEXT["emoji"]],
    "revenue": 1234567.5,
    "units": 42,
    "growth": -0.075,
    "audited": True,
    "restated": False,
    "comment": None,
    "empty": {},
}


@pytest.fixture(scope="module")
def pair(instance_with):
    return codec_pair(instance_with)


@pytest.fixture(scope="module")
def owner(pair) -> Actor:
    return create_user(pair["stdlib"])


@pytest.fixture(scope="module")
def admin(pair) -> Actor:
    return admin_of(pair["stdlib"])


def on(pair, actor: Actor, codec: str) -> Actor:
    """The account as it signs in to the instance running `codec`."""
    return dataclasses.replace(actor, base_url=pair[codec].base_url)


def provider(pair):
    """The one provider both instances call: the connection lives in the shared database."""
    return pair["stdlib"].upstream


def on_both(pair, actor: Actor, action) -> dict:
    """`action(client, upstream)` run once per codec; returns each outcome."""
    outcomes = {}
    for codec in CODECS:
        provider(pair).reset()
        with on(pair, actor, codec).client() as client:
            outcomes[codec] = action(client, provider(pair))
    return outcomes


def brief(value) -> str:
    return repr(value)[:600]


def assert_same(outcomes: dict, expected=None):
    # compared as booleans: pytest's diff of a long reply takes minutes
    same = outcomes["stdlib"] == outcomes["orjson"]
    assert same, (
        f"ENABLE_ORJSON off gave {brief(outcomes['stdlib'])}, on gave {brief(outcomes['orjson'])}"
    )
    if expected is not None:
        as_expected = outcomes["orjson"] == expected
        assert as_expected, f"both gave {brief(outcomes['orjson'])}, expected {brief(expected)}"


def message_text(message: dict) -> str:
    items = [item for item in message["output"] if item["type"] == "message"]
    return "".join(part["text"] for item in items for part in item["content"])


def reasoning_text(message: dict) -> list[str]:
    items = [item for item in message["output"] if item["type"] == "reasoning"]
    return [item["content"][0]["text"] for item in items]


def create_chat(client: httpx.Client, chat: dict, **extra) -> str:
    created = client.post("/api/v1/chats/new", json={"chat": chat, **extra})
    assert created.status_code == 200, created.text
    return created.json()["id"]


def read_chat(client: httpx.Client, chat_id: str) -> dict:
    fetched = client.get(f"/api/v1/chats/{chat_id}")
    assert fetched.status_code == 200, fetched.text
    return fetched.json()


def stored_chat_content(title: str = MIXED_TEXT["german"]) -> dict:
    """A conversation as the client saves it: metadata on a message, files and sources."""
    file_entry = {
        "type": "file",
        "id": str(uuid.uuid4()),
        "name": f"{MIXED_TEXT['french']}.pdf",
        "size": 20480,
        "status": "uploaded",
        "meta": nested(),
    }
    source = {
        "source": {"name": MIXED_TEXT["japanese"], "id": "doc-1", "type": "file"},
        "document": [ALL_MIXED],
        "metadata": [nested()],
        "distances": [0.25, 0.5],
    }
    user_message = {
        "id": "u1",
        "parentId": None,
        "childrenIds": ["a1"],
        "role": "user",
        "content": ALL_MIXED,
        "timestamp": 1767225600,
        "models": [MOCK_MODEL_ID],
        "files": [file_entry],
        "meta": nested(),
    }
    assistant_message = {
        "id": "a1",
        "parentId": "u1",
        "childrenIds": [],
        "role": "assistant",
        "content": MIXED_TEXT["markdown"] + MIXED_TEXT["pdf_paste"],
        "model": MOCK_MODEL_ID,
        "done": True,
        "timestamp": 1767225601,
        "sources": [source],
        "usage": {**USAGE, "details": nested()},
        "info": nested(),
    }
    return {
        "title": title,
        "models": [MOCK_MODEL_ID],
        "params": {"temperature": 0.7, "system": MIXED_TEXT["russian"]},
        "files": [file_entry],
        "history": {"currentId": "a1", "messages": {"u1": user_message, "a1": assistant_message}},
    }


def as_read(history: dict, author: Actor, *, copied: bool = False) -> dict:
    """The history as a read returns it: each user message names its author."""
    expected = json.loads(json.dumps(history))
    for message in expected["messages"].values():
        if message["role"] == "user":
            message["user_id"] = author.id
            message["user"] = {"id": author.id, "name": author.name}
        elif copied:
            message["user_id"] = author.id
    return expected


def exported_chats(client: httpx.Client) -> list[dict]:
    exported = client.get("/api/v1/chats/all")
    assert exported.status_code == 200, exported.text
    return [json.loads(line) for line in exported.text.split("\n") if line]


# --- a chat the way the web client sends it -------------------------------------------------------


def test_a_mixed_text_chat_is_stored_and_sent_to_the_provider_the_same(pair, owner):
    def chat(client, upstream):
        upstream.queue(reply.text(PIECES, reasoning=REASONING, usage=USAGE))
        turn, stored = ask(client, ALL_MIXED)
        stored_messages = read_chat(client, turn.chat_id)["chat"]["history"]["messages"]
        [sent] = upstream.chat_requests()
        return {
            "reply": message_text(stored),
            "content": stored["content"],
            "reasoning": reasoning_text(stored),
            "usage": stored["usage"],
            "user": stored_messages[turn.user_message_id]["content"],
            "sent": [entry["content"] for entry in sent["messages"] if entry["role"] == "user"],
            "sent_model": sent["model"],
        }

    outcomes = on_both(pair, owner, chat)

    expected_text = "".join(PIECES).strip()
    assert_same(outcomes)
    outcome = outcomes["orjson"]
    assert outcome["reply"] == expected_text
    assert outcome["content"] == expected_text
    assert outcome["reasoning"] == [REASONING]
    assert outcome["usage"]["total_tokens"] == USAGE["total_tokens"]
    assert outcome["user"] == ALL_MIXED
    assert outcome["sent"] == [ALL_MIXED]


def test_a_long_pasted_reply_is_stored_whole(pair, owner):
    def chat(client, upstream):
        upstream.queue(reply.text(LONG_PIECES))
        _, stored = ask(client, "print the whole document", timeout=120)
        return stored["content"]

    outcomes = on_both(pair, owner, chat)

    assert_same(outcomes, "".join(LONG_PIECES).strip())


# --- the OpenAI-compatible API --------------------------------------------------------------------


def test_the_completions_route_answers_the_same_json(pair, owner):
    def complete(client, upstream):
        upstream.queue(reply.text(PIECES, usage=USAGE))
        body = {
            "model": MOCK_MODEL_ID,
            "messages": [{"role": "user", "content": ALL_MIXED}],
            "stream": False,
        }
        answered = client.post("/api/chat/completions", json=body)
        assert answered.status_code == 200, answered.text
        [sent] = upstream.chat_requests()
        return answered.json(), [entry["content"] for entry in sent["messages"]]

    outcomes = on_both(pair, owner, complete)

    assert_same(outcomes)
    completion, sent = outcomes["orjson"]
    assert completion["choices"][0]["message"]["content"] == "".join(PIECES)
    assert completion["usage"]["total_tokens"] == USAGE["total_tokens"]
    assert sent == [ALL_MIXED]


def test_the_completions_route_streams_frames_that_all_parse(pair, owner):
    def stream(client, upstream):
        upstream.queue(reply.text(PIECES, reasoning=REASONING, usage=USAGE))
        body = {
            "model": MOCK_MODEL_ID,
            "messages": [{"role": "user", "content": ALL_MIXED}],
            "stream": True,
        }
        with client.stream("POST", "/api/chat/completions", json=body) as answered:
            assert answered.status_code == 200
            raw = "".join(answered.iter_text())
        frames = [frame for frame in raw.split("\n\n") if frame]
        assert all(frame.startswith("data: ") for frame in frames), frames
        payloads = [frame[len("data: ") :] for frame in frames]
        assert payloads[-1] == "[DONE]"
        chunks = [json.loads(payload) for payload in payloads[:-1]]
        deltas = [choice["delta"] for chunk in chunks for choice in chunk.get("choices", [])]
        [sent] = upstream.chat_requests()
        return {
            "content": "".join(delta.get("content") or "" for delta in deltas),
            "reasoning": "".join(delta.get("reasoning_content") or "" for delta in deltas),
            "sent": [entry["content"] for entry in sent["messages"]],
        }

    outcomes = on_both(pair, owner, stream)

    expected = {"content": "".join(PIECES), "reasoning": REASONING, "sent": [ALL_MIXED]}
    assert_same(outcomes, expected)


def _preset(codec: str) -> dict:
    return {
        "id": f"preset-{codec}-{uuid.uuid4().hex[:8]}",
        "base_model_id": MOCK_MODEL_ID,
        "name": MIXED_TEXT["german"],
        "meta": {
            "description": ALL_MIXED,
            "tags": [{"name": tag} for tag in TAGS],
            "capabilities": {"vision": True, "citations": False},
            "suggestion_prompts": [{"content": MIXED_TEXT["chinese"], "title": ["a", "b"]}],
        },
        "params": {"temperature": 0.7, "system": MIXED_TEXT["japanese"]},
        "access_grants": [EVERYONE_READS],
    }


def test_the_models_list_shows_the_same_presets(pair, admin):
    presets = {codec: _preset(codec) for codec in CODECS}
    for codec, preset in presets.items():
        with on(pair, admin, codec).client() as client:
            created = client.post("/api/v1/models/create", json=preset)
            assert created.status_code == 200, created.text

    def listed(client, upstream):
        models = client.get("/api/models", params={"refresh": "true"})
        assert models.status_code == 200, models.text
        entries = {model["id"]: model for model in models.json()["data"]}
        seen = {}
        for writer, preset in presets.items():
            stored = client.get("/api/v1/models/model", params={"id": preset["id"]})
            assert stored.status_code == 200, stored.text
            seen[writer] = {
                "listed_name": entries[preset["id"]]["name"],
                "meta": {key: stored.json()["meta"].get(key) for key in preset["meta"]},
                "params": {key: stored.json()["params"].get(key) for key in preset["params"]},
            }
        return seen

    outcomes = on_both(pair, admin, listed)

    template = _preset("any")
    written = {
        "listed_name": template["name"],
        "meta": template["meta"],
        "params": template["params"],
    }
    assert_same(outcomes, {codec: written for codec in CODECS})


# --- stored chats across the switch ---------------------------------------------------------------


@pytest.mark.parametrize(("writer", "reader"), CROSSINGS)
def test_a_saved_chat_reads_back_equal(pair, owner, writer, reader):
    content = stored_chat_content()
    with on(pair, owner, writer).client() as through_writer:
        chat_id = create_chat(through_writer, content, variables={"city": MIXED_TEXT["german"]})
        written = read_chat(through_writer, chat_id)
    with on(pair, owner, reader).client() as through_reader:
        read = read_chat(through_reader, chat_id)
        listed = through_reader.get("/api/v1/chats/", params={"page": 1}).json()
        found = through_reader.get("/api/v1/chats/search", params={"text": "Straße"}).json()

    assert read["chat"]["history"] == as_read(content["history"], owner)
    assert read["chat"]["files"] == content["files"]
    assert read["chat"]["params"] == content["params"]
    assert read["title"] == MIXED_TEXT["german"]
    assert read["chat"] == written["chat"]
    assert chat_id in [chat["id"] for chat in listed]
    assert chat_id in [chat["id"] for chat in found]


@pytest.mark.parametrize(("writer", "reader"), CROSSINGS)
def test_a_chat_updated_through_the_other_instance_keeps_the_new_text(pair, owner, writer, reader):
    content = stored_chat_content()
    with on(pair, owner, writer).client() as through_writer:
        chat_id = create_chat(through_writer, content)
    updated = stored_chat_content(title=MIXED_TEXT["korean"])
    updated["history"]["messages"]["a1"]["content"] = LONG_TEXT
    updated["history"]["messages"]["u1"]["content"] = MIXED_TEXT["emoji"]
    with on(pair, owner, reader).client() as through_reader:
        saved = through_reader.post(f"/api/v1/chats/{chat_id}", json={"chat": updated})
        assert saved.status_code == 200, saved.text
        retitled = through_reader.post(
            f"/api/v1/chats/{chat_id}", json={"chat": {"title": MIXED_TEXT["russian"]}}
        )
        assert retitled.status_code == 200, retitled.text
    with on(pair, owner, writer).client() as through_writer:
        read = read_chat(through_writer, chat_id)

    assert read["title"] == MIXED_TEXT["russian"]
    kept = read["chat"]["history"] == as_read(updated["history"], owner)
    assert kept, f"the history read back differs: {brief(read['chat']['history'])}"


@pytest.mark.parametrize(("writer", "reader"), CROSSINGS)
def test_pinned_tagged_and_archived_chats_read_back_equal(pair, owner, writer, reader):
    content = stored_chat_content()
    with on(pair, owner, writer).client() as through_writer:
        chat_id = create_chat(through_writer, content)
        assert through_writer.post(f"/api/v1/chats/{chat_id}/pin").status_code == 200
    with on(pair, owner, reader).client() as through_reader:
        for tag in TAGS:
            added = through_reader.post(f"/api/v1/chats/{chat_id}/tags", json={"name": tag})
            assert added.status_code == 200, added.text
    with on(pair, owner, writer).client() as through_writer:
        tags = through_writer.get(f"/api/v1/chats/{chat_id}/tags").json()
        all_tags = through_writer.get("/api/v1/chats/all/tags").json()
        tagged = through_writer.post(
            "/api/v1/chats/tags", json={"name": MIXED_TEXT["chinese"]}
        ).json()
        pinned = through_writer.get("/api/v1/chats/pinned").json()
        tagged_read = read_chat(through_writer, chat_id)
        assert through_writer.post(f"/api/v1/chats/{chat_id}/archive").status_code == 200
    with on(pair, owner, reader).client() as through_reader:
        archived = through_reader.get("/api/v1/chats/all/archived").json()
        archived_read = read_chat(through_reader, chat_id)

    assert sorted(tag["name"] for tag in tags) == sorted(TAGS)
    assert set(TAGS) <= {tag["name"] for tag in all_tags}
    assert chat_id in [chat["id"] for chat in tagged]
    assert chat_id in [chat["id"] for chat in pinned]
    assert tagged_read["pinned"] is True
    assert archived_read["archived"] is True
    assert archived_read["chat"]["history"] == as_read(content["history"], owner)
    assert sorted(archived_read["meta"]["tags"]) == sorted(TAG_IDS)
    assert chat_id in [chat["id"] for chat in archived]


@pytest.mark.parametrize(("writer", "reader"), CROSSINGS)
def test_a_cloned_and_a_shared_chat_read_back_equal(pair, owner, writer, reader):
    content = stored_chat_content()
    with on(pair, owner, writer).client() as through_writer:
        chat_id = create_chat(through_writer, content)
        cloned = through_writer.post(
            f"/api/v1/chats/{chat_id}/clone", json={"title": MIXED_TEXT["french"]}
        )
        assert cloned.status_code == 200, cloned.text
        shared = through_writer.post(f"/api/v1/chats/{chat_id}/share")
        assert shared.status_code == 200, shared.text
    share_id = shared.json()["share_id"]
    with on(pair, owner, reader).client() as through_reader:
        clone = read_chat(through_reader, cloned.json()["id"])
        copy = through_reader.get(f"/api/v1/chats/share/{share_id}")
        assert copy.status_code == 200, copy.text
        listed = through_reader.get("/api/v1/chats/shared").json()

    assert clone["title"] == MIXED_TEXT["french"]
    assert clone["chat"]["history"] == as_read(content["history"], owner, copied=True)
    assert copy.json()["chat"]["history"] == as_read(content["history"], owner)
    assert copy.json()["title"] == MIXED_TEXT["german"]
    assert chat_id in [entry["id"] for entry in listed]


@pytest.mark.parametrize(("writer", "reader"), CROSSINGS)
def test_an_exported_chat_imports_back_equal(pair, owner, writer, reader):
    content = stored_chat_content()
    with on(pair, owner, writer).client() as through_writer:
        chat_id = create_chat(through_writer, content)
        through_writer.post(f"/api/v1/chats/{chat_id}/pin")
    with on(pair, owner, reader).client() as through_reader:
        [exported] = [chat for chat in exported_chats(through_reader) if chat["id"] == chat_id]
        imported = through_reader.post(
            "/api/v1/chats/import",
            json={"chats": [{"chat": exported["chat"], "meta": exported["meta"], "pinned": True}]},
        )
        assert imported.status_code == 200, imported.text
    with on(pair, owner, writer).client() as through_writer:
        read = read_chat(through_writer, imported.json()[0]["id"])

    assert exported["chat"]["history"] == content["history"]
    assert read["chat"]["history"] == as_read(content["history"], owner, copied=True)
    assert read["chat"]["files"] == content["files"]
    assert read["title"] == MIXED_TEXT["german"]
    assert read["pinned"] is True


# --- tool calls and their results -----------------------------------------------------------------


def test_a_workspace_tool_result_reaches_the_provider_and_the_chat_the_same(pair, admin, owner):
    with python_tool(on(pair, admin, "stdlib"), TOOL_SOURCE, name="Reports") as tool_id:

        def call(client, upstream):
            upstream.queue(reply.tool_call("quarterly_report", TOOL_ARGUMENTS), reply.text("done"))
            turn, stored = ask(client, "report", tool_ids=[tool_id])
            first, second = upstream.chat_requests()
            [assistant] = [entry for entry in second["messages"] if entry.get("tool_calls")]
            [tool_message] = [entry for entry in second["messages"] if entry["role"] == "tool"]
            calls = [item for item in stored["output"] if item["type"] == "function_call"]
            results = [item for item in stored["output"] if item["type"] == "function_call_output"]
            [output] = results
            return {
                "offered": [tool["function"]["name"] for tool in first["tools"]],
                "sent_arguments": json.loads(assistant["tool_calls"][0]["function"]["arguments"]),
                "sent_result": json.loads(tool_message["content"]),
                "stored_arguments": json.loads(calls[0]["arguments"]),
                "stored_result": json.loads(output["output"][0]["text"]),
                "content": stored["content"],
            }

        outcomes = on_both(pair, owner, call)

    assert_same(outcomes)
    outcome = outcomes["orjson"]
    assert "quarterly_report" in outcome["offered"]
    assert outcome["sent_arguments"] == TOOL_ARGUMENTS
    assert outcome["sent_result"] == TOOL_RESULT
    assert outcome["stored_arguments"] == TOOL_ARGUMENTS
    assert outcome["stored_result"] == TOOL_RESULT
    assert outcome["content"] == "done"


def test_a_builtin_tool_returns_the_same_mixed_text_result(pair, owner):
    def search(client, upstream):
        word = f"heron{uuid.uuid4().hex[:8]}"
        body = f"{word} {ALL_MIXED}"
        created = client.post(
            "/api/v1/notes/create",
            json={
                "title": MIXED_TEXT["german"],
                "data": {"content": {"md": body}},
                "access_grants": [],
            },
        )
        assert created.status_code == 200, created.text
        upstream.queue(reply.tool_call("search_notes", {"query": word}), reply.text("done"))
        ask(client, "look in my notes")
        results = [
            entry["content"]
            for entry in upstream.chat_requests()[-1]["messages"]
            if entry["role"] == "tool"
        ]
        [found] = json.loads(results[-1])
        kept = {key: value for key, value in found.items() if key not in ("id", "updated_at")}
        return {**kept, "snippet": kept["snippet"].replace(word, "WORD")}

    outcomes = on_both(pair, owner, search)

    assert_same(outcomes)
    assert outcomes["orjson"]["title"] == MIXED_TEXT["german"]
    assert "WORD" in outcomes["orjson"]["snippet"]


# --- sources --------------------------------------------------------------------------------------


def test_a_citation_from_a_tool_is_stored_the_same(pair, admin, owner):
    with python_tool(on(pair, admin, "stdlib"), TOOL_SOURCE, name="Cited reports") as tool_id:

        def cite(client, upstream):
            upstream.queue(reply.tool_call("quarterly_report", TOOL_ARGUMENTS), reply.text("done"))
            turn, stored = ask(client, "report with sources", tool_ids=[tool_id])
            chat = read_chat(client, turn.chat_id)["chat"]
            return stored["sources"], chat["history"]["messages"][turn.assistant_message_id][
                "sources"
            ]

        outcomes = on_both(pair, owner, cite)

    assert_same(outcomes)
    sources, reloaded = outcomes["orjson"]
    assert reloaded == sources
    [cited] = [entry for entry in sources if entry["source"].get("name") == MIXED_TEXT["german"]]
    assert cited["document"] == [ALL_MIXED]
    assert cited["metadata"] == [{"source": MIXED_TEXT["windows_path"], "score": 0.75}]


ATTACHED_NAME = "Übersicht_日本語_🚀.txt"
# text extraction folds full-width punctuation and drops joiners, on both values
ATTACHED_KEYS = ("german", "arabic", "russian", "math")
ATTACHED_TEXT = "\n\n".join(MIXED_TEXT[key] for key in ATTACHED_KEYS)


def test_a_file_attached_to_the_chat_is_cited_the_same(pair, owner):
    def cite(client, upstream):
        uploaded = client.post(
            "/api/v1/files/",
            params={"process": "true", "process_in_background": "false"},
            files={"file": (ATTACHED_NAME, ATTACHED_TEXT.encode(), "text/plain")},
        )
        assert uploaded.status_code == 200, uploaded.text
        file_id = uploaded.json()["id"]
        upstream.queue(reply.text("noted", match=reply.answering("Was steht in der Datei?")))
        attached = [{"type": "file", "id": file_id, "name": ATTACHED_NAME}]
        _, stored = ask(client, "Was steht in der Datei?", files=attached)
        sent = "\n".join(
            str(entry["content"]) for entry in upstream.chat_requests()[-1]["messages"]
        )
        [source] = stored["sources"]
        return {
            "cited_file": source["source"]["id"] == file_id,
            "names": sorted({entry["name"] for entry in source["metadata"]}),
            "document": "\n\n".join(source["document"]),
            "text_sent": all(MIXED_TEXT[key] in sent for key in ATTACHED_KEYS),
        }

    outcomes = on_both(pair, owner, cite)

    assert_same(outcomes)
    outcome = outcomes["orjson"]
    assert outcome["cited_file"] and outcome["text_sent"]
    assert outcome["names"] == [ATTACHED_NAME]
    assert all(MIXED_TEXT[key] in outcome["document"] for key in ATTACHED_KEYS)


# --- what a tab receives --------------------------------------------------------------------------


def test_a_tab_receives_the_same_final_message_for_a_streamed_chat(pair, owner):
    outcomes = {}
    for codec in CODECS:
        upstream = provider(pair)
        upstream.reset()
        upstream.queue(reply.text(PIECES, reasoning=REASONING, usage=USAGE))
        actor = on(pair, owner, codec)
        with connected(actor) as tab, actor.client() as client:
            turn = send_message(client, ALL_MIXED, session_id=tab.client.sid)
            final = tab.wait_for(turn.chat_id, "chat:completion", done=True)
            stored = wait_for_reply(client, turn)
        streamed = [
            message_text(event["data"])
            for event in tab.events_of(turn.chat_id)
            if event.get("type") == "chat:completion" and event["data"].get("output")
        ]
        outcomes[codec] = {
            "final": message_text(final["data"]),
            "last_streamed": streamed[-1],
            "stored": stored["content"],
        }

    expected = "".join(PIECES).strip()
    assert_same(outcomes, {"final": expected, "last_streamed": expected, "stored": expected})
