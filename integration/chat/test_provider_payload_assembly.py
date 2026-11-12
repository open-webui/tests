"""What Open WebUI sends the model provider, read off the provider's own request log.

Nine 0.11.1 fixes land on the request body that reaches the provider:

* #28292 (`ff74bfa6a1`, `d22bb6703f`): memory context rendered rows in arrival order, so the
  system prompt changed between turns and prefix caching missed. Sections are now sorted.
* #28241 (`11739a2de8`): a `custom_params` entry on the request replaced every custom param of
  the global defaults instead of merging with them.
* PR #27661 (`fcc130c9b`): only the web client asked for `stream_options.include_usage`, so
  every other caller of a usage-capable model got no token counts.
* `a32a17965c`: replayed history kept its bookkeeping keys (`id`, `usage`, ...).
* `b6dc70c93b`: an Anthropic thinking block without its signature was replayed and rejected;
  reasoning details nested under `provider_specific_fields` were never captured.
* PR #28788 (`2e7df5467`): an attached chat carries an id and no url, so the url filter dropped
  it from `<attached_files>`.
* #28590 (`0b27fa5e87`): llama.cpp's `cache_n` was added on top of a `prompt_tokens` that
  already includes it, compacting a 39k chat as if it held 77k.
* #27603 (`5093a99389`): compaction ran on the configured task model, not the chat's own.
* #28240 (`c4b3e6840f`): the stored model of a reply was dropped on reload, so a previous
  model's provider-bound reasoning was replayed to whatever model came next.
* `3258330729`: the reasoning format of a replayed reply was keyed on a `provider` field Ollama
  models never carry. Each provider now gets stored reasoning in its own field: `thinking` for
  Ollama (pinned in integration/chat/test_ollama_chat.py), `reasoning_content` for llama.cpp
  and none for a strict OpenAI-style provider.
* #28292 (`ff74bfa6a1`): memories sharing a timestamp came back from search in row order; every
  memory sort now breaks ties on the memory id. One update call stamps all its memories with the
  same second, which is how the ties are made.
* PR #27661 (`fcc130c9b`): the usage capability is read before the custom-model fallback
  (`ENABLE_CUSTOM_MODEL_FALLBACK`, an instance of its own) swaps in the default model, so a
  custom model on the fallback still gets token counts by its own capability.
* #33940: since 0.13.1 the skill list and recalled memories reach the model as tool rounds Open
  WebUI adds itself, with no reasoning stored. On a llama.cpp connection those tool call turns
  went out without `reasoning_content`, which DeepSeek's thinking mode rejects whenever tools are
  sent, so the first message of a chat failed. They now carry an empty `reasoning_content`; a
  strict OpenAI-style provider still gets no reasoning field on them.
* #34257: those rounds were added before every reply and saved with it, so the fourth question
  carried four copies of the same skill list, knowledge list and memories; the tools they call
  were never in the request, and the empty text of their assistant turns let proxies insert
  placeholder text. Now a list goes out only when it changed or is no longer among the messages
  sent (first turn, after compaction), memories already in them are not recalled again, the
  three tools are declared and those turns carry no text. Memories still go out only with Memory
  on in the chat and the memories permission, also when a tool of the user's own is named
  recall_memories.

Stored history is created through the chats API and continued the way the web client
continues a chat, so the server rebuilds the payload from what it stored.

Twin of unit/chat/test_provider_payload_assembly.py.

Discriminates: passes on dev bbfa876af and ef67cc3fa. The narrow tests fail with memory sections in
row order, `custom_params` splatted, the include_usage block removed, only `output` stripped from
replayed messages, the thinking-signature filter removed, the other-model reasoning strip removed,
provider-nested reasoning details ignored, chat attachments filtered by url, `cache_n` added to
`prompt_tokens`, and compaction sent to the task model; dropping `model` from the replay keys fails
the two same-model replay cases. `think_tags` as the default reasoning format fails the llama.cpp
and strict-provider tests, `memory.id` dropped from the sort keys fails both tie tests, and the
capability read moved below the fallback swap fails the fallback usage test. Retargeted for
3aacd9ffa and e6ae368f5, where recalled memories reach the model as a recall_memories result Open
WebUI preloads after the user's message; passes on dev da9bca0fc. The memory order test also
pins that round's place (0.13.1): right after the question and out of the system prompt; it passes
on dev e8c0ad85e and fails in a backend copy that puts the preloaded rounds before the question.
The DeepSeek test fails on dev b13d02e09 (the first message gets DeepSeek's 400) and passes on the
fix branch fix-hidden-tool-turns-reasoning-content; with the fix but the stored history replayed
without the llama.cpp reasoning format, the second message fails.
The #34257 once-per-chat, declared-tools and no-text tests fail on dev 2547992b8 (four copies
each, none of the three tools declared, empty text) and pass on the fix branch
fix/skills-list-repeated-every-turn (e5e2fbbf0); the new-skill and compaction tests pass on both.
The own-tool memory test passes on dev 2547992b8 and fails on e5e2fbbf0, which sent the memories
with Memory off and without the permission whenever a tool named recall_memories was enabled.
The two tool turn without text tests (#34264), one replaying a stored reply and one sent by an
Anthropic client, fail on dev a14385816, which sends that turn with an empty text, and pass on
the fix branch fix/litellm-placeholder-after-tool-call.
"""

from __future__ import annotations

import time
import uuid

import pytest

from harness import second_provider
from harness.chat import ChatTurn, send_message, wait_for_reply
from harness.knowledge_bases import knowledge_base
from harness.listener import Answer, ReceivedRequest, json_answer
from harness.python_tools import EVERYONE_READS, python_tool
from harness.upstream import CONTEXT_TOOLS, MOCK_MODEL_ID, context_results, conversation

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

MODELS_CONFIG = ("/api/v1/configs/models", "/api/v1/configs/models")
CHATS_CONFIG = ("/api/v1/chats/config", "/api/v1/chats/config")
TASK_MODEL_ID = "mock-task-model"


def _store_chat(client, messages: list[dict]) -> tuple[str, str]:
    """A chat holding `messages` as one linear branch; returns its id and the last message id."""
    history: dict[str, dict] = {}
    parent_id = None
    for message in messages:
        message_id = message.get("id") or str(uuid.uuid4())
        history[message_id] = {
            "model": MOCK_MODEL_ID,
            **message,
            "id": message_id,
            "parentId": parent_id,
            "childrenIds": [],
            "timestamp": int(time.time()),
        }
        if parent_id:
            history[parent_id]["childrenIds"].append(message_id)
        parent_id = message_id
    chat = {"title": "stored", "models": [MOCK_MODEL_ID], "history": {"messages": history}}
    chat["history"]["currentId"] = parent_id
    created = client.post("/api/v1/chats/new", json={"chat": chat})
    assert created.status_code == 200, created.text
    return created.json()["id"], parent_id


def _continue(client, chat_id: str, parent_id: str, content: str = "and now?", **extra) -> dict:
    turn = send_message(client, content, chat_id=chat_id, parent_id=parent_id, **extra)
    return wait_for_reply(client, turn)


def _replayed_assistant(upstream) -> dict:
    sent = upstream.chat_requests()[-1]["messages"]
    return next(entry for entry in sent if entry["role"] == "assistant")


def _reasoning_reply(detail: dict, model: str = MOCK_MODEL_ID) -> dict:
    summary = [{"type": "output_text", "text": "step one"}]
    answer = {"type": "message", "content": [{"type": "output_text", "text": "hello"}]}
    reasoning = {"type": "reasoning", "summary": summary, "reasoning_details": [detail]}
    return {"role": "assistant", "content": "hello", "model": model, "output": [reasoning, answer]}


# --- #28292: memory context is sorted -------------------------------------------------------


def test_memory_context_lists_memories_in_a_stable_order(make_user, upstream):
    account = make_user()
    with account.client() as client:
        for content in ("Zeta note", "alpha note", "Beta note"):
            added = client.post(
                "/api/v1/memories/add",
                json={"content": content, "type": "user", "path": "work/project"},
            )
            assert added.status_code == 200, added.text
        turn = send_message(client, "what do you remember?", features={"memory": True})
        wait_for_reply(client, turn)

    # recalled memories reach the model as a recall_memories result Open WebUI preloads
    [recalled] = context_results(upstream.chat_requests()[-1], "recall_memories")
    entries = [line[2:] for line in recalled.splitlines() if line.startswith("- ")]
    assert entries == [
        "work/project: alpha note",
        "work/project: Beta note",
        "work/project: Zeta note",
    ], "memory context follows row order, so the cached prompt prefix changes (#28292)"

    # 0.13.1: the recall is a hidden tool round after the question, out of the system prompt
    messages = upstream.chat_requests()[-1]["messages"]
    last_user = max(index for index, entry in enumerate(messages) if entry["role"] == "user")
    after = messages[last_user + 1 :]
    assert [entry["role"] for entry in after] == ["assistant", "tool"], (
        f"the memories do not follow the question as a tool round: {after}"
    )
    assert [call["function"]["name"] for call in after[0]["tool_calls"]] == ["recall_memories"]
    system = " ".join(str(entry["content"]) for entry in messages if entry["role"] == "system")
    assert "alpha note" not in system, "the memories are still in the system prompt"


def test_no_memories_adds_no_memory_context(make_user, upstream):
    with make_user().client() as client:
        wait_for_reply(client, send_message(client, "hello", features={"memory": True}))

    sent = upstream.chat_requests()[-1]
    assert context_results(sent, "recall_memories") == []
    assert [entry["role"] for entry in conversation(sent)] == ["user"]


# --- #28241: custom params from the defaults survive a request override ----------------------


def test_custom_params_merge_with_the_global_defaults(admin, make_user, upstream, preserve):
    preserve(MODELS_CONFIG)
    with admin.client() as client:
        current = client.get(MODELS_CONFIG[0]).json()
        defaults = {"custom_params": {"x_global": "kept", "x_shared": "global"}}
        client.post(MODELS_CONFIG[1], json={**current, "DEFAULT_MODEL_PARAMS": defaults})

    with make_user().client() as client:
        turn = send_message(client, "hello", params={"custom_params": {"x_shared": "request"}})
        wait_for_reply(client, turn)

    sent = upstream.chat_requests()[-1]
    assert sent.get("x_shared") == "request"
    assert sent.get("x_global") == "kept", (
        "one custom param on the request discarded the global custom params (#28241)"
    )


# --- PR #27661: include_usage is requested for a usage-capable model -------------------------


@pytest.fixture
def usage_model(admin):
    model_id = f"usage-model-{uuid.uuid4().hex[:8]}"
    form = {
        "id": model_id,
        "name": "Usage model",
        "base_model_id": MOCK_MODEL_ID,
        "meta": {"capabilities": {"usage": True}},
        "params": {},
    }
    with admin.client() as client:
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
        client.get("/api/models").raise_for_status()  # registers the new model
    yield model_id
    with admin.client() as client:
        client.post("/api/v1/models/model/delete", json={"id": model_id})


def test_a_usage_capable_model_is_asked_for_token_counts(admin, upstream, usage_model):
    with admin.client() as client:
        wait_for_reply(client, send_message(client, "count me", model=usage_model))

    stream_options = upstream.chat_requests()[-1].get("stream_options") or {}
    assert stream_options.get("include_usage") is True, (
        "a caller other than the web client got no token counts for a usage model (PR #27661)"
    )


def test_a_model_without_the_usage_capability_is_not_asked(admin, upstream):
    with admin.client() as client:
        wait_for_reply(client, send_message(client, "no counting"))

    assert "stream_options" not in upstream.chat_requests()[-1]


# --- a32a17965c: replayed history carries no bookkeeping keys -------------------------------


def test_replayed_history_carries_only_payload_fields(make_user, upstream):
    with make_user().client() as client:
        chat_id, last_id = _store_chat(
            client,
            [
                {"role": "user", "content": "first question", "files": []},
                {
                    "role": "assistant",
                    "content": "first answer",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                    "files": [{"type": "file", "id": "f1", "url": "/api/v1/files/f1"}],
                },
            ],
        )
        _continue(client, chat_id, last_id)

    replayed = conversation(upstream.chat_requests()[-1])
    leaked = [sorted(set(entry) - {"role", "content"}) for entry in replayed]
    assert leaked == [[], [], []], f"bookkeeping keys reached the provider: {leaked}"


# --- b6dc70c93b: unsignable reasoning is not replayed --------------------------------------


def test_an_unsigned_anthropic_thinking_block_is_not_replayed(make_user, upstream):
    with make_user().client() as client:
        chat_id, last_id = _store_chat(
            client,
            [
                {"role": "user", "content": "think"},
                _reasoning_reply({"format": "anthropic-claude-v1", "text": "step one"}),
            ],
        )
        _continue(client, chat_id, last_id)

    assert "reasoning_details" not in _replayed_assistant(upstream), (
        "a thinking block without its signature was replayed, which Anthropic rejects"
    )


@pytest.mark.parametrize(
    "detail",
    [
        {"format": "anthropic-claude-v1", "text": "step one", "signature": "sig"},
        {"format": "openai-responses-v1", "text": "step one"},
    ],
    ids=["signed-anthropic", "other-provider"],
)
def test_replayable_reasoning_of_the_same_model_is_replayed(make_user, upstream, detail):
    with make_user().client() as client:
        chat_id, last_id = _store_chat(
            client, [{"role": "user", "content": "think"}, _reasoning_reply(detail)]
        )
        _continue(client, chat_id, last_id)

    assert _replayed_assistant(upstream).get("reasoning_details") == [detail]


# --- #28240: another model's reasoning is not replayed -------------------------------------


def test_another_models_reasoning_is_not_replayed(make_user, upstream):
    signed = {"format": "anthropic-claude-v1", "text": "step one", "signature": "sig"}
    with make_user().client() as client:
        chat_id, last_id = _store_chat(
            client,
            [{"role": "user", "content": "think"}, _reasoning_reply(signed, model="o3-mini")],
        )
        _continue(client, chat_id, last_id)

    assert "reasoning_details" not in _replayed_assistant(upstream), (
        "the previous model's reasoning was replayed to a different model (#28240)"
    )


# --- b6dc70c93b: reasoning details nested by the provider are captured ---------------------

NESTED_MODEL = "nested-reasoning-model"


def test_reasoning_details_nested_by_the_provider_are_kept(admin, listener, preserve):
    preserve(second_provider.OPENAI_CONFIG)
    detail = {"format": "anthropic-claude-v1", "text": "step one", "signature": "sig"}
    with admin.client() as client:
        second_provider.attach(client, listener, NESTED_MODEL)
    nested = {"provider_specific_fields": {"reasoning_details": [detail]}}
    reply = second_provider.sse({"role": "assistant", "content": ""}, nested, {"content": "hello"})
    listener.route("POST", "/v1/chat/completions", reply)

    with admin.client() as client:
        message = wait_for_reply(client, send_message(client, "think", model=NESTED_MODEL))

    stored = [item.get("reasoning_details") for item in message["output"]]
    assert [detail] in stored, f"nested reasoning details were lost: {message['output']}"


# --- PR #28788: an attached chat is listed for the model ----------------------------------


def test_an_attached_chat_is_listed_in_the_attached_files(make_user, upstream):
    attached = {"type": "chat", "id": "chat-abc", "name": "Design notes"}
    with make_user().client() as client:
        chat_id, last_id = _store_chat(
            client,
            [
                {"role": "user", "content": "look at this", "files": [attached]},
                {"role": "assistant", "content": "looking"},
            ],
        )
        _continue(client, chat_id, last_id)

    first_question = upstream.chat_requests()[-1]["messages"][0]["content"]
    assert '<file type="chat" id="chat-abc" name="Design notes"/>' in first_question, (
        f"the attached chat never reached the model (PR #28788): {first_question!r}"
    )


@pytest.mark.parametrize(
    ("attached", "listed"),
    [
        (
            {"type": "file", "id": "f1", "url": "/api/v1/files/f1", "name": "spec.pdf"},
            'id="f1" url="/api/v1/files/f1"',
        ),
        ({"type": "image", "id": "i1", "url": "data:image/png;base64,AAAA"}, None),
    ],
    ids=["file-with-url", "inline-image"],
)
def test_other_attachments_are_listed_only_by_url(make_user, upstream, attached, listed):
    with make_user().client() as client:
        chat_id, last_id = _store_chat(
            client,
            [
                {"role": "user", "content": "look at this", "files": [attached]},
                {"role": "assistant", "content": "looking"},
            ],
        )
        _continue(client, chat_id, last_id)

    first_question = str(upstream.chat_requests()[-1]["messages"][0]["content"])
    if listed:
        assert listed in first_question
    else:
        assert "<attached_files>" not in first_question


# --- #28590 and #27603: when compaction runs, and on which model ---------------------------


@pytest.fixture
def compaction_at(admin, preserve):
    """Turns context compaction on at the given token threshold."""
    preserve(CHATS_CONFIG)

    def enable(threshold: int, model: str = "") -> None:
        with admin.client() as client:
            current = client.get(CHATS_CONFIG[0]).json()
            updated = client.post(
                CHATS_CONFIG[1],
                json={
                    **current,
                    "ENABLE_CONTEXT_COMPACTION": True,
                    "CONTEXT_COMPACTION_TOKEN_THRESHOLD": threshold,
                    "CONTEXT_COMPACTION_TOKEN_CAP": threshold,
                    "CONTEXT_COMPACTION_MODEL": model,
                },
            )
            assert updated.status_code == 200, updated.text

    return enable


def _continue_long_chat(client, last_usage: dict) -> dict:
    chat_id, last_id = _store_chat(
        client,
        [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
            {"role": "assistant", "content": "second answer", "usage": last_usage},
        ],
    )
    return _continue(client, chat_id, last_id, "third question")


def _summary_requests(upstream) -> list[dict]:
    return [request for request in upstream.chat_requests() if not request.get("stream")]


@pytest.mark.parametrize(
    ("usage", "compacts"),
    [
        ({"prompt_tokens": 39000, "completion_tokens": 500, "cache_n": 38000}, False),
        ({"prompt_tokens": 90000, "completion_tokens": 500}, True),
        ({"prompt_n": 40000, "cache_n": 38000}, True),
        ({"prompt_eval_count": 50000, "eval_count": 30000}, True),
        ({"input_tokens": 50000, "output_tokens": 30000}, True),
        ({"prompt_tokens": 50000, "completion_tokens": 10000}, False),
    ],
    ids=[
        "cached-slice-counted-once",
        "oversized",
        "llama-cpp-native-halves",
        "ollama",
        "responses-api",
        "under-the-threshold",
    ],
)
def test_compaction_counts_cached_prompt_tokens_once(
    make_user, upstream, compaction_at, usage, compacts
):
    compaction_at(70000)
    with make_user().client() as client:
        _continue_long_chat(client, usage)

    assert bool(_summary_requests(upstream)) is compacts, (
        f"usage {usage} was measured wrongly against a 70k threshold (#28590)"
    )


@pytest.fixture
def task_model(admin, upstream, preserve):
    """A second provider model, configured as the task model; the next reset drops it."""
    preserve("tasks")
    upstream.models.append(TASK_MODEL_ID)
    time.sleep(1.1)  # the provider model list is cached for a second
    with admin.client() as client:
        assert TASK_MODEL_ID in {model["id"] for model in client.get("/api/models").json()["data"]}
        current = client.get("/api/v1/tasks/config").json()
        client.post(
            "/api/v1/tasks/config/update",
            json={**current, "TASK_MODEL": TASK_MODEL_ID, "TASK_MODEL_EXTERNAL": TASK_MODEL_ID},
        ).raise_for_status()
    return TASK_MODEL_ID


@pytest.mark.parametrize(
    ("compaction_model", "summarised_by"),
    [("", MOCK_MODEL_ID), (TASK_MODEL_ID, TASK_MODEL_ID), ("deleted-model", MOCK_MODEL_ID)],
    ids=["unset-uses-the-chat-model", "explicit-compaction-model", "unknown-falls-back"],
)
def test_compaction_summarises_with_the_chats_own_model(
    admin, upstream, compaction_at, task_model, compaction_model, summarised_by
):
    compaction_at(1000, model=compaction_model)
    with admin.client() as client:
        _continue_long_chat(client, {"prompt_tokens": 5000, "completion_tokens": 10})

    summaries = _summary_requests(upstream)
    assert summaries, "the chat was never compacted"
    assert summaries[0]["model"] == summarised_by, (
        f"compaction ran on {summaries[0]['model']!r} instead of {summarised_by!r} (#27603)"
    )


# --- 3258330729: each provider gets stored reasoning back in its own field ------------------

LLAMA_MODEL = "llama-reasoner"


def _plain_reasoning_reply(model: str) -> dict:
    summary = [{"type": "output_text", "text": "step one"}]
    answer = {"type": "message", "content": [{"type": "output_text", "text": "answer"}]}
    output = [{"type": "reasoning", "summary": summary}, answer]
    return {"role": "assistant", "content": "answer", "model": model, "output": output}


def test_llama_cpp_gets_the_reasoning_back_as_reasoning_content(admin, listener, preserve):
    preserve(second_provider.OPENAI_CONFIG)
    with admin.client() as client:
        second_provider.attach(client, listener, LLAMA_MODEL, provider="llama.cpp")
        listener.route("POST", "/v1/chat/completions", second_provider.sse({"content": "ok"}))
        chat_id, last_id = _store_chat(
            client, [{"role": "user", "content": "think"}, _plain_reasoning_reply(LLAMA_MODEL)]
        )
        _continue(client, chat_id, last_id, model=LLAMA_MODEL)

    sent = listener.requests_to("/v1/chat/completions")[-1].json()["messages"]
    replayed = next(entry for entry in sent if entry["role"] == "assistant")
    assert replayed.get("reasoning_content") == "step one", replayed
    assert replayed["content"] == "answer"
    assert "thinking" not in replayed


def test_a_strict_provider_gets_no_reasoning_field(make_user, upstream):
    with make_user().client() as client:
        chat_id, last_id = _store_chat(
            client, [{"role": "user", "content": "think"}, _plain_reasoning_reply(MOCK_MODEL_ID)]
        )
        _continue(client, chat_id, last_id)

    replayed = _replayed_assistant(upstream)
    assert not {"thinking", "reasoning_content"} & set(replayed), replayed
    assert replayed["content"] == "answer", "the reasoning was pasted into the content"


# --- #33940: the tool rounds Open WebUI adds pass DeepSeek's thinking mode --------------------

DEEPSEEK_MODEL = "deepseek-thinker"
DEEPSEEK_ERROR = "The `reasoning_content` in the thinking mode must be passed back to the API."


def _deepseek_thinking_mode(request: ReceivedRequest) -> Answer:
    """Answers like DeepSeek: with tools sent, every tool call turn must carry reasoning_content."""
    body = request.json()
    unreasoned = [
        entry
        for entry in body["messages"]
        if entry["role"] == "assistant"
        and entry.get("tool_calls")
        and "reasoning_content" not in entry
    ]
    if body.get("tools") and unreasoned:
        return json_answer({"error": {"message": DEEPSEEK_ERROR}}, status=400)
    return second_provider.sse({"reasoning_content": "thinking"}, {"content": "ok"})


def _context_tool_turns(body: dict) -> list[dict]:
    """The assistant turns of `body` that call the context tools Open WebUI adds itself."""
    return [
        entry
        for entry in body["messages"]
        if any(call["function"]["name"] in CONTEXT_TOOLS for call in entry.get("tool_calls") or [])
    ]


@pytest.fixture
def skilled_chat(admin, make_user):
    """`skilled_chat(base_model)`: a user with a memory and a native model with a skill."""
    tag = uuid.uuid4().hex[:8]
    created_models = []
    skill_id = f"release-notes-{tag}"
    with admin.client() as client:
        skill = client.post(
            "/api/v1/skills/create",
            json={
                "id": skill_id,
                "name": f"Release notes {tag}",
                "description": "Writes release notes.",
                "content": "Write release notes in a flat voice.",
                "meta": {},
                "access_grants": [EVERYONE_READS],
            },
        )
        assert skill.status_code == 200, skill.text
    account = make_user()
    with account.client() as client:
        remembered = client.post(
            "/api/v1/memories/add",
            json={"content": "Prefers short answers", "type": "user", "path": "style"},
        )
        assert remembered.status_code == 200, remembered.text

    def chat_model(base_model: str, **meta) -> tuple:
        model_id = f"skilled-{base_model}-{tag}"
        with admin.client() as client:
            if base_model != MOCK_MODEL_ID:
                # a user reaches a custom model only through a base model they can read
                published = client.post(
                    "/api/v1/models/model/access/update",
                    json={"id": base_model, "name": base_model, "access_grants": [EVERYONE_READS]},
                )
                assert published.status_code == 200, published.text
                created_models.append(base_model)
            made = client.post(
                "/api/v1/models/create",
                json={
                    "id": model_id,
                    "base_model_id": base_model,
                    "name": model_id,
                    "meta": {"skillIds": [skill_id], **meta},
                    "params": {"function_calling": "native"},
                    "access_grants": [EVERYONE_READS],
                },
            )
            assert made.status_code == 200, made.text
            client.get("/api/models", params={"refresh": "true"}).raise_for_status()
        created_models.append(model_id)
        return account, model_id

    yield chat_model
    with admin.client() as client:
        for model_id in created_models:
            client.post("/api/v1/models/model/delete", json={"id": model_id})
        client.delete(f"/api/v1/skills/id/{skill_id}/delete")


def test_a_deepseek_thinking_chat_with_a_skill_and_memories_answers(
    admin, listener, preserve, skilled_chat
):
    preserve(second_provider.OPENAI_CONFIG)
    with admin.client() as client:
        second_provider.attach(client, listener, DEEPSEEK_MODEL, provider="llama.cpp")
    listener.route("POST", "/v1/chat/completions", _deepseek_thinking_mode)
    account, model_id = skilled_chat(DEEPSEEK_MODEL)

    with account.client() as client:
        first_turn = send_message(client, "hi there", model=model_id, features={"memory": True})
        first = wait_for_reply(client, first_turn)
        assert not first.get("error"), f"the first message failed on DeepSeek (#33940): {first}"
        second_turn = send_message(
            client,
            "and again",
            model=model_id,
            chat_id=first_turn.chat_id,
            parent_id=first_turn.assistant_message_id,
            features={"memory": True},
        )
        second = wait_for_reply(client, second_turn)

    assert not second.get("error"), f"the second message failed on DeepSeek (#33940): {second}"
    sent = listener.requests_to("/v1/chat/completions")[-1].json()
    names = {
        call["function"]["name"]
        for turn in _context_tool_turns(sent)
        for call in turn["tool_calls"]
    }
    assert {"list_available_skills", "recall_memories"} <= names, (
        f"the skill list and the memories never reached the model as tool rounds: {sorted(names)}"
    )


def test_a_strict_provider_gets_no_reasoning_field_on_added_tool_rounds(upstream, skilled_chat):
    account, model_id = skilled_chat(MOCK_MODEL_ID)
    with account.client() as client:
        wait_for_reply(
            client, send_message(client, "hi there", model=model_id, features={"memory": True})
        )

    turns = _context_tool_turns(upstream.chat_requests()[-1])
    assert turns, "the skill list and the memories never reached the model as tool rounds"
    leaked = [sorted({"thinking", "reasoning_content"} & set(turn)) for turn in turns]
    assert leaked == [[]] * len(turns), f"a strict provider got a reasoning field: {leaked}"


# --- #34257: unchanged context is sent once, by tools the model has ------------------------------


@pytest.fixture
def context_chat(admin, skilled_chat):
    """A user with a memory, and a native model with a skill and a knowledge base attached."""
    name = f"Harbour charts {uuid.uuid4().hex[:6]}"
    with admin.client() as client, knowledge_base(client, name=name) as kb_id:
        knowledge = [{"type": "collection", "id": kb_id, "name": "Harbour charts"}]
        yield skilled_chat(MOCK_MODEL_ID, knowledge=knowledge)


def _chat_turns(client, model_id: str, count: int, turn: ChatTurn | None = None) -> ChatTurn:
    """`count` more questions in the chat of `turn` (a new chat without one); returns the last."""
    for number in range(count):
        turn = send_message(
            client,
            f"question {number + 1}",
            model=model_id,
            chat_id=turn.chat_id if turn else None,
            parent_id=turn.assistant_message_id if turn else None,
            features={"memory": True},
        )
        replied = wait_for_reply(client, turn)
        assert not replied.get("error"), f"the chat failed: {replied}"
    return turn


def _last_chat_request(upstream) -> dict:
    """The chat's own last request, not a compaction summary request."""
    return [body for body in upstream.chat_requests() if body.get("stream")][-1]


def _context_copies(body: dict) -> dict[str, int]:
    return {name: len(context_results(body, name)) for name in sorted(CONTEXT_TOOLS)}


def test_unchanged_skills_knowledge_and_memories_are_sent_once_per_chat(upstream, context_chat):
    account, model_id = context_chat
    with account.client() as client:
        _chat_turns(client, model_id, 4)

    copies = _context_copies(_last_chat_request(upstream))
    assert copies == dict.fromkeys(sorted(CONTEXT_TOOLS), 1), (
        f"the fourth question carries the unchanged lists and memories again (#34257): {copies}"
    )


def test_the_model_has_every_context_tool_it_is_shown_calling(upstream, context_chat):
    account, model_id = context_chat
    with account.client() as client:
        _chat_turns(client, model_id, 1)

    sent = _last_chat_request(upstream)
    shown = {
        call["function"]["name"]
        for turn in _context_tool_turns(sent)
        for call in turn["tool_calls"]
    }
    declared = {tool["function"]["name"] for tool in sent.get("tools") or []}
    assert shown == set(CONTEXT_TOOLS), f"not every context round reached the model: {shown}"
    assert shown <= declared, (
        f"the model is shown calling tools it was not given (#34257): {shown - declared}"
    )


def test_context_tool_turns_carry_no_text(upstream, context_chat):
    account, model_id = context_chat
    with account.client() as client:
        _chat_turns(client, model_id, 2)

    turns = _context_tool_turns(_last_chat_request(upstream))
    assert turns, "the context never reached the model as tool rounds"
    contents = [turn["content"] for turn in turns]
    # an empty string is what proxies fill with placeholder text
    assert contents == [None] * len(turns), f"a context tool turn carries text (#34257): {contents}"


def test_the_models_own_tool_turn_without_text_carries_no_text(make_user, upstream):
    tool_call = {
        "type": "function_call",
        "id": "call_noon",
        "call_id": "call_noon",
        "name": "get_current_timestamp",
        "arguments": "{}",
        "status": "completed",
    }
    tool_result = {
        "type": "function_call_output",
        "id": "fco_noon",
        "call_id": "call_noon",
        "output": [{"type": "input_text", "text": "noon"}],
        "status": "completed",
    }
    answer = {"type": "message", "content": [{"type": "output_text", "text": "It is noon."}]}
    with make_user().client() as client:
        chat_id, last_id = _store_chat(
            client,
            [
                {"role": "user", "content": "what time is it?"},
                {
                    "role": "assistant",
                    "content": "It is noon.",
                    "output": [tool_call, tool_result, answer],
                },
            ],
        )
        _continue(client, chat_id, last_id)

    turns = [entry for entry in upstream.chat_requests()[-1]["messages"] if entry.get("tool_calls")]
    contents = [turn["content"] for turn in turns]
    # LiteLLM 1.81.14 to 1.108 fills an empty string with placeholder text the model then repeats
    assert contents == [None], f"the model's own tool turn carries text (#34264): {contents}"


def test_a_tool_call_without_text_from_an_anthropic_client_carries_no_text(make_user, upstream):
    question = f"what time is it? [{uuid.uuid4().hex[:8]}]"
    request = {
        "model": MOCK_MODEL_ID,
        "max_tokens": 64,
        "tools": [
            {"name": "get_time", "description": "Current time", "input_schema": {"type": "object"}}
        ],
        "messages": [
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_noon", "name": "get_time", "input": {}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_noon", "content": "noon"}
                ],
            },
        ],
    }
    with make_user().client() as client:
        response = client.post("/api/v1/messages", json=request)
    assert response.status_code == 200, response.text

    [sent] = [
        body
        for body in upstream.chat_requests()
        if any(message.get("content") == question for message in body.get("messages", []))
    ]
    contents = [message["content"] for message in sent["messages"] if message.get("tool_calls")]
    assert contents == [None], (
        f"an Anthropic client's tool call turn carries text (#34264): {contents}"
    )


def test_a_new_skill_lists_the_skills_again(admin, upstream, context_chat):
    account, model_id = context_chat
    tag = uuid.uuid4().hex[:8]
    with account.client() as client:
        turn = _chat_turns(client, model_id, 2)
        with admin.client() as admin_client:
            created = admin_client.post(
                "/api/v1/skills/create",
                json={
                    "id": f"tide-tables-{tag}",
                    "name": f"Tide tables {tag}",
                    "description": "Reads tide tables.",
                    "content": "Read the tide tables.",
                    "meta": {},
                    "access_grants": [EVERYONE_READS],
                },
            )
            assert created.status_code == 200, created.text
        try:
            _chat_turns(client, model_id, 1, turn)
        finally:
            with admin.client() as admin_client:
                admin_client.delete(f"/api/v1/skills/id/tide-tables-{tag}/delete")

    listed = context_results(_last_chat_request(upstream), "list_available_skills")
    assert f"Tide tables {tag}" in listed[-1], (
        f"the new skill is missing from the latest list: {listed}"
    )


def test_compaction_lists_the_context_again(admin, preserve, upstream, context_chat):
    account, model_id = context_chat
    preserve(CHATS_CONFIG)
    with account.client() as client:
        turn = _chat_turns(client, model_id, 3)
        with admin.client() as admin_client:
            current = admin_client.get(CHATS_CONFIG[0]).json()
            compacting = {
                **current,
                "ENABLE_CONTEXT_COMPACTION": True,
                "CONTEXT_COMPACTION_TOKEN_THRESHOLD": 1,
                "CONTEXT_COMPACTION_TOKEN_CAP": 1,
                "CONTEXT_COMPACTION_RETENTION_PERCENTAGE": 40,
            }
            assert admin_client.post(CHATS_CONFIG[1], json=compacting).status_code == 200
        _chat_turns(client, model_id, 1, turn)

    assert [body for body in upstream.chat_requests() if not body.get("stream")], (
        "nothing was compacted"
    )
    copies = _context_copies(_last_chat_request(upstream))
    missing = [name for name, count in copies.items() if not count]
    assert not missing, f"after compaction the model no longer has {missing}: {copies}"


OWN_RECALL_TOOL = '''
class Tools:
    async def recall_memories(self) -> str:
        """Retrieve all stored memories for the user."""
        return "nothing stored here"
'''


@pytest.mark.parametrize("gate", ["memory switched off", "memories not permitted"])
def test_an_own_recall_memories_tool_sends_no_memories_past_the_gate(
    admin, preserve, upstream, context_chat, gate
):
    account, model_id = context_chat
    features = {"memory": True}
    if gate == "memory switched off":
        features = {}
    else:
        preserve("permissions")
        with admin.client() as client:
            current = client.get("/api/v1/users/default/permissions").json()
            barred = {**current, "features": {**current["features"], "memories": False}}
            client.post("/api/v1/users/default/permissions", json=barred).raise_for_status()
    with (
        python_tool(admin, OWN_RECALL_TOOL, name="Memory mimic") as tool_id,
        account.client() as client,
    ):
        turn = send_message(client, "hello", model=model_id, tool_ids=[tool_id], features=features)
        replied = wait_for_reply(client, turn)
        assert not replied.get("error"), f"the chat failed: {replied}"

    recalled = context_results(_last_chat_request(upstream), "recall_memories")
    assert recalled == [], f"memories reached the model with {gate}: {recalled}"


# --- #28292: memories sharing a timestamp come out in one order ------------------------------

TIED_PATH = "work/tied"


def _add_in_one_batch(client, contents: list[str]) -> list[dict]:
    """Memories added by one update call, which stamps them all with the same second."""
    operations = [
        {"action": "add", "content": content, "type": "user", "path": TIED_PATH}
        for content in contents
    ]
    applied = client.post("/api/v1/memories/update", json={"operations": operations})
    assert applied.status_code == 200, applied.text
    return [result["memory"] for result in applied.json()]


@pytest.fixture
def tied_memories(make_user):
    """A client and six memories on one path, all updated in the same second."""
    with make_user().client() as client:
        added = _add_in_one_batch(client, [f"tied note {index}" for index in range(6)])
        assert len({memory["updated_at"] for memory in added}) == 1, added
        yield client, added


def _by_id(memories: list[dict]) -> list[str]:
    return sorted(memory["id"] for memory in memories)


def test_memory_search_breaks_timestamp_ties_on_the_memory_id(tied_memories):
    client, added = tied_memories

    found = client.post("/api/v1/memories/search", json={"path": TIED_PATH})

    assert found.status_code == 200, found.text
    assert [memory["id"] for memory in found.json()] == _by_id(added), (
        "memories sharing a timestamp came back in row order, not by id (#28292)"
    )


def test_reading_a_memory_path_breaks_timestamp_ties_on_the_memory_id(tied_memories):
    client, added = tied_memories

    read = client.post("/api/v1/memories/path", json={"path": TIED_PATH})

    assert read.status_code == 200, read.text
    assert [memory["id"] for memory in read.json()["memories"]] == _by_id(added), (
        "memories sharing a timestamp came back in row order, not by id (#28292)"
    )


# --- PR #27661: a fallen-back custom model keeps its own usage capability ----------------

FALLBACK_ENV = {"ENABLE_CUSTOM_MODEL_FALLBACK": "true", "DEFAULT_MODELS": MOCK_MODEL_ID}


@pytest.fixture(scope="module")
def fallback_instance(instance_with):
    """An instance where a custom model whose base model is gone falls back to the default."""
    return instance_with(FALLBACK_ENV)


def _orphaned_model(instance, usage: bool) -> str:
    """A custom model on a base model that no connection serves, with the usage capability."""
    model_id = f"orphaned-{uuid.uuid4().hex[:8]}"
    form = {
        "id": model_id,
        "name": "Orphaned preset",
        "base_model_id": "retired-base-model",
        "meta": {"capabilities": {"usage": usage}},
        "params": {},
    }
    with instance.client() as client:
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
        client.get("/api/models").raise_for_status()
    return model_id


@pytest.mark.slow
@pytest.mark.parametrize("usage", [True, False], ids=["usage-model", "plain-model"])
def test_a_model_on_the_fallback_is_asked_for_token_counts_by_its_own_capability(
    fallback_instance, usage
):
    model_id = _orphaned_model(fallback_instance, usage)
    fallback_instance.upstream.reset()
    with fallback_instance.client() as client:
        answered = client.post(
            "/api/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

    assert answered.status_code == 200, answered.text
    sent = fallback_instance.upstream.chat_requests()[-1]
    assert sent["model"] == MOCK_MODEL_ID, "the request did not fall back to the default model"
    asked = (sent.get("stream_options") or {}).get("include_usage") is True
    assert asked is usage, (
        "the fallback model's capability decided whether token counts were requested, not the "
        "custom model's own (PR #27661)"
    )
