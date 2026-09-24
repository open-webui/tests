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

Stored history is created through the chats API and continued the way the web client
continues a chat, so the server rebuilds the payload from what it stored.

Twin of unit/chat/test_provider_payload_assembly.py.

Discriminates: passes on dev bbfa876af. The narrow tests fail with memory sections in row order,
`custom_params` splatted, the include_usage block removed, only `output` stripped from replayed
messages, the thinking-signature filter removed, the other-model reasoning strip removed,
provider-nested reasoning details ignored, chat attachments filtered by url, `cache_n` added to
`prompt_tokens`, and compaction sent to the task model; dropping `model` from the replay keys
fails the two same-model replay cases.
"""

from __future__ import annotations

import time
import uuid

import pytest

from harness import second_provider
from harness.chat import send_message, wait_for_reply
from harness.upstream import MOCK_MODEL_ID

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

    system = upstream.chat_requests()[-1]["messages"][0]
    assert system["role"] == "system"
    entries = [line[2:] for line in system["content"].splitlines() if line.startswith("- ")]
    assert entries == [
        "work/project: alpha note",
        "work/project: Beta note",
        "work/project: Zeta note",
    ], "memory context follows row order, so the cached prompt prefix changes (#28292)"


def test_no_memories_adds_no_memory_context(make_user, upstream):
    with make_user().client() as client:
        wait_for_reply(client, send_message(client, "hello", features={"memory": True}))

    sent = upstream.chat_requests()[-1]["messages"]
    assert [entry["role"] for entry in sent] == ["user"]


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

    replayed = upstream.chat_requests()[-1]["messages"]
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
