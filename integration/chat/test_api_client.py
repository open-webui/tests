"""Journey: Open WebUI as an OpenAI-compatible API for a script, the way the API docs describe it.

A client holding an API key or a session token lists the models at `/api/models` and chats
through `/api/chat/completions` with the official `openai` SDK pointed at `/api`, streamed and
not; the parameters it sends reach the provider and the reply parses as the SDK's own types. The
documented request options do what they say: `files` with an uploaded file or a knowledge
collection puts that text in front of the model, `tool_ids` offers a workspace tool and hands
the model's call back to the client, and `features` turns on the built-in tools it names. A file
is uploaded, polled until processed and added to a knowledge base over the API, then asked
about. A script that creates a chat first and names it and the empty answer has the server run
the tool loop and write the answer into that chat, polling the chat's tasks until they are done;
without a session id the call itself returns once the answer is stored. A wrong key, a model
that does not exist and a model the account may not use are refused before anything reaches the
provider, the last two alike so a private model's name stays hidden.

Discriminates: on dev 0f5a58f5f, in backend copies: `get_current_user_by_api_key` finding no user
fails every API-key run; an unknown key answered 403 fails the wrong-key test; dropping the
`/api/v1/models` alias fails both model list runs; dropping the client's `temperature`,
`max_tokens`, `seed` and `stop` fails the parameters test; an empty `files` list in the retrieval
handler fails both file and both knowledge runs; discarding `tool_ids` fails both tool runs and
both saved chat loop runs; answering a saved chat without a session id before its reply is stored
fails both blocking runs; reading `features.memory` as always off fails the memory-on runs and as
always on the memory-off runs; an unknown model falling back to the first one fails the
missing-model test; `check_model_access` letting every model through fails the inaccessible-model
test; the OpenAI router stripping `data: ` from the relayed stream fails every streamed test; and
dropping `usage` from a non-streamed reply fails both completion runs.
"""

from __future__ import annotations

import time
import uuid
from typing import Iterator

import httpx
import openai
import pytest

from harness import upstream as reply
from harness.chat import NO_BACKGROUND_TASKS
from harness.knowledge_bases import knowledge_base
from harness.python_tools import python_tool
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

ADMIN_CONFIG = "/api/v1/auths/admin/config"
DEFAULT_PERMISSIONS = "/api/v1/users/default/permissions"
MEMORY_TOOLS = {"search_memories", "add_memory"}
TOOL_SOURCE = '''
class Tools:
    def lookup_ticket(self, ticket: str) -> str:
        """
        Look a support ticket up.
        :param ticket: The ticket number.
        """
        return f"ticket {ticket} is open"
'''


@pytest.fixture
def api_access(admin, preserve):
    """API keys on without endpoint restrictions, and users allowed keys and knowledge bases."""
    preserve("admin_config", "permissions")
    with admin.client() as client:
        config = client.get(ADMIN_CONFIG).json()
        saved = client.post(
            ADMIN_CONFIG,
            json={
                **config,
                "ENABLE_API_KEYS": True,
                "ENABLE_API_KEYS_ENDPOINT_RESTRICTIONS": False,
            },
        )
        assert saved.status_code == 200, saved.text
        permissions = client.get(DEFAULT_PERMISSIONS).json()
        permissions["features"]["api_keys"] = True
        permissions["workspace"]["knowledge"] = True
        saved = client.post(DEFAULT_PERMISSIONS, json=permissions)
        assert saved.status_code == 200, saved.text


@pytest.fixture(params=["api key", "session token"])
def credential(request, make_user, api_access) -> str:
    """A fresh account's API key, or the session token the sign-in gave it."""
    account = make_user()
    if request.param == "session token":
        return account.token
    with account.client() as client:
        generated = client.post("/api/v1/auths/api_key")
    assert generated.status_code == 200, generated.text
    return generated.json()["api_key"]


@pytest.fixture
def api_key(make_user, api_access) -> str:
    with make_user().client() as client:
        generated = client.post("/api/v1/auths/api_key")
    assert generated.status_code == 200, generated.text
    return generated.json()["api_key"]


def sdk(instance, token: str) -> openai.OpenAI:
    """The official client the way the docs point it at Open WebUI."""
    return openai.OpenAI(api_key=token, base_url=f"{instance.base_url}/api", max_retries=0)


def complete(client: httpx.Client, prompt: str, **options) -> httpx.Response:
    payload = {"model": MOCK_MODEL_ID, "messages": [{"role": "user", "content": prompt}]}
    return client.post("/api/chat/completions", json={**payload, "stream": False, **options})


def upload_and_wait(client: httpx.Client, filename: str, text: str) -> str:
    """Upload as the docs do, then poll the processing status until it is done."""
    uploaded = client.post(
        "/api/v1/files/", files={"file": (filename, text.encode(), "text/plain")}
    )
    assert uploaded.status_code == 200, uploaded.text
    file_id = uploaded.json()["id"]
    deadline = time.monotonic() + 60
    status = {}
    while time.monotonic() < deadline:
        status = client.get(f"/api/v1/files/{file_id}/process/status").json()
        if status.get("status") in ("completed", "failed"):
            break
        time.sleep(0.2)
    assert status.get("status") == "completed", f"processing {filename} ended as {status}"
    return file_id


def sent_text(upstream) -> str:
    [request] = upstream.chat_requests()
    return str(request["messages"])


def offered_tools(upstream) -> set[str]:
    [request] = upstream.chat_requests()
    return {tool["function"]["name"] for tool in request.get("tools") or []}


@pytest.fixture
def private_preset(make_user) -> Iterator[str]:
    """A preset on the scripted model that only its owner may use."""
    owner = make_user(role="admin")
    preset_id = f"private-{uuid.uuid4().hex[:8]}"
    with owner.client() as client:
        created = client.post(
            "/api/v1/models/create",
            json={
                "id": preset_id,
                "base_model_id": MOCK_MODEL_ID,
                "name": "Private preset",
                "meta": {},
                "params": {},
                "access_grants": [],
            },
        )
        assert created.status_code == 200, created.text
        yield preset_id
        client.post("/api/v1/models/model/delete", json={"id": preset_id})


# --- the model list and chat completions through the SDK ------------------------------------


def test_the_sdk_lists_the_models_the_account_may_use(instance, credential):
    listed = sdk(instance, credential).models.list()
    with instance.client(credential) as client:
        aliased = client.get("/api/v1/models")

    assert MOCK_MODEL_ID in [model.id for model in listed.data]
    assert aliased.status_code == 200, aliased.text
    assert [model["id"] for model in aliased.json()["data"]] == [model.id for model in listed.data]


def test_a_completion_parses_as_the_sdks_chat_completion(instance, credential, upstream):
    upstream.queue(reply.text("Paris.", usage={"prompt_tokens": 7, "completion_tokens": 2}))

    completion = sdk(instance, credential).chat.completions.create(
        model=MOCK_MODEL_ID,
        messages=[{"role": "user", "content": "Capital of France?"}],
    )

    assert completion.object == "chat.completion"
    assert completion.choices[0].message.role == "assistant"
    assert completion.choices[0].message.content == "Paris."
    assert completion.choices[0].finish_reason == "stop"
    assert completion.usage.prompt_tokens == 7
    assert "Capital of France?" in sent_text(upstream)


def test_a_streamed_completion_parses_chunk_by_chunk(instance, credential, upstream):
    upstream.queue(reply.text(["The ", "sky ", "is ", "blue."]))

    stream = sdk(instance, credential).chat.completions.create(
        model=MOCK_MODEL_ID,
        messages=[{"role": "user", "content": "Why is the sky blue?"}],
        stream=True,
    )
    chunks = list(stream)

    pieces = [chunk.choices[0].delta.content for chunk in chunks if chunk.choices]
    assert "".join(piece or "" for piece in pieces) == "The sky is blue."
    assert [piece for piece in pieces if piece] == ["The ", "sky ", "is ", "blue."]
    assert chunks[-1].choices[0].finish_reason == "stop"


def test_the_stream_is_server_sent_events_ending_in_done(instance, api_key, upstream):
    upstream.queue(reply.text(["one ", "two"]))

    with instance.client(api_key) as client:
        with client.stream(
            "POST",
            "/api/chat/completions",
            json={
                "model": MOCK_MODEL_ID,
                "messages": [{"role": "user", "content": "count"}],
                "stream": True,
            },
        ) as response:
            content_type = response.headers.get("content-type", "")
            events = [line for line in response.iter_lines() if line]

    assert response.status_code == 200
    assert content_type.startswith("text/event-stream")
    assert all(event.startswith("data: ") for event in events), events
    assert events[-1] == "data: [DONE]"


def test_the_parameters_a_client_sends_reach_the_provider(instance, api_key, upstream):
    sdk(instance, api_key).chat.completions.create(
        model=MOCK_MODEL_ID,
        messages=[{"role": "user", "content": "Pick a number."}],
        temperature=0.25,
        max_tokens=42,
        seed=7,
        stop=["END"],
    )

    [sent] = upstream.chat_requests()
    assert (sent["temperature"], sent["max_tokens"], sent["seed"]) == (0.25, 42, 7)
    assert sent["stop"] == ["END"]


# --- documented request options ----------------------------------------------------------------


def test_an_uploaded_file_in_files_reaches_a_streamed_answer(instance, credential, upstream):
    upstream.queue(reply.text(["On a ", "Tuesday."]))
    with instance.client(credential) as client:
        file_id = upload_and_wait(client, "orders.txt", "Order 4471 shipped on a Tuesday.")

    stream = sdk(instance, credential).chat.completions.create(
        model=MOCK_MODEL_ID,
        messages=[{"role": "user", "content": "When did order 4471 ship?"}],
        stream=True,
        extra_body={"files": [{"type": "file", "id": file_id}]},
    )
    pieces = [chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices]

    assert "".join(pieces) == "On a Tuesday."
    assert "Order 4471 shipped on a Tuesday." in sent_text(upstream)


def test_a_file_added_to_a_knowledge_base_over_the_api_answers_a_question(
    instance, credential, upstream
):
    upstream.queue(reply.text("Brass, since 1911."))
    with instance.client(credential) as client:
        with knowledge_base(client, "Lighthouse facts") as knowledge_id:
            file_id = upload_and_wait(
                client,
                "lighthouse.txt",
                "The lighthouse bell is made of brass and dates from 1911.",
            )
            added = client.post(
                f"/api/v1/knowledge/{knowledge_id}/file/add", json={"file_id": file_id}
            )
            assert added.status_code == 200, added.text
            answered = complete(
                client,
                "What is the lighthouse bell made of?",
                files=[{"type": "collection", "id": knowledge_id}],
            )

    assert answered.status_code == 200, answered.text
    assert answered.json()["choices"][0]["message"]["content"] == "Brass, since 1911."
    assert "made of brass and dates from 1911" in sent_text(upstream)


def test_tool_ids_offer_the_tool_and_hand_its_call_back(instance, admin, credential, upstream):
    upstream.queue(reply.tool_call("lookup_ticket", {"ticket": "T-17"}))

    with python_tool(admin, TOOL_SOURCE, name="Tickets") as tool_id:
        completion = sdk(instance, credential).chat.completions.create(
            model=MOCK_MODEL_ID,
            messages=[{"role": "user", "content": "Is T-17 open?"}],
            extra_body={"tool_ids": [tool_id]},
        )

    assert "lookup_ticket" in offered_tools(upstream)
    [call] = completion.choices[0].message.tool_calls
    assert (call.function.name, call.function.arguments) == ("lookup_ticket", '{"ticket": "T-17"}')


@pytest.mark.parametrize("memory_on", [True, False], ids=["memory on", "memory off"])
def test_features_turn_on_the_built_in_tools_they_name(instance, credential, upstream, memory_on):
    with instance.client(credential) as client:
        answered = complete(
            client,
            "What do you remember about me?",
            features={"memory": memory_on},
            session_id=f"api-{uuid.uuid4().hex[:8]}",
        )

    assert answered.status_code == 200, answered.text
    assert (MEMORY_TOOLS <= offered_tools(upstream)) is memory_on


# --- the server-side loop: a saved chat the answer is written into ---------------------------


def create_chat(client: httpx.Client, prompt: str) -> tuple[str, str]:
    """A chat holding the question and an empty answer, as the docs create one; returns both ids."""
    user_message_id, assistant_message_id = str(uuid.uuid4()), str(uuid.uuid4())
    now = int(time.time())
    messages = {
        user_message_id: {
            "id": user_message_id,
            "role": "user",
            "content": prompt,
            "timestamp": now,
            "models": [MOCK_MODEL_ID],
            "childrenIds": [assistant_message_id],
        },
        assistant_message_id: {
            "id": assistant_message_id,
            "role": "assistant",
            "content": "",
            "parentId": user_message_id,
            "childrenIds": [],
            "model": MOCK_MODEL_ID,
            "modelIdx": 0,
            "done": False,
            "timestamp": now + 1,
        },
    }
    history = {"currentId": assistant_message_id, "messages": messages}
    created = client.post(
        "/api/v1/chats/new",
        json={"chat": {"title": "API run", "models": [MOCK_MODEL_ID], "history": history}},
    )
    assert created.status_code == 200, created.text
    return created.json()["id"], assistant_message_id


def run_in_chat(client: httpx.Client, chat_id: str, message_id: str, prompt: str, **options):
    return client.post(
        "/api/chat/completions",
        json={
            "model": MOCK_MODEL_ID,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "chat_id": chat_id,
            "id": message_id,
            "background_tasks": NO_BACKGROUND_TASKS,
            **options,
        },
    )


def wait_until_idle(client: httpx.Client, chat_id: str) -> None:
    deadline = time.monotonic() + 60
    while client.get(f"/api/tasks/chat/{chat_id}").json()["task_ids"]:
        assert time.monotonic() < deadline, "the chat's task never finished"
        time.sleep(0.2)


def stored_answer(client: httpx.Client, chat_id: str, message_id: str) -> dict:
    chat = client.get(f"/api/v1/chats/{chat_id}").json()["chat"]
    return chat["history"]["messages"][message_id]


def test_a_saved_chat_runs_the_tool_loop_and_stores_the_answer(
    instance, admin, credential, upstream
):
    prompt = "Is ticket T-17 still open?"
    upstream.queue(
        reply.tool_call("lookup_ticket", {"ticket": "T-17"}),
        reply.text("Yes, T-17 is still open."),
    )
    with python_tool(admin, TOOL_SOURCE, name="Tickets") as tool_id:
        with instance.client(credential) as client:
            chat_id, message_id = create_chat(client, prompt)
            accepted = run_in_chat(
                client,
                chat_id,
                message_id,
                prompt,
                tool_ids=[tool_id],
                session_id=f"api-{uuid.uuid4().hex[:8]}",
            )
            assert accepted.status_code == 200, accepted.text
            wait_until_idle(client, chat_id)
            answer = stored_answer(client, chat_id, message_id)

    assert accepted.json()["chat_id"] == chat_id and accepted.json()["task_ids"]
    assert answer["done"] is True
    assert answer["content"] == "Yes, T-17 is still open."
    follow_up = upstream.chat_requests()[1]
    assert "ticket T-17 is open" in str(follow_up["messages"])


def test_without_a_session_id_the_call_returns_once_the_answer_is_stored(
    instance, credential, upstream
):
    prompt = "Say hello."
    upstream.queue(reply.text("Hello there."))
    with instance.client(credential) as client:
        chat_id, message_id = create_chat(client, prompt)
        returned = run_in_chat(client, chat_id, message_id, prompt)
        answer = stored_answer(client, chat_id, message_id)

    assert returned.status_code == 200, returned.text
    assert returned.json() is None
    assert (answer["done"], answer["content"]) == (True, "Hello there.")


# --- what a client is told when it gets something wrong ----------------------------------------


def test_a_wrong_key_is_refused_with_401(instance, upstream, api_access):
    client = sdk(instance, "sk-" + "0" * 32)

    with pytest.raises(openai.AuthenticationError):
        client.models.list()
    with pytest.raises(openai.AuthenticationError):
        client.chat.completions.create(
            model=MOCK_MODEL_ID, messages=[{"role": "user", "content": "hello"}]
        )
    assert upstream.chat_requests() == []


def test_a_model_that_does_not_exist_is_refused(instance, api_key, upstream):
    with instance.client(api_key) as client:
        refused = complete(client, "hello", model="no-such-model")

    assert refused.status_code == 400, refused.text
    assert refused.json()["detail"] == "Model not found"
    assert upstream.chat_requests() == []


def test_a_model_the_account_may_not_use_is_refused_like_a_missing_one(
    instance, api_key, upstream, private_preset
):
    with instance.client(api_key) as client:
        listed = [model["id"] for model in client.get("/api/models").json()["data"]]
        refused = complete(client, "hello", model=private_preset)

    assert private_preset not in listed
    assert refused.status_code == 400, refused.text
    assert refused.json()["detail"] == "Model not found"
    assert upstream.chat_requests() == []
