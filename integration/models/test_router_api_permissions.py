"""Journey: a router model over the API, who may use it, and what routing costs.

An API client that may pick a router model chats with it through `/api/chat/completions` and the
OpenAI-compatible `/api/v1/chat/completions`, whole and streamed, and gets the answer of the
model the rules pick. An account without access never sees the router model and is refused,
over the API and when it continues a shared router chat. An admin chats with every router
model, set up in the panel or through `ROUTER_MODELS`, with `BYPASS_ADMIN_ACCESS_CONTROL` on and
off. The Decisions API key stays with the admins. A pattern route costs no Decisions API call,
a condition route one per message, however many conditions and streamed pieces; the decider's
tokens stay out of the reply's usage. Titles, tags and follow-ups are written by the model that
answered, with no further Decisions API call. Autocomplete, the call emoji (Display Emoji in Call)
and Merge Responses in a router chat are routed on the user's text they carry, as a message with
that text would be: autocomplete on the chat and the text being typed, the emoji on the spoken
sentence, a merge on the question and the answers it merges. The Decisions API never judges their
built-in instructions.

The autocomplete, emoji and merge tests are red on dev 639e25ae3, where the router judges each
task's instructions: the emoji instructions match the code pattern by their own backticks. They
stay red with the first follow-up for open-webui/open-webui#34887 (b01b7d916), which routed only
the tasks' chat history, so text typed after a chat with history no longer reached the code
model. They pass with 828b30373, which routes each task on the text it carries.

Two tests stay red on dev d599f7a08. An admin with the bypass off is refused a router model set
through `ROUTER_MODELS` with "Model not found" (open-webui/open-webui#34786, fix PR #34791).
With no task model set, each background task of a reply is routed again on the task's own
instructions: the word "default" in the title and tags instructions matches a `def ` pattern
(saved without its trailing space), so a legal question gets its title and tags from the code
model, and the follow-ups cost a second Decisions API call (open-webui/open-webui#34887).

Discriminates, in backend copies of dev d599f7a08: patterns that never match turn the API test
and the cost test red; skipping the access check for router models turns the stranger and the
shared chat tests red; one Decisions API call per condition turns the cost test red; router
settings copied into `/api/models` turn the key test red. The admin test passes with PR #34791
applied, and the task test passes in a copy that runs background tasks on the model of the
chat's latest reply.
"""

from __future__ import annotations

import json
import time
import uuid

import httpx
import pytest

from harness import upstream as reply
from harness.access import grant
from harness.actors import admin_of
from harness.chat import ask, send_message, wait_for_reply
from harness.router_journeys import (
    DECIDER_KEY,
    Decider,
    condition,
    default,
    make_targets,
    models_listed,
    pattern,
    remove_targets,
    save_router,
)
from harness.router_models import ROUTER_MODELS_CONFIG
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

CODE_PATTERN = "```|def "
LEGAL = "the user asks about contracts or legal questions"
MEDICAL = "the user asks about health"
TRAVEL = "the user plans a trip"


@pytest.fixture
def decider(listener) -> Decider:
    return Decider(listener)


@pytest.fixture
def targets(admin, upstream):
    made = make_targets(admin, upstream, "code", "legal", "general", "medical", "travel")
    yield made
    remove_targets(admin, upstream, made)


@pytest.fixture
def smart_router(admin, preserve, targets, decider):
    """Code by pattern, legal, medical and travel by condition, general by default."""
    preserve(ROUTER_MODELS_CONFIG)
    return save_router(
        admin,
        [
            pattern(targets["code"], CODE_PATTERN),
            condition(targets["legal"], LEGAL),
            condition(targets["medical"], MEDICAL),
            condition(targets["travel"], TRAVEL),
            default(targets["general"]),
        ],
        decider=decider,
    )


def chat_body(model: str, prompt: str, stream: bool) -> dict:
    return {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": stream}


def streamed_text(client: httpx.Client, path: str, body: dict) -> str:
    with client.stream("POST", path, json=body) as response:
        assert response.status_code == 200, response.read().decode()
        lines = [line for line in response.iter_lines() if line.startswith("data: ")]
    pieces = [json.loads(line[6:]) for line in lines if line != "data: [DONE]"]
    return "".join(
        choice.get("delta", {}).get("content") or ""
        for piece in pieces
        for choice in piece.get("choices", [])
    )


def provider_models(upstream, prompt: str) -> list[str]:
    """The models the provider was asked to answer `prompt` with."""
    return [body["model"] for body in upstream.chat_requests() if reply.answering(prompt)(body)]


def clone_shared(client: httpx.Client, share_id: str) -> dict:
    cloned = client.post(f"/api/v1/chats/{share_id}/clone/shared")
    assert cloned.status_code == 200, cloned.text
    return cloned.json()


# ---------------------------------------------------------------- API clients


def test_an_api_client_gets_the_answer_of_the_routed_model_whole_and_streamed(
    make_user, upstream, targets, smart_router, decider
):
    client_user = make_user()
    code_prompt = "fix this: def broken(:"
    plain_prompt = f"what is a good name for a cat {uuid.uuid4().hex[:4]}"
    upstream.queue(
        reply.text("code answer", match=reply.answering(code_prompt)),
        reply.text(["general ", "answer"], match=reply.answering(plain_prompt)),
    )
    with client_user.client() as client:
        listed = models_listed(client)
        assert smart_router.id in listed, "a user with access does not see the router model"
        assert listed[smart_router.id]["name"] == smart_router.name

        whole = client.post(
            "/api/chat/completions", json=chat_body(smart_router.id, code_prompt, False)
        )
        assert whole.status_code == 200, whole.text
        streamed = streamed_text(
            client, "/api/v1/chat/completions", chat_body(smart_router.id, plain_prompt, True)
        )

    assert whole.json()["choices"][0]["message"]["content"] == "code answer"
    assert streamed == "general answer"
    assert provider_models(upstream, code_prompt) == [targets["code"].id]
    assert provider_models(upstream, plain_prompt) == [targets["general"].id]


def test_an_account_without_access_never_sees_or_reaches_the_router_model(
    admin, make_user, preserve, upstream, targets
):
    preserve(ROUTER_MODELS_CONFIG)
    member, stranger = make_user(), make_user()
    router = save_router(
        admin,
        [default(targets["general"])],
        access_grants=[grant("user", member.id, "read")],
    )
    prompt = f"am I allowed {uuid.uuid4().hex[:4]}"
    upstream.queue(reply.text("member answer", match=reply.answering(prompt)))

    with stranger.client() as client:
        assert router.id not in models_listed(client), "a stranger is offered the router model"
        for path in ("/api/chat/completions", "/api/v1/chat/completions"):
            refused = client.post(path, json=chat_body(router.id, prompt, False))
            assert refused.status_code == 403 or (
                refused.status_code in (400, 404) and "Model not found" in refused.text
            ), f"{path} did not refuse a stranger: {refused.status_code} {refused.text}"
    assert provider_models(upstream, prompt) == [], "a stranger's message reached a model"

    with member.client() as client:
        assert router.id in models_listed(client)
        answered = client.post("/api/chat/completions", json=chat_body(router.id, prompt, False))
    assert answered.status_code == 200, answered.text
    assert answered.json()["choices"][0]["message"]["content"] == "member answer"


def test_a_shared_router_chat_is_continued_only_by_a_reader_who_may_pick_the_router(
    admin, make_user, preserve, upstream, targets
):
    preserve(ROUTER_MODELS_CONFIG)
    owner, member, outsider = make_user(), make_user(), make_user()
    router = save_router(
        admin,
        [default(targets["general"])],
        access_grants=[grant("user", owner.id, "read"), grant("user", member.id, "read")],
    )
    upstream.queue(reply.text("first answer", match=reply.answering("first question")))
    with owner.client() as client:
        turn, _ = ask(client, "first question", model=router.id)
        shared = client.post(f"/api/v1/chats/{turn.chat_id}/share")
        assert shared.status_code == 200, shared.text
        readers = [grant("user", reader.id, "read") for reader in (member, outsider)]
        granted = client.post(
            f"/api/v1/chats/shared/{turn.chat_id}/access/update", json={"access_grants": readers}
        )
        assert granted.status_code == 200, granted.text
    share_id = shared.json()["share_id"]

    follow_up = f"and then {uuid.uuid4().hex[:4]}"
    upstream.queue(reply.text("continued answer", match=reply.answering(follow_up)))
    with member.client() as client:
        clone = clone_shared(client, share_id)
        next_turn = send_message(
            client,
            follow_up,
            model=router.id,
            chat_id=clone["id"],
            parent_id=clone["chat"]["history"]["currentId"],
        )
        assert wait_for_reply(client, next_turn)["content"] == "continued answer"
    with outsider.client() as client:
        clone = clone_shared(client, share_id)
        refused = client.post(
            "/api/chat/completions",
            json={
                **chat_body(router.id, follow_up, True),
                "chat_id": clone["id"],
                "parent_id": clone["chat"]["history"]["currentId"],
            },
        )
    assert refused.status_code in (400, 403, 404), (
        f"a reader without access to the router model continued its chat: {refused.text}"
    )
    assert provider_models(upstream, follow_up) == [targets["general"].id]


@pytest.mark.slow
@pytest.mark.parametrize("bypass", ["true", "false"])
def test_an_admin_chats_with_every_router_model_with_the_admin_bypass_on_and_off(
    instance_with, bypass
):
    env_router = {
        "id": "env-router",
        "name": "Env Router",
        "meta": {"routes": [{"model_id": MOCK_MODEL_ID}], "strategy": "first"},
    }
    booted = instance_with(
        {"ROUTER_MODELS": json.dumps([env_router]), "BYPASS_ADMIN_ACCESS_CONTROL": bypass}
    )
    admin = admin_of(booted)
    panel_router = save_router(admin, [{"model_id": MOCK_MODEL_ID}], access_grants=[])

    answers = {}
    with admin.client() as client:
        listed = models_listed(client)
        for router_id in ("env-router", panel_router.id):
            assert router_id in listed, f"the admin is not offered {router_id}"
            prompt = f"hello {router_id}"
            booted.upstream.queue(reply.text(f"answer {router_id}", match=reply.answering(prompt)))
            answered = client.post(
                "/api/chat/completions", json=chat_body(router_id, prompt, False)
            )
            answers[router_id] = (
                answered.json()["choices"][0]["message"]["content"]
                if answered.status_code == 200
                else f"HTTP {answered.status_code} {answered.text}"
            )

    assert answers == {
        "env-router": "answer env-router",
        panel_router.id: f"answer {panel_router.id}",
    }, "the admin is refused a router model it is offered (open-webui/open-webui#34786)"


def test_the_decisions_api_key_stays_with_the_admins(admin, make_user, smart_router):
    member = make_user()
    with member.client() as client:
        seen = {
            "/api/models": json.dumps(models_listed(client)),
            "/api/v1/models/model": client.get(
                "/api/v1/models/model", params={"id": smart_router.id}
            ).text,
            "/api/v1/models/list": client.get("/api/v1/models/list").text,
            ROUTER_MODELS_CONFIG[0]: client.get(ROUTER_MODELS_CONFIG[0]).text,
        }
    with admin.client() as client:
        admin_view = client.get(ROUTER_MODELS_CONFIG[0]).text

    assert DECIDER_KEY in admin_view, "the admin cannot read the saved Decisions API key back"
    leaked = [path for path, body in seen.items() if DECIDER_KEY in body]
    assert leaked == [], f"a user reads the Decisions API key from {leaked}"


# ---------------------------------------------------------------- cost


def test_a_pattern_route_costs_no_decisions_call_and_a_condition_route_one_per_message(
    user, upstream, targets, smart_router, decider
):
    code_prompt = "```python print(1)```"
    legal_prompt = "can my landlord keep the deposit"
    usage = {"prompt_tokens": 11, "completion_tokens": 22, "total_tokens": 33}
    upstream.queue(
        reply.text([f"code {n} " for n in range(20)], match=reply.answering(code_prompt)),
        reply.text(
            [f"legal {n} " for n in range(20)], usage=usage, match=reply.answering(legal_prompt)
        ),
    )
    with user.client() as client:
        ask(client, code_prompt, model=smart_router.id)
        assert decider.calls() == [], "a pattern match called the Decisions API"

        decider.scores = {LEGAL: 0.9}
        _, legal_reply = ask(client, legal_prompt, model=smart_router.id)

    calls = decider.calls()
    assert len(calls) == 1, f"one condition route made {len(calls)} Decisions API calls"
    asked = sorted(question["instructions"] for question in calls[0]["questions"])
    assert len(asked) == 3 and all(
        any(text in question for question in asked) for text in (LEGAL, MEDICAL, TRAVEL)
    ), f"the three conditions were not scored in one call: {asked}"
    assert provider_models(upstream, legal_prompt) == [targets["legal"].id]
    assert legal_reply["content"].strip() == "".join(f"legal {n} " for n in range(20)).strip()
    assert legal_reply.get("usage", {}).get("total_tokens") == 33, (
        f"the reply's usage is not the answering model's: {legal_reply.get('usage')}"
    )


def test_titles_tags_and_follow_ups_come_from_the_model_that_answered(
    admin, preserve, make_user, upstream, targets, smart_router, decider
):
    preserve("tasks")
    with admin.client() as client:
        current = client.get("/api/v1/tasks/config").json()
        enabled = {
            **current,
            "TASK_MODEL": "",
            "TASK_MODEL_EXTERNAL": "",
            "ENABLE_TITLE_GENERATION": True,
            "ENABLE_TAGS_GENERATION": True,
            "ENABLE_FOLLOW_UP_GENERATION": True,
        }
        client.post("/api/v1/tasks/config/update", json=enabled).raise_for_status()

    prompt = f"can I break my rental contract early {uuid.uuid4().hex[:4]}"
    decider.scores = {LEGAL: 0.9}
    upstream.queue(reply.text("legal answer", match=lambda body: bool(body.get("stream"))))
    asker = make_user()
    with asker.client() as client:
        turn = send_message(
            client,
            prompt,
            model=smart_router.id,
            background_tasks={
                "title_generation": True,
                "tags_generation": True,
                "follow_up_generation": True,
            },
        )
        answer = wait_for_reply(client, turn)
        assert answer["content"] == "legal answer"
        tasks = wait_for_tasks(upstream, 3)

    task_models = {kind: model for kind, model in tasks}
    decisions_calls = len(decider.calls())
    assert (task_models, decisions_calls) == (
        {
            "title": targets["legal"].id,
            "tags": targets["legal"].id,
            "follow-ups": targets["legal"].id,
        },
        1,
    ), (
        "the background tasks did not run on the model that answered, with the one Decisions "
        f"API call of the reply: tasks went to {task_models}, {decisions_calls} Decisions API calls"
    )


TASK_KINDS = {
    "title": "Generate a concise",
    "tags": "Generate 1-3 broad tags",
    "follow-ups": "Predict 3-5 useful messages",
}


def wait_for_tasks(upstream, count: int) -> list[tuple[str, str]]:
    deadline = time.monotonic() + 30
    found: list[tuple[str, str]] = []
    while time.monotonic() < deadline:
        found = [
            (kind, body["model"])
            for body in upstream.chat_requests()
            if not body.get("stream")
            for kind, marker in TASK_KINDS.items()
            if marker in json.dumps(body.get("messages", []))
        ]
        if len(found) >= count:
            return found
        time.sleep(0.2)
    raise AssertionError(f"the background tasks never all reached a model: {found}")


AUTOCOMPLETE = "You are an autocompletion system"
EMOJI = "reflect the speaker's likely facial expression"
MERGE = "synthesize these responses into a single"
CAT_CHAT = [
    {"role": "user", "content": "what is a good name for a cat"},
    {"role": "assistant", "content": "Whiskers is a classic."},
]


def task_answer(target, marker: str, content: str) -> reply.Reply:
    """`content` from `target` when it is asked for the task whose instructions hold `marker`."""
    return reply.text(
        content,
        match=lambda body: (
            body["model"] == target.id and marker in json.dumps(body.get("messages", []))
        ),
    )


def task_models(upstream, marker: str) -> list[str]:
    """The models the provider was asked to run the task whose instructions hold `marker` on."""
    return [
        body["model"]
        for body in upstream.chat_requests()
        if marker in json.dumps(body.get("messages", []))
    ]


def judged_instructions(decider: Decider, marker: str) -> list[str]:
    return [call["input"] for call in decider.calls() if marker in call["input"]]


def test_autocomplete_in_a_router_chat_is_routed_on_the_typed_text_like_a_message(
    admin, preserve, make_user, upstream, targets, smart_router, decider
):
    preserve("tasks")
    with admin.client() as client:
        current = client.get("/api/v1/tasks/config").json()
        enabled = {
            **current,
            "TASK_MODEL": "",
            "TASK_MODEL_EXTERNAL": "",
            "ENABLE_AUTOCOMPLETE_GENERATION": True,
        }
        client.post("/api/v1/tasks/config/update", json=enabled).raise_for_status()

    tag = uuid.uuid4().hex[:4]
    code_text, plain_text = f"and how do I write def load_lease( {tag}", f"and for a dog {tag}"
    upstream.queue(
        reply.text(
            "code answer",
            match=lambda body: (
                reply.answering(code_text)(body) and AUTOCOMPLETE not in json.dumps(body)
            ),
        ),
        task_answer(targets["code"], AUTOCOMPLETE, json.dumps({"text": "path):"})),
        task_answer(targets["general"], AUTOCOMPLETE, json.dumps({"text": "like Rex"})),
    )
    with make_user().client() as client:
        sent = client.post(
            "/api/chat/completions",
            json={
                "model": smart_router.id,
                "messages": [*CAT_CHAT, {"role": "user", "content": code_text}],
                "stream": False,
            },
        )
        assert sent.status_code == 200, sent.text
        suggestions = []
        for text in (code_text, plain_text):
            # what the chat input sends while the user types in a chat with history
            suggested = client.post(
                "/api/v1/tasks/auto/completions",
                json={
                    "model": smart_router.id,
                    "prompt": text,
                    "messages": CAT_CHAT,
                    "type": "general",
                    "stream": False,
                },
            )
            assert suggested.status_code == 200, suggested.text
            suggestions.append(suggested.json()["choices"][0]["message"]["content"])

    assert sent.json()["choices"][0]["message"]["content"] == "code answer"
    assert task_models(upstream, AUTOCOMPLETE) == [targets["code"].id, targets["general"].id], (
        "autocomplete was not routed on the typed text the way sending it as a message is"
    )
    assert suggestions == [json.dumps({"text": "path):"}), json.dumps({"text": "like Rex"})]
    assert judged_instructions(decider, AUTOCOMPLETE) == [], (
        "the Decisions API judged the autocomplete instructions"
    )


def test_a_call_emoji_in_a_router_chat_is_routed_on_the_spoken_sentence(
    make_user, upstream, targets, smart_router, decider
):
    tag = uuid.uuid4().hex[:4]
    plain_sentence, code_sentence = f"Glad that helped {tag}.", f"Start with def main( {tag}."
    upstream.queue(
        task_answer(targets["general"], EMOJI, "😊"),
        task_answer(targets["code"], EMOJI, "🤓"),
    )
    emojis = []
    with make_user().client() as client:
        for sentence in (plain_sentence, code_sentence):
            # what a call sends for each sentence it speaks with Show emoji in call on
            answered = client.post(
                "/api/v1/tasks/emoji/completions",
                json={"model": smart_router.id, "prompt": sentence},
            )
            assert answered.status_code == 200, answered.text
            emojis.append(answered.json()["choices"][0]["message"]["content"])

    assert task_models(upstream, EMOJI) == [targets["general"].id, targets["code"].id], (
        "the call emoji was not routed on the spoken sentence"
    )
    assert emojis == ["😊", "🤓"]
    assert judged_instructions(decider, EMOJI) == [], (
        "the Decisions API judged the emoji instructions"
    )


def test_merged_responses_in_a_router_chat_are_routed_on_the_question_and_the_responses(
    make_user, upstream, targets, smart_router, decider
):
    tag = uuid.uuid4().hex[:4]
    question = f"can I break my rental contract early {tag}"
    plain_responses = ["Usually not without a clause.", "Check the notice period."]
    code_responses = ["Usually not.", "Here is a checker:\n```python\nprint(1)\n```"]
    decider.scores = {LEGAL: 0.9}
    upstream.queue(
        task_answer(targets["legal"], MERGE, "legal merge"),
        task_answer(targets["code"], MERGE, "code merge"),
    )
    merged = []
    with make_user().client() as client:
        for responses in (plain_responses, code_responses):
            # what Merge Responses sends under a router model's answers
            merged.append(
                streamed_text(
                    client,
                    "/api/v1/tasks/moa/completions",
                    {
                        "model": smart_router.id,
                        "prompt": question,
                        "responses": responses,
                        "stream": True,
                    },
                )
            )

    assert task_models(upstream, MERGE) == [targets["legal"].id, targets["code"].id], (
        "the merge was not routed on the question and the responses it merges"
    )
    assert merged == ["legal merge", "code merge"]
    assert judged_instructions(decider, MERGE) == [], (
        "the Decisions API judged the merge instructions"
    )
