"""Journey: a router model sends each message to the model its rules pick.

A router model has patterns (regular expressions, not case-sensitive, checked first and in list
order against the current branch of the conversation, kept as written, spaces included),
conditions (plain-language text that one
Decisions API call scores together, the most likely one winning if it reaches the threshold) and
default models picked by a strategy (First: the top one; Random: any of them per message). A
pattern match never asks the Decisions API. A Decisions API that fails, stalls or answers nonsense
hands the message to the default models, and the user still gets an answer. The Decisions API gets
the router's model and key and one question per condition; its input is the default prompt with the
recent chat history, or the admin's prompt with its modifiers filled in, and never the target
model's system prompt. Saving refuses a router that routes to a router, an invalid pattern and a
threshold outside 0 to 1. A router without default models fails a message that matches nothing
with an error. A router switched off or deleted refuses new messages, and a deleted one's chats
still open. A user who may use the router but not the model a rule picks gets an error, never
another model's answer. `ROUTER_MODELS` makes a router at boot that routes like a saved one.
Behaviour from docs/features/administration/router-models.md and Admin Settings > Models.

Every target is a preset of the scripted model whose system prompt marks the provider requests
routed to it, and the Decisions API is a local listener scripted per test (`harness.router_models`).

The spaces test is red on dev d599f7a08: saving strips a pattern's trailing space, so "function "
also matches "functionality" (issue not filed yet).

Discriminates (dev d599f7a08, backend copies): patterns checked case-sensitively turns the pattern
test red; the last matching pattern winning turns the order test red; routing on the newest
message only turns the earlier-message test red; the least likely condition winning, and a
threshold compared with `>`, each turn the conditions test red; the last default under First turns
the threshold test red; Random always taking the first default turns the Random test red; a
Decisions failure re-raised turns all nine failure cases red; the Decisions timeout raised to 60
seconds turns the stalled test red; the default prompt sending the whole chat turns the history and
long chat tests red; a custom prompt ignored turns the custom prompt test red; routing on the
client's messages with system messages kept turns the system prompt test red; the three access
checks on the picked model removed turn the access test red; a route to a router allowed turns the
router-to-router test red; a switched-off row still listed turns the switch test red; a delete that
saves nothing turns the delete test red; `ROUTER_MODELS` read as empty turns its test red.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import uuid
from typing import Callable, Iterator

import httpx
import pytest

from harness import upstream as reply
from harness.actors import admin_of
from harness.chat import ChatTurn, ask, send_message, wait_for_reply
from harness.listener import ReceivedRequest, json_answer, text_answer
from harness.router_models import (
    DECIDER_KEY,
    DECIDER_MODEL,
    EVERYONE_READS,
    ROUTER_MODELS_CONFIG,
    Decider,
    Target,
    conditions_asked,
    create_target,
    default_routes,
    held_answer,
    queue_target_answers,
    save_router_model,
)
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

LEGAL = "The user asks about contracts or other legal questions"
MEDICAL = "The conversation includes personal or medical details"
CODE_PATTERN = r"def \w+\("


@pytest.fixture
def make_target(admin) -> Iterator[Callable[..., Target]]:
    """`make_target(label)`: a preset of the scripted model everyone may use, unless private."""
    made: list[str] = []

    def create(label: str, private: bool = False) -> Target:
        with admin.client() as client:
            target = create_target(client, label, private=private)
        made.append(target.id)
        return target

    yield create
    with admin.client() as client:
        for model_id in made:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


@pytest.fixture
def decider(listener) -> Decider:
    return Decider(listener)


@pytest.fixture
def make_router(admin, preserve) -> Callable[..., str]:
    """`make_router(routes, ...)`: a public router model saved as the panel saves it."""
    preserve(ROUTER_MODELS_CONFIG)

    def create(routes: list[dict], strategy: str = "first", config: dict | None = None) -> str:
        router_id = f"router-{uuid.uuid4().hex[:8]}"
        with admin.client() as client:
            save_router_model(
                client,
                router_id,
                f"Router {router_id}",
                routes=routes,
                access_grants=[EVERYONE_READS],
                strategy=strategy,
                config=config,
            )
            client.get("/api/models", params={"refresh": True}).raise_for_status()
        return router_id

    return create


def route_through(
    client: httpx.Client, upstream, router_id: str, prompt: str, targets: list[Target], **send
) -> tuple[ChatTurn, dict]:
    """Send `prompt` to the router, each target ready to answer it as itself."""
    queue_target_answers(upstream, prompt, targets)
    return ask(client, prompt, model=router_id, **send)


def answered_by(answer: dict, targets: list[Target]) -> Target | None:
    return next((target for target in targets if answer.get("content") == target.answer), None)


def expect_answered_by(answer: dict, target: Target, targets: list[Target], why: str) -> None:
    picked = answered_by(answer, targets)
    assert picked == target, (
        f"{why}: expected {target.name} to answer, got {picked.name if picked else None} "
        f"(content {answer.get('content')!r}, error {answer.get('error')})"
    )
    assert answer.get("selectedModelId") == target.id, (
        f"the reply names {answer.get('selectedModelId')} as its model, not {target.id}"
    )
    assert not answer.get("error"), f"the reply carries an error: {answer.get('error')}"


def history_of(*turns: tuple[str, dict]) -> list[dict]:
    """The messages of a branch, as the web client sends them with the next message."""
    messages = []
    for prompt, answer in turns:
        messages += [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer.get("content", "")},
        ]
    return messages


def send_outcome(client: httpx.Client, router_id: str, prompt: str, **send) -> str:
    """What the user is told when a message cannot be answered: the refusal or the stored error."""
    try:
        turn = send_message(client, prompt, model=router_id, **send)
    except AssertionError as refused:
        return str(refused)
    stored = wait_for_reply(client, turn, timeout=30)
    return json.dumps(stored.get("error") or "") if stored.get("error") else ""


# ---------------------------------------------------------------- patterns


def test_a_pattern_match_answers_from_its_model_without_asking_the_decisions_api(
    make_user, upstream, decider, make_target, make_router
):
    code, legal, general = make_target("code"), make_target("legal"), make_target("general")
    targets = [code, legal, general]
    router = make_router(
        [
            {"model_id": code.id, "pattern": CODE_PATTERN},
            {"model_id": legal.id, "condition": LEGAL},
            {"model_id": general.id},
        ],
        config=decider.config(),
    )
    decider.scores = {LEGAL: 0.99}
    tag = uuid.uuid4().hex[:6]

    with make_user().client() as client:
        # upper case: patterns are not case-sensitive
        _, answer = route_through(
            client, upstream, router, f"Why does DEF parse(text): fail on contracts? {tag}", targets
        )
        expect_answered_by(answer, code, targets, "a message matching the pattern")
        assert decider.requests() == [], "a pattern match still asked the Decisions API"

        _, answer = route_through(
            client, upstream, router, f"Can I leave this contract early? {tag}", targets
        )
    expect_answered_by(answer, legal, targets, "a message only the condition matches")
    assert len(decider.requests()) == 1


def test_a_pattern_keeps_its_spaces_and_matches_only_as_written(
    admin, make_user, upstream, make_target, make_router
):
    code, general = make_target("code"), make_target("general")
    targets = [code, general]
    # the docs' own example pattern: "def " and "function " end in a space on purpose;
    # dev strips the trailing one on save (issue not filed yet)
    pattern = "```|def |function "
    router = make_router([{"model_id": code.id, "pattern": pattern}, {"model_id": general.id}])
    with admin.client() as client:
        saved = client.get(ROUTER_MODELS_CONFIG[0]).json()["ROUTER_MODELS"]
    [stored] = [model for model in saved if model["id"] == router]
    tag = uuid.uuid4().hex[:6]

    with make_user().client() as client:
        _, answer = route_through(
            client, upstream, router, f"Explain function hoisting {tag}", targets
        )
        expect_answered_by(answer, code, targets, "a message with 'function ' in it")
        _, answer = route_through(
            client, upstream, router, f"What is the default functionality? {tag}", targets
        )
    expect_answered_by(answer, general, targets, "'default' and 'functionality' hold no 'def '")
    assert stored["meta"]["routes"][0]["pattern"] == pattern, (
        f"saving changed the pattern {pattern!r} to {stored['meta']['routes'][0]['pattern']!r}"
    )


def test_the_first_listed_pattern_wins_when_several_match(
    make_user, upstream, make_target, make_router
):
    first, second, general = make_target("first"), make_target("second"), make_target("general")
    targets = [first, second, general]
    in_order = make_router(
        [
            {"model_id": first.id, "pattern": "invoice"},
            {"model_id": second.id, "pattern": "invoice|python"},
            {"model_id": general.id},
        ]
    )
    swapped = make_router(
        [
            {"model_id": second.id, "pattern": "invoice|python"},
            {"model_id": first.id, "pattern": "invoice"},
            {"model_id": general.id},
        ]
    )
    prompt = f"Write python that totals an invoice {uuid.uuid4().hex[:6]}"

    with make_user().client() as client:
        _, answer = route_through(client, upstream, in_order, prompt, targets)
        expect_answered_by(answer, first, targets, "both patterns match, the first is listed first")
        _, answer = route_through(client, upstream, swapped, prompt + " again", targets)
    expect_answered_by(answer, second, targets, "both patterns match, the second is listed first")


def test_a_pattern_matches_an_earlier_message_of_the_branch_but_not_one_edited_away(
    make_user, upstream, make_target, make_router
):
    code, general = make_target("code"), make_target("general")
    targets = [code, general]
    router = make_router([{"model_id": code.id, "pattern": "```"}, {"model_id": general.id}])
    tag = uuid.uuid4().hex[:6]
    snippet = f"Fix this:\n```\nprint(1\n```\n{tag}"

    with make_user().client() as client:
        first, first_answer = route_through(client, upstream, router, snippet, targets)
        expect_answered_by(first_answer, code, targets, "a message with a code block")

        follow_up = f"And what is the weather like? {tag}"
        _, answer = route_through(
            client,
            upstream,
            router,
            follow_up,
            targets,
            chat_id=first.chat_id,
            parent_id=first.assistant_message_id,
            history=history_of((snippet, first_answer)),
        )
        expect_answered_by(answer, code, targets, "a follow-up after a code block in the branch")

        # the first message edited: a new branch from the top of the same chat
        edited = f"What is the weather like today? {tag}"
        _, answer = route_through(
            client, upstream, router, edited, targets, chat_id=first.chat_id, parent_id=None
        )
    expect_answered_by(answer, general, targets, "a branch without the code block")


# ---------------------------------------------------------------- conditions and defaults


def test_the_decisions_api_scores_every_condition_in_one_call_and_the_most_likely_wins(
    make_user, upstream, decider, make_target, make_router
):
    legal, medical, general = make_target("legal"), make_target("medical"), make_target("general")
    targets = [legal, medical, general]
    router = make_router(
        [
            {"model_id": legal.id, "condition": LEGAL},
            {"model_id": medical.id, "condition": MEDICAL},
            {"model_id": general.id},
        ],
        config=decider.config(threshold=0.5),
    )
    tag = uuid.uuid4().hex[:6]

    with make_user().client() as client:
        decider.scores = {LEGAL: 0.6, MEDICAL: 0.9}
        _, answer = route_through(
            client, upstream, router, f"My doctor and my landlord both wrote {tag}", targets
        )
        expect_answered_by(answer, medical, targets, "the medical condition scored highest")

        [request] = decider.requests()
        assert request.headers.get("Authorization") == f"Bearer {DECIDER_KEY}", (
            f"the router's key was not sent: {request.headers.get('Authorization')}"
        )
        assert request.json().get("model") == DECIDER_MODEL
        assert sorted(conditions_asked(request, [LEGAL, MEDICAL])) == sorted([LEGAL, MEDICAL]), (
            f"each condition is not its own question: {request.json().get('questions')}"
        )
        assert len(request.json()["questions"]) == 2

        decider.scores = {LEGAL: 0.5, MEDICAL: 0.2}
        _, answer = route_through(
            client, upstream, router, f"Is this clause enforceable? {tag}", targets
        )
    expect_answered_by(answer, legal, targets, "the legal condition scored exactly the threshold")
    assert len(decider.requests()) == 2, "the Decisions API was not called once per message"


def test_when_no_condition_reaches_the_threshold_the_top_default_model_answers(
    make_user, upstream, decider, make_target, make_router
):
    legal, general, spare = make_target("legal"), make_target("general"), make_target("spare")
    targets = [legal, general, spare]
    router = make_router(
        [
            {"model_id": legal.id, "condition": LEGAL},
            {"model_id": general.id},
            {"model_id": spare.id},
        ],
        strategy="first",
        config=decider.config(threshold=0.8),
    )
    decider.scores = {LEGAL: 0.79}

    with make_user().client() as client:
        for turn in range(3):
            _, answer = route_through(
                client, upstream, router, f"Read this lease {turn} {uuid.uuid4().hex[:6]}", targets
            )
            expect_answered_by(answer, general, targets, "no condition reached 0.8, strategy First")
    assert len(decider.requests()) == 3


def test_random_spreads_unmatched_messages_over_the_default_models_only(
    make_user, upstream, make_target, make_router
):
    code, left, right = make_target("code"), make_target("left"), make_target("right")
    targets = [code, left, right]
    router = make_router(
        [
            {"model_id": code.id, "pattern": "```"},
            {"model_id": left.id},
            {"model_id": right.id},
        ],
        strategy="random",
    )

    picked = []
    with make_user().client() as client:
        # 16 draws miss one of two defaults once in 32768 runs
        for turn in range(16):
            _, answer = route_through(
                client, upstream, router, f"Tell me a story {turn} {uuid.uuid4().hex[:6]}", targets
            )
            assert not answer.get("error"), f"a Random reply failed: {answer.get('error')}"
            picked.append(answered_by(answer, targets))
    names = {target.name if target else None for target in picked}
    assert names == {left.name, right.name}, f"Random did not pick among the defaults: {names}"


def test_a_router_without_conditions_never_calls_the_decisions_api(
    make_user, upstream, decider, make_target, make_router
):
    code, general, legal = make_target("code"), make_target("general"), make_target("legal")
    targets = [code, general, legal]
    plain = make_router(
        [{"model_id": code.id, "pattern": "```"}, {"model_id": general.id}],
        config=decider.config(),
    )
    with_condition = make_router(
        [{"model_id": legal.id, "condition": LEGAL}, {"model_id": general.id}],
        config=decider.config(),
    )
    decider.scores = {LEGAL: 0.1}
    tag = uuid.uuid4().hex[:6]

    with make_user().client() as client:
        for turn in range(2):
            _, answer = route_through(client, upstream, plain, f"Hello {turn} {tag}", targets)
            expect_answered_by(
                answer, general, targets, "nothing matched on a router without conditions"
            )
        assert decider.requests() == [], "a router without conditions called the Decisions API"

        _, answer = route_through(client, upstream, with_condition, f"Hello there {tag}", targets)
    expect_answered_by(answer, general, targets, "the condition scored low")
    assert len(decider.requests()) == 1


# ---------------------------------------------------------------- a failing Decisions API


def _closed_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _names(request: ReceivedRequest) -> list[str]:
    return [question["name"] for question in request.json()["questions"]]


def _predicates(request: ReceivedRequest, probability, names: list[str] | None = None) -> dict:
    asked = names if names is not None else _names(request)
    return {
        "answers": [
            {"type": "predicate", "name": name, "probability": probability} for name in asked
        ]
    }


BROKEN_ANSWERS = {
    "server error": lambda request: json_answer({"error": "overloaded"}, status=500),
    "rejected key": lambda request: json_answer({"error": "invalid api key"}, status=401),
    "not json": lambda request: text_answer("the route is legal", content_type="text/plain"),
    "no probabilities": lambda request: json_answer({"result": "legal"}),
    "unknown condition": lambda request: json_answer(
        _predicates(request, 0.99, names=["a-condition-nobody-asked"])
    ),
    "probability above one": lambda request: json_answer(_predicates(request, 1.7)),
    "negative probability": lambda request: json_answer(_predicates(request, -0.5)),
    "a condition left out": lambda request: json_answer(
        _predicates(request, 0.99, names=_names(request)[:1])
    ),
}


@pytest.mark.parametrize("failure", [*BROKEN_ANSWERS, "unreachable"])
def test_a_failing_decisions_api_hands_the_message_to_the_default_model(
    failure, make_user, upstream, decider, make_target, make_router
):
    legal, medical, general = make_target("legal"), make_target("medical"), make_target("general")
    targets = [legal, medical, general]
    if failure == "unreachable":
        config = decider.config(base_url=f"http://127.0.0.1:{_closed_port()}/v1")
    else:
        decider.answer = BROKEN_ANSWERS[failure]
        config = decider.config()
    router = make_router(
        [
            {"model_id": legal.id, "condition": LEGAL},
            {"model_id": medical.id, "condition": MEDICAL},
            {"model_id": general.id},
        ],
        config=config,
    )

    with make_user().client() as client:
        _, answer = route_through(
            client, upstream, router, f"Is my contract valid? {uuid.uuid4().hex[:6]}", targets
        )
    expect_answered_by(answer, general, targets, f"the Decisions API failed ({failure})")
    expected_calls = 0 if failure == "unreachable" else 1
    assert len(decider.requests()) == expected_calls


def test_a_stalled_decisions_api_is_given_up_on_within_seconds(
    make_user, upstream, decider, make_target, make_router
):
    legal, general = make_target("legal"), make_target("general")
    targets = [legal, general]
    release = threading.Event()
    decider.answer = held_answer(
        release, json_answer({"answers": [{"type": "predicate", "name": "x", "probability": 1}]})
    )
    router = make_router(
        [{"model_id": legal.id, "condition": LEGAL}, {"model_id": general.id}],
        config=decider.config(),
    )

    try:
        with make_user().client() as client:
            started = time.monotonic()
            _, answer = route_through(
                client, upstream, router, f"Review my contract {uuid.uuid4().hex[:6]}", targets
            )
            waited = time.monotonic() - started
    finally:
        release.set()
    expect_answered_by(answer, general, targets, "the Decisions API never answered")
    assert waited < 15, f"the reply waited {waited:.0f}s on a Decisions API that never answered"
    assert len(decider.requests()) == 1, "the stalled Decisions API was asked more than once"


# ---------------------------------------------------------------- what the Decisions API gets


def _converse(
    client: httpx.Client, upstream, router: str, prompts: list[str], targets: list[Target]
) -> list[dict]:
    """Send `prompts` as one chat, each after the reply to the one before; returns the replies."""
    turns: list[tuple[str, dict]] = []
    chat_id, parent_id = None, None
    for prompt in prompts:
        turn, answer = route_through(
            client,
            upstream,
            router,
            prompt,
            targets,
            chat_id=chat_id,
            parent_id=parent_id,
            history=history_of(*turns),
        )
        assert not answer.get("error"), f"a reply failed: {answer.get('error')}"
        turns.append((prompt, answer))
        chat_id, parent_id = turn.chat_id, turn.assistant_message_id
    return [answer for _, answer in turns]


def test_the_default_prompt_gives_the_decisions_api_the_recent_history_only(
    make_user, upstream, decider, make_target, make_router
):
    legal, general = make_target("legal"), make_target("general")
    targets = [legal, general]
    router = make_router(
        [{"model_id": legal.id, "condition": LEGAL}, {"model_id": general.id}],
        config=decider.config(),
    )
    tag = uuid.uuid4().hex[:6]
    markers = [f"turn-{number}-{tag}" for number in range(10)]

    with make_user().client() as client:
        _converse(
            client, upstream, router, [f"Note {marker} please" for marker in markers], targets
        )

    last_input = decider.inputs()[-1]
    assert markers[-1] in last_input, (
        f"the Decisions API did not get the new message: {last_input!r}"
    )
    assert markers[-2] in last_input, f"the Decisions API got no earlier turn: {last_input!r}"
    assert markers[0] not in last_input, (
        f"the Decisions API got the whole chat, the first of ten turns included: {last_input!r}"
    )


def test_a_long_chat_keeps_the_decisions_request_bounded(
    make_user, upstream, decider, make_target, make_router
):
    legal, general = make_target("legal"), make_target("general")
    targets = [legal, general]
    router = make_router(
        [{"model_id": legal.id, "condition": LEGAL}, {"model_id": general.id}],
        config=decider.config(),
    )
    tag = uuid.uuid4().hex[:6]
    long_message = f"start-{tag} " + "the quick brown fox jumps over the lazy dog " * 10_000

    with make_user().client() as client:
        _converse(client, upstream, router, [long_message, f"short-{tag}", long_message], targets)

    [*_, last] = decider.requests()
    assert len(last.body) < 50_000, (
        f"the Decisions request grew with the chat: {len(last.body)} bytes for a chat of "
        f"{2 * len(long_message)} characters"
    )
    assert f"start-{tag}" in str(last.json()["input"]), "the new message was cut from the input"


def test_a_custom_prompt_replaces_the_decisions_input_and_fills_its_modifiers(
    make_user, upstream, decider, make_target, make_router
):
    legal, general = make_target("legal"), make_target("general")
    targets = [legal, general]
    routes = [{"model_id": legal.id, "condition": LEGAL}, {"model_id": general.id}]
    newest_only = make_router(routes, config=decider.config(prompt="Route this: {{prompt}}"))
    last_two = make_router(routes, config=decider.config(prompt="Recent:\n{{MESSAGES:END:2}}"))
    tag = uuid.uuid4().hex[:6]
    prompts = [f"first-{tag}", f"second-{tag}", f"third-{tag}"]

    with make_user().client() as client:
        _converse(client, upstream, newest_only, prompts, targets)
        newest_input = decider.inputs()[-1]
        replies = _converse(client, upstream, last_two, prompts, targets)
    last_two_input = decider.inputs()[-1]

    assert newest_input.strip() == f"Route this: third-{tag}", (
        f"a prompt without {{{{MESSAGES}}}} sent more than the new message: {newest_input!r}"
    )
    assert last_two_input.startswith("Recent:"), (
        f"the custom prompt was not used: {last_two_input!r}"
    )
    assert f"third-{tag}" in last_two_input and replies[1]["content"] in last_two_input, (
        f"{{{{MESSAGES:END:2}}}} did not hold the last two messages: {last_two_input!r}"
    )
    assert f"second-{tag}" not in last_two_input and f"first-{tag}" not in last_two_input, (
        f"{{{{MESSAGES:END:2}}}} held more than the last two messages: {last_two_input!r}"
    )


def test_the_decisions_api_sees_neither_the_targets_nor_the_chats_system_prompt(
    make_user, upstream, decider, make_target, make_router
):
    legal, general = make_target("legal"), make_target("general")
    targets = [legal, general]
    router = make_router(
        [{"model_id": legal.id, "condition": LEGAL}, {"model_id": general.id}],
        config=decider.config(),
    )
    tag = uuid.uuid4().hex[:6]
    chat_system = f"The user's own system prompt {tag}."
    decider.scores = {LEGAL: 0.9}

    with make_user().client() as client:
        first_prompt = f"Can my landlord keep the deposit? {tag}"
        first, first_answer = route_through(
            client,
            upstream,
            router,
            first_prompt,
            targets,
            history=[{"role": "system", "content": chat_system}],
        )
        expect_answered_by(first_answer, legal, targets, "the legal condition scored high")
        _, answer = route_through(
            client,
            upstream,
            router,
            f"And the notice period? {tag}",
            targets,
            chat_id=first.chat_id,
            parent_id=first.assistant_message_id,
            history=[
                {"role": "system", "content": chat_system},
                *history_of((first_prompt, first_answer)),
            ],
        )
    expect_answered_by(answer, legal, targets, "the follow-up")

    inputs = decider.inputs()
    assert len(inputs) == 2 and first_prompt in inputs[-1], f"unexpected Decisions inputs: {inputs}"
    leaked = [
        text
        for text in (legal.marker, general.marker, chat_system)
        if any(text in sent for sent in inputs)
    ]
    assert not leaked, f"the Decisions API was sent system prompts: {leaked}"


# ---------------------------------------------------------------- access and lifecycle


def test_a_user_without_access_to_the_picked_model_gets_an_error_not_another_models_answer(
    make_user, upstream, make_target, make_router
):
    secret, general = make_target("secret", private=True), make_target("general")
    targets = [secret, general]
    router = make_router([{"model_id": secret.id, "pattern": "payroll"}, {"model_id": general.id}])
    prompt = f"Show me the payroll numbers {uuid.uuid4().hex[:6]}"
    queue_target_answers(upstream, prompt, targets)

    with make_user().client() as client:
        told = send_outcome(client, router, prompt)

    assert told, "a user without access to the picked model got no error"
    assert secret.answer not in told and general.answer not in told, (
        f"the user was answered: {told}"
    )
    reached = [body for body in upstream.chat_requests() if secret.was_sent(body)]
    assert not reached, "the model the user may not use was sent the message"
    assert not any(general.was_sent(body) for body in upstream.chat_requests()), (
        "the message went to the default model in place of the one the rule picked"
    )


def test_saving_a_router_that_routes_to_a_router_is_refused(admin, preserve, make_router):
    preserve(ROUTER_MODELS_CONFIG)
    existing = make_router(default_routes(MOCK_MODEL_ID))
    with admin.client() as client:
        before = client.get(ROUTER_MODELS_CONFIG[0]).json()["ROUTER_MODELS"]
        for routes, what in [
            (default_routes(existing), "another router model"),
            (default_routes("self-router"), "itself"),
        ]:
            router = {
                "id": "self-router",
                "name": "Self Router",
                "meta": {"routes": routes, "strategy": "first"},
            }
            saved = client.post(ROUTER_MODELS_CONFIG[1], json={"ROUTER_MODELS": [*before, router]})
            assert saved.status_code == 400, f"a router routing to {what} was saved: {saved.text}"
            assert "router" in saved.json().get("detail", "").lower(), (
                f"the refusal does not say why: {saved.text}"
            )
        after = client.get(ROUTER_MODELS_CONFIG[0]).json()["ROUTER_MODELS"]
    assert after == before


@pytest.mark.parametrize(
    "route, decisions, reason",
    [
        ({"pattern": "([unclosed"}, None, "pattern"),
        ({"condition": LEGAL}, {"threshold": 1.5}, "threshold"),
        ({"condition": LEGAL}, {"threshold": -0.1}, "threshold"),
    ],
    ids=["invalid regex", "threshold above one", "negative threshold"],
)
def test_saving_a_rule_the_router_cannot_use_is_refused_with_the_reason(
    route, decisions, reason, admin, preserve
):
    preserve(ROUTER_MODELS_CONFIG)
    config = (
        {
            "decisions": {
                "api_base_url": "http://127.0.0.1:9/v1",
                "model": DECIDER_MODEL,
                **decisions,
            }
        }
        if decisions
        else {}
    )
    router = {
        "id": f"router-{uuid.uuid4().hex[:8]}",
        "name": "Broken Router",
        "meta": {"routes": [{"model_id": MOCK_MODEL_ID, **route}, {"model_id": MOCK_MODEL_ID}]},
        "config": config,
    }
    with admin.client() as client:
        before = client.get(ROUTER_MODELS_CONFIG[0]).json()["ROUTER_MODELS"]
        saved = client.post(ROUTER_MODELS_CONFIG[1], json={"ROUTER_MODELS": [*before, router]})
        after = client.get(ROUTER_MODELS_CONFIG[0]).json()["ROUTER_MODELS"]
    assert 400 <= saved.status_code < 500, f"the broken rule was saved: {saved.text}"
    assert reason in saved.text.lower(), f"the refusal does not name the {reason}: {saved.text}"
    assert after == before


def test_a_message_matching_nothing_on_a_router_without_defaults_fails_with_an_error(
    make_user, upstream, make_target, make_router
):
    code = make_target("code")
    router = make_router([{"model_id": code.id, "pattern": "```"}])
    prompt = f"Hello there {uuid.uuid4().hex[:6]}"
    queue_target_answers(upstream, prompt, [code])

    with make_user().client() as client:
        told = send_outcome(client, router, prompt)

    assert told, "a message no rule matched got no error"
    assert code.answer not in told
    assert not any(reply.answering(prompt)(body) for body in upstream.chat_requests())


def _listed(client: httpx.Client, model_id: str) -> bool:
    models = client.get("/api/models")
    assert models.status_code == 200, models.text
    return any(model["id"] == model_id for model in models.json()["data"])


def test_a_router_switched_off_refuses_messages_until_switched_back_on(
    admin, make_user, upstream, make_target, make_router
):
    general = make_target("general")
    router = make_router(default_routes(general.id))
    member = make_user()

    def toggle() -> None:
        with admin.client() as client:
            toggled = client.post("/api/v1/models/model/toggle", params={"id": router})
            assert toggled.status_code == 200, toggled.text

    toggle()
    with member.client() as client:
        assert not _listed(client, router), "a router switched off is still offered"
        told = send_outcome(client, router, f"Hello {uuid.uuid4().hex[:6]}")
        assert told, "a router switched off still answered"

    toggle()
    with member.client() as client:
        assert _listed(client, router), "a router switched back on is not offered"
        _, answer = route_through(
            client, upstream, router, f"Hello {uuid.uuid4().hex[:6]}", [general]
        )
    expect_answered_by(answer, general, [general], "the router switched back on")


def test_a_deleted_routers_chat_still_opens_and_a_new_message_there_gets_an_error(
    admin, make_user, upstream, make_target, make_router
):
    general = make_target("general")
    router = make_router(default_routes(general.id))
    member = make_user()
    prompt = f"Remember this {uuid.uuid4().hex[:6]}"

    with member.client() as client:
        turn, answer = route_through(client, upstream, router, prompt, [general])
        expect_answered_by(answer, general, [general], "before the router was deleted")

    with admin.client() as client:
        current = client.get(ROUTER_MODELS_CONFIG[0]).json()["ROUTER_MODELS"]
        kept = [model for model in current if model["id"] != router]
        deleted = client.post(ROUTER_MODELS_CONFIG[1], json={"ROUTER_MODELS": kept})
        assert deleted.status_code == 200, deleted.text

    with member.client() as client:
        assert not _listed(client, router), "a deleted router is still offered"
        stored = client.get(f"/api/v1/chats/{turn.chat_id}")
        assert stored.status_code == 200, (
            f"the deleted router's chat no longer opens: {stored.text}"
        )
        messages = stored.json()["chat"]["history"]["messages"]
        assert messages[turn.assistant_message_id].get("content") == general.answer
        told = send_outcome(
            client,
            router,
            f"Still there? {uuid.uuid4().hex[:6]}",
            chat_id=turn.chat_id,
            parent_id=turn.assistant_message_id,
            history=history_of((prompt, answer)),
        )
    assert "not found" in told.lower(), (
        f"a new message to a deleted router did not say so: {told!r}"
    )


@pytest.mark.slow
def test_a_router_from_router_models_routes_like_one_saved_in_the_panel(instance_with):
    tag = uuid.uuid4().hex[:6]
    code_id, general_id = f"env-code-{tag}", f"env-general-{tag}"
    router = {
        "id": f"env-router-{tag}",
        "name": f"Env Router {tag}",
        "meta": {
            "routes": [{"model_id": code_id, "pattern": "```"}, {"model_id": general_id}],
            "strategy": "first",
        },
    }
    booted = instance_with({"ROUTER_MODELS": json.dumps([router])})
    admin = admin_of(booted)
    with admin.client() as client:
        code = create_target(client, "env code", model_id=code_id)
        general = create_target(client, "env general", model_id=general_id)

        saved = client.get(ROUTER_MODELS_CONFIG[0]).json()["ROUTER_MODELS"]
        assert router["id"] in [model["id"] for model in saved], (
            f"ROUTER_MODELS was not loaded: {saved}"
        )
        client.get("/api/models", params={"refresh": True}).raise_for_status()
        assert _listed(client, router["id"]), "the router from ROUTER_MODELS is not offered"

        targets = [code, general]
        _, answer = route_through(client, booted.upstream, router["id"], f"```x``` {tag}", targets)
        expect_answered_by(answer, code, targets, "a code block, from ROUTER_MODELS")
        _, answer = route_through(client, booted.upstream, router["id"], f"Hi {tag}", targets)
    expect_answered_by(answer, general, targets, "nothing matched, from ROUTER_MODELS")
