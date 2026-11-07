"""Journey: how an uploaded file, a knowledge base, a referenced chat and a note reach the model.

Since 0.13.1 (bd6fc15be) a model's Preload files capability (File Context until 131783e54) decides
whether an attachment's text goes into the prompt up front or the model reads it through its builtin
tools. Left unset it follows the function calling mode: off in native mode, where the model is
offered the reader of that kind (`view_file` for an uploaded file, `view_knowledge_file` for a
knowledge base, `view_chat` for a referenced chat, `view_note` for a note) and calling it returns
the text, and on in legacy mode, whether the model or the chat (its Chat Controls) picks legacy.
Ticked, the text goes into the prompt in native mode as well; unticked, it stays out in legacy mode
too. A model with Preload files off and the matching builtin tool category off gets nothing on
purpose (closed open-webui/open-webui#33818), so that pairing is not tested here.

The retrieval settings behind the preloaded text are pinned too: since 98f7a6146 Top K defaults
to 10, the chunk size to 1024 and the new Full Text Threshold to 50000 characters, which the
admin saves in Admin Settings > Documents, the env var seeds on a fresh instance and every
signed-in account reads from `/api/config` (the file dialog decides with it whether to show its
whole-document switch); Always Use Full Text shows there as full context. A negative threshold is
refused. The threshold boundary itself is in integration/retrieval/test_chat_retrieval.py, the
API path without a session in integration/chat/test_api_client.py.

Discriminates: on dev f659c0d12, in backend copies: `is_file_context_enabled` answering True
fails the four native default and the four legacy unticked cases; answering False fails the four
native ticked and the four legacy default cases; answering the mode default whatever the
capability fails the native ticked and legacy unticked cases; the threshold left out of the
settings response fails the defaults, round trip, negative and env tests; dropping `ge=0` fails the
negative threshold test; `/api/config` without the threshold fails the round trip test; the env
var ignored fails the env test; Top K defaulting to 3 again fails the defaults test; and the
mode read from the model only, not the chat, fails the four chat-picked legacy cases.
"""

from __future__ import annotations

import contextlib
import uuid
from typing import Callable

import pytest

from harness import upstream as reply
from harness.access import grant
from harness.actors import Actor, admin_of
from harness.chat import ask
from harness.chat_history import seed_chat
from harness.knowledge_bases import add_text_file, knowledge_base
from harness.model_presets import preset_with_capabilities
from harness.upstream import conversation
from harness.web_retrieval import RETRIEVAL_CONFIG

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

# the tool a model in native mode reads each kind of attachment with
READERS = {
    "file": "view_file",
    "collection": "view_knowledge_file",
    "chat": "view_chat",
    "note": "view_note",
}


def attach_file(
    stack: contextlib.ExitStack, admin: Actor, person: Actor, secret: str
) -> tuple[dict, dict]:
    with person.client() as client:
        uploaded = client.post(
            "/api/v1/files/",
            params={"process": "true", "process_in_background": "false"},
            files={"file": ("locker.txt", secret.encode(), "text/plain")},
        )
    assert uploaded.status_code == 200, f"uploading the file failed: {uploaded.text}"
    file_id = uploaded.json()["id"]
    return {"type": "file", "id": file_id, "name": "locker.txt"}, {"file_id": file_id}


def attach_knowledge(
    stack: contextlib.ExitStack, admin: Actor, person: Actor, secret: str
) -> tuple[dict, dict]:
    # a user may not create knowledge by default, so the admin shares one for reading
    client = stack.enter_context(admin.client())
    base = knowledge_base(client, "Locker codes", [grant("user", person.id, "read")])
    knowledge_id = stack.enter_context(base)
    file_id = add_text_file(client, knowledge_id, "locker.txt", secret)
    attachment = {"type": "collection", "id": knowledge_id, "name": "Locker codes"}
    return attachment, {"file_id": file_id}


def attach_chat(
    stack: contextlib.ExitStack, admin: Actor, person: Actor, secret: str
) -> tuple[dict, dict]:
    with person.client() as client:
        chat_id, _ = seed_chat(
            client,
            [
                {"role": "user", "content": "Remember the locker code for me."},
                {"role": "assistant", "content": secret},
            ],
        )
    return {"type": "chat", "id": chat_id, "name": "Seeded chat"}, {"chat_id": chat_id}


def attach_note(
    stack: contextlib.ExitStack, admin: Actor, person: Actor, secret: str
) -> tuple[dict, dict]:
    with person.client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={"title": "Locker", "data": {"content": {"md": secret}}},
        )
    assert created.status_code == 200, f"creating the note failed: {created.text}"
    note_id = created.json()["id"]
    return {"type": "note", "id": note_id, "name": "Locker"}, {"note_id": note_id}


ATTACH: dict[str, Callable[..., tuple[dict, dict]]] = {
    "file": attach_file,
    "collection": attach_knowledge,
    "chat": attach_chat,
    "note": attach_note,
}


def prompt_text(request: dict) -> str:
    """Everything the model was handed before it called any tool of its own."""
    return str(conversation(request))


def ask_with(
    person: Actor, upstream, model_id: str, attachment: dict, *replies, **options
) -> list[dict]:
    """Ask on `model_id` with `attachment`, the way the web client sends it; the requests the
    provider got for the question, oldest first."""
    question = f"what is the locker code? {uuid.uuid4().hex[:8]}"
    answering = reply.answering(question)
    upstream.queue(*[scripted(answering) for scripted in replies])
    with person.client() as client:
        ask(client, question, model=model_id, chat_files=[attachment], **options)
    requests = [request for request in upstream.chat_requests() if answering(request)]
    assert requests, f"the provider never received {question!r}"
    return requests


KINDS = pytest.mark.parametrize("kind", list(ATTACH))


@KINDS
def test_in_native_mode_the_model_reads_an_attachment_through_its_tool(
    kind, admin, make_user, upstream
):
    person = make_user()
    secret = f"The locker code is {uuid.uuid4().hex[:6]}."
    with contextlib.ExitStack() as stack:
        attachment, arguments = ATTACH[kind](stack, admin, person, secret)
        model_id = stack.enter_context(preset_with_capabilities(admin, {}))
        first, *_, follow_up = ask_with(
            person,
            upstream,
            model_id,
            attachment,
            lambda match: reply.tool_call(READERS[kind], arguments, match=match),
            lambda match: reply.text("Got it.", match=match),
        )

    assert secret not in prompt_text(first), f"the {kind} was preloaded in native mode: {first}"
    offered = {tool["function"]["name"] for tool in first.get("tools") or []}
    assert READERS[kind] in offered, f"{READERS[kind]} was not offered, only {sorted(offered)}"
    results = [entry["content"] for entry in follow_up["messages"] if entry["role"] == "tool"]
    assert any(secret in str(result) for result in results), (
        f"{READERS[kind]} did not return the {kind}'s text: {results}"
    )


@KINDS
@pytest.mark.parametrize(
    ("function_calling", "capabilities", "preloaded"),
    [
        ("native", {"file_context": True}, True),
        ("legacy", {}, True),
        ("legacy", {"file_context": False}, False),
    ],
    ids=["native ticked", "legacy default", "legacy unticked"],
)
def test_preload_files_puts_the_text_in_the_prompt_only_when_on(
    kind, function_calling, capabilities, preloaded, admin, make_user, upstream
):
    person = make_user()
    secret = f"The locker code is {uuid.uuid4().hex[:6]}."
    with contextlib.ExitStack() as stack:
        attachment, _ = ATTACH[kind](stack, admin, person, secret)
        preset = preset_with_capabilities(admin, capabilities, function_calling)
        model_id = stack.enter_context(preset)
        [request] = ask_with(
            person, upstream, model_id, attachment, lambda match: reply.text("ok", match=match)
        )

    sent = prompt_text(request)
    assert (secret in sent) == preloaded, (
        f"{function_calling} mode with {capabilities or 'Preload files unset'}: the {kind}'s text "
        f"was {'missing from' if preloaded else 'put into'} the prompt: {sent}"
    )


@KINDS
def test_legacy_function_calling_chosen_for_the_chat_preloads_the_text(
    kind, admin, make_user, upstream
):
    # Chat Controls (or the account's own Advanced Params) send the mode with each message
    person = make_user()
    secret = f"The locker code is {uuid.uuid4().hex[:6]}."
    with contextlib.ExitStack() as stack:
        attachment, _ = ATTACH[kind](stack, admin, person, secret)
        model_id = stack.enter_context(preset_with_capabilities(admin, {}))
        [request] = ask_with(
            person,
            upstream,
            model_id,
            attachment,
            lambda match: reply.text("ok", match=match),
            params={"function_calling": "legacy"},
        )

    assert secret in prompt_text(request), (
        f"the {kind} was not preloaded with legacy chosen for the chat: {prompt_text(request)}"
    )


# --- the retrieval settings behind the preloaded text ------------------------------------------


def test_the_retrieval_defaults_send_ten_pieces_of_1024_characters_or_files_up_to_50000_whole(
    admin,
):
    with admin.client() as client:
        current = client.get(RETRIEVAL_CONFIG[0])
    assert current.status_code == 200, current.text
    config = current.json()
    defaults = {
        "TOP_K": config["TOP_K"],
        "TOP_K_RERANKER": config["TOP_K_RERANKER"],
        "CHUNK_SIZE": config["CHUNK_SIZE"],
        "RAG_FULL_TEXT_CHAR_THRESHOLD": config["RAG_FULL_TEXT_CHAR_THRESHOLD"],
        "RAG_FULL_CONTEXT": config["RAG_FULL_CONTEXT"],
    }
    assert defaults == {
        "TOP_K": 10,
        "TOP_K_RERANKER": 10,
        "CHUNK_SIZE": 1024,
        "RAG_FULL_TEXT_CHAR_THRESHOLD": 50000,
        "RAG_FULL_CONTEXT": False,
    }, f"the retrieval defaults changed: {defaults}"


def test_the_full_text_threshold_is_saved_and_every_account_reads_it(admin, make_user, preserve):
    preserve(RETRIEVAL_CONFIG)
    with admin.client() as client:
        saved = client.post(RETRIEVAL_CONFIG[1], json={"RAG_FULL_TEXT_CHAR_THRESHOLD": 1234})
        assert saved.status_code == 200, saved.text
        assert saved.json()["RAG_FULL_TEXT_CHAR_THRESHOLD"] == 1234, saved.json()
        assert client.get(RETRIEVAL_CONFIG[0]).json()["RAG_FULL_TEXT_CHAR_THRESHOLD"] == 1234

    with make_user().client() as client:
        seen = client.get("/api/config").json()["file"]
    assert seen["full_text_char_threshold"] == 1234, f"a user read the file settings as {seen}"
    assert seen["full_context"] is False, seen


def test_always_use_full_text_shows_as_full_context_to_every_account(admin, make_user, preserve):
    preserve(RETRIEVAL_CONFIG)
    with admin.client() as client:
        saved = client.post(RETRIEVAL_CONFIG[1], json={"RAG_FULL_CONTEXT": True})
    assert saved.status_code == 200, saved.text
    assert saved.json()["RAG_FULL_CONTEXT"] is True, saved.json()

    with make_user().client() as client:
        seen = client.get("/api/config").json()["file"]
    assert seen["full_context"] is True, f"a user read the file settings as {seen}"


def test_a_negative_full_text_threshold_is_refused(admin, preserve):
    preserve(RETRIEVAL_CONFIG)
    with admin.client() as client:
        set_first = client.post(RETRIEVAL_CONFIG[1], json={"RAG_FULL_TEXT_CHAR_THRESHOLD": 777})
        assert set_first.status_code == 200, set_first.text
        refused = client.post(RETRIEVAL_CONFIG[1], json={"RAG_FULL_TEXT_CHAR_THRESHOLD": -1})
        kept = client.get(RETRIEVAL_CONFIG[0]).json()["RAG_FULL_TEXT_CHAR_THRESHOLD"]

    assert refused.status_code == 422, f"-1 was answered {refused.status_code}: {refused.text}"
    assert kept == 777, f"the refused threshold changed the saved one to {kept}"


@pytest.mark.slow
def test_the_threshold_env_var_seeds_a_fresh_instance(instance_with):
    launched = instance_with({"RAG_FULL_TEXT_CHAR_THRESHOLD": "321"})
    with admin_of(launched).client() as client:
        config = client.get(RETRIEVAL_CONFIG[0]).json()
    assert config["RAG_FULL_TEXT_CHAR_THRESHOLD"] == 321, config["RAG_FULL_TEXT_CHAR_THRESHOLD"]
