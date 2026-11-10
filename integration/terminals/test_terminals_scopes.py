"""Journey: a terminal's scope decides whose machine a request reaches, and nothing else does.

New on dev after 0.13.3 (titled `refac`, 6f2fc7bc7 to c15f28649). A terminal on a Terminals
orchestrator has a scope: one machine for everyone, one per user, one per chat or one per model.
Open WebUI names the user, the chat and the model to the orchestrator from what it checked
itself: the signed-in account, a saved chat that account may write to, a model that account may
use. A per-chat terminal has no machine for an unsaved chat and a per-model terminal none
without a model. Headers a client adds pick no other machine, and a chat of someone else opens
none. In a chat shared for replies a collaborator uses the terminal its owner picked, a per-chat
terminal runs in that chat's one machine as long as the owner may use the terminal, and the
collaborator may change the chat's models, terminal and working directory and nothing else.

The real Terminals service (open-webui/terminals) runs on Docker and starts a real Open Terminal
container per machine (`harness.terminals`); without its images or Docker the tests skip.

Discriminates: passes on dev c15f28649 with Terminals 0.4.1 and Open Terminal 0.17.7. Each
mutation below, made in a backend copy, turns its tests red. A new terminal without a scope
created as per user fails the one-machine test. The chat left out when a machine's state is read
fails the per-chat test (409 for a saved chat) and the state checks of the someone-else's-chat and
collaborator tests. One fixed model named to the orchestrator fails the per-model test (the
model's own machine never starts), and the model's name left out of the machine list fails it too.
The check that the person may use the model removed fails the model-access test (a private model's
machine starts). The caller's `X-User-Id`, `X-Terminal-Template-Id` and `X-Terminal-Chat-Id`
headers passed on over the resolved ones fails the headers test (the command prints the other
person's file). The chat access check removed fails the someone-else's-chat test. A `local:` chat
id taken for a saved chat fails the unsaved-chat test (403). The owner's-selection check removed
fails the collaborator test, the changed-selection check removed fails the selection test, and a
per-chat terminal checked against the caller in place of the chat's owner fails the owner's-access
test (the collaborator is refused in the owner's chat). `title` added to what a collaborator may
change fails the chat-fields test. The orchestrator's reason replaced by a fixed text fails the
unknown-scope test.
"""

from __future__ import annotations

import contextlib
import json
import uuid
from typing import Iterator

import httpx
import pytest

from harness.access import grant
from harness.python_tools import EVERYONE_READS
from harness.terminal_server import read_grant
from harness.terminals import (
    create_template,
    execute,
    instance_of,
    list_instances,
    output_of,
    share_template,
)
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

NEEDS_SAVED_CHAT = "This terminal requires a saved chat"
OWNERS_TERMINAL_ONLY = "Use the terminal selected by the chat owner"


def _reason(answer: httpx.Response) -> tuple[int, object]:
    return answer.status_code, answer.json().get("detail")


def _new_chat(client: httpx.Client, **chat) -> str:
    created = client.post("/api/v1/chats/new", json={"chat": {"title": "Terminal", **chat}})
    assert created.status_code == 200, f"creating the chat failed: {created.text}"
    return created.json()["id"]


def _share_chat(owner, chat_id: str, *members, share_mode: str | None = "continue") -> None:
    """Share the chat as the share dialog does; "continue" lets the members reply in it."""
    with owner.client() as client:
        shared = client.post(f"/api/v1/chats/{chat_id}/share", json={"share_mode": None})
        assert shared.status_code == 200, f"sharing the chat failed: {shared.text}"
        saved = client.post(
            f"/api/v1/chats/shared/{chat_id}/access/update",
            json={
                "access_grants": [grant("user", member.id, "read") for member in members],
                "share_mode": share_mode,
            },
        )
    assert saved.status_code == 200, f"saving the chat's access failed: {saved.text}"


def _terminals_for(admin, connect_terminals, people, *scopes) -> list[str]:
    """One terminal per scope on one connection, each shared with `people`; returns their ids."""
    grants = [read_grant(person.id) for person in people]
    connection = connect_terminals(*grants)
    terminal_ids = []
    with admin.client() as client:
        for scope in scopes:
            terminal_id = create_template(client, connection, scope=scope)["terminal_id"]
            share_template(client, terminal_id, *grants)
            terminal_ids.append(terminal_id)
    return terminal_ids


@contextlib.contextmanager
def _workspace_model(admin, name: str, *grants: dict) -> Iterator[str]:
    """A workspace model over the scripted one, readable by `grants`; deleted afterwards."""
    model_id = f"terminal-model-{uuid.uuid4().hex[:8]}"
    form = {
        "id": model_id,
        "name": name,
        "base_model_id": MOCK_MODEL_ID,
        "meta": {},
        "params": {},
        "access_grants": list(grants),
    }
    with admin.client() as client:
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, f"creating the model failed: {created.text}"
        try:
            client.get("/api/models").raise_for_status()  # registers the model
            yield model_id
        finally:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


def test_a_terminal_with_one_machine_for_all_gives_two_people_the_same_machine(
    admin, make_user, connect_terminals, terminals
):
    first, second = make_user(), make_user()
    (terminal_id,) = _terminals_for(admin, connect_terminals, [first, second], None)
    line = f"left for everyone {uuid.uuid4().hex[:8]}"

    with first.client() as client:
        output_of(execute(client, terminal_id, f"echo {line} > ~/shared.txt"))
        firsts = instance_of(client, terminal_id)
    with second.client() as client:
        printed = output_of(execute(client, terminal_id, "cat ~/shared.txt"))
        seconds = instance_of(client, terminal_id)

    assert line in printed, f"the second person does not see the first one's file: {printed!r}"
    assert (firsts["context_id"], firsts["status"]) == ("default", "running"), (
        f"the first person's machine is {firsts}"
    )
    assert seconds["id"] == firsts["id"], (
        f"the two people have machines of their own: {firsts} and {seconds}"
    )
    assert terminals.containers() == [f"terminal-{firsts['id']}"], (
        f"the terminal runs more than its one machine: {terminals.containers()}"
    )


def test_each_chat_of_a_per_chat_terminal_has_a_machine_of_its_own(
    admin, make_user, connect_terminals, terminals
):
    person = make_user()
    (terminal_id,) = _terminals_for(admin, connect_terminals, [person], "chat")
    tag = uuid.uuid4().hex[:8]
    name, own = f"first-{tag}.txt", f"second-{tag}.txt"

    with person.client() as client:
        first_chat, second_chat = _new_chat(client), _new_chat(client)
        before = instance_of(client, terminal_id, chat_id=first_chat)
        output_of(execute(client, terminal_id, f"echo first > ~/{name}", first_chat))
        first = instance_of(client, terminal_id, chat_id=first_chat)
        second_before = instance_of(client, terminal_id, chat_id=second_chat)
        seen = output_of(execute(client, terminal_id, f"touch ~/{own}; ls ~", second_chat)).split()
        second = instance_of(client, terminal_id, chat_id=second_chat)
        without_chat = client.get(f"/api/v1/terminals/templates/{terminal_id}/instance")

    assert before["status"] == "not_created", f"the chat had a machine before its use: {before}"
    assert (first["status"], first["context_id"]) == ("running", f"chat:{first_chat}"), (
        f"the first chat's machine is {first}"
    )
    assert second_before["status"] == "not_created", (
        f"the first chat's command gave the second chat a machine: {second_before}"
    )
    assert own in seen and name not in seen, (
        f"the second chat's machine shows the first chat's file: {seen}"
    )
    assert (second["status"], second["context_id"]) == ("running", f"chat:{second_chat}"), (
        f"the second chat's machine is {second}"
    )
    assert terminals.holding(name) == {f"chat:{first_chat}": "first\n"}, (
        f"on disk the first chat's file is in {terminals.holding(name)}"
    )
    assert _reason(without_chat) == (409, NEEDS_SAVED_CHAT), (
        f"a per-chat terminal's machine without a chat answered {without_chat.text}"
    )


def test_each_model_of_a_per_model_terminal_has_one_machine_for_everyone_using_it(
    admin, make_user, connect_terminals, terminals
):
    first, second = make_user(), make_user()
    (terminal_id,) = _terminals_for(admin, connect_terminals, [first, second], "model")
    tag = uuid.uuid4().hex[:8]
    name, model_name = f"model-{tag}.txt", f"Review Desk {tag}"

    with _workspace_model(admin, model_name, EVERYONE_READS) as preset:
        with first.client() as client:
            output_of(execute(client, terminal_id, f"echo base > ~/{name}", model_id=MOCK_MODEL_ID))
            base = instance_of(client, terminal_id, model_id=MOCK_MODEL_ID)
            on_preset = output_of(
                execute(client, terminal_id, f"touch ~/own-{tag}; ls ~", model_id=preset)
            ).split()
        with second.client() as client:
            on_base = output_of(
                execute(client, terminal_id, f"cat ~/{name}", model_id=MOCK_MODEL_ID)
            )
            preset_machine = instance_of(client, terminal_id, model_id=preset)
        with admin.client() as client:
            listed = list_instances(client, terminal_id)

    assert (base["status"], base["context_id"]) == ("running", f"model:{MOCK_MODEL_ID}"), (
        f"the first model's machine is {base}"
    )
    assert f"own-{tag}" in on_preset and name not in on_preset, (
        f"another model's machine shows the first model's file: {on_preset}"
    )
    assert "base" in on_base, (
        f"a second person on the same model does not reach its machine: {on_base!r}"
    )
    assert (preset_machine["status"], preset_machine["context_id"]) == (
        "running",
        f"model:{preset}",
    ), f"the second person sees the other model's machine as {preset_machine}"
    assert terminals.holding(name) == {f"model:{MOCK_MODEL_ID}": "base\n"}, (
        f"on disk the first model's file is in {terminals.holding(name)}"
    )
    named = {context: machine.get("label") for context, machine in listed.items()}
    assert named == {f"model:{MOCK_MODEL_ID}": MOCK_MODEL_ID, f"model:{preset}": model_name}, (
        f"the machine list names the models' machines {named}"
    )


def test_a_per_model_terminal_needs_a_model_the_person_may_use(
    admin, make_user, connect_terminals, terminals
):
    person = make_user()
    (terminal_id,) = _terminals_for(admin, connect_terminals, [person], "model")

    with _workspace_model(admin, "Private Desk") as private:
        with person.client() as client:
            no_model = execute(client, terminal_id, "echo ran")
            no_model_state = client.get(f"/api/v1/terminals/templates/{terminal_id}/instance")
            private_model = execute(client, terminal_id, "echo ran", model_id=private)
            private_state = client.get(
                f"/api/v1/terminals/templates/{terminal_id}/instance", params={"model_id": private}
            )
            unknown_model = execute(client, terminal_id, "echo ran", model_id="no-such-model")
            started_by_refused = terminals.instances()
            allowed = output_of(execute(client, terminal_id, "echo ran", model_id=MOCK_MODEL_ID))
        with admin.client() as client:
            admins = output_of(execute(client, terminal_id, "echo ran", model_id=private))

    for label, answer in (("a command", no_model), ("the machine's state", no_model_state)):
        assert _reason(answer) == (409, "This terminal requires a model"), (
            f"{label} without a model answered {answer.status_code} {answer.text}"
        )
    for label, answer in (
        ("a command with a model the person may not use", private_model),
        ("the machine's state for a model the person may not use", private_state),
        ("a command with a model that does not exist", unknown_model),
    ):
        assert _reason(answer) == (403, "Model access denied"), (
            f"{label} answered {answer.status_code} {answer.text}"
        )
    assert started_by_refused == [], f"a refused request started a machine: {started_by_refused}"
    assert allowed.strip() == "ran", f"with a model the person may use it printed {allowed!r}"
    assert admins.strip() == "ran", f"the private model's own admin got {admins!r}"


def test_headers_a_client_adds_pick_no_other_machine(
    admin, make_user, connect_terminals, terminals
):
    person, other = make_user(), make_user()
    per_user, per_chat = _terminals_for(admin, connect_terminals, [person, other], "user", "chat")
    tag = uuid.uuid4().hex[:8]
    secret, mine = f"secret-{tag}.txt", f"mine-{tag}.txt"

    with other.client() as client:
        others_chat = _new_chat(client)
        output_of(execute(client, per_user, f"echo theirs > ~/{secret}"))
        output_of(execute(client, per_chat, f"echo theirs > ~/{secret}", others_chat))
    forged = {
        "X-User-Id": other.id,
        "X-Terminal-Template-Id": per_chat.rsplit(":", 1)[1],
        "X-Terminal-Chat-Id": others_chat,
    }
    look = f"touch ~/{mine}; ls ~; cat ~/{secret}"
    with person.client() as client:
        own_chat = _new_chat(client)
        on_user = client.post(
            f"/api/v1/terminals/{per_user}/execute",
            params={"wait": 30},
            headers=forged,
            json={"command": look},
        )
        on_chat = client.post(
            f"/api/v1/terminals/{per_chat}/execute",
            params={"wait": 30},
            headers={**forged, "X-Session-Id": own_chat},
            json={"command": look},
        )
        named_user = instance_of(client, per_user, user_id=other.id)

    assert set(terminals.holding(secret)) == {f"user:{other.id}", f"chat:{others_chat}"}, (
        f"the other person's files are not where the test put them: {terminals.holding(secret)}"
    )
    for label, answer in (("per-user", on_user), ("per-chat", on_chat)):
        printed = output_of(answer)
        assert mine in printed.split() and "theirs" not in printed, (
            f"forged headers on the {label} terminal reached the other person's machine: "
            f"{printed!r}"
        )
    assert set(terminals.holding(mine)) == {f"user:{person.id}", f"chat:{own_chat}"}, (
        f"the person's commands ran in these machines: {set(terminals.holding(mine))}"
    )
    assert named_user["context_id"] == f"user:{person.id}", (
        f"naming another user gave the person that user's machine state: {named_user}"
    )


def test_a_chat_of_someone_else_opens_no_machine(admin, make_user, connect_terminals, terminals):
    owner, stranger = make_user(), make_user()
    per_chat, per_user = _terminals_for(admin, connect_terminals, [owner, stranger], "chat", "user")
    name = f"private-{uuid.uuid4().hex[:8]}.txt"

    with owner.client() as client:
        chat_id = _new_chat(client, terminal_id=per_chat)
        output_of(execute(client, per_chat, f"echo owner > ~/{name}", chat_id))
    machines_before = terminals.instances()
    answers = {}
    for label, account in (("another person", stranger), ("an admin", admin)):
        with account.client() as client:
            answers[f"{label}'s command"] = execute(client, per_chat, f"cat ~/{name}", chat_id)
            answers[f"{label}'s command on a per-user terminal"] = execute(
                client, per_user, "echo ran", chat_id
            )
            answers[f"{label}'s look at the machine's state"] = client.get(
                f"/api/v1/terminals/templates/{per_chat}/instance", params={"chat_id": chat_id}
            )

    assert [machine["context_id"] for machine in machines_before] == [f"chat:{chat_id}"], (
        f"the owner's chat did not get its machine: {machines_before}"
    )
    for label, answer in answers.items():
        assert _reason(answer) == (403, "Chat access denied"), (
            f"{label} naming someone else's chat answered {answer.status_code} {answer.text}"
        )
    assert terminals.instances() == machines_before, (
        f"a refused request changed the machines: {terminals.instances()}"
    )


def test_an_unsaved_chat_runs_in_the_persons_machine_and_has_no_per_chat_one(
    admin, make_user, connect_terminals, terminals
):
    person = make_user()
    per_user, per_chat = _terminals_for(admin, connect_terminals, [person], "user", "chat")
    unsaved_chats = [f"temporary:{uuid.uuid4()}", f"local:{uuid.uuid4()}"]

    with person.client() as client:
        ran = {chat: execute(client, per_user, "echo ran", chat) for chat in unsaved_chats}
        refused = {chat: execute(client, per_chat, "echo ran", chat) for chat in unsaved_chats}

    for chat, answer in ran.items():
        assert output_of(answer).strip() == "ran", f"the per-user terminal failed for {chat}"
    for chat, answer in refused.items():
        assert _reason(answer) == (409, NEEDS_SAVED_CHAT), (
            f"the per-chat terminal answered {answer.status_code} {answer.text} for {chat}"
        )
    assert [machine["context_id"] for machine in terminals.instances()] == [f"user:{person.id}"], (
        f"unsaved chats ran somewhere other than the person's one machine: {terminals.instances()}"
    )


def test_a_collaborator_uses_only_the_terminal_the_chats_owner_picked(
    admin, make_user, connect_terminals, terminals
):
    owner, guest = make_user(), make_user()
    picked, other = _terminals_for(admin, connect_terminals, [owner, guest], "user", "user")

    with owner.client() as client:
        chat_id = _new_chat(client, terminal_id=picked)
        chat_without_terminal = _new_chat(client)
    _share_chat(owner, chat_id, guest)
    _share_chat(owner, chat_without_terminal, guest)
    with guest.client() as client:
        on_picked = execute(client, picked, "echo ran", chat_id)
        on_other = execute(client, other, "echo ran", chat_id)
        on_other_state = client.get(
            f"/api/v1/terminals/templates/{other}/instance", params={"chat_id": chat_id}
        )
        nothing_picked = execute(client, picked, "echo ran", chat_without_terminal)
        own_chat = output_of(execute(client, other, "echo ran", _new_chat(client)))
        machine = instance_of(client, picked, chat_id=chat_id)

    assert output_of(on_picked).strip() == "ran", "the collaborator cannot use the picked terminal"
    assert machine["context_id"] == f"user:{guest.id}", (
        f"on a per-user terminal the collaborator did not get their own machine: {machine}"
    )
    for label, answer in (
        ("a command on another terminal", on_other),
        ("the machine state of another terminal", on_other_state),
        ("a command in a chat without a terminal", nothing_picked),
    ):
        assert _reason(answer) == (403, OWNERS_TERMINAL_ONLY), (
            f"{label} in someone else's chat answered {answer.status_code} {answer.text}"
        )
    assert own_chat.strip() == "ran", "the collaborator cannot use that terminal in their own chat"


def test_a_chat_that_stores_another_terminal_refuses_the_one_asked_for(
    admin, make_user, connect_terminals, terminals
):
    person = make_user()
    first, second = _terminals_for(admin, connect_terminals, [person], "user", "user")
    changed = "Chat terminal selection changed; refresh the chat"

    with person.client() as client:
        chat_id = _new_chat(client, terminal_id=first)
        on_stored = output_of(execute(client, first, "echo ran", chat_id))
        on_other = execute(client, second, "echo ran", chat_id)
        saved = client.post(f"/api/v1/chats/{chat_id}", json={"chat": {"terminal_id": second}})
        on_new = output_of(execute(client, second, "echo ran", chat_id))
        on_old = execute(client, first, "echo ran", chat_id)

    assert on_stored.strip() == "ran", f"the chat's own terminal printed {on_stored!r}"
    assert _reason(on_other) == (409, changed), (
        f"a terminal the chat does not store answered {on_other.status_code} {on_other.text}"
    )
    assert saved.status_code == 200, f"saving the chat's new terminal failed: {saved.text}"
    assert on_new.strip() == "ran", f"the newly picked terminal printed {on_new!r}"
    assert _reason(on_old) == (409, changed), (
        f"the terminal picked before answered {on_old.status_code} {on_old.text}"
    )


def test_a_per_chat_terminal_in_a_shared_chat_follows_the_owners_access(
    admin, make_user, connect_terminals, terminals
):
    owner, guest = make_user(), make_user()
    (terminal_id,) = _terminals_for(admin, connect_terminals, [owner, guest], "chat")
    name = f"together-{uuid.uuid4().hex[:8]}.txt"

    with owner.client() as client:
        chat_id = _new_chat(client, terminal_id=terminal_id)
    _share_chat(owner, chat_id, guest)
    with admin.client() as client:
        share_template(client, terminal_id, read_grant(owner.id))
    with guest.client() as client:
        output_of(execute(client, terminal_id, f"echo guest > ~/{name}", chat_id))
        in_own_chat = execute(client, terminal_id, "echo ran", _new_chat(client))
    with owner.client() as client:
        owners_view = output_of(execute(client, terminal_id, f"cat ~/{name}", chat_id))
    held = terminals.holding(name)
    with admin.client() as client:
        share_template(client, terminal_id, read_grant(guest.id))
    with guest.client() as client:
        owner_lost_access = execute(client, terminal_id, "echo ran", chat_id)
        in_own_chat_now = output_of(execute(client, terminal_id, "echo ran", _new_chat(client)))

    assert held == {f"chat:{chat_id}": "guest\n"}, (
        f"the collaborator's file is not in the chat's one machine: {held}"
    )
    assert "guest" in owners_view, (
        f"the owner does not see the collaborator's file: {owners_view!r}"
    )
    assert _reason(in_own_chat) == (403, "Terminal access denied"), (
        f"a terminal not shared with the collaborator ran in their own chat: {in_own_chat.text}"
    )
    assert _reason(owner_lost_access) == (403, "Terminal access denied"), (
        "the chat's machine still ran after its owner lost the terminal: "
        f"{owner_lost_access.status_code} {owner_lost_access.text}"
    )
    assert in_own_chat_now.strip() == "ran", (
        f"the terminal now shared with the collaborator printed {in_own_chat_now!r}"
    )


def test_a_collaborator_changes_only_a_chats_models_terminal_and_working_directory(
    admin, make_user, connect_terminals, terminals
):
    owner, guest, reader = make_user(), make_user(), make_user()
    picked, other = _terminals_for(admin, connect_terminals, [owner, guest], "user", "user")
    title = f"Budget {uuid.uuid4().hex[:8]}"

    with owner.client() as client:
        chat_id = _new_chat(client, title=title, terminal_id=picked)
        read_only_chat = _new_chat(client, title=title, terminal_id=picked)
    _share_chat(owner, chat_id, guest)
    _share_chat(owner, read_only_chat, reader, share_mode=None)
    allowed = {"terminal_id": other, "working_directory": "/home/user/plans", "models": ["m"]}
    with guest.client() as client:
        saved = client.post(f"/api/v1/chats/{chat_id}", json={"chat": allowed})
        on_other = output_of(execute(client, other, "echo ran", chat_id))
        refused = {
            "the title": client.post(f"/api/v1/chats/{chat_id}", json={"chat": {"title": "Mine"}}),
            "the title beside the terminal": client.post(
                f"/api/v1/chats/{chat_id}", json={"chat": {"title": "Mine", "terminal_id": picked}}
            ),
            "the messages": client.post(
                f"/api/v1/chats/{chat_id}", json={"chat": {"history": {"messages": {}}}}
            ),
        }
    with reader.client() as client:
        refused["the terminal of a chat shared for reading"] = client.post(
            f"/api/v1/chats/{read_only_chat}", json={"chat": {"terminal_id": other}}
        )
    with owner.client() as client:
        stored = client.get(f"/api/v1/chats/{chat_id}").json()
        stored_read_only = client.get(f"/api/v1/chats/{read_only_chat}").json()

    assert saved.status_code == 200, f"the collaborator's change failed: {saved.text}"
    for label, answer in refused.items():
        assert answer.status_code == 401, (
            f"a collaborator changed {label}: {answer.status_code} {answer.text}"
        )
    kept = {key: stored["chat"].get(key) for key in allowed}
    assert kept == allowed, f"the owner's chat does not hold the collaborator's change: {kept}"
    assert (stored["title"], stored["chat"]["title"]) == (title, title), (
        f"the chat's title changed: {stored['title']!r} {stored['chat']['title']!r}"
    )
    assert stored_read_only["chat"].get("terminal_id") == picked, (
        f"a reader changed the chat's terminal: {stored_read_only['chat']}"
    )
    assert on_other.strip() == "ran", (
        f"the terminal the collaborator picked for the chat printed {on_other!r}"
    )


def test_a_terminal_with_an_unknown_scope_is_not_created(admin, connect_terminals, terminals):
    connection = connect_terminals()

    with admin.client() as client:
        refused = client.post(
            "/api/v1/terminals/templates",
            json={
                "id": str(uuid.uuid4()),
                "connection_id": connection["id"],
                "data": {"name": "Per galaxy", "scope": "galaxy"},
            },
        )
        menu_after_refusal = client.get("/api/v1/terminals/").json()
        created = create_template(client, connection, scope="model")
        menu = client.get("/api/v1/terminals/").json()

    assert refused.status_code == 422, (
        f"a terminal with an unknown scope answered {refused.status_code} {refused.text}"
    )
    assert "scope" in json.dumps(refused.json().get("detail")), (
        f"the refusal does not name the scope: {refused.text}"
    )
    assert menu_after_refusal == [], f"the refused terminal is in the menu: {menu_after_refusal}"
    assert [(entry["id"], entry["scope"]) for entry in menu] == [
        (created["terminal_id"], "model")
    ], f"a terminal with a known scope is listed as {menu}"
