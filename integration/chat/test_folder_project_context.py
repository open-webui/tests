"""Journey: a folder's system prompt reaches the turns of the chats filed in it, and only those.

Every turn of a chat in a folder carries the folder's system prompt, the first and the later
ones; a chat outside never does. Moving a chat in over the API gives its next turn the prompt and
moving it out takes it away. A chat in a subfolder gets the subfolder's prompt and not its
parent's, and one in a subfolder without a prompt gets none. A member a folder is shared with for
writing gets the owner's prompt in the chat they start there, and an automation filed in a folder
runs under its prompt. Each turn is read from the request the scripted provider received.

Twin of e2e/chat/test_folder_project_context.py.

`test_an_automation_filed_in_a_folder_runs_under_its_prompt` is red on dev 62f70a844: since
de73bb830 a chat request whose reply message is already stored in the chat, the way automations,
sub-agents and timers prepare their reply, is refused with 409 and the reply is never written
(open-webui/open-webui#32066).

Discriminates: passes on dev ebc6add67; in a backend copy with the middleware's folder lookup
returning no folder every test that expects the prompt goes red, and with the prompt taken from
the folder's top-level ancestor the two subfolder tests go red.
"""

from __future__ import annotations

import time
import uuid

import pytest

from harness import upstream as reply
from harness.access import make_group
from harness.chat import ask
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

HARBOUR_PROMPT = "Answer as the harbour master of Port Ellen."
GARDEN_PROMPT = "Answer as the head gardener of Kew."
SEEDS_PROMPT = "Answer as the seed keeper of the vault."


def create_folder(client, name: str, system_prompt: str | None = None, parent_id=None) -> str:
    data = {"system_prompt": system_prompt} if system_prompt else None
    created = client.post(
        "/api/v1/folders/", json={"name": name, "parent_id": parent_id, "data": data}
    )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def system_sent(upstream, prompt: str) -> str:
    """The system text of the one provider request that answered `prompt`."""
    [request] = [body for body in upstream.chat_requests() if reply.answering(prompt)(body)]
    return "\n".join(
        str(message["content"]) for message in request["messages"] if message["role"] == "system"
    )


def turn(client, upstream, prompt: str, **options):
    upstream.queue(reply.text(f"answer to {prompt}", match=reply.answering(prompt)))
    sent, answer = ask(client, prompt, **options)
    assert answer["content"] == f"answer to {prompt}", answer
    return sent, answer


def follow_up(client, upstream, earlier, prompt: str):
    return turn(
        client,
        upstream,
        prompt,
        chat_id=earlier.chat_id,
        parent_id=earlier.assistant_message_id,
        history=[{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "yes"}],
    )


def test_every_turn_in_the_folder_carries_its_prompt_and_no_chat_outside(make_user, upstream):
    owner = make_user()
    with owner.client() as client:
        folder_id = create_folder(client, "Harbour", HARBOUR_PROMPT)
        first, _ = turn(client, upstream, "when is high tide?", folder_id=folder_id)
        follow_up(client, upstream, first, "and low tide?")
        turn(client, upstream, "what is for lunch?")

    assert system_sent(upstream, "when is high tide?").startswith(HARBOUR_PROMPT)
    assert system_sent(upstream, "and low tide?").startswith(HARBOUR_PROMPT)
    assert HARBOUR_PROMPT not in system_sent(upstream, "what is for lunch?")


def test_moving_a_chat_in_and_out_decides_its_next_turn(make_user, upstream):
    owner = make_user()
    with owner.client() as client:
        folder_id = create_folder(client, "Harbour", HARBOUR_PROMPT)
        loose, _ = turn(client, upstream, "before the move?")
        moved = client.post(f"/api/v1/chats/{loose.chat_id}/folder", json={"folder_id": folder_id})
        assert moved.status_code == 200, moved.text
        filed, _ = follow_up(client, upstream, loose, "after the move?")
        out = client.post(f"/api/v1/chats/{loose.chat_id}/folder", json={"folder_id": None})
        assert out.status_code == 200, out.text
        follow_up(client, upstream, filed, "after leaving?")

    assert HARBOUR_PROMPT not in system_sent(upstream, "before the move?")
    assert system_sent(upstream, "after the move?").startswith(HARBOUR_PROMPT)
    assert HARBOUR_PROMPT not in system_sent(upstream, "after leaving?")


def test_a_subfolder_chat_gets_the_subfolders_prompt_and_not_its_parents(make_user, upstream):
    owner = make_user()
    with owner.client() as client:
        garden_id = create_folder(client, "Garden", GARDEN_PROMPT)
        seeds_id = create_folder(client, "Seeds", SEEDS_PROMPT, parent_id=garden_id)
        turn(client, upstream, "which beans keep longest?", folder_id=seeds_id)

    in_seeds = system_sent(upstream, "which beans keep longest?")
    assert in_seeds.startswith(SEEDS_PROMPT), in_seeds
    assert GARDEN_PROMPT not in in_seeds


def test_a_subfolder_without_a_prompt_does_not_borrow_its_parents(make_user, upstream):
    owner = make_user()
    with owner.client() as client:
        garden_id = create_folder(client, "Garden", GARDEN_PROMPT)
        sheds_id = create_folder(client, "Sheds", parent_id=garden_id)
        turn(client, upstream, "where are the rakes?", folder_id=sheds_id)

    assert GARDEN_PROMPT not in system_sent(upstream, "where are the rakes?")


def test_a_writing_member_gets_the_owners_prompt_in_the_folder(make_user, admin, upstream):
    owner, member = make_user(), make_user()
    group_id = make_group(admin, [owner, member], {"sharing": {"folders": True}})
    with owner.client() as client:
        folder_id = create_folder(client, "Harbour", HARBOUR_PROMPT)
        grants = [
            {"principal_type": "group", "principal_id": group_id, "permission": permission}
            for permission in ("read", "write")
        ]
        shared = client.post(
            f"/api/v1/folders/{folder_id}/access/update", json={"access_grants": grants}
        )
        assert shared.status_code == 200, shared.text
    with member.client() as client:
        sent, _ = turn(client, upstream, "where may I moor?", folder_id=folder_id)
        assert client.get(f"/api/v1/chats/{sent.chat_id}").json()["folder_id"] == folder_id

    assert system_sent(upstream, "where may I moor?").startswith(HARBOUR_PROMPT)


def wait_for_run(client, automation_id: str, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        runs = client.get(f"/api/v1/automations/{automation_id}/runs").json()
        if runs:
            return runs[0]
        time.sleep(0.2)
    raise AssertionError("the automation never recorded a run")


def test_an_automation_filed_in_a_folder_runs_under_its_prompt(make_user, upstream):
    owner = make_user(role="admin")
    prompt = f"List the ships in port, batch {uuid.uuid4().hex[:6]}."
    upstream.queue(reply.text("Two trawlers.", match=reply.answering(prompt)))
    with owner.client() as client:
        folder_id = create_folder(client, "Harbour", HARBOUR_PROMPT)
        created = client.post(
            "/api/v1/automations/create",
            json={
                "name": "Harbour log",
                "folder_id": folder_id,
                "is_active": False,
                "data": {
                    "prompt": prompt,
                    "model_id": MOCK_MODEL_ID,
                    "rrule": "DTSTART:20990101T090000\nRRULE:FREQ=DAILY;BYHOUR=9;BYMINUTE=0",
                    "target": {"type": "chat"},
                },
            },
        )
        assert created.status_code == 200, created.text
        automation_id = created.json()["id"]
        try:
            assert client.post(f"/api/v1/automations/{automation_id}/run").status_code == 200
            run = wait_for_run(client, automation_id)
            assert run["status"] == "success", run
            assert client.get(f"/api/v1/chats/{run['chat_id']}").json()["folder_id"] == folder_id
        finally:
            client.delete(f"/api/v1/automations/{automation_id}/delete")

    assert system_sent(upstream, prompt).startswith(HARBOUR_PROMPT)
