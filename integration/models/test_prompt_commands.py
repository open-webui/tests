"""Journey: a prompt's slash command is unique across all accounts.

Creating a prompt whose command is already taken is refused with the "already registered"
message, whether the first prompt is the same account's or another account's private one, and the
first prompt keeps its text. Renaming a prompt's command to a taken one is refused the same way
through both the editor's name and command autosave and its full save, and the prompt keeps the
command it had. A command that was freed by a rename can be created again.

Discriminates: in a backend copy where the command lookup never finds a prompt, the two create
tests and both rename tests go red (the second prompt or the rename is not refused with the
message); where the metadata update leaves the command as it was, the freed-command test goes red.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Callable, Iterator

import pytest

from harness.access import make_group
from harness.actors import Actor

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

TAKEN = "Uh-oh! This command is already registered. Please choose another command string."


@dataclass
class Writers:
    first: Actor
    second: Actor
    save: Callable[[Actor, str, str], dict]


@pytest.fixture
def writers(admin, make_user) -> Iterator[Writers]:
    """Two accounts that may write prompts; `save(account, command, content)` posts a prompt."""
    first, second = make_user(), make_user()
    make_group(admin, [first, second], {"workspace": {"prompts": True}})
    created: list[tuple[Actor, str]] = []

    def save(account: Actor, command: str, content: str) -> dict:
        form = {"command": command, "name": f"Prompt {command}", "content": content}
        with account.client() as client:
            response = client.post("/api/v1/prompts/create", json=form)
        if response.status_code == 200:
            created.append((account, response.json()["id"]))
        return {"status": response.status_code, "body": response.json()}

    yield Writers(first, second, save)
    for account, prompt_id in created:
        with account.client() as client:
            client.delete(f"/api/v1/prompts/id/{prompt_id}/delete")


def _texts(account: Actor, command: str) -> list[str]:
    with account.client() as client:
        listed = client.get("/api/v1/prompts/").json()
    return [prompt["content"] for prompt in listed if prompt["command"] == command]


def _command_of(account: Actor, prompt_id: str) -> str:
    with account.client() as client:
        return client.get(f"/api/v1/prompts/id/{prompt_id}").json()["command"]


def test_a_second_prompt_with_the_same_command_is_refused_and_the_first_keeps_its_text(writers):
    first, save = writers.first, writers.save
    command = f"taken-{uuid.uuid4().hex[:8]}"
    assert save(first, command, "The first text.")["status"] == 200

    again = save(first, command, "The second text.")

    assert again["status"] == 400
    assert again["body"]["detail"] == TAKEN
    assert _texts(first, command) == ["The first text."]


def test_another_accounts_private_prompt_still_holds_its_command(writers):
    first, second, save = writers.first, writers.second, writers.save
    command = f"private-{uuid.uuid4().hex[:8]}"
    assert save(first, command, "Mine alone.")["status"] == 200
    assert _texts(second, command) == []

    refused = save(second, command, "Taking it over.")

    assert refused["status"] == 400
    assert refused["body"]["detail"] == TAKEN
    assert _texts(first, command) == ["Mine alone."]
    assert _texts(second, command) == []


@pytest.mark.parametrize("route", ["update/meta", "update"])
def test_renaming_a_command_to_a_taken_one_is_refused_and_keeps_the_old_command(writers, route):
    first, save = writers.first, writers.save
    suffix = uuid.uuid4().hex[:8]
    held, moving = f"held-{suffix}", f"moving-{suffix}"
    save(first, held, "Held text.")
    prompt_id = save(first, moving, "Moving text.")["body"]["id"]
    form = {"name": "Renamed", "command": held}
    if route == "update":
        form["content"] = "Moving text."

    with first.client() as client:
        refused = client.post(f"/api/v1/prompts/id/{prompt_id}/{route}", json=form)

    assert refused.status_code == 400
    assert refused.json()["detail"] == TAKEN
    assert _command_of(first, prompt_id) == moving
    assert _texts(first, held) == ["Held text."]


def test_a_command_freed_by_a_rename_can_be_created_again(writers):
    first, second, save = writers.first, writers.second, writers.save
    suffix = uuid.uuid4().hex[:8]
    old, new = f"old-{suffix}", f"new-{suffix}"
    prompt_id = save(first, old, "Original.")["body"]["id"]
    with first.client() as client:
        renamed = client.post(
            f"/api/v1/prompts/id/{prompt_id}/update/meta", json={"name": "Renamed", "command": new}
        )
    assert renamed.status_code == 200, renamed.text

    assert save(second, old, "Reused.")["status"] == 200
    assert save(second, new, "Too late.")["status"] == 400
    assert _texts(second, old) == ["Reused."]
