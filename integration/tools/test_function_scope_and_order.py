"""Journey: which filters, actions and pipe models a chat gets, and in which order.

A filter attached to a model runs on that model's chats and no other until it is made global.
Filters run in the order of their `priority` valve, lowest first, and a changed valve reorders
the next chat. A filter's user valves reach it per user. Actions are listed on every model in the
order of their `priority` valve, sub-actions in their own order, and a switched-off action leaves
the list. A manifold pipe adds one model per entry of `pipes`, named with the class's `name`
before the entry's name, and each answers with its own id; a switched-off pipe leaves the list.
Browser twins: e2e/admin/test_filter_functions.py, test_action_functions.py and
test_pipe_functions.py.

Discriminates: passes on dev ebc6add67. In a backend copy whose filter pipeline ignores a model's
own filters the scope test fails, in one that sorts filters by id alone the filter order test
fails, in one that hands every filter the default user valves the user valves test fails, in one
that orders actions by id alone the action order test fails, in one whose model list keeps
switched-off actions and pipes the two switch tests fail and in one whose model list skips
manifolds the manifold test fails.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Iterator

import pytest

from harness import upstream as reply
from harness.actors import Actor
from harness.chat import ask
from harness.plugins import installed_function
from harness.python_tools import EVERYONE_READS
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]


def tagging_filter(tag: str, priority: int = 0) -> str:
    return f"""from pydantic import BaseModel


class Filter:
    class Valves(BaseModel):
        priority: int = {priority}

    def __init__(self):
        self.valves = self.Valves()

    def inlet(self, body: dict) -> dict:
        body["messages"][-1]["content"] += " [{tag}]"
        return body
"""


USER_TAGGING_FILTER = """from pydantic import BaseModel


class Filter:
    class UserValves(BaseModel):
        tag: str = "guest"

    def inlet(self, body: dict, __user__: dict) -> dict:
        body["messages"][-1]["content"] += f" [{__user__['valves'].tag}]"
        return body
"""


def noting_action(priority: int) -> str:
    return f"""from pydantic import BaseModel


class Action:
    class Valves(BaseModel):
        priority: int = {priority}

    def __init__(self):
        self.valves = self.Valves()

    async def action(self, body: dict):
        return None
"""


STAMPING_ACTION = """class Action:
    actions = [{"id": "arrival", "name": "Stamp arrival"}, {"id": "departure"}]

    async def action(self, body: dict):
        return None
"""

ECHO_PIPE = """class Pipe:
    def pipe(self, body: dict) -> str:
        return "answered by " + body["model"]
"""

MANIFOLD_PIPE = """class Pipe:
    def __init__(self):
        self.name = "Pier/"

    def pipes(self):
        return [{"id": "north", "name": "North quay"}, {"id": "south", "name": "South quay"}]

    def pipe(self, body: dict) -> str:
        return "answered by " + body["model"]
"""


def sent_text(account: Actor, upstream, question: str, model: str = MOCK_MODEL_ID) -> str:
    """Ask `question` on `model`; the last message the provider got for it."""
    upstream.queue(reply.text("noted", match=reply.answering(question)))
    with account.client() as client:
        ask(client, question, model=model)
    request = next(filter(reply.answering(question), upstream.chat_requests()))
    return request["messages"][-1]["content"]


@contextmanager
def preset_with_filter(admin: Actor, filter_id: str) -> Iterator[str]:
    model_id = f"gate-{uuid.uuid4().hex[:8]}"
    with admin.client() as client:
        created = client.post(
            "/api/v1/models/create",
            json={
                "id": model_id,
                "name": model_id,
                "base_model_id": MOCK_MODEL_ID,
                "meta": {"filterIds": [filter_id]},
                "params": {},
                "access_grants": [EVERYONE_READS],
            },
        )
        assert created.status_code == 200, created.text
        try:
            client.get("/api/models").raise_for_status()  # the web client lists models first
            yield model_id
        finally:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


def test_a_models_own_filter_runs_only_on_that_model_until_made_global(admin, make_user, upstream):
    person = make_user()
    with installed_function(admin, tagging_filter("gate")) as filter_id:
        with preset_with_filter(admin, filter_id) as preset_id:
            assert sent_text(person, upstream, "moor here?", model=preset_id) == "moor here? [gate]"
            assert sent_text(person, upstream, "moor there?") == "moor there?"

            with admin.client() as client:
                client.post(f"/api/v1/functions/id/{filter_id}/toggle/global").raise_for_status()
            assert sent_text(person, upstream, "moor anywhere?") == "moor anywhere? [gate]"


def test_filters_run_in_the_order_of_their_priority_valve(admin, make_user, upstream):
    person = make_user()
    with (
        installed_function(admin, tagging_filter("north", 1), is_global=True) as north,
        installed_function(admin, tagging_filter("south", 2), is_global=True),
    ):
        assert sent_text(person, upstream, "which way?") == "which way? [north] [south]"

        with admin.client() as client:
            saved = client.post(f"/api/v1/functions/id/{north}/valves/update", json={"priority": 3})
        assert saved.status_code == 200, saved.text
        assert sent_text(person, upstream, "which way now?") == "which way now? [south] [north]"


def test_each_user_gets_their_own_filter_user_valves(admin, make_user, upstream):
    first, second = make_user(), make_user()
    with installed_function(admin, USER_TAGGING_FILTER, is_global=True) as filter_id:
        with first.client() as client:
            saved = client.post(
                f"/api/v1/functions/id/{filter_id}/valves/user/update", json={"tag": "captain"}
            )
        assert saved.status_code == 200, saved.text

        assert sent_text(first, upstream, "who am I?") == "who am I? [captain]"
        assert sent_text(second, upstream, "and me?") == "and me? [guest]"


def listed_models(account: Actor) -> dict[str, dict]:
    with account.client() as client:
        listed = client.get("/api/models")
    assert listed.status_code == 200, listed.text
    return {model["id"]: model for model in listed.json()["data"]}


def listed_actions(account: Actor, own: set[str]) -> list[str]:
    """The ids of the mock model's actions that belong to `own` functions, in listed order."""
    actions = listed_models(account)[MOCK_MODEL_ID].get("actions", [])
    return [action["id"] for action in actions if action["id"].split(".")[0] in own]


def test_actions_are_listed_in_the_order_of_their_priority_valve(admin, make_user):
    person = make_user()
    with (
        installed_function(admin, noting_action(1), is_global=True) as first,
        installed_function(admin, noting_action(2), is_global=True) as second,
        installed_function(admin, STAMPING_ACTION, is_global=True) as stamping,
    ):
        own = {first, second, stamping}
        stamps = [f"{stamping}.arrival", f"{stamping}.departure"]
        assert listed_actions(person, own) == [*stamps, first, second]

        with admin.client() as client:
            saved = client.post(f"/api/v1/functions/id/{first}/valves/update", json={"priority": 3})
        assert saved.status_code == 200, saved.text
        assert listed_actions(person, own) == [*stamps, second, first]

        names = {
            action["id"]: action["name"]
            for action in listed_models(person)[MOCK_MODEL_ID]["actions"]
        }
        assert names[stamps[0]] == "Stamp arrival"
        assert names[stamps[1]] == f"{stamping} (departure)"


def test_a_switched_off_action_leaves_every_model(admin, make_user):
    person = make_user()
    with installed_function(admin, noting_action(0), is_global=True) as action_id:
        assert listed_actions(person, {action_id}) == [action_id]

        with admin.client() as client:
            client.post(f"/api/v1/functions/id/{action_id}/toggle").raise_for_status()
        assert listed_actions(person, {action_id}) == []


def test_a_manifold_adds_one_model_per_entry_and_each_answers_as_itself(admin, upstream):
    with installed_function(admin, MANIFOLD_PIPE) as pipe_id:
        models = listed_models(admin)
        quays = {f"{pipe_id}.north": "Pier/North quay", f"{pipe_id}.south": "Pier/South quay"}
        assert {model_id: models[model_id]["name"] for model_id in quays} == quays
        assert pipe_id not in models

        with admin.client() as client:
            for model_id in quays:
                _, answer = ask(client, "who answers?", model=model_id)
                assert answer["content"] == f"answered by {model_id}"


def test_a_switched_off_pipe_leaves_the_model_list(admin):
    with installed_function(admin, ECHO_PIPE) as pipe_id:
        assert pipe_id in listed_models(admin)

        with admin.client() as client:
            client.post(f"/api/v1/functions/id/{pipe_id}/toggle").raise_for_status()
        assert pipe_id not in listed_models(admin)
