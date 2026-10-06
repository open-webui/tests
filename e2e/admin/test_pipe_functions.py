"""Journey: pipes as models a person picks in the chat's model selector and talks to.

A pipe an admin adds is private to admins until it is shared; then it shows in a person's model
selector under its function's name, and the chat on it is answered by the pipe's code, which is
handed the person's message. A manifold pipe lists one
model per entry of its `pipes`, each named with the class's `name` before the entry's name, and
each answers as itself. A pipe switched off in Admin Panel > Functions leaves the selector.

Discriminates: passes on dev ebc6add67. In a backend copy whose model list skips manifolds the
manifold test fails, in one that hands the pipe an empty message list the pipe test fails, and
in one whose model list keeps switched-off pipes the switch test fails.
"""

from __future__ import annotations

import re
import uuid
from contextlib import contextmanager
from typing import Iterator

import pytest
from playwright.sync_api import Page, expect

from harness.actors import Actor
from harness.python_tools import EVERYONE_READS
from utils.chat_ui import chat_input, expect_reply, send
from utils.model_selector import model_options, select_model

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

ECHO_PIPE = """class Pipe:
    def pipe(self, body: dict) -> str:
        return "The pier office heard: " + body["messages"][-1]["content"]
"""

MANIFOLD_PIPE = """class Pipe:
    def __init__(self):
        self.name = "Pier/"

    def pipes(self):
        return [{"id": "north", "name": "North quay"}, {"id": "south", "name": "South quay"}]

    def pipe(self, body: dict) -> str:
        return "Answered by " + body["model"]
"""


@contextmanager
def named_function(admin: Actor, name: str, source: str) -> Iterator[str]:
    """A switched-on function called `name`, the way the editor saves one; deleted afterwards."""
    function_id = f"pipe_{uuid.uuid4().hex[:8]}"
    with admin.client() as client:
        created = client.post(
            "/api/v1/functions/create",
            json={
                "id": function_id,
                "name": name,
                "content": source,
                "meta": {"description": "a pipe for the pier"},
            },
        )
        assert created.status_code == 200, created.text
        try:
            client.post(f"/api/v1/functions/id/{function_id}/toggle").raise_for_status()
            yield function_id
        finally:
            client.delete(f"/api/v1/functions/id/{function_id}/delete")


@contextmanager
def shared_with_everyone(admin: Actor, models: dict[str, str]) -> Iterator[None]:
    """Share each pipe model (id to name) with every account, as the admin's Models page does."""
    with admin.client() as client:
        client.get("/api/models").raise_for_status()
        try:
            for model_id, name in models.items():
                shared = client.post(
                    "/api/v1/models/model/access/update",
                    json={"id": model_id, "name": name, "access_grants": [EVERYONE_READS]},
                )
                assert shared.status_code == 200, shared.text
            yield
        finally:
            for model_id in models:
                client.post("/api/v1/models/model/delete", json={"id": model_id})


def new_chat(page: Page) -> None:
    page.goto("/")
    expect(chat_input(page)).to_be_visible()


def test_a_pipe_shared_with_everyone_is_picked_by_its_name_and_answers(page_for, admin, make_user):
    name = f"Pier office {uuid.uuid4().hex[:6]}"
    with named_function(admin, name, ECHO_PIPE) as pipe_id:
        page = page_for(make_user())
        new_chat(page)
        expect(model_options(page, name), "a new pipe is private to admins").to_have_count(0)

        with shared_with_everyone(admin, {pipe_id: name}):
            new_chat(page)
            select_model(page, name)
            expect(page.get_by_role("button", name=f"Selected model: {name}")).to_be_visible()
            send(page, "is berth nine free?")
            expect_reply(page, "The pier office heard: is berth nine free?")


def test_a_manifold_lists_each_of_its_models_and_each_answers_as_itself(page_for, admin, make_user):
    with named_function(admin, f"Quays {uuid.uuid4().hex[:6]}", MANIFOLD_PIPE) as pipe_id:
        quays = {f"{pipe_id}.north": "Pier/North quay", f"{pipe_id}.south": "Pier/South quay"}
        with shared_with_everyone(admin, quays):
            page = page_for(make_user())
            new_chat(page)
            expect(model_options(page, "Pier/North quay")).to_have_count(1)
            expect(model_options(page, "Pier/South quay")).to_have_count(1)

            for model_id, name in quays.items():
                new_chat(page)
                select_model(page, name)
                send(page, f"who answers at {name}?")
                expect_reply(page, f"Answered by {model_id}")


def test_a_pipe_switched_off_leaves_the_model_selector(page_for, admin, make_user):
    name = f"Pier office {uuid.uuid4().hex[:6]}"
    with (
        named_function(admin, name, ECHO_PIPE) as pipe_id,
        shared_with_everyone(admin, {pipe_id: name}),
    ):
        page = page_for(make_user())
        new_chat(page)
        expect(model_options(page, name)).to_have_count(1)

        admin_page = page_for(admin)
        admin_page.goto("/admin/functions")
        admin_page.get_by_placeholder("Search Functions").fill(pipe_id)
        card = admin_page.get_by_role("main").get_by_role(
            "button", name=re.compile(f"^pipe {name}")
        )
        card.get_by_role("switch").click()
        expect(card.get_by_role("switch")).not_to_be_checked()

        new_chat(page)
        expect(model_options(page, name)).to_have_count(0)
