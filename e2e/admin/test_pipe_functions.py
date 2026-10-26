"""Journey: pipes as models a person picks in the chat's model selector and talks to.

A pipe an admin adds shows in the admin's model selector under its function's name and stays
private to admins until it is shared; then a person picks it by that name, and the chat on it is
answered by the pipe's code, which is handed the person's message. A manifold pipe lists one
model per entry of its `pipes`, each named with the class's `name` before the entry's name, and
each answers as itself. A pipe switched off in Admin Panel > Functions leaves the selector. A
pipe with user valves shows a Valves knob beside the message input, and the value a person saves
there shapes the pipe's next answer. Without Allow Chat Valves a user is not offered that knob,
while an admin still is.

Discriminates: passes on dev ebc6add67. One backend copy listing a single pipe under its id,
every manifold as one plain pipe and switched-off pipes as well, and handing every pipe the
default user valves, turned each test red on its own edit. The Allow Chat Valves test fails in
the pre-fix build of 0.12.4 (46cdd70d7 reverted): the user is offered the knob.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.actors import Actor
from harness.python_tools import EVERYONE_READS
from utils.chat_ui import chat_input, expect_reply, send
from utils.model_selector import model_options, select_model
from utils.tooltips import tooltip_button
from utils.valves import customise, valve
from utils.workspace_list import item_row

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

WELCOME_PIPE = """from pydantic import BaseModel, Field


class Pipe:
    class UserValves(BaseModel):
        name: str = Field("traveller", description="What the pier calls you")

    def pipe(self, body: dict, __user__: dict) -> str:
        return f"Welcome aboard, {__user__['valves'].name}."
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


def test_a_pipe_is_listed_by_its_name_private_until_shared_and_answers(page_for, admin, make_user):
    name = f"Pier office {uuid.uuid4().hex[:6]}"
    with named_function(admin, name, ECHO_PIPE) as pipe_id:
        admin_page = page_for(admin)
        new_chat(admin_page)
        expect(model_options(admin_page, name)).to_have_count(1)
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

        admin_page = page_for(admin, path="/admin/functions")
        admin_page.get_by_placeholder("Search Functions").fill(pipe_id)
        card = item_row(admin_page, name)
        card.get_by_role("switch").click()
        expect(card.get_by_role("switch")).not_to_be_checked()

        new_chat(page)
        expect(model_options(page, name)).to_have_count(0)


def test_a_pipes_user_valves_set_beside_the_message_input_shape_its_answer(
    page_for, admin, make_user
):
    name = f"Pier welcome {uuid.uuid4().hex[:6]}"
    with named_function(admin, name, WELCOME_PIPE) as pipe_id:
        with shared_with_everyone(admin, {pipe_id: name}):
            page = page_for(make_user())
            new_chat(page)
            select_model(page, name)
            send(page, "hello?")
            expect_reply(page, "Welcome aboard, traveller.")

            composer = page.locator("form").filter(has=chat_input(page))
            tooltip_button(composer, "Valves").click()
            dialog = page.get_by_role("dialog").filter(has_text="Valves")
            name_valve = valve(dialog, "Name", "What the pier calls you")
            customise(name_valve)
            name_valve.get_by_role("textbox").fill("Ada")
            dialog.get_by_role("button", name="Save").click()
            expect(page.get_by_text("Valves updated successfully")).to_be_visible()
            page.keyboard.press("Escape")
            expect(dialog).to_have_count(0)

            send(page, "hello again?")
            expect_reply(page, "Welcome aboard, Ada.")


def valves_knob(page: Page) -> Locator:
    composer = page.locator("form").filter(has=chat_input(page))
    return composer.get_by_role("button", name="Valves", exact=True)


def _withhold_chat_valves(admin: Actor) -> None:
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["chat"]["valves"] = False
        saved = client.post("/api/v1/users/default/permissions", json=permissions)
    assert saved.status_code == 200, saved.text


def test_without_allow_chat_valves_a_pipe_offers_no_valves_beside_the_message_input(
    page_for, admin, make_user, preserve
):
    """#32772 (fix 46cdd70d7): a pipe's Valves knob showed whatever Allow Chat Valves said."""
    preserve("permissions")
    _withhold_chat_valves(admin)
    name = f"Pier welcome {uuid.uuid4().hex[:6]}"
    with named_function(admin, name, WELCOME_PIPE) as pipe_id:
        with shared_with_everyone(admin, {pipe_id: name}):
            pages = {"admin": page_for(make_user(role="admin")), "user": page_for(make_user())}
            for page in pages.values():
                new_chat(page)
                select_model(page, name)
                expect(page.get_by_role("button", name=f"Selected model: {name}")).to_be_visible()

            expect(valves_knob(pages["admin"]), "an admin keeps the knob").to_be_visible()
            expect(valves_knob(pages["user"]), "the user is offered the knob").to_have_count(0)
