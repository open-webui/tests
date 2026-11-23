"""Journey: chatting with a router model in the browser, reply by reply.

A user picks a router model that sends code to a code model by pattern, legal questions to a
legal model by condition and the rest to a general model. Each reply streams in like any other,
its header names the router model and the model that answered, and nothing of the Decisions API
shows in the chat. Every follow-up is routed again, so one chat holds replies of several models;
after a reload each reply still names its model and the next message goes to the router model,
not to the model of the last reply. A shared link shows the same names. Editing a question routes
the new branch on its own words, and Regenerate routes the question again, each version naming
its own model. A router model picking at random hides the model in the header, after a reload
and in a shared link, until the user rates the reply. A title generated with Generate in the
chat's title or in the sidebar's rename, for the open chat and for another one, is written by
the model that answered, as in a direct chat with it, with no Decisions API call of its own.

`test_a_generated_title_is_written_by_the_model_that_answered` is red on dev bd150cdf6
(open-webui/open-webui#34887): Generate asks the router model again, which matches the `def `
pattern (saved without its trailing space) against the word "default" in the built-in title
instructions, so the code model writes the title of a legal chat. It passes with the fix
(build of 7b7806ca2), which asks the model the router picked for the latest reply.

Discriminates: passes on dev d599f7a08 with build c1f3b6fd1. In a copy of that build whose reply
header never names the answering model, the routing and edit tests turn red; in one that names
it for a random pick too, the random test turns red. In a backend copy whose patterns never
match, the routing and edit tests turn red.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.router_journeys import (
    Decider,
    Target,
    condition,
    default,
    make_targets,
    pattern,
    remove_targets,
    save_router,
)
from harness.router_models import ROUTER_MODELS_CONFIG
from utils.chat_surface import open_sidebar, surface_of
from utils.chat_ui import conversation, expect_reply, last_reply, replies, send
from utils.model_selector import select_model
from utils.sidebar_chats import chat_menu, chat_row

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

CODE_PATTERN = "```|def "
LEGAL = "the user asks about contracts or legal questions"
TITLE_TASK = "Generate a concise"


@pytest.fixture
def decider(listener) -> Decider:
    return Decider(listener)


@pytest.fixture
def targets(admin, upstream):
    made = make_targets(admin, upstream, "code", "legal", "general")
    yield made
    remove_targets(admin, upstream, made)


@pytest.fixture
def smart_router(admin, preserve, targets, decider) -> Target:
    preserve(ROUTER_MODELS_CONFIG)
    return save_router(
        admin,
        [
            pattern(targets["code"], CODE_PATTERN),
            condition(targets["legal"], LEGAL),
            default(targets["general"]),
        ],
        decider=decider,
    )


def reply_headers(page: Page) -> Locator:
    """The line over each reply naming who wrote it."""
    return conversation(page).get_by_role("heading", level=3)


def named(router: Target, target: Target) -> re.Pattern:
    return re.compile(rf"^\s*{re.escape(router.name)}\s+{re.escape(target.name)}\s*$")


def ask_on_page(page: Page, upstream, prompt: str, answer: str) -> None:
    upstream.queue(reply.text(answer, match=reply.answering(prompt)))
    send(page, prompt)
    expect_reply(page, answer)


def share(owner, chat_id: str, *readers) -> str:
    with owner.client() as client:
        shared = client.post(f"/api/v1/chats/{chat_id}/share")
        assert shared.status_code == 200, shared.text
        grants = [
            {"principal_type": "user", "principal_id": reader.id, "permission": "read"}
            for reader in readers
        ]
        granted = client.post(
            f"/api/v1/chats/shared/{chat_id}/access/update", json={"access_grants": grants}
        )
        assert granted.status_code == 200, granted.text
    return shared.json()["share_id"]


def only_chat_id(owner) -> str:
    with owner.client() as client:
        chats = client.get("/api/v1/chats/").json()
    assert len(chats) == 1, f"expected the one chat of this test, found {len(chats)}"
    return chats[0]["id"]


def test_each_reply_names_the_model_the_router_picked_through_follow_ups_a_reload_and_a_share(
    page_for, make_user, upstream, targets, smart_router, decider
):
    tag = uuid.uuid4().hex[:6]
    asker = make_user()
    page = page_for(asker)
    select_model(page, smart_router.name)

    # the first reply streams: its first piece shows while the provider holds the rest
    greeting = f"hello there {tag}"
    rest = threading.Event()
    upstream.queue(
        reply.text(
            ["general starts ", "and general ends"],
            hold=rest,
            hold_after=2,
            match=reply.answering(greeting),
        )
    )
    send(page, greeting)
    expect(last_reply(page)).to_contain_text("general starts")
    expect(last_reply(page)).not_to_contain_text("and general ends")
    rest.set()
    expect_reply(page, "general starts and general ends")

    decider.scores = {LEGAL: 0.9}
    ask_on_page(page, upstream, f"can my landlord keep the deposit {tag}", "legal answer")
    expect(reply_headers(page)).to_have_text(
        [named(smart_router, targets["general"]), named(smart_router, targets["legal"])]
    )

    page.reload()
    expect(reply_headers(page)).to_have_text(
        [named(smart_router, targets["general"]), named(smart_router, targets["legal"])]
    )
    decider.scores = {}
    ask_on_page(page, upstream, f"thanks, anything else {tag}", "general again")
    ask_on_page(page, upstream, f"def add(a, b): {tag}", "code answer")

    order = [targets[label] for label in ("general", "legal", "general", "code")]
    expect(reply_headers(page)).to_have_text([named(smart_router, model) for model in order])
    expect(replies(page)).to_have_count(4)
    expect(conversation(page)).not_to_contain_text(decider.note)
    assert len(decider.calls()) == 3, (
        f"expected one Decisions API call per unmatched message, got {len(decider.calls())}"
    )

    viewer = make_user()
    share_id = share(asker, only_chat_id(asker), viewer)
    shared_page = page_for(viewer, path=f"/s/{share_id}")
    expect(reply_headers(shared_page)).to_have_text([named(smart_router, model) for model in order])


def test_an_edited_question_and_a_regenerated_reply_are_routed_on_their_own(
    page_for, make_user, upstream, targets, smart_router, decider
):
    tag = uuid.uuid4().hex[:6]
    page = page_for(make_user())
    select_model(page, smart_router.name)
    ask_on_page(page, upstream, f"def sort(items): {tag}", "code answer")

    rewrite = f"can I sublet my flat {tag}"
    decider.scores = {LEGAL: 0.9}
    upstream.queue(reply.text("legal answer", match=reply.answering(rewrite)))
    question = conversation(page).locator(".chat-user").last
    question.hover()
    question.get_by_role("button", name="Edit").click()
    question.locator("textarea").fill(rewrite)
    question.get_by_role("button", name="Send").click()
    expect_reply(page, "legal answer")
    expect(reply_headers(page)).to_have_text([named(smart_router, targets["legal"])])

    decider.scores = {}
    upstream.queue(reply.text("general answer", match=reply.answering(rewrite)))
    last_reply(page).hover()
    conversation(page).get_by_role("button", name="Regenerate").last.click()
    page.get_by_text("Try Again").click()
    expect_reply(page, "general answer")
    expect(reply_headers(page)).to_have_text([named(smart_router, targets["general"])])

    last_reply(page).hover()
    conversation(page).get_by_role("button", name="Previous message").last.click()
    expect_reply(page, "legal answer")
    expect(reply_headers(page)).to_have_text([named(smart_router, targets["legal"])])

    question = conversation(page).locator(".chat-user").last
    question.hover()
    question.get_by_role("button", name="Previous message").click()
    expect_reply(page, "code answer")
    expect(reply_headers(page)).to_have_text([named(smart_router, targets["code"])])


def test_a_random_pick_stays_hidden_until_the_reply_is_rated(
    page_for, make_user, admin, preserve, upstream, targets
):
    preserve(ROUTER_MODELS_CONFIG)
    blind = save_router(
        admin, [default(targets["code"]), default(targets["general"])], strategy="random"
    )
    by_id = {target.id: target for target in targets.values()}
    rater = make_user()
    page = page_for(rater)
    select_model(page, blind.name)
    ask_on_page(page, upstream, f"which tea is best {uuid.uuid4().hex[:6]}", "green tea")
    hidden = re.compile(rf"^\s*{re.escape(blind.name)}\s*$")
    expect(reply_headers(page)).to_have_text([hidden])

    page.reload()
    expect_reply(page, "green tea")
    expect(reply_headers(page)).to_have_text([hidden])

    viewer = make_user()
    chat_id = only_chat_id(rater)
    shared_page = page_for(viewer, path=f"/s/{share(rater, chat_id, viewer)}")
    expect(conversation(shared_page)).to_contain_text("green tea")
    expect(reply_headers(shared_page)).to_have_text([hidden])
    for target in targets.values():
        expect(conversation(shared_page)).not_to_contain_text(target.name)

    with rater.client() as client:
        stored = client.get(f"/api/v1/chats/{chat_id}").json()["chat"]["history"]["messages"]
    picked = next(message for message in stored.values() if message["role"] == "assistant")
    picked_name = by_id[picked["selectedModelId"]].name
    expect(conversation(page)).not_to_contain_text(picked_name)
    replies(page).last.hover()
    with page.expect_response(
        lambda response: (
            response.request.method == "POST" and "/evaluations/feedback" in response.url
        )
    ):
        conversation(page).get_by_role("button", name="Good Response").click()
    expect(conversation(page)).to_contain_text(f'This response was generated by "{picked_name}"')


@pytest.fixture
def tasks_on_the_chat_model(admin, preserve):
    """No task model set, so a chat's title is written by the model the user chats with."""
    preserve("tasks")
    with admin.client() as client:
        current = client.get("/api/v1/tasks/config").json()
        updated = client.post(
            "/api/v1/tasks/config/update",
            json={
                **current,
                "TASK_MODEL": "",
                "TASK_MODEL_EXTERNAL": "",
                "ENABLE_TITLE_GENERATION": True,
            },
        )
        assert updated.status_code == 200, updated.text


def title_written_by(target: Target, title: str) -> reply.Reply:
    """The title `target` writes; any other model asked for a title gets the provider's default."""
    return reply.text(
        json.dumps({"title": title}),
        match=lambda body: body["model"] == target.id and reply.answering(TITLE_TASK)(body),
    )


def title_requests(upstream, count: int) -> list[str]:
    """The models asked for a chat title, in order, once `count` were asked."""
    deadline = time.monotonic() + 20
    while True:
        asked = [
            body["model"] for body in upstream.chat_requests() if reply.answering(TITLE_TASK)(body)
        ]
        if len(asked) >= count or time.monotonic() > deadline:
            return asked
        time.sleep(0.2)


def generate_in_sidebar(page: Page, title: str) -> None:
    sidebar = open_sidebar(page)
    chat_menu(sidebar, title).get_by_role("menuitem", name="Rename").click()
    sidebar.get_by_role("button", name="Generate").click()


def test_a_generated_title_is_written_by_the_model_that_answered(
    page_for, make_user, upstream, targets, smart_router, decider, tasks_on_the_chat_model
):
    tag = uuid.uuid4().hex[:6]
    # titles only come when asked for, with Generate
    asker = make_user()
    with asker.client() as client:
        saved = client.post(
            "/api/v1/users/user/settings/update",
            json={"ui": {"title": {"auto": False}, "autoTags": False, "autoFollowUps": False}},
        )
        assert saved.status_code == 200, saved.text
    page = page_for(asker)
    select_model(page, smart_router.name)
    decider.scores = {LEGAL: 0.9}
    question = f"can I break my rental contract early {tag}"
    ask_on_page(page, upstream, question, "legal answer")
    expect(reply_headers(page)).to_have_text([named(smart_router, targets["legal"])])
    legal = targets["legal"]

    header = surface_of(page)
    upstream.queue(title_written_by(legal, f"Ending a lease early {tag}"))
    header.tab.dblclick()
    title_box = header.tab_bar.get_by_role("textbox", name="Title")
    expect(title_box).to_be_focused()
    header.tab_bar.get_by_role("button", name="Generate").click()
    asked = title_requests(upstream, 1)
    assert asked == [legal.id], f"Generate in the chat's title asked {asked}, not {legal.id}"
    expect(header.tab).to_have_text(f"Ending a lease early {tag}")

    upstream.queue(title_written_by(legal, f"Leaving a rental contract {tag}"))
    generate_in_sidebar(page, f"Ending a lease early {tag}")
    asked = title_requests(upstream, 2)
    assert asked == [legal.id] * 2, f"Generate in the sidebar for the open chat asked {asked}"
    expect(chat_row(page, f"Leaving a rental contract {tag}")).to_be_visible()

    open_sidebar(page).get_by_role("link", name="New Chat").last.click()
    page.wait_for_url(re.compile(r"/$"))
    upstream.queue(title_written_by(legal, f"Breaking a tenancy early {tag}"))
    generate_in_sidebar(page, f"Leaving a rental contract {tag}")
    asked = title_requests(upstream, 3)
    assert asked == [legal.id] * 3, f"Generate in the sidebar for another chat asked {asked}"
    expect(chat_row(page, f"Breaking a tenancy early {tag}")).to_be_visible()

    with asker.client() as client:
        assert client.get("/api/v1/chats/").json()[0]["title"] == f"Breaking a tenancy early {tag}"
    assert len(decider.calls()) == 1, (
        f"expected only the reply's Decisions API call, got {len(decider.calls())}"
    )
