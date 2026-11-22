"""Journey: ratings given while comparing two models rank them on the admin's Evaluations page.

An account with two default models sends one prompt and gets both models' replies side by side;
rating one reply is a win or a loss against the other. The Leaderboard then lists the two models
in Elo order with their won and lost counts, its column headers re-sort it both ways, and a model's
row opens the tags its ratings were given in the rating form. The Feedback tab lists each rating,
filters by model, lets the admin delete an entry and exports what the filter shows as JSON or CSV.
Deleting the last rating of the model the tab is narrowed to shows all ratings again.
A router model the admin adds under Router Models in Admin Settings > Models, picking at random
between two models and made public, answers a user through one of them, and rating its reply names
the model that answered. Switching message rating off in the admin settings removes the thumbs
from a reply.

Two tests stay red on dev ebc6add67. The CSV export reads each rating's chat from the rating data,
where it is never stored (the chat sits in the rating's meta), so its chat_id column is always
empty (open-webui/open-webui#32021, fixed by b6a7820a2). Rating the reply of a comparison that is
not the selected one selects it first, which rebuilds the reply and closes the rating form it just
opened, so the rating is saved without the form for a score, a reason, a comment or tags
(open-webui/open-webui#32022).

Discriminates: passes on dev 176d31d1d; in a frontend copy, sorting the leaderboard the other
way round turns the ranking test red, swapping its won and lost columns turns the counts test
red, a model filter that is never applied turns the filter test red, a delete that never reaches
the server turns the delete test red and always showing the thumbs turns the message rating test
red. On dev ebc6add67, in one frontend build, column headers that do nothing turn the header
test red, a model dialog that leaves out the top tags the tags test, an export that ignores the
model filter the JSON test and a rating form without the arena's answer the added arena test, now
the added router model test, retargeted to router models on dev 53d741157, where it passes. The
last rating test passes on dev f659c0d12 and fails in a frontend copy with 88cf7f209 (#33794)
undone (the tab stays on the deleted model and lists nothing).
"""

from __future__ import annotations

import csv
import io
import json
import re
import uuid
from typing import Callable, Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.actors import Actor
from harness.python_tools import EVERYONE_READS
from harness.router_models import ROUTER_MODELS_CONFIG
from harness.upstream import MOCK_MODEL_ID
from utils.access_control import choose_visibility
from utils.chat_ui import conversation, expect_reply, replies, send
from utils.model_selector import select_model

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

QUESTION = "which fruit keeps longest in a lunchbox"
ANSWER = "an apple"


@pytest.fixture
def make_contender(admin) -> Iterator[Callable[[], tuple[str, str]]]:
    """`make_contender()`: the id and name of a new preset on the scripted model."""
    created_ids: list[str] = []

    def create() -> tuple[str, str]:
        tag = uuid.uuid4().hex[:8]
        model_id, name = f"contender-{tag}", f"Contender {tag}"
        form = {
            "id": model_id,
            "base_model_id": MOCK_MODEL_ID,
            "name": name,
            "meta": {},
            "params": {},
            "access_grants": [EVERYONE_READS],
        }
        with admin.client() as client:
            created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
        created_ids.append(model_id)
        return model_id, name

    yield create
    with admin.client() as client:
        for model_id in created_ids:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


@pytest.fixture
def duel(make_contender, make_user, page_for, upstream):
    """`duel(ratings)`: two new models, compared once per rating given to one of their replies.

    A rating is the reply's position (0 for the first model, 1 for the second), the button
    that rates it and optionally a tag the rater adds in the rating form. Returns the rater,
    then the id and name of each model.
    """

    def compare(ratings: list[tuple[int, str]]):
        first, second = make_contender(), make_contender()
        rater = make_user()
        with rater.client() as client:
            settings = {"ui": {"models": [first[0], second[0]]}}
            client.post("/api/v1/users/user/settings/update", json=settings).raise_for_status()
        page = page_for(rater)
        for position, verdict, *tags in ratings:
            upstream.queue(reply.text(ANSWER, match=reply.answering(QUESTION)))
            upstream.queue(reply.text(ANSWER, match=reply.answering(QUESTION)))
            page.goto("/")
            send(page, QUESTION)
            expect(replies(page)).to_have_count(2)
            expect(replies(page).first).to_contain_text(ANSWER)
            expect(replies(page).last).to_contain_text(ANSWER)
            _rate(page, position, verdict)
            for tag in tags:
                _tag_rating(page, tag)
        return rater, first, second

    return compare


def _is_feedback_write(response) -> bool:
    return response.request.method == "POST" and "/evaluations/feedback" in response.url


def _rate(page: Page, position: int, verdict: str) -> None:
    # the thumbs sit beside each reply, not inside it
    replies(page).nth(position).hover()
    with page.expect_response(_is_feedback_write):
        conversation(page).get_by_role("button", name=verdict).nth(position).click()


def _tag_rating(page: Page, tag: str) -> None:
    form = conversation(page)
    tags = form.get_by_placeholder("Add a tag...")
    tags.fill(tag)
    tags.press("Enter")
    expect(form.get_by_text(tag, exact=True)).to_be_visible()
    with page.expect_response(_is_feedback_write):
        form.get_by_role("button", name="Save", exact=True).click()
    expect(page.get_by_text("Thanks for your feedback!")).to_be_visible()


def _count(number: int) -> re.Pattern:
    # a cell also holds a percentage that only shows on hover
    return re.compile(rf"(^|\s){number}\s*$")


def _open_evaluations(admin: Actor, page_for, tab: str) -> Page:
    page = page_for(admin, path=f"/admin/evaluations/{tab}")
    return page


def _feedback_row(page: Page, rater: Actor, model_id: str) -> Locator:
    return page.get_by_role("row", name=re.compile(rf"^{re.escape(rater.name)} {model_id}\b"))


def test_the_leaderboard_ranks_the_compared_models_by_their_ratings(duel, admin, page_for):
    _, (_, winner), (_, loser) = duel([(0, "Good Response"), (0, "Good Response")])

    page = _open_evaluations(admin, page_for, "leaderboard")

    expect(page.get_by_role("row").filter(has_text=winner)).to_have_count(1)
    expect(page.get_by_role("row").filter(has_text=loser)).to_have_count(1)
    ranked = [
        text
        for text in page.get_by_role("row").all_inner_texts()
        if winner in text or loser in text
    ]
    assert [winner in text for text in ranked] == [True, False], ranked


def test_the_leaderboard_counts_each_models_wins_and_losses(duel, admin, page_for):
    ratings = [(0, "Good Response"), (0, "Good Response"), (0, "Bad Response")]
    _, (_, first), (_, second) = duel(ratings)

    page = _open_evaluations(admin, page_for, "leaderboard")

    first_cells = page.get_by_role("row").filter(has_text=first).get_by_role("cell")
    second_cells = page.get_by_role("row").filter(has_text=second).get_by_role("cell")
    expect(first_cells.nth(2)).to_have_text("1012")
    expect(first_cells.nth(3)).to_have_text(_count(2))
    expect(first_cells.nth(4)).to_have_text(_count(1))
    expect(second_cells.nth(2)).to_have_text("988")
    expect(second_cells.nth(3)).to_have_text(_count(1))
    expect(second_cells.nth(4)).to_have_text(_count(2))


def test_the_feedback_tab_can_be_narrowed_to_one_model(duel, admin, page_for):
    rater, (first_id, _), (second_id, _) = duel([(0, "Good Response"), (1, "Bad Response")])
    page = _open_evaluations(admin, page_for, "feedback")
    expect(_feedback_row(page, rater, first_id)).to_have_count(1)
    expect(_feedback_row(page, rater, second_id)).to_have_count(1)

    _narrow_to(page, second_id)

    expect(_feedback_row(page, rater, second_id)).to_have_count(1)
    expect(_feedback_row(page, rater, first_id)).to_have_count(0)


def test_an_admin_deletes_a_rating_from_the_feedback_tab(duel, admin, page_for):
    rater, (first_id, _), (second_id, _) = duel([(0, "Good Response"), (1, "Good Response")])
    page = _open_evaluations(admin, page_for, "feedback")
    expect(_feedback_row(page, rater, first_id)).to_have_count(1)

    _feedback_row(page, rater, second_id).get_by_role("button", name="More").click()
    page.get_by_role("menuitem", name="Delete").click()

    expect(page.get_by_text("Feedback deleted successfully")).to_be_visible()
    expect(_feedback_row(page, rater, second_id)).to_have_count(0)
    expect(_feedback_row(page, rater, first_id)).to_have_count(1)
    with rater.client() as client:
        kept = client.get("/api/v1/evaluations/feedbacks/user").json()["items"]
    assert [item["data"]["model_id"] for item in kept] == [first_id]


def test_deleting_the_last_rating_of_the_chosen_model_shows_all_ratings_again(
    duel, admin, page_for
):
    """#33794: the filter stayed on a model with no ratings left, so the tab showed none."""
    rater, (first_id, _), (second_id, _) = duel([(0, "Good Response"), (1, "Good Response")])
    page = _open_evaluations(admin, page_for, "feedback")
    expect(_feedback_row(page, rater, first_id)).to_have_count(1)
    _narrow_to(page, second_id)
    expect(_feedback_row(page, rater, first_id)).to_have_count(0)

    _feedback_row(page, rater, second_id).get_by_role("button", name="More").click()
    page.get_by_role("menuitem", name="Delete").click()

    expect(page.get_by_text("Feedback deleted successfully")).to_be_visible()
    expect(_feedback_row(page, rater, second_id)).to_have_count(0)
    expect(
        _feedback_row(page, rater, first_id), "the filter stayed on the model with no ratings"
    ).to_have_count(1)


def test_switching_message_rating_off_removes_the_thumbs_from_a_reply(
    admin, preserve, make_user, page_for, upstream
):
    preserve("admin_config")
    with admin.client() as client:
        current = client.get("/api/v1/auths/admin/config").json()
        saved = client.post(
            "/api/v1/auths/admin/config", json={**current, "ENABLE_MESSAGE_RATING": False}
        )
    saved.raise_for_status()
    page = page_for(make_user())
    upstream.queue(reply.text(ANSWER, match=reply.answering(QUESTION)))

    send(page, QUESTION)

    expect(replies(page).first).to_contain_text(ANSWER)
    replies(page).first.hover()
    expect(conversation(page).get_by_role("button", name="Copy").last).to_be_visible()
    expect(conversation(page).get_by_role("button", name="Good Response")).to_have_count(0)
    expect(conversation(page).get_by_role("button", name="Bad Response")).to_have_count(0)


def test_rating_the_first_reply_of_a_comparison_opens_the_rating_form(
    make_contender, make_user, page_for, upstream
):
    first, second = make_contender(), make_contender()
    rater = make_user()
    with rater.client() as client:
        settings = {"ui": {"models": [first[0], second[0]]}}
        client.post("/api/v1/users/user/settings/update", json=settings).raise_for_status()
    page = page_for(rater)
    upstream.queue(reply.text(ANSWER, match=reply.answering(QUESTION)))
    upstream.queue(reply.text(ANSWER, match=reply.answering(QUESTION)))
    send(page, QUESTION)
    expect(replies(page).first).to_contain_text(ANSWER)
    expect(replies(page).last).to_contain_text(ANSWER)

    _rate(page, 0, "Bad Response")

    expect(
        conversation(page).get_by_text("How would you rate this response?"),
        "rating the reply that is not the selected one saves it and drops its rating form "
        "(open-webui/open-webui#32022)",
    ).to_be_visible()


def _ranked(page: Page, *names: str) -> list[str]:
    """`names` in the order the leaderboard shows them."""
    expect(page.get_by_role("row").filter(has_text=names[0])).to_have_count(1)
    rows = page.get_by_role("row").all_inner_texts()
    return [name for row in rows for name in names if name in row]


def test_the_leaderboard_column_headers_reorder_the_models(duel, admin, page_for):
    _, (_, winner), (_, loser) = duel([(0, "Good Response"), (0, "Good Response")])
    page = _open_evaluations(admin, page_for, "leaderboard")
    assert _ranked(page, winner, loser) == [winner, loser]

    page.get_by_role("columnheader", name="Lost").click()
    expect(page.get_by_role("columnheader", name="Lost")).to_contain_text("Lost")
    expect(page.get_by_role("row").filter(has_text=loser).get_by_role("cell").nth(4)).to_have_text(
        _count(2)
    )
    assert _ranked(page, winner, loser) == [loser, winner]

    page.get_by_role("columnheader", name="Lost").click()
    assert _ranked(page, winner, loser) == [winner, loser]


def test_a_leaderboard_row_shows_the_tags_its_ratings_were_given(duel, admin, page_for):
    _, _, (_, winner) = duel([(1, "Good Response", "packed lunches")])
    page = _open_evaluations(admin, page_for, "leaderboard")

    page.get_by_role("row").filter(has_text=winner).click()

    details = page.get_by_role("dialog").filter(has_text="Activity")
    expect(details).to_contain_text(winner)
    expect(details).to_contain_text(re.compile(r"packed lunches\s*1"))


def _export(page: Page, choice: str) -> str:
    page.get_by_role("button", name="Export", exact=True).first.click()
    with page.expect_download() as download:
        page.get_by_role("menuitem", name=choice).click()
    with open(download.value.path(), encoding="utf-8") as saved:
        return saved.read()


def _narrow_to(page: Page, model_id: str) -> None:
    page.get_by_role("button", name="All").click()
    # each row names its model in a button too; the filter's choices are the pressable ones
    choice = page.get_by_role("button", name=model_id, exact=True)
    choice.and_(page.locator("[aria-pressed]")).click()


def test_the_json_export_holds_the_chosen_models_ratings(duel, admin, page_for):
    rater, (first_id, _), (second_id, _) = duel([(0, "Good Response"), (1, "Bad Response")])
    page = _open_evaluations(admin, page_for, "feedback")
    expect(_feedback_row(page, rater, first_id)).to_have_count(1)
    _narrow_to(page, second_id)
    expect(_feedback_row(page, rater, first_id)).to_have_count(0)

    exported = json.loads(_export(page, "Export as JSON"))

    assert [entry["data"]["model_id"] for entry in exported] == [second_id]
    [rating] = exported
    assert rating["user_id"] == rater.id
    assert rating["data"]["rating"] == -1
    assert rating["data"]["sibling_model_ids"] == [first_id]
    assert QUESTION in json.dumps(rating["snapshot"])


def test_the_csv_export_lists_each_rating_with_its_chat(duel, admin, page_for):
    rater, (first_id, _), (second_id, _) = duel([(0, "Good Response")])
    with rater.client() as client:
        [chat] = client.get("/api/v1/chats/").json()
    page = _open_evaluations(admin, page_for, "feedback")
    _narrow_to(page, first_id)
    expect(_feedback_row(page, rater, first_id)).to_have_count(1)

    [row] = csv.DictReader(io.StringIO(_export(page, "Export as CSV")))

    assert (row["user_id"], row["model_id"], row["rating"]) == (rater.id, first_id, "1")
    assert row["sibling_model_ids"] == second_id
    assert row["chat_id"] == chat["id"], (
        "the CSV export reads the chat from the rating data and leaves it out "
        "(open-webui/open-webui#32021)"
    )


def _stored_reply(rater: Actor) -> dict:
    """The rater's only reply, as the server stored it."""
    with rater.client() as client:
        [chat] = client.get("/api/v1/chats/").json()
        stored = client.get(f"/api/v1/chats/{chat['id']}").json()
    return next(
        message
        for message in stored["chat"]["history"]["messages"].values()
        if message["role"] == "assistant"
    )


def test_a_router_model_added_in_the_settings_answers_through_its_model_and_names_it(
    page_for, make_user, make_contender, preserve, admin, upstream
):
    preserve(ROUTER_MODELS_CONFIG)
    contenders = dict([make_contender(), make_contender()])
    name = f"Blind Duel {uuid.uuid4().hex[:8]}"
    admin_page = page_for(make_user(role="admin"), path="/admin/settings/models")
    admin_page.get_by_role("button", name="Add Router Model").click()
    router = admin_page.get_by_role("dialog", name="Add Router Model")
    router.get_by_placeholder("Model Name").fill(name)
    for contender in contenders.values():
        router.get_by_role("button", name="Add model").click()
        router.get_by_role("combobox", name="Select a model").last.select_option(label=contender)
    router.get_by_role("combobox", name="Strategy").select_option(label="Random")
    router.get_by_role("button", name="Access", exact=True).click()
    access = admin_page.get_by_role("dialog", name="Access Control")
    choose_visibility(access, "Public")
    expect(access.get_by_text("Accessible to all users")).to_be_visible()
    admin_page.keyboard.press("Escape")
    expect(access).to_be_hidden()
    router.get_by_role("button", name="Save", exact=True).click()
    expect(router).to_be_hidden()
    expect(admin_page.get_by_text("Settings saved successfully!").first).to_be_visible()

    rater = make_user()
    page = page_for(rater)
    select_model(page, name)
    upstream.queue(reply.text(ANSWER, match=reply.answering(QUESTION)))
    send(page, QUESTION)
    expect_reply(page, ANSWER)
    _rate(page, 0, "Bad Response")

    picked = _stored_reply(rater).get("selectedModelId")
    assert picked in contenders, f"the router answered through {picked}, not one of its models"
    expect(conversation(page)).to_contain_text(
        f'This response was generated by "{contenders[picked]}"'
    )
    feedback = _open_evaluations(admin, page_for, "feedback")
    expect(_feedback_row(feedback, rater, picked)).to_have_count(1)
