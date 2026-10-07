"""Journey: ratings given while comparing two models rank them on the admin's Evaluations page.

An account with two default models sends one prompt and gets both models' replies side by side;
rating one reply is a win or a loss against the other. The Leaderboard then lists the two models
in Elo order with their won and lost counts, its column headers re-sort it both ways, and a model's
row opens the tags its ratings were given in the rating form. The Feedback tab lists each rating,
filters by model, lets the admin delete an entry and exports what the filter shows as JSON or CSV.
An arena model the admin adds in the Evaluations settings, limited to one model and made public,
answers a user through that model, and rating its reply names the model that answered. The admin
settings decide where evaluations show: switching message rating off removes the thumbs from a
reply, switching arena models off removes their section from the Evaluations settings and the
model from the selector.

Two tests stay red on dev ebc6add67. The CSV export reads each rating's chat from the rating data,
where it is never stored (the chat sits in the rating's meta), so its chat_id column is always
empty (open-webui/open-webui#32021). Rating the reply of a comparison that is not the selected one
selects it first, which rebuilds the reply and closes the rating form it just opened, so the
rating is saved without the form for a score, a reason, a comment or tags
(open-webui/open-webui#32022).

Discriminates: passes on dev 176d31d1d; in a frontend copy, sorting the leaderboard the other
way round turns the ranking test red, swapping its won and lost columns turns the counts test
red, a model filter that is never applied turns the filter test red, a delete that never reaches
the server turns the delete test red, always showing the thumbs turns the message rating test
red and always showing the arena section turns the arena test red. On dev ebc6add67, in one
frontend build, column headers that do nothing turn the header test red, a model dialog that
leaves out the top tags the tags test, an export that ignores the model filter the JSON test and
a rating form without the arena's answer the added arena test.
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
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import conversation, expect_reply, replies, send
from utils.model_selector import select_model
from utils.tooltips import tooltip_button

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
    page = page_for(admin)
    page.goto(f"/admin/evaluations/{tab}")
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

    page.get_by_role("button", name="All").click()
    page.get_by_role("button", name=second_id, exact=True).click()

    expect(_feedback_row(page, rater, second_id)).to_have_count(1)
    expect(_feedback_row(page, rater, first_id)).to_have_count(0)


def test_an_admin_deletes_a_rating_from_the_feedback_tab(duel, admin, page_for):
    rater, (first_id, _), (second_id, _) = duel([(0, "Good Response"), (1, "Good Response")])
    page = _open_evaluations(admin, page_for, "feedback")
    expect(_feedback_row(page, rater, first_id)).to_have_count(1)

    _feedback_row(page, rater, second_id).locator("button").click()
    page.get_by_role("button", name="Delete").click()

    expect(page.get_by_text("Feedback deleted successfully")).to_be_visible()
    expect(_feedback_row(page, rater, second_id)).to_have_count(0)
    expect(_feedback_row(page, rater, first_id)).to_have_count(1)
    with rater.client() as client:
        kept = client.get("/api/v1/evaluations/feedbacks/user").json()["items"]
    assert [item["data"]["model_id"] for item in kept] == [first_id]


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


@pytest.fixture
def arena_model(admin, preserve) -> Iterator[str]:
    """The name of a public arena model that the admin's settings list."""
    preserve(("/api/v1/evaluations/config", "/api/v1/evaluations/config"))
    tag = uuid.uuid4().hex[:8]
    name = f"Arena {tag}"
    grants = [{"principal_type": "user", "principal_id": "*", "permission": "read"}]
    arena = {
        "id": f"arena-{tag}",
        "name": name,
        "meta": {"model_ids": [MOCK_MODEL_ID], "access_grants": grants},
    }
    with admin.client() as client:
        saved = client.post(
            "/api/v1/evaluations/config",
            json={"ENABLE_EVALUATION_ARENA_MODELS": True, "EVALUATION_ARENA_MODELS": [arena]},
        )
        assert saved.status_code == 200, saved.text
        yield name
        client.post("/api/v1/models/model/delete", json={"id": arena["id"]})


def _offered_in_selector(page: Page, name: str) -> Locator:
    page.goto("/")
    page.get_by_role("button", name=re.compile("^Selected model")).click()
    page.get_by_role("textbox", name="Search In Models").fill(name)
    available = page.get_by_role("listbox", name="Available models")
    return available.get_by_role("option", name=f"Select {name} model")


def test_switching_arena_models_off_hides_them_in_the_settings_and_the_selector(
    page_for, make_user, arena_model
):
    admin_page = page_for(make_user(role="admin"))
    admin_page.goto("/admin/settings/evaluations")
    settings = admin_page.get_by_role("dialog")
    expect(settings.get_by_text(arena_model)).to_be_visible()
    expect(_offered_in_selector(page_for(make_user()), arena_model)).to_be_visible()

    admin_page.get_by_role("switch", name="Arena Models").click()
    expect(settings.get_by_text(arena_model)).to_have_count(0)
    settings.get_by_role("button", name="Save", exact=True).click()
    expect(admin_page.get_by_text("Settings saved successfully!").first).to_be_visible()

    expect(_offered_in_selector(page_for(make_user()), arena_model)).to_have_count(0)


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
        page.get_by_role("button", name=choice).click()
    with open(download.value.path(), encoding="utf-8") as saved:
        return saved.read()


def _narrow_to(page: Page, model_id: str) -> None:
    page.get_by_role("button", name="All").click()
    page.get_by_role("button", name=model_id, exact=True).click()


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


@pytest.fixture
def arena_settings(admin, preserve) -> Iterator[list[str]]:
    """Restores the evaluation settings; delete the arena model ids appended to the list."""
    preserve(("/api/v1/evaluations/config", "/api/v1/evaluations/config"))
    added: list[str] = []
    yield added
    with admin.client() as client:
        for model_id in added:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


def test_an_arena_model_added_in_the_settings_answers_through_its_model_and_names_it(
    page_for, make_user, make_contender, arena_settings, admin, upstream
):
    contender_id, contender = make_contender()
    tag = uuid.uuid4().hex[:8]
    name = f"Blind Duel {tag}"
    arena_settings.append(f"blind-duel-{tag}")
    admin_page = page_for(make_user(role="admin"))
    admin_page.goto("/admin/settings/evaluations")
    settings = admin_page.get_by_role("dialog")
    switch = settings.get_by_role("switch", name="Arena Models")
    if switch.get_attribute("aria-checked") != "true":
        switch.click()
    tooltip_button(settings, "Add Arena Model").click()
    arena = admin_page.get_by_role("dialog").filter(has=admin_page.get_by_placeholder("Model Name"))
    arena.get_by_placeholder("Model Name").fill(name)
    choices = arena.get_by_role("combobox").filter(has_text="Select a model")
    choices.select_option(label=contender)
    choices.locator("xpath=following-sibling::div//button").click()
    expect(arena.get_by_text(contender, exact=True)).to_be_visible()
    arena.get_by_role("combobox").filter(has_text="Public").select_option("public")
    arena.get_by_role("button", name="Save", exact=True).click()
    expect(arena).to_be_hidden()
    expect(settings.get_by_text(name)).to_be_visible()
    settings.get_by_role("button", name="Save", exact=True).click()
    expect(admin_page.get_by_text("Settings saved successfully!").first).to_be_visible()

    rater = make_user()
    page = page_for(rater)
    select_model(page, name)
    upstream.queue(reply.text(ANSWER, match=reply.answering(QUESTION)))
    send(page, QUESTION)
    expect_reply(page, ANSWER)
    _rate(page, 0, "Bad Response")

    expect(conversation(page)).to_contain_text(f'This response was generated by "{contender}"')
    feedback = _open_evaluations(admin, page_for, "feedback")
    expect(_feedback_row(feedback, rater, contender_id)).to_have_count(1)
