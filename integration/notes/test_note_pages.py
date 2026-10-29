"""Journey: the pages of a note, added, renamed, ordered, nested, copied and deleted over the API.

Since v0.12.5 (b34885efc) a note holds pages and subpages. The page panel sends each change to
`POST /api/v1/notes/{id}/pages/update` as one action (`create`, `rename`, `duplicate`,
`reorder`, `move`, `delete`) with the `revision` of the page list it saw; a later GET of the
note returns the pages parent first in `data.pages`. A change made against an older revision is
refused with 409. The default page (`main`) stays first and cannot be moved or deleted, a page
cannot be nested inside itself or its subpages, deleting a page moves its subpages up a level,
and a duplicate copies the page with its subpages. A new page id must be a UUID, and a create
retried with the same id adds nothing. Text is written page by page through `/update` with
`page_id`, and the notes search answers with the page holding the match.

Discriminates: journey, no fix. In backend copies of dev 05e115366: skipping the revision check
turned the stale-revision test red; dropping the guard on `main` turned the default-page test
red; dropping the subtree check on move together with the cycle check when the pages are
ordered (the move check alone stays green) turned the nesting test red; deleting the subpages
with their parent turned the delete test red; copying only the page turned the duplicate test
red; `matching_page` always answering the first page turned the search test red.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from harness.actors import Actor

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]


def _create_note(owner: Actor, title: str, body: str = "") -> str:
    with owner.client() as client:
        created = client.post(
            "/api/v1/notes/create", json={"title": title, "data": {"content": {"md": body}}}
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _note(owner: Actor, note_id: str) -> dict:
    with owner.client() as client:
        fetched = client.get(f"/api/v1/notes/{note_id}")
    assert fetched.status_code == 200, fetched.text
    return fetched.json()


def _send(owner: Actor, note_id: str, **operation) -> httpx.Response:
    """One page action against the page list the owner reads first, as the page panel sends it."""
    operation.setdefault("revision", _note(owner, note_id)["data"]["pages_revision"])
    with owner.client() as client:
        return client.post(f"/api/v1/notes/{note_id}/pages/update", json=operation)


def _change(owner: Actor, note_id: str, **operation) -> dict:
    answered = _send(owner, note_id, **operation)
    assert answered.status_code == 200, f"{operation}: HTTP {answered.status_code} {answered.text}"
    return answered.json()


def _add_page(owner: Actor, note_id: str, title: str, parent_id: str | None = None) -> str:
    page_id = str(uuid.uuid4())
    _change(owner, note_id, action="create", new_id=page_id, title=title, parent_id=parent_id)
    return page_id


def _write_page(owner: Actor, note_id: str, page_id: str, text: str) -> None:
    with owner.client() as client:
        written = client.post(
            f"/api/v1/notes/{note_id}/update",
            json={"data": {"content": {"md": text}}, "page_id": page_id},
        )
    assert written.status_code == 200, written.text


def _tree(owner: Actor, note_id: str) -> list[tuple[str, str | None]]:
    """The stored pages in order, each as (title, parent title)."""
    pages = _note(owner, note_id)["data"]["pages"]
    titles = {page["id"]: page["title"] for page in pages}
    return [(page["title"], titles.get(page.get("parent_id"))) for page in pages]


@pytest.fixture
def outline(make_user) -> tuple[Actor, str, dict[str, str]]:
    """A note with pages Plan (with subpage Plan detail) and Budget, after the default page."""
    owner = make_user()
    note_id = _create_note(owner, "Trip")
    ids = {"main": "main"}
    ids["plan"] = _add_page(owner, note_id, "Plan")
    ids["budget"] = _add_page(owner, note_id, "Budget")
    ids["detail"] = _add_page(owner, note_id, "Plan detail", parent_id=ids["plan"])
    return owner, note_id, ids


def test_added_pages_and_subpages_are_stored_parent_first(outline):
    owner, note_id, _ = outline

    assert _tree(owner, note_id) == [
        ("Trip", None),
        ("Plan", None),
        ("Plan detail", "Plan"),
        ("Budget", None),
    ]
    assert _note(owner, note_id)["data"]["pages_revision"] == 3


def test_a_renamed_page_keeps_its_title_and_the_default_page_stops_following_the_note(outline):
    owner, note_id, ids = outline

    _change(owner, note_id, action="rename", page_id=ids["budget"], title="  Costs  ")
    _change(owner, note_id, action="rename", page_id="main", title="Overview")
    with owner.client() as client:
        renamed = client.post(f"/api/v1/notes/{note_id}/update", json={"title": "Holiday"})
    assert renamed.status_code == 200, renamed.text

    titles = [title for title, _ in _tree(owner, note_id)]
    assert titles == ["Overview", "Plan", "Plan detail", "Costs"], titles
    assert _note(owner, note_id)["title"] == "Holiday"


def test_an_unrenamed_default_page_follows_the_note_title(make_user):
    owner = make_user()
    note_id = _create_note(owner, "Trip")
    _add_page(owner, note_id, "Plan")

    with owner.client() as client:
        renamed = client.post(f"/api/v1/notes/{note_id}/update", json={"title": "Holiday"})
    assert renamed.status_code == 200, renamed.text

    assert _tree(owner, note_id) == [("Holiday", None), ("Plan", None)]


def test_reorder_stores_the_given_order_with_the_default_page_first(outline):
    owner, note_id, ids = outline

    _change(
        owner, note_id, action="reorder", order=["main", ids["budget"], ids["plan"], ids["detail"]]
    )

    assert _tree(owner, note_id) == [
        ("Trip", None),
        ("Budget", None),
        ("Plan", None),
        ("Plan detail", "Plan"),
    ]


@pytest.mark.parametrize(
    "order",
    [["budget", "main", "plan", "detail"], ["main", "plan", "detail"]],
    ids=["default-page-not-first", "page-missing"],
)
def test_a_reorder_that_moves_the_default_page_or_drops_a_page_is_refused(outline, order):
    owner, note_id, ids = outline
    before = _tree(owner, note_id)

    answered = _send(owner, note_id, action="reorder", order=[ids[name] for name in order])

    assert answered.status_code == 400, answered.text
    assert _tree(owner, note_id) == before


def test_moving_a_page_under_another_takes_its_subpages_along(outline):
    owner, note_id, ids = outline

    _change(owner, note_id, action="move", page_id=ids["plan"], parent_id=ids["budget"])

    assert _tree(owner, note_id) == [
        ("Trip", None),
        ("Budget", None),
        ("Plan", "Budget"),
        ("Plan detail", "Plan"),
    ]


def test_moving_a_page_before_a_sibling_puts_it_there(outline):
    owner, note_id, ids = outline

    _change(
        owner, note_id, action="move", page_id=ids["budget"], parent_id=None, before_id=ids["plan"]
    )

    assert _tree(owner, note_id) == [
        ("Trip", None),
        ("Budget", None),
        ("Plan", None),
        ("Plan detail", "Plan"),
    ]


@pytest.mark.parametrize("parent", ["plan", "detail"], ids=["itself", "its-subpage"])
def test_a_page_cannot_be_nested_inside_itself_or_its_subpages(outline, parent):
    owner, note_id, ids = outline
    before = _tree(owner, note_id)

    answered = _send(owner, note_id, action="move", page_id=ids["plan"], parent_id=ids[parent])

    assert answered.status_code == 400, f"a page was nested under {parent}: {answered.text}"
    assert _tree(owner, note_id) == before


@pytest.mark.parametrize(
    "operation",
    [
        {"action": "move", "page_id": "main", "parent_id": "plan"},
        {"action": "delete", "page_id": "main"},
        {"action": "move", "page_id": "budget", "parent_id": None, "before_id": "main"},
    ],
    ids=["move-default", "delete-default", "move-before-default"],
)
def test_the_default_page_stays_first_and_cannot_be_deleted(outline, operation):
    owner, note_id, ids = outline
    before = _tree(owner, note_id)
    named = {key: ids.get(value, value) for key, value in operation.items()}

    answered = _send(owner, note_id, **named)

    assert answered.status_code == 400, f"{named} was accepted: {answered.text}"
    assert _tree(owner, note_id) == before


def test_the_only_page_of_a_note_cannot_be_deleted(make_user):
    owner = make_user()
    note_id = _create_note(owner, "Single", "keep me")

    answered = _send(owner, note_id, action="delete", page_id="main")

    assert answered.status_code == 400, answered.text
    assert _note(owner, note_id)["data"]["content"]["md"] == "keep me"


def test_deleting_a_page_keeps_its_subpages_one_level_up(outline):
    owner, note_id, ids = outline
    _write_page(owner, note_id, ids["detail"], "ferry at nine")

    _change(owner, note_id, action="delete", page_id=ids["plan"])

    assert _tree(owner, note_id) == [("Trip", None), ("Plan detail", None), ("Budget", None)]
    pages = {page["id"]: page for page in _note(owner, note_id)["data"]["pages"]}
    assert pages[ids["detail"]]["content"]["md"] == "ferry at nine"


def test_duplicating_a_page_copies_it_with_its_subpages_after_the_original(outline):
    owner, note_id, ids = outline
    _write_page(owner, note_id, ids["plan"], "pack light")
    _write_page(owner, note_id, ids["detail"], "ferry at nine")
    copy_id = str(uuid.uuid4())

    _change(owner, note_id, action="duplicate", page_id=ids["plan"], new_id=copy_id)

    assert _tree(owner, note_id) == [
        ("Trip", None),
        ("Plan", None),
        ("Plan detail", "Plan"),
        ("Plan (Copy)", None),
        ("Plan detail", "Plan (Copy)"),
        ("Budget", None),
    ]
    pages = _note(owner, note_id)["data"]["pages"]
    copy, detail_copy = pages[3], pages[4]
    assert copy["id"] == copy_id
    assert detail_copy["id"] not in ids.values()
    texts = (copy["content"]["md"], detail_copy["content"]["md"])
    assert texts == ("pack light", "ferry at nine"), texts


def test_a_change_made_against_an_older_page_list_is_refused(outline):
    owner, note_id, ids = outline
    stale = _note(owner, note_id)["data"]["pages_revision"]
    _change(owner, note_id, action="rename", page_id=ids["plan"], title="Route")
    before = _tree(owner, note_id)

    answered = _send(
        owner, note_id, action="rename", page_id=ids["budget"], title="Costs", revision=stale
    )

    assert answered.status_code == 409, f"a stale page change was applied: {answered.text}"
    assert _tree(owner, note_id) == before


@pytest.mark.parametrize(
    "operation",
    [
        {"action": "create", "new_id": "page-two", "title": "Two"},
        {"action": "create", "title": "Two"},
        {"action": "duplicate", "page_id": "main", "new_id": "page-two"},
    ],
    ids=["create-non-uuid", "create-without-id", "duplicate-non-uuid"],
)
def test_a_new_page_needs_a_uuid(outline, operation):
    owner, note_id, _ = outline
    before = _tree(owner, note_id)

    answered = _send(owner, note_id, **operation)

    assert answered.status_code == 400, f"{operation} was accepted: {answered.text}"
    assert _tree(owner, note_id) == before


def test_a_create_retried_with_the_same_id_adds_no_second_page(make_user):
    owner = make_user()
    note_id = _create_note(owner, "Trip")
    page_id = str(uuid.uuid4())
    sent = {"action": "create", "new_id": page_id, "title": "Plan", "revision": 0}

    first = _send(owner, note_id, **sent)
    retried = _send(owner, note_id, **sent)

    assert (first.status_code, retried.status_code) == (200, 200), retried.text
    assert _tree(owner, note_id) == [("Trip", None), ("Plan", None)]


def test_text_is_written_to_the_named_page_only(outline):
    owner, note_id, ids = outline

    _write_page(owner, note_id, ids["budget"], "two hundred euros")
    with owner.client() as client:
        unnamed = client.post(
            f"/api/v1/notes/{note_id}/update", json={"data": {"content": {"md": "lost"}}}
        )

    assert unnamed.status_code == 409, f"text without a page landed somewhere: {unnamed.text}"
    pages = _note(owner, note_id)["data"]["pages"]
    texts = {page["title"]: page["content"]["md"] for page in pages}
    assert texts == {"Trip": "", "Plan": "", "Plan detail": "", "Budget": "two hundred euros"}


@pytest.mark.parametrize(
    "body",
    [
        {"title": "Trip", "page_generation": 0},
        {"data": {"pages": []}},
    ],
    ids=["page-generation", "whole-page-list"],
)
def test_a_plain_update_cannot_manage_the_pages(outline, body):
    owner, note_id, _ = outline
    before = _tree(owner, note_id)

    with owner.client() as client:
        answered = client.post(f"/api/v1/notes/{note_id}/update", json=body)

    assert answered.status_code == 400, f"{body} was accepted: {answered.text}"
    assert _tree(owner, note_id) == before


def test_the_notes_search_answers_with_the_page_holding_the_match(outline):
    owner, note_id, ids = outline
    word = f"walrus{uuid.uuid4().hex[:8]}"
    _write_page(owner, note_id, ids["main"], "general packing list")
    _write_page(owner, note_id, ids["budget"], f"the {word} tour costs extra")

    with owner.client() as client:
        found = client.get("/api/v1/notes/search", params={"query": word})
    assert found.status_code == 200, found.text

    [hit] = [item for item in found.json()["items"] if item["id"] == note_id]
    data = hit["data"]
    assert (data["page_id"], data["page_title"]) == (ids["budget"], "Budget"), data
    assert word in data["content"]["md"], data
    assert data["page_count"] == 4
