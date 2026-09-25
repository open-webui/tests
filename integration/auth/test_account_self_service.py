"""Journey: what a signed-in person changes and reads about their own account.

The account settings save a profile (name, picture, bio, gender, date of birth), a status (emoji,
message, expiry) that others see on the person's profile card, free-form info that each save
merges into, and user variables that a system prompt reads as `{{user.variables.<key>}}`. The
usage page sums the account's own chats and messages, tokens included, and the chat statistics
list its own chats. None of these routes takes an account id, so each only ever reads or writes
the caller's account: a second person saving the same things leaves the first one's untouched,
and a pending account is refused. The status routes are refused while the admin has switched
user status off.

Discriminates: in a backend copy, merging the saved info into an empty dict instead of the stored
one turns the info test red, rendering user variables from an empty dict turns the system-prompt
test red, and counting the usage of every account instead of the caller's turns the usage test
red.
"""

from __future__ import annotations

import base64
import json

import pytest

from harness import upstream as reply
from harness.chat import ask
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)
ADMIN_CONFIG = "/api/v1/auths/admin/config"


def session(actor) -> dict:
    with actor.client() as client:
        answer = client.get("/api/v1/auths/")
    answer.raise_for_status()
    return answer.json()


def save_profile(actor, **changes):
    form = {"name": actor.name, "profile_image_url": "/user.png", **changes}
    with actor.client() as client:
        return client.post("/api/v1/auths/update/profile", json=form)


# --------------------------------------------------------------------------- profile


def test_a_saved_profile_is_what_the_session_reads_back(make_user):
    person, other = make_user(), make_user()

    saved = save_profile(
        person,
        name="Grace Harbour",
        profile_image_url=PNG_DATA_URL,
        bio="Keeps the lighthouse.",
        gender="female",
        date_of_birth="1990-04-12",
    )

    assert saved.status_code == 200, saved.text
    profile = session(person)
    assert (profile["name"], profile["bio"], profile["gender"], profile["date_of_birth"]) == (
        "Grace Harbour",
        "Keeps the lighthouse.",
        "female",
        "1990-04-12",
    )
    with other.client() as client:
        served = client.get(f"/api/v1/users/{person.id}/profile/image")
    assert served.headers["content-type"] == "image/png"
    assert served.content == base64.b64decode(PNG_DATA_URL.split(",", 1)[1])
    assert session(other)["name"] == other.name, "saving a profile changed another account"


@pytest.mark.parametrize(
    "picture", ["javascript:alert(1)", "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4="]
)
def test_a_profile_with_an_unsafe_picture_is_refused_and_changes_nothing(make_user, picture):
    person = make_user()

    refused = save_profile(person, name="Renamed", profile_image_url=picture)

    assert refused.status_code == 422, refused.text
    assert session(person)["name"] == person.name


def test_a_pending_account_cannot_save_a_profile(make_user):
    pending = make_user(role="pending")
    assert save_profile(pending, name="Sneaky").status_code == 401


# --------------------------------------------------------------------------- status


def save_status(actor, **status):
    with actor.client() as client:
        return client.post("/api/v1/users/user/status/update", json=status)


def test_a_status_is_saved_for_the_account_and_shown_to_others(make_user):
    person, other = make_user(), make_user()

    saved = save_status(
        person, status_emoji="🛟", status_message="On the water", status_expires_at=4102444800
    )

    assert saved.status_code == 200, saved.text
    with person.client() as client:
        own = client.get("/api/v1/users/user/status").json()
    with other.client() as client:
        card = client.get(f"/api/v1/users/{person.id}/info").json()
        others_own = client.get("/api/v1/users/user/status").json()
    expected = ("🛟", "On the water", 4102444800)
    assert (own["status_emoji"], own["status_message"], own["status_expires_at"]) == expected
    assert (card["status_emoji"], card["status_message"], card["status_expires_at"]) == expected
    assert others_own["status_message"] is None, "saving a status changed another account"


def test_a_status_update_keeps_the_fields_it_leaves_out(make_user):
    person = make_user()
    save_status(person, status_emoji="⛵", status_message="Sailing").raise_for_status()

    save_status(person, status_message="Back on shore").raise_for_status()

    with person.client() as client:
        stored = client.get("/api/v1/users/user/status").json()
    assert (stored["status_emoji"], stored["status_message"]) == ("⛵", "Back on shore")


def test_status_routes_are_refused_while_user_status_is_off(admin, make_user, preserve):
    preserve("admin_config")
    person = make_user()
    with admin.client() as client:
        current = client.get(ADMIN_CONFIG).json()
        client.post(ADMIN_CONFIG, json={**current, "ENABLE_USER_STATUS": False}).raise_for_status()

    with person.client() as client:
        read = client.get("/api/v1/users/user/status")
    saved = save_status(person, status_message="hidden")

    assert (read.status_code, saved.status_code) == (403, 403)


# --------------------------------------------------------------------------- info


def save_info(actor, fields: dict) -> dict:
    with actor.client() as client:
        saved = client.post("/api/v1/users/user/info/update", json=fields)
    assert saved.status_code == 200, saved.text
    return saved.json()


def read_info(actor):
    with actor.client() as client:
        answer = client.get("/api/v1/users/user/info")
    answer.raise_for_status()
    return answer.json()


def test_saved_info_merges_into_what_is_stored(make_user):
    person, other = make_user(), make_user()

    save_info(person, {"location": "Harbour", "shift": "early"})
    merged = save_info(person, {"shift": "late", "desk": 4})

    assert merged == {"location": "Harbour", "shift": "late", "desk": 4}
    assert read_info(person) == merged
    assert not read_info(other), "saving info changed another account"


# --------------------------------------------------------------------------- variables


def save_variables(actor, variables):
    with actor.client() as client:
        return client.post("/api/v1/users/user/variables/update", json={"variables": variables})


def read_variables(actor) -> dict:
    with actor.client() as client:
        answer = client.get("/api/v1/users/user/variables")
    answer.raise_for_status()
    return answer.json()["variables"]


@pytest.mark.parametrize(
    "variables",
    [{"Team": "harbour"}, {"team name": "harbour"}, {"team": 4}, ["team"]],
    ids=["capital-key", "space-in-key", "number-value", "not-an-object"],
)
def test_invalid_variables_are_refused_and_keep_the_stored_ones(make_user, variables):
    person = make_user()
    save_variables(person, {"team": "harbour"}).raise_for_status()

    refused = save_variables(person, variables)

    assert refused.status_code in (400, 422), refused.text
    assert read_variables(person) == {"team": "harbour"}


def test_a_system_prompt_reads_the_callers_own_variables(make_user, upstream):
    person, other = make_user(), make_user()
    saved = save_variables(person, {"team": "harbour pilots", "shift": "line one\r\nline two"})
    assert saved.status_code == 200, saved.text
    assert saved.json()["variables"]["shift"] == "line one\nline two"
    system = {"role": "system", "content": "Team: [{{user.variables.team}}]"}

    upstream.queue(reply.text("ok"), reply.text("ok"))
    with person.client() as client:
        ask(client, "who am I?", history=[system])
    with other.client() as client:
        ask(client, "who am I?", history=[system])

    person_request, other_request = upstream.chat_requests()[-2:]
    assert person_request["messages"][0] == {"role": "system", "content": "Team: [harbour pilots]"}
    assert other_request["messages"][0] == {"role": "system", "content": "Team: []"}


# --------------------------------------------------------------------------- usage


def usage(actor, **params):
    with actor.client() as client:
        return client.get("/api/v1/users/usage", params=params)


def test_usage_counts_the_callers_own_chats_messages_and_tokens(make_user, upstream):
    person, other = make_user(), make_user()
    tokens = {"prompt_tokens": 12, "completion_tokens": 30, "total_tokens": 42}
    upstream.queue(reply.text("first answer", usage=tokens), reply.text("second answer"))
    with person.client() as client:
        ask(client, "first question")
    with other.client() as client:
        ask(client, "someone else's question")

    answer = usage(person)

    assert answer.status_code == 200, answer.text
    totals = answer.json()["totals"]
    counted = {
        key: totals[key]
        for key in ("total_chats", "messages", "user_messages", "assistant_messages")
    }
    assert counted == {"total_chats": 1, "messages": 2, "user_messages": 1, "assistant_messages": 1}
    assert (totals["input_tokens"], totals["output_tokens"]) == (12, 30)
    assert totals["lifetime_tokens"] == 42
    assert answer.json()["insights"]["most_used_model"] == MOCK_MODEL_ID
    empty = usage(make_user()).json()["totals"]
    assert (empty["total_chats"], empty["messages"], empty["lifetime_tokens"]) == (0, 0, 0)


@pytest.mark.parametrize(
    ("params", "status"),
    [({"days": 3}, 422), ({"start_date": 2_000_000_000, "end_date": 1_000_000_000}, 400)],
    ids=["period-too-short", "start-after-end"],
)
def test_usage_refuses_a_period_it_cannot_cover(make_user, params, status):
    assert usage(make_user(), **params).status_code == status


def test_chat_statistics_list_only_the_callers_chats(make_user, upstream):
    person, other = make_user(), make_user()
    upstream.queue(reply.text("an answer"), reply.text("another answer"))
    with person.client() as client:
        turn, _ = ask(client, "a question")
    with other.client() as client:
        others_turn, _ = ask(client, "another question")

    with person.client() as client:
        listed = client.get("/api/v1/chats/stats/usage")

    assert listed.status_code == 200, listed.text
    [stats] = listed.json()["items"]
    assert stats["id"] == turn.chat_id
    assert (stats["message_count"], stats["history_user_message_count"]) == (2, 1)
    assert stats["models"] == {MOCK_MODEL_ID: 1}
    assert others_turn.chat_id not in json.dumps(listed.json())
