"""Regression: SCIM PATCH accepted what it cannot apply and touched accounts for nothing.

open-webui 0.11.4 commit `ad9da9816`. The PATCH handler dispatched on the operation's path with
no else branch, so an attribute Open WebUI does not support and an operation other than
`replace` both passed as a silent success, and a value of the wrong type (a string for `active`,
a number for `userName`) was written through to the account. Every operation is now validated
before anything is stored, a refused one answers a SCIM 400 and fields whose value equals the
stored one are dropped, so a sync that changes nothing leaves `meta.lastModified` alone. The
store's externalId update (behind PUT) now stamps the account only when the id really changed.

Twin of unit/models/test_scim_provisioning_validation.py.

Discriminates: passes on dev bbfa876af; with the handler's pre-`ad9da9816` if-chain restored the
unsupported path and op answer 200 and the wrongly typed values are stored or fail with a 500;
with the unchanged-field pruning removed a same-value sync moves `meta.lastModified`; with the
externalId update stamping never or always, a new id leaves it or a resent one moves it.
"""

from __future__ import annotations

import time

import pytest

from harness.scim import SCIM_ENV, patch, provision, scim_client

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]


@pytest.fixture(scope="module")
def directory(instance_with):
    with scim_client(instance_with(SCIM_ENV)) as client:
        yield client


def _stored(directory, user_id: str) -> dict:
    fetched = directory.get(f"/Users/{user_id}")
    assert fetched.status_code == 200, fetched.text
    return fetched.json()


def _identity(resource: dict) -> tuple:
    return resource["userName"], resource["displayName"], resource["active"]


# ---------------------------------------------------------------- narrow: refused, not ignored


@pytest.mark.parametrize(
    "operation",
    [
        {"op": "replace", "path": "nickName", "value": "bob"},
        {"op": "replace", "path": "phoneNumbers[primary eq true].value", "value": "123"},
        {"op": "remove", "path": "displayName"},
        {"op": "delete", "path": "displayName"},
    ],
    ids=["unknown-path", "filtered-path", "remove-a-required-name", "unknown-op"],
)
def test_an_operation_open_webui_cannot_apply_is_refused(directory, operation):
    user = provision(directory)

    answered = patch(directory, user["id"], operation)

    assert answered.status_code == 400, (
        f"{operation} was answered {answered.status_code}, so the directory believes a change "
        "Open WebUI never made"
    )
    assert _identity(_stored(directory, user["id"])) == _identity(user)


@pytest.mark.parametrize(
    ("path", "value"),
    [("active", "yes"), ("userName", 123), ("displayName", ["New"]), ("name.formatted", 5)],
)
def test_a_value_of_the_wrong_type_is_refused(directory, path, value):
    user = provision(directory)

    answered = patch(directory, user["id"], {"op": "replace", "path": path, "value": value})

    assert answered.status_code == 400, f"{value!r} for {path!r}: HTTP {answered.status_code}"
    assert _identity(_stored(directory, user["id"])) == _identity(user)


def test_one_bad_operation_refuses_the_whole_patch(directory):
    user = provision(directory)

    answered = patch(
        directory,
        user["id"],
        {"op": "replace", "path": "displayName", "value": "Renamed"},
        {"op": "replace", "path": "nickName", "value": "bob"},
    )

    assert answered.status_code == 400, answered.text
    assert _stored(directory, user["id"])["displayName"] == user["displayName"]


# ---------------------------------------------------------------- narrow: a no-op sync


@pytest.mark.parametrize(
    "operation",
    [
        {"op": "replace", "path": "displayName", "value": "Directory User"},
        {"op": "replace", "value": {"displayName": "Directory User", "active": True}},
    ],
    ids=["same-name", "same-attributes-without-a-path"],
)
def test_a_sync_that_changes_nothing_keeps_last_modified(directory, operation):
    user = provision(directory)
    time.sleep(1.1)  # lastModified has one-second resolution

    answered = patch(directory, user["id"], operation)

    assert answered.status_code == 200, answered.text
    assert answered.json()["meta"]["lastModified"] == user["meta"]["lastModified"], (
        "a sync that re-sent the stored values marked the account as modified"
    )


def test_resending_the_same_external_id_keeps_last_modified(directory):
    user = provision(directory)
    time.sleep(1.1)

    answered = patch(
        directory, user["id"], {"op": "replace", "path": "externalId", "value": user["externalId"]}
    )

    assert answered.status_code == 200, answered.text
    assert answered.json()["meta"]["lastModified"] == user["meta"]["lastModified"]


@pytest.mark.parametrize(
    ("external_id", "moves"),
    [(None, False), ("ext-replaced", True)],
    ids=["same-external-id", "new-external-id"],
)
def test_a_full_update_touches_the_account_only_for_a_new_external_id(
    directory, external_id, moves
):
    user = provision(directory)
    time.sleep(1.1)

    replaced = directory.put(
        f"/Users/{user['id']}",
        json={"externalId": external_id or user["externalId"], "displayName": user["displayName"]},
    )

    assert replaced.status_code == 200, replaced.text
    last_modified = _stored(directory, user["id"])["meta"]["lastModified"]
    assert (last_modified != user["meta"]["lastModified"]) is moves, (
        f"lastModified went from {user['meta']['lastModified']} to {last_modified}"
    )


# ---------------------------------------------------------------- nearby: real changes still land


def test_a_changed_name_is_stored_and_moves_last_modified(directory):
    user = provision(directory)
    time.sleep(1.1)

    answered = patch(
        directory,
        user["id"],
        {"op": "replace", "path": "displayName", "value": "Renamed User"},
        {"op": "add", "path": "externalId", "value": "ext-renamed"},
    )

    assert answered.status_code == 200, answered.text
    stored = _stored(directory, user["id"])
    assert (stored["displayName"], stored["externalId"]) == ("Renamed User", "ext-renamed")
    assert stored["meta"]["lastModified"] > user["meta"]["lastModified"]


def test_deactivating_is_still_a_boolean_replace(directory):
    user = provision(directory)

    answered = patch(directory, user["id"], {"op": "replace", "path": "active", "value": False})

    assert answered.status_code == 200, answered.text
    assert _stored(directory, user["id"])["active"] is False
