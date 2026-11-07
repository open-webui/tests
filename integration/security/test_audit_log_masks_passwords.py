"""Regression: the audit log recorded new passwords in plain text.

Fix `d3dde3609` (#31622): with request auditing on, the logged request body was
masked only for a field named exactly `password`, so `new_password` (a user changing their own
password), the LDAP `app_dn_password` and the YaCy `YACY_PASSWORD` were written to the audit
log as typed. Any JSON field whose name ends in `password`, in any letter case, is now replaced
with asterisks before the entry is logged; the rest of the request still is.

Fix `3ef0d1543` (#31659): the mask stopped at the first double quote, so a password holding one
was logged from that quote on, and a body with whitespace before the colon was not masked at all.
Response bodies were logged unmasked, so the LDAP settings, which come back with the application
password in them, put it in the log on every save and every read. The mask now runs to the
closing quote, allows whitespace around the colon and covers responses as well.

Since 24e30d1cb the audit log keeps no request or response body for the sign-in routes under
`/api/v1/auths` (the password change, adding a user, the LDAP settings), only that the call
was made; the mask is checked on the admin reset and the retrieval settings, which still log
their bodies.

Fix `a7e51207c` (#33746): the mask covered passwords only, so API keys, secrets and tokens
(the connection keys of the OpenAI settings, a web search engine's API key) were logged as typed
in both the request and the response. Fields named for a key, secret, token or authorization are
now masked too, a list of them as a whole; the fields around them still are logged.

The tests run on an instance of their own with request and response auditing on, reads
included and bodies kept whole, and read the audit log file in its data directory.

Discriminates: passes on dev b859124f9; with the masking pattern reverted to the exact
`"password"` field the YaCy test goes red (the new value is in the log). With `3ef0d1543`
reverted the double quote, spaced colon and retrieval response tests go red. With bodies
captured again under `/api/v1/auths` the self-service change, add-user and LDAP tests go red.
The key tests pass on dev f659c0d12 and fail with a7e51207c reverted (the keys are logged).
"""

from __future__ import annotations

import json
import time
import uuid

import pytest

from harness.actors import create_user
from harness.instance import ADMIN_EMAIL

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

MASK = "********"
AUDIT_ENV = {
    "AUDIT_LOG_LEVEL": "REQUEST_RESPONSE",
    "AUDIT_EXCLUDED_PATHS": "",
    "ENABLE_AUDIT_GET_REQUESTS": "true",
    # the retrieval settings come back larger than the default cut
    "MAX_BODY_LOG_SIZE": "1000000",
}
RETRIEVAL_CONFIG = ("/api/v1/retrieval/config", "/api/v1/retrieval/config/update")
LDAP_SERVER = "/api/v1/auths/admin/config/ldap/server"


@pytest.fixture
def audited(instance_with):
    return instance_with(AUDIT_ENV)


def _secret(label: str) -> str:
    return f"Sec-{label}-{uuid.uuid4().hex[:12]}"


def _entries_for(instance, path_suffix: str, user_email: str) -> list[dict]:
    audit_log = instance.data_dir / "audit.log"
    if not audit_log.exists():
        return []
    lines = [line for line in audit_log.read_text().splitlines() if line.strip()]
    entries = [json.loads(line) for line in lines]
    return [
        entry
        for entry in entries
        if entry["request_uri"].split("?")[0].endswith(path_suffix)
        and entry["user"].get("email") == user_email
    ]


def _logged_entry(instance, path_suffix: str, user_email: str, verb: str, marker: str = "") -> dict:
    """The newest entry for this route, method and account that carries `marker`, once it lands."""
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        entries = [
            entry
            for entry in _entries_for(instance, path_suffix, user_email)
            if entry["verb"] == verb
            and marker in entry["request_object"] + entry["response_object"]
        ]
        if entries:
            return entries[-1]
        time.sleep(0.2)
    raise AssertionError(f"no audit entry for {verb} {path_suffix} by {user_email}")


def _logged_request(instance, path_suffix: str, user_email: str, marker: str = "") -> str:
    return _logged_entry(instance, path_suffix, user_email, "POST", marker)["request_object"]


def _logged_response(instance, path_suffix: str, user_email: str, verb: str, marker: str) -> str:
    return _logged_entry(instance, path_suffix, user_email, verb, marker)["response_object"]


def _audit_text(instance) -> str:
    audit_log = instance.data_dir / "audit.log"
    return audit_log.read_text() if audit_log.exists() else ""


def _bodyless_entry(instance, path_suffix: str, user_email: str, verb: str, seen: int) -> dict:
    """The entry for this call, once it lands after the `seen` entries already logged."""
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        entries = [
            entry
            for entry in _entries_for(instance, path_suffix, user_email)
            if entry["verb"] == verb
        ]
        if len(entries) > seen:
            return entries[-1]
        time.sleep(0.2)
    raise AssertionError(f"no audit entry for {verb} {path_suffix} by {user_email}")


def _directory_host() -> str:
    # each test saves its own host, so an earlier test's entry is never taken for its own
    return f"ldap-{uuid.uuid4().hex[:8]}.audit.invalid"


def _ldap_settings(app_password: str, host: str) -> dict:
    return {
        "label": "Audit directory",
        "host": host,
        "app_dn": "cn=reader,dc=example,dc=com",
        "app_dn_password": app_password,
        "search_base": "dc=example,dc=com",
        "use_tls": False,
    }


def test_a_users_own_password_change_logs_neither_password(audited):
    account = create_user(audited)
    new_password = _secret("new")

    with audited.client(account.token) as client:
        changed = client.post(
            "/api/v1/auths/update/password",
            json={"password": account.password, "new_password": new_password},
        )
    assert changed.status_code == 200, changed.text

    entry = _bodyless_entry(audited, "/auths/update/password", account.email, "POST", 0)
    assert entry["request_object"] == "" and entry["response_object"] == ""
    audit_text = _audit_text(audited)
    assert new_password not in audit_text, "the new password reached the audit log (#31622)"
    assert account.password not in audit_text


def test_an_admin_reset_logs_the_name_but_not_the_new_password(audited):
    account = create_user(audited)
    new_password = _secret("reset")
    new_name = f"Renamed {uuid.uuid4().hex[:8]}"

    with audited.client() as client:
        reset = client.post(
            f"/api/v1/users/{account.id}/update",
            json={"name": new_name, "password": new_password},
        )
    assert reset.status_code == 200, reset.text

    logged = _logged_request(audited, f"/users/{account.id}/update", ADMIN_EMAIL)
    assert new_password not in logged
    assert json.loads(logged) == {"name": new_name, "password": MASK}


def test_adding_a_user_logs_the_call_without_the_password(audited):
    password = _secret("added")
    email = f"added-{uuid.uuid4().hex[:8]}@example.com"
    seen = len(_entries_for(audited, "/auths/add", ADMIN_EMAIL))

    with audited.client() as client:
        added = client.post(
            "/api/v1/auths/add",
            json={"name": "Added", "email": email, "password": password, "role": "user"},
        )
    assert added.status_code == 200, added.text

    entry = _bodyless_entry(audited, "/auths/add", ADMIN_EMAIL, "POST", seen)
    assert entry["request_object"] == "" and entry["response_object"] == ""
    assert password not in _audit_text(audited)


def test_an_ldap_app_password_saved_and_read_back_stays_out_of_the_log(audited):
    # Unset LDAP settings cannot be posted back, so the module's own instance is left as saved.
    app_password = _secret("ldap")
    host = _directory_host()

    seen = len(_entries_for(audited, "/admin/config/ldap/server", ADMIN_EMAIL))

    with audited.client() as client:
        saved = client.post(LDAP_SERVER, json=_ldap_settings(app_password, host))
        read = client.get(LDAP_SERVER)
    assert saved.status_code == 200, saved.text
    assert read.json()["app_dn_password"] == app_password

    entry = _bodyless_entry(audited, "/admin/config/ldap/server", ADMIN_EMAIL, "POST", seen)
    assert entry["request_object"] == "" and entry["response_object"] == ""
    assert app_password not in _audit_text(audited), (
        "the LDAP password, saved or read back, reached the audit log (#31622, #31659)"
    )


def test_a_yacy_password_in_capitals_is_masked_in_the_retrieval_settings_save(audited, preserve):
    preserve(RETRIEVAL_CONFIG, on=audited)
    yacy_password = _secret("yacy")
    yacy_user = f"yacy-{uuid.uuid4().hex[:8]}"

    with audited.client() as client:
        saved = client.post(
            RETRIEVAL_CONFIG[1],
            json={"web": {"YACY_USERNAME": yacy_user, "YACY_PASSWORD": yacy_password}},
        )
    assert saved.status_code == 200, saved.text

    logged = _logged_request(audited, "/retrieval/config/update", ADMIN_EMAIL)
    assert yacy_password not in logged, (
        f"the YaCy password reached the audit log (#31622): {logged}"
    )
    assert json.loads(logged)["web"] == {"YACY_USERNAME": yacy_user, "YACY_PASSWORD": MASK}


def test_a_password_holding_a_double_quote_is_masked_to_its_end(audited):
    account = create_user(audited)
    tail = _secret("tail")
    new_password = f'Q"{tail}'

    with audited.client() as client:
        reset = client.post(f"/api/v1/users/{account.id}/update", json={"password": new_password})
    assert reset.status_code == 200, reset.text

    logged = _logged_request(audited, f"/users/{account.id}/update", ADMIN_EMAIL)
    assert tail not in logged, (
        f"the password after its quote reached the audit log (#31659): {logged}"
    )
    assert json.loads(logged) == {"password": MASK}


def test_a_password_field_spaced_before_its_colon_is_masked(audited):
    account = create_user(audited)
    new_password = _secret("spaced")
    body = f'{{"password" : "{new_password}"}}'

    with audited.client() as client:
        reset = client.post(
            f"/api/v1/users/{account.id}/update",
            content=body,
            headers={"Content-Type": "application/json"},
        )
    assert reset.status_code == 200, reset.text

    logged = _logged_request(audited, f"/users/{account.id}/update", ADMIN_EMAIL)
    assert new_password not in logged, f"a spaced password field was logged (#31659): {logged}"
    assert json.loads(logged) == {"password": MASK}


def test_the_yacy_password_sent_back_is_masked_in_the_logged_responses(audited, preserve):
    preserve(RETRIEVAL_CONFIG, on=audited)
    yacy_password = _secret("yacy-response")
    yacy_user = f"yacy-{uuid.uuid4().hex[:8]}"

    with audited.client() as client:
        saved = client.post(
            RETRIEVAL_CONFIG[1],
            json={"web": {"YACY_USERNAME": yacy_user, "YACY_PASSWORD": yacy_password}},
        )
        read = client.get(RETRIEVAL_CONFIG[0])
    assert saved.status_code == 200, saved.text
    assert read.json()["web"]["YACY_PASSWORD"] == yacy_password

    for verb, path_suffix in (("POST", RETRIEVAL_CONFIG[1]), ("GET", RETRIEVAL_CONFIG[0])):
        logged = _logged_response(audited, path_suffix, ADMIN_EMAIL, verb, yacy_user)
        assert yacy_password not in logged, (
            f"the YaCy password in the {verb} response reached the audit log (#31659): {logged}"
        )
        assert json.loads(logged)["web"]["YACY_PASSWORD"] == MASK


OPENAI_CONFIG = ("/openai/config", "/openai/config/update")


def test_connection_api_keys_are_masked_in_the_logged_request_and_response(audited, preserve):
    preserve(OPENAI_CONFIG, on=audited)
    with audited.client() as client:
        current = client.get(OPENAI_CONFIG[0])
        assert current.status_code == 200, current.text
        urls = current.json()["OPENAI_API_BASE_URLS"]
        keys = [_secret(f"sk-{index}") for index in range(len(urls))]
        saved = client.post(OPENAI_CONFIG[1], json={**current.json(), "OPENAI_API_KEYS": keys})
    assert saved.status_code == 200, saved.text
    assert saved.json()["OPENAI_API_KEYS"] == keys

    entry = _logged_entry(audited, OPENAI_CONFIG[1], ADMIN_EMAIL, "POST", urls[0])
    for side in ("request_object", "response_object"):
        logged = entry[side]
        leaked = [key for key in keys if key in logged]
        assert leaked == [], f"connection API keys reached the audit log's {side} (#33746)"
        assert json.loads(logged)["OPENAI_API_KEYS"] == MASK
        assert json.loads(logged)["OPENAI_API_BASE_URLS"] == urls


def test_a_web_search_api_key_is_masked_in_the_retrieval_settings_save_and_read(audited, preserve):
    preserve(RETRIEVAL_CONFIG, on=audited)
    api_key = _secret("brave")
    engine_user = f"yacy-{uuid.uuid4().hex[:8]}"

    with audited.client() as client:
        saved = client.post(
            RETRIEVAL_CONFIG[1],
            json={"web": {"YACY_USERNAME": engine_user, "BRAVE_SEARCH_API_KEY": api_key}},
        )
        read = client.get(RETRIEVAL_CONFIG[0])
    assert saved.status_code == 200, saved.text
    assert read.json()["web"]["BRAVE_SEARCH_API_KEY"] == api_key

    logged = {
        "the request": _logged_request(audited, RETRIEVAL_CONFIG[1], ADMIN_EMAIL, engine_user),
        "the read back": _logged_response(
            audited, RETRIEVAL_CONFIG[0], ADMIN_EMAIL, "GET", engine_user
        ),
    }
    for where, text in logged.items():
        assert api_key not in text, f"the search API key in {where} reached the audit log (#33746)"
        assert json.loads(text)["web"]["BRAVE_SEARCH_API_KEY"] == MASK
        assert json.loads(text)["web"]["YACY_USERNAME"] == engine_user
