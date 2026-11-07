"""Journey: a native app signs in through the browser and keeps its own renewable session.

The app opens a login request with a loopback callback (`http://127.0.0.1:<port>`) and a PKCE
challenge. A signed-in person approves it in the browser, which hands the app a one-time code on
that callback. The app trades the code plus its verifier for an access token and a refresh token,
and later trades the refresh token for a new pair. The access token works as the account's bearer
token; the code and the refresh token never do. The account lists its sessions and revokes them,
which cuts the app off at once. Only an approved account (user or admin) gets or keeps one, and
only a signed-in person can approve, never an API key. Added in 404d27a36 with 99f4541e3,
69189cbaa and c16b77ba0 (0.13.1). The browser side is e2e/auth/test_native_app_sign_in.py; which
role reaches each route is in integration/security/test_route_role_sweep.py.

Every request carries an address of its own (`X-Forwarded-For`), since the sessions routes allow
30 requests a minute per address.

Discriminates: on dev f659c0d12, each of these edits in a backend copy turns its test red: an
exchange that skips the consumed check and keeps the request open (code reuse), dropping the
challenge check (wrong verifier), the callback check (wrong callback), the 127.0.0.1 host check
(localhost and other hosts), the already-approved checks (second approval), the refresh-token-id
match on refresh (old refresh token), the request-id check on a retried refresh, the session
lookup in token validation (revoke), the revoke on sign-out, the account filter in the list or in
revoke, the role check in the proof (pending and deactivated account), the human-session
dependency (API key), refresh tokens among the accepted token types and a limit of 3000.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import uuid

import httpx
import pytest

from harness.actors import sign_in
from harness.mfa import new_address

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

SESSIONS = "/api/v1/auths/sessions"
OWN_SESSION = "/api/v1/auths/"
ADMIN_CONFIG = "/api/v1/auths/admin/config"
LOGIN_EXPIRED = "Login request expired or already authorized."


class App:
    """A native app on its own address, holding the PKCE pair of its pending sign-in."""

    def __init__(self, instance, name: str | None = None):
        self.instance = instance
        self.address = new_address()
        self.name = name or f"Desktop {uuid.uuid4().hex[:8]}"
        self.callback = f"http://127.0.0.1:{40000 + secrets.randbelow(20000)}/callback"
        self.verifier = secrets.token_urlsafe(48)
        self.state = secrets.token_urlsafe(32)

    @property
    def challenge(self) -> str:
        digest = hashlib.sha256(self.verifier.encode()).digest()
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()

    def client(self, token: str | None = None) -> httpx.Client:
        headers = {"X-Forwarded-For": self.address}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return httpx.Client(base_url=self.instance.base_url, headers=headers, timeout=60.0)

    def post(self, path: str, body: dict | None = None, token: str | None = None):
        with self.client(token) as client:
            return client.post(f"{SESSIONS}{path}", json=body)

    def open_request(self, **overrides) -> httpx.Response:
        body = {
            "name": self.name,
            "redirect_uri": self.callback,
            "state": self.state,
            "code_challenge": self.challenge,
            **overrides,
        }
        return self.post("/requests", body)

    def opened(self) -> str:
        opened = self.open_request()
        assert opened.status_code == 200, f"opening the login request failed: {opened.text}"
        return opened.json()["session_id"]

    def exchange(self, session_id: str, code: str, **overrides) -> httpx.Response:
        body = {
            "session_id": session_id,
            "code": code,
            "code_verifier": self.verifier,
            "redirect_uri": self.callback,
            **overrides,
        }
        return self.post("/exchange", body)

    def refresh(self, refresh_token: str, **extra) -> httpx.Response:
        return self.post("/refresh", {"refresh_token": refresh_token, **extra})


def approve(app: App, session_id: str, token: str) -> httpx.Response:
    """What the approval page sends for the signed-in person in the browser."""
    return app.post(f"/{session_id}/authorize", token=token)


def code_from(approval: httpx.Response, app: App) -> str:
    assert approval.status_code == 200, f"approving the login request failed: {approval.text}"
    callback = httpx.URL(approval.json()["redirect_uri"])
    assert str(callback.copy_with(query=None)) == app.callback, callback
    assert callback.params["state"] == app.state, "the callback lost the app's state"
    return callback.params["code"]


def human_token(instance, account) -> str:
    """A session from the sign-in form, which carries the proof of how the person signed in."""
    return sign_in(instance, account.email, account.password)


def signed_in_app(instance, account, name: str | None = None) -> tuple[App, dict]:
    app = App(instance, name)
    session_id = app.opened()
    code = code_from(approve(app, session_id, human_token(instance, account)), app)
    exchanged = app.exchange(session_id, code)
    assert exchanged.status_code == 200, f"the code exchange failed: {exchanged.text}"
    return app, exchanged.json()


def status_with(instance, token: str, path: str = OWN_SESSION) -> int:
    with instance.client(token) as client:
        return client.get(path).status_code


def listed(instance, token: str) -> list[dict]:
    with instance.client(token) as client:
        answer = client.get(SESSIONS)
    assert answer.status_code == 200, answer.text
    return answer.json()


def test_an_approved_app_trades_its_code_for_a_session_of_that_account(instance, make_user):
    account = make_user()
    app = App(instance)
    session_id = app.opened()
    person = human_token(instance, account)

    with instance.client(person) as client:
        pending = client.get(f"{SESSIONS}/{session_id}/request")
    assert pending.status_code == 200, pending.text
    assert pending.json()["name"] == app.name
    code = code_from(approve(app, session_id, person), app)
    exchanged = app.exchange(session_id, code)

    assert exchanged.status_code == 200, exchanged.text
    tokens = exchanged.json()
    assert tokens["session_id"] == session_id
    assert tokens["token_type"] == "Bearer"
    assert tokens["refresh_expires_at"] >= tokens["expires_at"]
    with instance.client(tokens["token"]) as client:
        me = client.get(OWN_SESSION)
    assert me.status_code == 200, f"the app's access token is refused: {me.text}"
    assert me.json()["id"] == account.id
    assert [entry["name"] for entry in listed(instance, person)] == [app.name]


def test_a_login_code_works_once(instance, make_user):
    account = make_user()
    app = App(instance)
    session_id = app.opened()
    code = code_from(approve(app, session_id, human_token(instance, account)), app)

    assert app.exchange(session_id, code).status_code == 200
    reused = app.exchange(session_id, code)

    assert reused.status_code == 401, f"a used login code was accepted again: {reused.text}"


def test_a_code_needs_the_apps_own_verifier_and_callback(instance, make_user):
    account = make_user()
    app = App(instance)
    session_id = app.opened()
    code = code_from(approve(app, session_id, human_token(instance, account)), app)

    wrong_verifier = app.exchange(session_id, code, code_verifier=secrets.token_urlsafe(48))
    other_port = app.callback.replace("/callback", "/elsewhere")
    wrong_callback = app.exchange(session_id, code, redirect_uri=other_port)
    other_request = app.exchange(str(uuid.uuid4()), code)

    assert wrong_verifier.status_code == 401, f"a wrong verifier got tokens: {wrong_verifier.text}"
    assert wrong_callback.status_code == 401, f"a wrong callback got tokens: {wrong_callback.text}"
    assert other_request.status_code == 401, other_request.text
    # the refused attempts did not use up the code for the real app
    assert app.exchange(session_id, code).status_code == 200


@pytest.mark.parametrize(
    "callback",
    [
        "http://localhost:43123/callback",
        "https://127.0.0.1:43123/callback",
        "http://127.0.0.1/callback",
        "http://127.0.0.1:43123/callback?next=x",
        "http://evil.example:43123/callback",
        "http://user:pass@127.0.0.1:43123/callback",
    ],
)
def test_a_login_request_only_calls_back_to_a_loopback_port(instance, callback):
    refused = App(instance).open_request(redirect_uri=callback)

    assert refused.status_code == 422, f"{callback} was accepted as a callback: {refused.text}"


def test_a_login_request_is_approved_once_and_only_while_it_exists(instance, make_user):
    account, other = make_user(), make_user()
    app = App(instance)
    session_id = app.opened()
    code = code_from(approve(app, session_id, human_token(instance, account)), app)

    again = approve(app, session_id, human_token(instance, other))
    unknown = approve(app, str(uuid.uuid4()), human_token(instance, other))

    assert again.status_code == 409, f"an approved request was approved again: {again.text}"
    assert again.json()["detail"] == LOGIN_EXPIRED
    assert unknown.status_code == 409, unknown.text
    assert app.exchange(session_id, code).status_code == 200
    assert approve(app, session_id, human_token(instance, other)).status_code == 409
    with instance.client(human_token(instance, other)) as client:
        signed_in = client.get(f"{SESSIONS}/{session_id}/request")
    assert signed_in.status_code == 404, (
        f"a finished sign-in still shows as pending: {signed_in.text}"
    )


def test_refresh_rotates_and_the_old_refresh_token_is_refused(instance, make_user):
    app, first = signed_in_app(instance, make_user())

    rotated = app.refresh(first["refresh_token"])
    assert rotated.status_code == 200, rotated.text
    second = rotated.json()
    assert second["session_id"] == first["session_id"]
    assert second["refresh_token"] != first["refresh_token"]
    stale = app.refresh(first["refresh_token"])
    after = app.refresh(second["refresh_token"])

    assert stale.status_code == 401, f"the rotated-out refresh token still works: {stale.text}"
    assert after.status_code == 200, f"the new refresh token is refused: {after.text}"
    assert status_with(instance, after.json()["token"]) == 200


def test_a_retried_refresh_gets_its_answer_again_and_only_for_that_request(instance, make_user):
    app, first = signed_in_app(instance, make_user())
    request_id = str(uuid.uuid4())

    answered = app.refresh(first["refresh_token"], request_id=request_id)
    retried = app.refresh(first["refresh_token"], request_id=request_id)
    other_request = app.refresh(first["refresh_token"], request_id=str(uuid.uuid4()))

    assert answered.status_code == 200, answered.text
    assert retried.status_code == 200, f"a lost refresh answer is not recoverable: {retried.text}"
    assert retried.json()["refresh_token"] == answered.json()["refresh_token"]
    assert other_request.status_code == 401, (
        f"the rotated-out refresh token works for a new request: {other_request.text}"
    )


def test_revoking_a_session_cuts_the_app_off(instance, make_user):
    account = make_user()
    app, tokens = signed_in_app(instance, account)
    person = human_token(instance, account)
    assert status_with(instance, tokens["token"]) == 200

    with instance.client(person) as client:
        revoked = client.delete(f"{SESSIONS}/{tokens['session_id']}")
        again = client.delete(f"{SESSIONS}/{tokens['session_id']}")

    assert revoked.status_code == 200, revoked.text
    assert again.status_code == 404
    assert status_with(instance, tokens["token"]) == 401, "a revoked session's token still works"
    assert app.refresh(tokens["refresh_token"]).status_code == 401
    assert listed(instance, person) == []
    assert status_with(instance, person) == 200, "revoking the app signed the browser out"


def test_signing_out_with_the_apps_token_ends_its_session(instance, make_user):
    account = make_user()
    app, tokens = signed_in_app(instance, account)

    with app.client(tokens["token"]) as client:
        assert client.post("/api/v1/auths/signout").status_code == 200

    assert app.refresh(tokens["refresh_token"]).status_code == 401
    assert listed(instance, human_token(instance, account)) == []


def test_an_account_lists_and_revokes_only_its_own_sessions(instance, make_user):
    owner, stranger = make_user(), make_user()
    _, owned = signed_in_app(instance, owner, name=f"Laptop {uuid.uuid4().hex[:8]}")
    _, theirs = signed_in_app(instance, stranger)
    App(instance).opened()  # a request nobody approved yet is no session
    stranger_token = human_token(instance, stranger)

    names = [entry["name"] for entry in listed(instance, human_token(instance, owner))]
    with instance.client(stranger_token) as client:
        foreign = client.delete(f"{SESSIONS}/{owned['session_id']}")

    assert len(names) == 1, f"the owner sees sessions that are not theirs: {names}"
    assert [entry["id"] for entry in listed(instance, stranger_token)] == [theirs["session_id"]]
    assert foreign.status_code == 404, f"another account revoked the session: {foreign.text}"
    assert status_with(instance, owned["token"]) == 200


def test_a_pending_account_cannot_approve_a_login_request(instance, make_user):
    pending = make_user(role="pending")
    app = App(instance)
    session_id = app.opened()

    refused = approve(app, session_id, human_token(instance, pending))

    assert refused.status_code == 403, f"a pending account approved a sign-in: {refused.text}"
    assert refused.json()["detail"] == "Account approval is required."


def test_a_deactivated_account_loses_its_app_session(instance, admin, make_user):
    account = make_user()
    app = App(instance)
    session_id = app.opened()
    code = code_from(approve(app, session_id, human_token(instance, account)), app)
    _, active = signed_in_app(instance, account)

    with admin.client() as client:
        demoted = client.post(f"/api/v1/users/{account.id}/update", json={"role": "pending"})
    assert demoted.status_code == 200, demoted.text

    exchanged = app.exchange(session_id, code)
    refreshed = App(instance).refresh(active["refresh_token"])
    assert exchanged.status_code == 403, f"a demoted account got app tokens: {exchanged.text}"
    assert refreshed.status_code == 403, f"a demoted account refreshed: {refreshed.text}"


def test_the_code_and_refresh_token_are_no_bearer_tokens(instance, make_user):
    account = make_user()
    app = App(instance)
    session_id = app.opened()
    code = code_from(approve(app, session_id, human_token(instance, account)), app)

    assert status_with(instance, code) == 401, "the one-time login code works as a bearer token"
    tokens = app.exchange(session_id, code).json()
    assert status_with(instance, tokens["refresh_token"]) == 401, (
        "the refresh token works as a bearer token"
    )
    assert status_with(instance, tokens["token"]) == 200


def test_an_api_key_cannot_approve_or_open_a_session(instance, make_user, preserve):
    preserve("admin_config")
    keyholder = make_user(role="admin")
    with keyholder.client() as client:
        current = client.get(ADMIN_CONFIG).json()
        assert client.post(ADMIN_CONFIG, json={**current, "ENABLE_API_KEYS": True}).is_success
        api_key = client.post("/api/v1/auths/api_key").json()["api_key"]
    app = App(instance)
    session_id = app.opened()

    approved = approve(app, session_id, api_key)
    opened = app.post("", {"name": "From a key"}, token=api_key)

    assert approved.status_code == 403, f"an API key approved a sign-in: {approved.text}"
    assert opened.status_code == 403, f"an API key opened a session: {opened.text}"
    assert approve(app, session_id, human_token(instance, keyholder)).status_code == 200


def test_a_signed_in_person_opens_a_session_directly(instance, make_user):
    account = make_user()
    app = App(instance)

    opened = app.post("", {"name": app.name}, token=human_token(instance, account))

    assert opened.status_code == 200, opened.text
    tokens = opened.json()
    assert app.refresh(tokens["refresh_token"]).status_code == 200
    assert [entry["name"] for entry in listed(instance, tokens["token"])] == [app.name]


def test_too_many_requests_from_one_address_are_cut_off(instance):
    app = App(instance)
    answers = [app.open_request().status_code for _ in range(30)]

    assert set(answers) == {200}, answers
    limited = app.open_request()
    assert limited.status_code == 429, f"the 31st request in a minute went through: {limited.text}"
    assert App(instance).open_request().status_code == 200
