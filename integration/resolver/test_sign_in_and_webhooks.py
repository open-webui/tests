"""Journey: SSO sign-in and outgoing webhooks reached by host name, under both resolvers.

`AIOHTTP_CLIENT_ASYNC_DNS_RESOLVER` (off by default since c5ec01b1f, PR #28242, after c-ares broke
name resolution in #28013 and #28215) decides which resolver every aiohttp connector Open WebUI
opens uses. The SSO instance here is named `localhost` in its provider URL, as an operator writes
it, and the fake OIDC provider names its other endpoints by `localhost`, a hosts-file name for
`::1` alone or a hosts-file name for another local address. Under both resolvers a new account gets
the picture the provider names, an expiring session token is refreshed at the token endpoint the
discovery document names and forwarded, a token exchange asks the provider's introspection endpoint
which client the token was minted for, and signing out finds the end-session endpoint. A user's
notification target, the admin's event webhook and a finished chat's `always` target are each
delivered to a service named by host. A picture, refresh endpoint or webhook whose name does not
resolve fails the same way under both: the default avatar, no forwarded token and a refused
test button, with the account still created. Since ab73749ca (#34585) a refresh endpoint that does
not resolve counts as a provider outage: the session is kept, and once the name resolves again the
next call refreshes and forwards a token. That test passes on dev b6e71d4d7 and fails on
ab73749ca~1 (the session is gone). Where the fetch guard is on (the default) a webhook naming a
local service or an unresolvable host is refused at once, identically under both.

The requests, httpx and authlib calls are not affected by the flag: the code and token exchange
of the sign-in itself, its userinfo and key fetches (authlib over httpx), the channels' incoming
webhooks (no outgoing call), the GitHub email lookup (a hardcoded public host) and the terminal
working-directory call of automations (terminals). Sign-out with a provider that does not
resolve is left out: the session that sign-out reads only exists after a sign-in through that
provider. The introspection endpoint is read from the discovery document once, at the first
sign-in, so it is named `localhost` for the instance's life.

Discriminates: on dev 176d31d1d, a backend copy whose `env.py` installs a resolver that fails every
lookup when the flag is on turns every c-ares run of a by-name test red and leaves every threaded
run green; the same resolver installed for the flag off does the reverse. A failure test stays green
under a failing resolver by design, and a resolver that takes 20 seconds to refuse an unknown name
turns it red. Its c-ares runs skip, naming the reason, on a machine whose DNS server keeps c-ares
from refusing an unknown name at once (a cached reply with a stale EDNS cookie, c-ares issues 1081
and 1271).
"""

from __future__ import annotations

import base64
import time
from urllib.parse import urlsplit

import pytest

from harness.actors import admin_of, create_user
from harness.host_names import (
    FAILS_WITHIN,
    RESOLVERS,
    UNRESOLVABLE,
    name_forms,
    serving_by_name,
    timed,
)
from harness.host_names import (
    name_form as form_named,
)
from harness.inflight import start_slow_reply
from harness.listener import Listener, json_answer
from harness.oidc_provider import (
    CLIENT_ID,
    OidcProvider,
    oauth_settings,
    serve,
    session_user,
    sign_in,
    sso_env,
)
from harness.web_retrieval import LOCAL_WEB_FETCH

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source, pytest.mark.slow]

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)
EXPIRING_SOON = 60  # inside the five minutes before expiry in which Open WebUI refreshes
TOOL_SPEC = {"openapi": "3.0.0", "info": {"title": "SSO tools", "version": "1"}, "paths": {}}
EXCHANGE = "/api/v1/auths/oauth/oidc/token/exchange"
ADMIN_CONFIG = "/api/v1/auths/admin/config"
PERMISSIONS = "/api/v1/users/default/permissions"
TARGETS = "/api/v1/notifications/targets"
EVENT_WEBHOOKS = "/api/events/webhooks"
FAILED_DELIVERY = "Webhook delivery failed"
INVALID_URL = "The URL you provided is invalid. Please double-check and try again."


def _local_addresses() -> list[str]:
    """Addresses of the hosts-file name forms this machine has, for the provider to listen on."""
    addresses = []
    for label in ("ipv6-only", "hosts-file"):
        try:
            addresses.append(form_named(label).address)
        except pytest.skip.Exception:
            continue
    return addresses


@pytest.fixture(scope="module")
def idp():
    provider, shutdown = serve(_local_addresses())
    port = urlsplit(provider.base_url).port
    provider.endpoint_bases = {"introspection_endpoint": f"http://localhost:{port}"}
    yield provider
    shutdown()


@pytest.fixture
def sso(resolver, idp, package_instance_with):
    """The SSO instance for this resolver, its provider named `localhost` and local fetching on."""
    discovery = f"http://localhost:{urlsplit(idp.base_url).port}/.well-known/openid-configuration"
    env = {
        **RESOLVERS[resolver],
        **LOCAL_WEB_FETCH,
        **sso_env(idp),
        "OPENID_PROVIDER_URL": discovery,
        "OAUTH_TOKEN_EXCHANGE_TRUSTED_CLIENT_IDS": CLIENT_ID,
    }
    instance = package_instance_with(env)
    idp.reset()
    return instance


@pytest.fixture
def tool_server(listener):
    listener.route("GET", "/openapi.json", json_answer(TOOL_SPEC))
    return listener


def _named_base(idp: OidcProvider, host: str) -> str:
    return f"http://{host}:{urlsplit(idp.base_url).port}"


def _signed_in_user(sso) -> dict:
    result = sign_in(sso)
    assert result.token, f"the sign-in failed: {result.error}"
    return session_user(sso, result.token)


def _sign_in_as_admin(sso, idp):
    """An SSO admin's browser, holding the session and the `oauth_session_id` cookie."""
    with oauth_settings(sso, ENABLE_OAUTH_ROLE_MANAGEMENT=True):
        idp.sign_in_as(roles=["admin"])
        result = sign_in(sso)
    assert result.token, f"the sign-in failed: {result.error}"
    return result.browser


def _forwarded_token(browser, tool_server) -> str | None:
    """Verify a `system_oauth` tool server; returns the bearer token it was sent."""
    browser.post(
        "/api/v1/configs/tool_servers/verify",
        json={
            "url": tool_server.base_url,
            "path": "openapi.json",
            "type": "openapi",
            "auth_type": "system_oauth",
            "key": "",
            "config": {},
        },
    )
    fetch = tool_server.requests_to("/openapi.json")[-1]
    authorization = fetch.headers.get("Authorization")
    return authorization.removeprefix("Bearer ") if authorization else None


def _refresh_grants(idp: OidcProvider):
    entries = idp.requests_to("/token")
    return [entry for entry in entries if entry.form.get("grant_type") == "refresh_token"]


# --- the picture a new account gets ------------------------------------------------------------


@pytest.mark.parametrize("name_form", name_forms())
def test_a_new_account_gets_the_picture_fetched_by_name(resolver, name_form, sso, idp):
    with serving_by_name(name_form) as pictures:
        pictures.route("GET", "/avatar.png", (200, {"Content-Type": "image/png"}, PNG))
        idp.sign_in_as(picture=f"{pictures.base_url}/avatar.png")
        account = _signed_in_user(sso)

    assert account["profile_image_url"].startswith("data:image/png;base64,")
    [fetch] = pictures.requests_to("/avatar.png")
    assert fetch.headers["Authorization"] == f"Bearer {idp.issued[-1]['access_token']}"
    assert fetch.headers["Host"] == urlsplit(pictures.base_url).netloc


@pytest.mark.usefixtures("refuses_unknown_names")
def test_a_picture_that_does_not_resolve_leaves_the_default_avatar(resolver, sso, idp):
    idp.sign_in_as(picture=f"http://{UNRESOLVABLE}/avatar.png")

    account, seconds = timed(_signed_in_user, sso)

    assert account["profile_image_url"] == "/user.png"
    assert seconds < FAILS_WITHIN, f"the sign-in took {seconds:.1f}s to give up on the picture"


# --- the session token, refreshed at the endpoint the discovery document names ----------------


@pytest.mark.parametrize("name_form", name_forms())
def test_an_expiring_token_is_refreshed_at_an_endpoint_named_by_host(
    resolver, name_form, sso, idp, tool_server
):
    host = _form_host(name_form)
    idp.token_lifetime = EXPIRING_SOON
    browser = _sign_in_as_admin(sso, idp)
    signed_in = idp.issued[-1]
    idp.endpoint_bases["token_endpoint"] = _named_base(idp, host)
    try:
        forwarded = _forwarded_token(browser, tool_server)
    finally:
        del idp.endpoint_bases["token_endpoint"]

    [refresh] = _refresh_grants(idp)
    assert refresh.form["refresh_token"] == signed_in["refresh_token"]
    assert refresh.headers["Host"] == f"{host}:{urlsplit(idp.base_url).port}"
    assert forwarded == idp.issued[-1]["access_token"], (
        "the expiring token was forwarded unrefreshed"
    )


@pytest.mark.usefixtures("refuses_unknown_names")
def test_a_refresh_endpoint_that_does_not_resolve_forwards_nothing_and_keeps_the_session(
    resolver, sso, idp, tool_server
):
    idp.token_lifetime = EXPIRING_SOON
    browser = _sign_in_as_admin(sso, idp)
    signed_in = idp.issued[-1]
    idp.endpoint_bases["token_endpoint"] = _named_base(idp, UNRESOLVABLE)
    try:
        forwarded, seconds = timed(_forwarded_token, browser, tool_server)
    finally:
        del idp.endpoint_bases["token_endpoint"]
    after_it_resolves = _forwarded_token(browser, tool_server)

    assert forwarded is None, "a token that could not be refreshed was forwarded"
    assert seconds < FAILS_WITHIN, f"the refresh took {seconds:.1f}s to fail"
    assert after_it_resolves == idp.issued[-1]["access_token"], (
        "the failed refresh dropped the SSO session, so nothing was forwarded once the token "
        "endpoint resolved again"
    )
    assert _refresh_grants(idp)[-1].form["refresh_token"] == signed_in["refresh_token"]


# --- the token exchange and sign-out ------------------------------------------------------------


def _exchange(sso, idp, person: dict):
    token = idp.issue_access_token(person)
    with sso.client() as client:
        return client.post(EXCHANGE, json={"token": token})


def test_a_token_exchange_asks_the_introspection_endpoint_who_the_token_is_for(resolver, sso, idp):
    person = idp.sign_in_as()
    _signed_in_user(sso)

    trusted = _exchange(sso, idp, person)
    idp.introspected_client_id = "somebody-elses-client"
    try:
        untrusted = _exchange(sso, idp, person)
    finally:
        idp.introspected_client_id = CLIENT_ID

    assert trusted.status_code == 200, trusted.text
    assert untrusted.status_code == 403, untrusted.text
    asked = idp.requests_to("/introspect")
    assert len(asked) == 2
    assert asked[0].headers["Host"] == f"localhost:{urlsplit(idp.base_url).port}"
    assert asked[0].headers["Authorization"].startswith("Basic ")
    assert asked[0].form["token_type_hint"] == "access_token"


def test_signing_out_of_sso_finds_the_end_session_endpoint(resolver, sso, idp):
    idp.sign_in_as()
    result = sign_in(sso)
    idp.endpoint_bases["end_session_endpoint"] = _named_base(idp, "localhost")
    try:
        signed_out = result.browser.post("/api/v1/auths/signout")
    finally:
        del idp.endpoint_bases["end_session_endpoint"]

    assert signed_out.status_code == 200, signed_out.text
    redirect = signed_out.json()["redirect_url"]
    assert redirect.startswith(f"{_named_base(idp, 'localhost')}/logout?id_token_hint=")
    fetches = idp.requests_to("/.well-known/openid-configuration")
    assert [entry for entry in fetches if "aiohttp" in entry.headers["User-Agent"]]


# --- webhooks -----------------------------------------------------------------------------------


def _wait_for(condition, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return False


def _save(client, path: str, **changes) -> None:
    current = client.get(path)
    current.raise_for_status()
    saved = client.post(path, json={**current.json(), **changes})
    assert saved.status_code == 200, saved.text


def _allow_user_webhooks(client) -> None:
    _save(client, ADMIN_CONFIG, ENABLE_USER_WEBHOOKS=True)
    features = client.get(PERMISSIONS).json()["features"]
    _save(client, PERMISSIONS, features={**features, "webhooks": True})


def _form_host(label: str) -> str:
    return form_named(label).host


def _host_of(service: Listener) -> str:
    return urlsplit(service.base_url).netloc


@pytest.fixture
def allowed_webhooks(sso, preserve):
    preserve("admin_config", "permissions", on=sso)
    with admin_of(sso).client() as client:
        _allow_user_webhooks(client)
    return sso


@pytest.mark.parametrize("name_form", name_forms())
def test_a_notification_target_is_tested_by_name(resolver, name_form, allowed_webhooks):
    with serving_by_name(name_form) as service:
        service.route("POST", "/hook", json_answer({}))
        account = create_user(allowed_webhooks)
        with account.client() as client:
            created = client.post(
                TARGETS, json={"id": "ops", "config": {"url": f"{service.base_url}/hook"}}
            )
            tested = client.post(f"{TARGETS}/ops/test")

    assert created.status_code == 200, created.text
    assert tested.status_code == 200 and tested.json() == {"ok": True}, tested.text
    [delivered] = service.requests_to("/hook")
    assert delivered.json() == {"action": "test", "user_id": account.id}
    assert delivered.headers["Host"] == _host_of(service)


@pytest.mark.parametrize("name_form", name_forms())
def test_a_new_account_is_delivered_to_an_event_webhook_by_name(
    resolver, name_form, allowed_webhooks
):
    with serving_by_name(name_form) as service, admin_of(allowed_webhooks).client() as admin:
        service.route("POST", "/events", json_answer({}))
        created = admin.post(
            EVENT_WEBHOOKS, json={"url": f"{service.base_url}/events", "events": ["user.created"]}
        )
        assert created.status_code == 200, created.text
        try:
            account = create_user(allowed_webhooks)
            assert _wait_for(lambda: service.requests_to("/events")), "no event was delivered"
        finally:
            admin.delete(f"{EVENT_WEBHOOKS}/{created.json()['id']}")

    [delivered] = service.requests_to("/events")
    assert delivered.json()["event"] == "user.created"
    assert delivered.json()["subject"] == {"type": "user", "id": account.id}
    assert delivered.headers["Host"] == _host_of(service)


@pytest.mark.parametrize("name_form", name_forms())
def test_a_finished_chat_is_delivered_to_an_always_target_by_name(
    resolver, name_form, allowed_webhooks
):
    account = create_user(allowed_webhooks)
    with serving_by_name(name_form) as service:
        service.route("POST", "/done", json_answer({}))
        target = {
            "id": "done",
            "config": {"url": f"{service.base_url}/done"},
            "events": ["chat.finished"],
            "delivery": "always",
        }
        with account.client() as client:
            assert client.post(TARGETS, json=target).status_code == 200
            start_slow_reply(client, allowed_webhooks.upstream, chunk_delay=0.01)
        assert _wait_for(lambda: service.requests_to("/done")), "the finished chat was not sent"

    [delivered] = service.requests_to("/done")
    assert delivered.json()["action"] == "chat"
    assert delivered.json()["message"].startswith("part-0 ")
    assert delivered.headers["Host"] == _host_of(service)


@pytest.mark.usefixtures("refuses_unknown_names")
def test_a_notification_target_that_does_not_resolve_fails_its_test_button(
    resolver, allowed_webhooks
):
    account = create_user(allowed_webhooks)
    with account.client() as client:
        created = client.post(
            TARGETS, json={"id": "gone", "config": {"url": f"http://{UNRESOLVABLE}/hook"}}
        )
        tested, seconds = timed(client.post, f"{TARGETS}/gone/test")

    assert created.status_code == 200, created.text
    assert (tested.status_code, tested.json()) == (400, {"detail": FAILED_DELIVERY})
    assert seconds < FAILS_WITHIN, f"the delivery took {seconds:.1f}s to fail"


@pytest.mark.usefixtures("refuses_unknown_names")
def test_an_event_webhook_that_does_not_resolve_leaves_account_creation_and_the_next_webhook_alone(
    resolver, allowed_webhooks
):
    with serving_by_name("localhost") as service, admin_of(allowed_webhooks).client() as admin:
        service.route("POST", "/events", json_answer({}))
        gone = admin.post(
            EVENT_WEBHOOKS,
            json={"url": f"http://{UNRESOLVABLE}/events", "events": ["user.created"]},
        )
        working = admin.post(
            EVENT_WEBHOOKS, json={"url": f"{service.base_url}/events", "events": ["user.created"]}
        )
        assert gone.status_code == 200 and working.status_code == 200
        try:
            _, seconds = timed(create_user, allowed_webhooks)
            assert _wait_for(lambda: service.requests_to("/events")), "no event was delivered"
        finally:
            admin.delete(f"{EVENT_WEBHOOKS}/{gone.json()['id']}")
            admin.delete(f"{EVENT_WEBHOOKS}/{working.json()['id']}")

    assert seconds < FAILS_WITHIN, f"adding the account took {seconds:.1f}s"


# --- where the fetch guard is on, a webhook naming a local service is refused -------------------


@pytest.fixture
def guarded_webhooks(resolving_instance, resolving_admin, preserve):
    preserve("admin_config", "permissions", on=resolving_instance)
    with resolving_admin.client() as client:
        _allow_user_webhooks(client)
    return resolving_instance


@pytest.mark.parametrize("name_form", name_forms())
def test_a_target_naming_a_local_service_is_refused_at_once(resolver, name_form, guarded_webhooks):
    with serving_by_name(name_form) as service:
        with create_user(guarded_webhooks).client() as client:
            created = client.post(TARGETS, json={"config": {"url": f"{service.base_url}/hook"}})
            listed = client.get(TARGETS)

    assert (created.status_code, created.json()) == (400, {"detail": INVALID_URL})
    assert listed.json() == {"targets": []}
    assert service.received == []


@pytest.mark.parametrize("name_form", name_forms())
def test_an_event_webhook_naming_a_local_service_is_refused_at_once(
    resolver, name_form, guarded_webhooks, resolving_admin
):
    with serving_by_name(name_form) as service, resolving_admin.client() as admin:
        created = admin.post(
            EVENT_WEBHOOKS, json={"url": f"{service.base_url}/events", "events": ["user.created"]}
        )
        listed = admin.get(EVENT_WEBHOOKS).json()

    assert created.status_code == 400, created.text
    assert INVALID_URL in created.text
    assert all(f"{service.base_url}/events" != webhook["url"] for webhook in listed)
    assert service.received == []


def test_a_target_that_does_not_resolve_is_refused_at_once(resolver, guarded_webhooks):
    with create_user(guarded_webhooks).client() as client:
        created, seconds = timed(
            client.post, TARGETS, json={"config": {"url": f"http://{UNRESOLVABLE}/hook"}}
        )

    assert (created.status_code, created.json()) == (400, {"detail": INVALID_URL})
    assert seconds < FAILS_WITHIN, f"the refusal took {seconds:.1f}s"
