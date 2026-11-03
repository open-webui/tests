"""Regression: an SSO provider saved in the admin panel never showed up on the sign-in page.

open-webui/open-webui#26917. With `ENABLE_OAUTH_PERSISTENT_CONFIG` on, the admin panel stores
the OAuth / OIDC settings, but the sign-in providers were built once at startup from the
environment alone. A provider saved there never got a sign-in button, not even after a restart,
and its login link answered 404. Since bab66b2c2 the providers are built from the stored settings
when the server starts (the admin panel says a restart applies them), and a save with a value no
provider can be built with is refused before it is stored.

Discriminates: passes on dev d22c4449c, fails on bab66b2c2^ (the sign-in page lists no provider
for the saved settings after a restart, the login link answers 404 and an unusable value is
stored with a 200).
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from harness.oidc_provider import (
    OAUTH_CONFIG_PATH,
    browser_for,
    oauth_settings,
    shared_provider,
    sso_env,
)

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

PERSISTENT = {"ENABLE_OAUTH_PERSISTENT_CONFIG": "true"}
IMPORT_CONFIG = "/api/v1/configs/import"
GITHUB_AUTHORIZE = "https://github.com/login/oauth/authorize"


@pytest.fixture
def idp():
    return shared_provider()


@pytest.fixture
def persistent(instance_with):
    return instance_with(PERSISTENT)


def _sign_in_providers(instance) -> dict:
    answer = httpx.get(f"{instance.base_url}/api/config", timeout=30.0)
    assert answer.status_code == 200, f"/api/config answered {answer.status_code}"
    return answer.json()["oauth"]["providers"]


def _login_redirect(instance, provider: str) -> httpx.Response:
    with browser_for(instance) as browser:
        return browser.get(f"/oauth/{provider}/login")


def _saved_provider(idp, name: str) -> dict:
    return {
        "OAUTH_CLIENT_ID": idp.client_id,
        "OAUTH_CLIENT_SECRET": idp.client_secret,
        "OPENID_PROVIDER_URL": idp.discovery_url,
        "OAUTH_PROVIDER_NAME": name,
    }


def _import_settings(instance, settings: dict) -> None:
    with instance.client() as admin:
        imported = admin.post(IMPORT_CONFIG, json={"config": settings})
    assert imported.status_code == 200, f"importing settings failed: {imported.text}"


def _assert_signs_in_through(instance, idp, name: str) -> None:
    providers = _sign_in_providers(instance)
    assert providers.get("oidc") == name, (
        f"the sign-in page offers {providers}, not the provider {name!r}"
    )
    login = _login_redirect(instance, "oidc")
    assert login.status_code == 302, (
        f"the SSO login link answered {login.status_code}: {login.text[:300]}"
    )
    assert login.headers["location"].startswith(f"{idp.base_url}/authorize"), (
        f"the SSO login sent the browser to {login.headers['location']}, not the provider"
    )


def test_a_provider_saved_in_the_admin_panel_gets_a_sign_in_button_after_a_restart(persistent, idp):
    """Narrow: the stored settings build the provider at startup, and clearing them removes it."""
    name = f"Panel SSO {uuid.uuid4().hex[:8]}"
    try:
        with oauth_settings(persistent, **_saved_provider(idp, name)):
            persistent.restart()
            _assert_signs_in_through(persistent, idp, name)
    finally:
        persistent.restart()

    assert "oidc" not in _sign_in_providers(persistent), (
        "the cleared provider still has a sign-in button after a restart"
    )
    assert _login_redirect(persistent, "oidc").status_code == 404


@pytest.mark.parametrize(
    ("field", "unusable"), [("OAUTH_CODE_CHALLENGE_METHOD", "plain"), ("OAUTH_TIMEOUT", "soon")]
)
def test_a_value_the_provider_cannot_use_is_rejected_on_save(persistent, idp, field, unusable):
    """Narrow: the save is refused before it is stored, and the provider works after a restart."""
    name = f"Panel SSO {uuid.uuid4().hex[:8]}"
    try:
        with oauth_settings(persistent, **_saved_provider(idp, name)):
            with persistent.client() as admin:
                refused = admin.post(OAUTH_CONFIG_PATH, json={field: unusable})
                stored = admin.get(OAUTH_CONFIG_PATH).json()[field]
            assert refused.status_code == 400, (
                f"saving {field}={unusable!r} answered {refused.status_code}"
            )
            assert stored != unusable, f"the refused {field} was stored anyway"
            persistent.restart()
            _assert_signs_in_through(persistent, idp, name)
    finally:
        persistent.restart()


def test_stored_settings_decide_every_provider_after_a_restart(persistent):
    """Broad: a provider the admin panel has no fields for follows the stored settings too."""
    github = {
        "oauth.github.client_id": f"gh-{uuid.uuid4().hex[:8]}",
        "oauth.github.client_secret": "gh-secret",
    }
    _import_settings(persistent, github)
    try:
        persistent.restart()
        assert "github" in _sign_in_providers(persistent), (
            "the stored GitHub settings got no sign-in button"
        )
        login = _login_redirect(persistent, "github")
        assert login.status_code == 302, f"the GitHub login link answered {login.status_code}"
        location = login.headers["location"]
        assert location.startswith(GITHUB_AUTHORIZE), (
            f"the GitHub login sent the browser to {location}"
        )
        assert f"client_id={github['oauth.github.client_id']}" in location, (
            f"wrong GitHub client in {location}"
        )
    finally:
        _import_settings(persistent, dict.fromkeys(github, ""))
        persistent.restart()


@pytest.mark.parametrize("persistence", ["true", "false"])
def test_a_provider_from_the_environment_still_gets_a_sign_in_button(
    instance_with, idp, persistence
):
    """Nearby: a provider set only in the environment keeps working with persistence on or off."""
    instance = instance_with({**sso_env(idp), "ENABLE_OAUTH_PERSISTENT_CONFIG": persistence})
    with instance.client() as admin:
        name = admin.get(OAUTH_CONFIG_PATH).json()["OAUTH_PROVIDER_NAME"]
    _assert_signs_in_through(instance, idp, name)
