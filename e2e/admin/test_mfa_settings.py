"""Journey: the admin requires an authenticator for everybody from Authentication settings.

Admin Panel > Settings > Authentication has a "Multi-factor authentication" section. Its main
switch, "Require an authenticator for all users", reveals the two exemptions (SSO and trusted
header sign-in) once it is on. Saving signs every device out, the admin's own browser included,
which lands on the sign-in page; signing in again with the password leads to the authenticator
setup, and the saved settings hold the switches as they were set.

Discriminates: in a backend copy, making `is_mfa_required` always answer False turns the test red
at the setup step (the password alone signs the admin in).
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import expect

from harness.actors import admin_of
from harness.mfa import admin_session
from utils.admin_settings import submit

pytestmark = [
    pytest.mark.journey,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

REQUIRE = "Require an authenticator for all users"
EXEMPT_SSO = "Allow OAuth sign-in without an authenticator"
EXEMPT_PROXY = "Allow trusted-header sign-in without an authenticator"


@pytest.fixture
def switchable(instance_with):
    instance = instance_with({"DEFAULT_USER_ROLE": "user"})
    if not instance.serves_frontend:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    return instance


def test_requiring_an_authenticator_signs_everyone_out_into_setup(switchable, page_for):
    admin = admin_of(switchable)
    page = page_for(admin, path="/admin/settings/authentication")
    settings = page.get_by_role("dialog")
    require = settings.get_by_role("switch", name=REQUIRE)
    expect(require).not_to_be_checked()
    expect(settings.get_by_role("switch", name=EXEMPT_SSO)).to_have_count(0)

    require.click()
    settings.get_by_role("switch", name=EXEMPT_SSO).click()
    expect(settings.get_by_role("switch", name=EXEMPT_PROXY)).not_to_be_checked()
    submit(settings)

    expect(page).to_have_url(re.compile(r"/auth"))
    with switchable.client(admin.token) as client:
        assert client.get("/api/v1/auths/").status_code == 401
    page.get_by_label("Email").fill(admin.email)
    page.get_by_label("Password", exact=True).fill(admin.password)
    page.get_by_role("button", name="Sign in", exact=True).click()
    expect(page.get_by_role("heading", name="Set up your authenticator")).to_be_visible()

    switchable.admin_token = admin_session(switchable, {})
    with switchable.client() as client:
        saved = client.get("/api/v1/auths/admin/config").json()
    assert (saved["ENABLE_MFA"], saved["MFA_ALLOW_OAUTH_BYPASS"]) == (True, True)
    assert saved["MFA_ALLOW_TRUSTED_HEADER_BYPASS"] is False
