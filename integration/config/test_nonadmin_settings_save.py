"""Regression: a regular user's settings save failed while the UI reported success.

open-webui 0.10.2, fix `9866a02863` (issue #26627): the non-admin `settings.interface` check in
`update_user_settings_by_session_user` read its default permissions from
`request.app.state.config.USER_PERMISSIONS`, which is not populated, so every non-admin save that
carried interface settings answered 500 and nothing was stored. Admins skip that check and were
unaffected. The fix reads `Config.get('user.permissions')`.

Twin of unit/config/test_nonadmin_settings_save.py.

Discriminates: passes on bbfa876af, fails with the check reading
`request.app.state.config.USER_PERMISSIONS` again (the user's save answers 500 and nothing reads
back); the admin test passes on both.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

SETTINGS = "/api/v1/users/user/settings"


def _save_and_reload(actor, ui: dict):
    with actor.client() as client:
        saved = client.post(f"{SETTINGS}/update", json={"ui": ui})
        stored = client.get(SETTINGS)
    return saved, stored.json()


def test_a_regular_users_interface_settings_are_saved(make_user):
    saved, stored = _save_and_reload(make_user(), {"widescreenMode": True, "chatBubble": False})

    assert saved.status_code == 200, (
        f"a regular user's settings save failed (#26627): HTTP {saved.status_code} {saved.text}"
    )
    assert stored["ui"]["widescreenMode"] is True
    assert stored["ui"]["chatBubble"] is False


def test_an_admins_interface_settings_are_saved(make_user):
    saved, stored = _save_and_reload(make_user(role="admin"), {"widescreenMode": True})

    assert saved.status_code == 200, saved.text
    assert stored["ui"]["widescreenMode"] is True
