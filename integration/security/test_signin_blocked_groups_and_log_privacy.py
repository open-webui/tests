"""Regression: a failed SSO callback logged the live tokens, and blocked groups typed as
comma-separated text blocked nothing.

open-webui 0.11.4 fixes `39c1e86e9` (#29709) and `3fc1146c1`:

- #29709: when the provider returned tokens but no user data, the callback's warning
  interpolated the whole token response, so the access, refresh and ID tokens landed in the
  application log. It now names the provider instead.
- `3fc1146c1`: group sync parsed `OAUTH_BLOCKED_GROUPS` with `JSONCodec.loads`, so comma
  separated admin text, and a list stored by a config import, read as an error and blocked
  nothing; the admin form also returned a stored list raw, which its own response model
  rejects. The check now accepts JSON text, a list or comma-separated text, and the form
  reads a list back as JSON text.

Twin of unit/security/test_signin_blocked_groups_and_log_privacy.py.

Discriminates: passes on dev bbfa876af; with 39c1e86e9 reverted in a copy the server log
carries the tokens, and with 3fc1146c1 reverted the person joins the blocked group from comma
text or an imported list and the admin OAuth settings answer 500.
"""

from __future__ import annotations

import json
import secrets

import pytest

from harness.oidc_provider import (
    OAUTH_CONFIG_PATH,
    group_member_ids,
    group_named,
    oauth_settings,
    session_user,
    shared_provider,
    sign_in,
    sso_env,
)

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]


@pytest.fixture
def idp():
    return shared_provider()


@pytest.fixture
def sso(instance_with, idp):
    return instance_with(sso_env(idp))


def save_blocked_groups(sso, how: str, groups: list[str]) -> None:
    """Store the blocked groups the way an admin can: form text, JSON text or a config import."""
    if how == "config import":
        path, body = "/api/v1/configs/import", {"config": {"oauth.blocked_groups": groups}}
    elif how == "JSON text":
        path, body = OAUTH_CONFIG_PATH, {"OAUTH_BLOCKED_GROUPS": json.dumps(groups)}
    else:
        path, body = OAUTH_CONFIG_PATH, {"OAUTH_BLOCKED_GROUPS": ", ".join(groups)}
    with sso.client() as admin:
        saved = admin.post(path, json=body)
    assert saved.status_code == 200, saved.text


# ── tokens stay out of the log (39c1e86e9, #29709) ─────────────────────────


def test_a_callback_without_user_data_keeps_the_tokens_out_of_the_log(sso, idp):
    """Narrow: the provider answered with tokens but an empty userinfo; none may be logged."""
    sub = f"no-profile-{secrets.token_hex(4)}"
    idp.sign_in_as(sub=sub, id_token_claims={"sub": sub}, userinfo={})
    log_offset = sso.log_size()

    result = sign_in(sso)

    assert result.token is None and result.error, "a sign-in without user data went through"
    tokens = idp.issued[-1]
    server_log = sso.log_since(log_offset)
    for name in ("access_token", "refresh_token", "id_token"):
        assert tokens[name] not in server_log, f"the server log leaked the {name}"


# ── blocked groups (3fc1146c1) ─────────────────────────────────────────────


@pytest.mark.parametrize("how", ["comma-separated text", "JSON text", "config import"])
def test_a_blocked_group_is_never_joined(sso, idp, how):
    """Narrow: however the blocked list was saved, group sync never adds anyone to it."""
    tag = secrets.token_hex(4)
    with (
        group_named(sso, f"blocked-{tag}") as blocked,
        group_named(sso, f"team-{tag}") as open_group,
        oauth_settings(sso, ENABLE_OAUTH_GROUP_MANAGEMENT=True),
    ):
        save_blocked_groups(sso, how, [blocked["name"], f"other-{tag}"])
        idp.sign_in_as(groups=[blocked["name"], open_group["name"]])
        result = sign_in(sso)
        assert result.token, f"the sign-in failed: {result.error}"
        account = session_user(sso, result.token)

        assert account["id"] in group_member_ids(sso, open_group)
        assert account["id"] not in group_member_ids(sso, blocked), "joined a blocked group"


@pytest.mark.parametrize(
    ("how", "junk"),
    [("comma-separated text", ["", " "]), ("config import", ["", None, 7])],
    ids=["blank-text", "imported-junk"],
)
def test_junk_in_the_blocked_list_blocks_nobody_and_breaks_nothing(sso, idp, how, junk):
    """Nearby: blank or non-text entries are dropped instead of blocking or failing the sync."""
    with (
        group_named(sso, f"team-{secrets.token_hex(4)}") as open_group,
        oauth_settings(sso, ENABLE_OAUTH_GROUP_MANAGEMENT=True),
    ):
        save_blocked_groups(sso, how, junk)
        idp.sign_in_as(groups=[open_group["name"]])
        result = sign_in(sso)
        assert result.token, f"the sign-in failed: {result.error}"
        account = session_user(sso, result.token)
        assert account["id"] in group_member_ids(sso, open_group)


def test_a_stored_blocked_groups_list_reads_back_with_its_commas(sso):
    """Narrow: the admin form reads a stored list as JSON text, so a comma in a name survives."""
    stored = ["team, with comma", "^vip-.*$"]
    with oauth_settings(sso, OAUTH_ADMIN_ROLES="admin,owner"):
        save_blocked_groups(sso, "config import", stored)
        with sso.client() as admin:
            form = admin.get(OAUTH_CONFIG_PATH)

    assert form.status_code == 200, form.text
    assert json.loads(form.json()["OAUTH_BLOCKED_GROUPS"]) == stored
    assert form.json()["OAUTH_ADMIN_ROLES"] == "admin,owner"
