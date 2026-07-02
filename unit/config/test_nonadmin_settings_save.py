"""Regression for open-webui/open-webui#26627.

Non-admin users could not save their interface settings: PATCH
`/api/v1/users/user/settings/update` returned a 500 while the UI wrongly
reported success. Admins were unaffected because the failing branch is
guarded by `user.role != 'admin'`.

Mechanism. In `update_user_settings_by_session_user`
(backend/open_webui/routers/users.py) the non-admin `settings.interface`
permission check fed `has_permission(...)` its `default_permissions` from
`request.app.state.config.USER_PERMISSIONS`. That attribute is not populated
on `app.state.config` in this codebase, so the access raised at request time
and the handler 500'd before the settings were ever written. `has_permission`
expects that argument to be the permissions *dict* (it does
`fill_missing_permissions(default_permissions, DEFAULT_USER_PERMISSIONS)`).

Fix (commit 9866a02863, 0.10.2): read the permissions from the live config
store via `await Config.get('user.permissions')` instead, which returns the
usable dict. The sibling `features.direct_tool_servers` check in the same
function already used `Config.get('user.permissions')` in 0.10.1 — only the
`settings.interface` check was still reading the dead app.state attribute, so
that specific pairing is the load-bearing signal here.

Source audit rather than behavioral: reproducing the 500 needs a running app
with a populated `app.state`, a DB, an authenticated non-admin session and the
FastAPI dependency graph. The bug is a single wrong permission source on one
line, and the fixed/buggy reads are textually distinct and unique, so a scoped
read of the function body discriminates cleanly without that machinery.

Discriminates: passes on dev/0.10.2 (`Config.get('user.permissions')`), fails
on 0.10.1 (`request.app.state.config.USER_PERMISSIONS`).
"""

from __future__ import annotations

import re

import pytest

pytestmark = pytest.mark.regression

FUNC = "update_user_settings_by_session_user"
BUGGY_READ = "app.state.config.USER_PERMISSIONS"


def _function_body(open_webui_backend) -> str:
    """Return the source of update_user_settings_by_session_user only.

    Sliced from its `async def` to the next top-level `async def`/`def`/
    `@router` so the audit is confined to the function the fix touched and
    can't be fooled by an identical read elsewhere in the module.
    """
    src = (open_webui_backend / "open_webui" / "routers" / "users.py").read_text(encoding="utf-8")
    start = re.search(rf"^async def {re.escape(FUNC)}\b", src, re.MULTILINE)
    assert start is not None, f"{FUNC} not found in routers/users.py — source moved?"

    rest = src[start.end() :]
    nxt = re.search(r"^(?:@router\b|async def |def )", rest, re.MULTILINE)
    return rest[: nxt.start()] if nxt else rest


def test_interface_check_reads_permissions_from_config_store(open_webui_backend):
    """The fixed behavior: the non-admin `settings.interface` gate must source
    its permissions from `Config.get('user.permissions')` (the live persisted
    dict), which is what makes the check work for non-admins."""
    body = _function_body(open_webui_backend)
    interface_check = [
        ln.strip()
        for ln in body.splitlines()
        if "settings.interface" in ln and "Config.get('user.permissions')" in ln
    ]
    assert interface_check, (
        "the settings.interface has_permission call no longer passes "
        "Config.get('user.permissions'); non-admin interface-settings saves "
        "regress to the #26627 500 if the permissions dict isn't sourced here"
    )


def test_settings_save_does_not_read_dead_app_state_permissions(open_webui_backend):
    """The bug mechanism: the function must not read
    `request.app.state.config.USER_PERMISSIONS`. That attribute isn't populated
    on app.state.config, so this access is what 500'd the non-admin save."""
    body = _function_body(open_webui_backend)
    offenders = [ln.strip() for ln in body.splitlines() if BUGGY_READ in ln]
    assert not offenders, (
        f"{FUNC} reads permissions from {BUGGY_READ!r}, the unpopulated "
        f"app.state attribute that raised and 500'd non-admin interface-settings "
        f"saves in #26627: {offenders}"
    )
