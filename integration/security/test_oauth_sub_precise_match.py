"""Regression: a numeric sub past the signed 64 bit range could not sign in at all.

`17cc56670` (open-webui 0.11.3). The SQLite sub lookup widened the query with `int(sub)` for any
sub passing `str.isdecimal()`, so a sub above 2**63 - 1 was handed to the SQLite driver as an
integer it cannot bind, and the sign-in errored. The fix widens only when the value fits in a
signed 64 bit integer and `str(int(sub)) == sub`. The zero padded and non-ASCII cases need
accounts stored by older releases, so they stay in the unit twin.

Twin of unit/security/test_oauth_sub_precise_match.py.

Discriminates: passes on dev bbfa876af; with 17cc56670 reverted in a copy the sub beyond the
64 bit range fails to sign in (OverflowError in the lookup).
"""

from __future__ import annotations

import secrets

import pytest

from harness.oidc_provider import session_user, shared_provider, sign_in, sso_env

pytestmark = [
    pytest.mark.regression,
    pytest.mark.api,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

INT64_MAX = 2**63 - 1


@pytest.fixture
def idp():
    return shared_provider()


@pytest.fixture
def sso(instance_with, idp):
    return instance_with(sso_env(idp))


@pytest.mark.parametrize(
    "sub",
    [str(INT64_MAX + 1 + secrets.randbelow(10**6)), str(INT64_MAX)],
    ids=["beyond-int64", "int64-boundary"],
)
def test_a_large_numeric_sub_signs_in_to_the_same_account_twice(sso, idp, sub):
    """Narrow (beyond-int64) and nearby (the boundary itself): both keep one account."""
    idp.sign_in_as(sub=sub)
    accounts = []
    for _ in range(2):
        result = sign_in(sso)
        assert result.token, f"the sign-in failed: {result.error}"
        accounts.append(session_user(sso, result.token)["id"])
    assert accounts[0] == accounts[1]
