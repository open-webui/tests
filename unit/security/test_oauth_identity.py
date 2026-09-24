"""Regression: OAuth identity handling that the HTTP twin cannot reach.

integration/security/test_oauth_identity.py drives the sign-in path for these fixes. What stays
here needs rows written by older releases, or a provider whose keys only its own client reaches:

* `a6834f089` (#28954, issue #27760): releases before it stored a numeric sub as a JSON number.
  The lookup must still find such a row from the string sub and coerce an int it is handed, and
  linking a sub merges into the provider entry and stores it as text.
* `73c1f5806` (#28624): a provider entry carrying keys besides `sub` must still match; the old
  `contains()` compared the whole serialised entry.
* `aeda6ff13`: back-channel logout fetched the provider's keys with an anonymous `PyJWKClient`,
  so a provider whose key endpoint only its configured client can reach never signed anyone
  out. The keys now come through the provider's authlib client.

Discriminates: passes on dev bbfa876af; in a copy with a6834f089 reverted the JSON-number row,
the int lookup and the merge fail, with 73c1f5806 reverted the entry with extra keys is missed,
and with the key fetch put back on `PyJWKClient` the logout answers 400.
"""

from __future__ import annotations

import time
import urllib.parse
import uuid
from unittest.mock import create_autospec, patch

import jwt
import pytest
from authlib.integrations.starlette_client import StarletteOAuth2App
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI
from jwt.algorithms import RSAAlgorithm
from starlette.requests import Request

from unit.security.memory_db import memory_database

pytestmark = pytest.mark.regression

ISSUER = "https://idp.example"
CLIENT_ID = "owui-client"


@pytest.fixture
def users(owui_module):
    return owui_module("open_webui.models.users")


async def add_account(users, oauth: dict) -> str:
    account_id = str(uuid.uuid4())
    await users.Users.insert_new_user(
        id=account_id, name=account_id, email=f"{account_id}@example.com", oauth=oauth
    )
    return account_id


# --------------------------------------------------------------------------- legacy rows


@pytest.mark.parametrize(
    ("stored_sub", "looked_up"), [(777, "777"), ("888", 888)], ids=["json-number", "int-lookup"]
)
@pytest.mark.asyncio
async def test_a_numeric_sub_matches_across_number_and_text(
    owui_module, users, stored_sub, looked_up
):
    """Narrow (#28954): an older release's JSON-number sub, or an int argument, still matches."""
    async with memory_database(owui_module, users.User):
        account_id = await add_account(users, {"github": {"sub": stored_sub}})
        found = await users.Users.get_user_by_oauth_sub(provider="github", sub=looked_up)
    assert found is not None and found.id == account_id


@pytest.mark.asyncio
async def test_a_provider_entry_with_extra_keys_still_matches(owui_module, users):
    """Narrow (#28624): the match is on the `sub` value, not on the serialised entry."""
    async with memory_database(owui_module, users.User):
        account_id = await add_account(users, {"oidc": {"sub": "12345", "email": "b@example.com"}})
        found = await users.Users.get_user_by_oauth_sub(provider="oidc", sub="12345")
    assert found is not None and found.id == account_id


@pytest.mark.asyncio
async def test_linking_a_sub_keeps_the_entrys_other_keys_and_stores_text(owui_module, users):
    """Narrow (#28954): linking merges into the provider entry instead of replacing it."""
    async with memory_database(owui_module, users.User):
        account_id = await add_account(users, {"oidc": {"sub": "12345", "email": "b@example.com"}})
        linked = await users.Users.update_user_oauth_by_id(id=account_id, provider="oidc", sub=999)
    assert linked.oauth["oidc"] == {"sub": "999", "email": "b@example.com"}


# --------------------------------------------------------------------------- back-channel logout


def logout_request(logout_token: str) -> Request:
    body = urllib.parse.urlencode({"logout_token": logout_token}).encode()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    headers = [(b"content-type", b"application/x-www-form-urlencoded")]
    scope = {"type": "http", "method": "POST", "path": "/oauth/backchannel-logout"}
    return Request({**scope, "headers": headers, "query_string": b""}, receive)


@pytest.mark.asyncio
async def test_back_channel_logout_reads_the_keys_through_the_providers_client(owui_module, users):
    """Narrow (aeda6ff13): a valid logout token is accepted although only the client has keys."""
    oauth = owui_module("open_webui.utils.oauth")
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = RSAAlgorithm.to_jwk(key.public_key(), as_dict=True)
    client = create_autospec(StarletteOAuth2App, instance=True)
    client.client_id = CLIENT_ID
    # nothing listens on the discovered jwks_uri; the keys are only reachable through the client
    client.load_server_metadata.return_value = {"issuer": ISSUER, "jwks_uri": "http://127.0.0.1:9/"}
    client.fetch_jwk_set.return_value = {"keys": [{**public_key, "kid": "k1", "use": "sig"}]}
    claims = {
        "iss": ISSUER,
        "aud": CLIENT_ID,
        "iat": int(time.time()),
        "sub": "someone-not-signed-in-here",
        "events": {"http://schemas.openid.net/event/backchannel-logout": {}},
    }
    logout_token = jwt.encode(claims, key, algorithm="RS256", headers={"kid": "k1"})

    providers = {"oidc": {"register": lambda registry: client}}
    async with memory_database(owui_module, users.User):
        with patch.dict(oauth.OAUTH_PROVIDERS, providers, clear=True):
            manager = oauth.OAuthManager(app=FastAPI())
            answer = await manager.handle_backchannel_logout(logout_request(logout_token), db=None)

    assert answer.status_code == 200, answer.body
