"""Regression tests for three 0.11.4 connection fixes.

* "Custom header encoding" (`7a4a4b93d`): the custom headers a connection sends
  were substituted as raw text, so a person's name or group carrying anything
  beyond plain ASCII, a line break included, broke the request or reached the
  other end as something else. `parse_custom_headers` now percent-encodes the
  substituted value, preserving ASCII punctuation and existing escapes.
* "Authentication type header" (`ee46e2664`): a request to an OpenAI or Ollama
  connection now carries X-OpenWebUI-Auth-Type, saying whether the person behind
  it signed in through the browser or called with an API key. The name is set
  with FORWARD_USER_INFO_HEADER_AUTH_TYPE and `{{AUTH_TYPE}}` works in custom
  headers.
* "Dotless host addresses" (`6f55514c5`, PR #29945, issue #28161): with local
  web fetching turned on, an address pointing at a container name on the same
  network, "http://apprise:8000" and the like, is now accepted rather than
  refused as invalid (validators.simple_host).

The 0.11.4 connection role check (PR #29619) is covered separately under
unit/security/test_connection_listing_roles.py.

Discriminates: passes on dev 344ea5306; on the pre-fix refs a non-ASCII header
value passes through raw, no auth-type header is attached, and a dotless host is
refused even with local fetch enabled.
"""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.regression


@pytest.fixture(scope="session")
def headers_module(owui_module):
    return owui_module("open_webui.utils.headers")


@pytest.fixture(scope="session")
def user():
    return SimpleNamespace(id="u1", role="user", name="Zoë Müller", email="zoe@example.com")


def _request(auth_type=None):
    state = SimpleNamespace(auth_type=auth_type) if auth_type else None
    return SimpleNamespace(state=state, headers={"user-agent": "ua"})


# ── narrow: substituted header values are encoded ─────────────────────────


def test_non_ascii_name_is_encoded(headers_module, user):
    parsed = headers_module.parse_custom_headers(
        {"X-Owner": "{{USER_NAME}}"}, user=user, request=_request()
    )

    assert "%C3%AB" in parsed["X-Owner"] and "%C3%BC" in parsed["X-Owner"], (
        "a person's name carrying non-ASCII characters passed into the header raw, "
        "breaking the request or reaching the other end rewritten (7a4a4b93d)"
    )


def test_line_break_in_a_value_is_encoded(headers_module, user):
    groups = [SimpleNamespace(id="g1", name="Team\nInjected: true")]
    parsed = headers_module.parse_custom_headers(
        {"X-Groups": "{{USER_GROUPS}}"}, user=user, request=_request(), user_groups=groups
    )

    assert "\n" not in parsed["X-Groups"], (
        "a line break in a substituted header value reached the request raw, letting "
        "a group name split or smuggle headers (7a4a4b93d)"
    )


def test_ascii_punctuation_stays_readable(headers_module, user):
    parsed = headers_module.parse_custom_headers(
        {"X-Tag": "team-a/b (core)"}, user=user, request=_request()
    )

    assert parsed["X-Tag"] == "team-a/b (core)", (
        "ordinary ASCII punctuation was percent-encoded, making headers unreadable "
        "for no reason (7a4a4b93d)"
    )


def test_existing_percent_escapes_survive(headers_module, user):
    parsed = headers_module.parse_custom_headers(
        {"X-Path": "files/a%20b.txt"}, user=user, request=_request()
    )

    assert parsed["X-Path"] == "files/a%20b.txt", (
        "an already-escaped value was double-encoded, corrupting paths the caller "
        "had encoded themselves (7a4a4b93d)"
    )


# ── narrow: the auth-type header and the {{AUTH_TYPE}} placeholder ────────


def test_browser_signins_carry_the_auth_type_header(headers_module, user):
    parsed = headers_module.include_user_info_headers({}, user, request=_request("jwt"))

    assert parsed.get("X-OpenWebUI-Auth-Type") == "jwt", (
        "a request signed in through the browser carries no auth-type header, so the "
        "upstream cannot tell browser sessions from API keys (ee46e2664)"
    )


def test_api_keys_carry_their_auth_type(headers_module, user):
    parsed = headers_module.include_user_info_headers({}, user, request=_request("api_key"))

    assert parsed.get("X-OpenWebUI-Auth-Type") == "api_key", (
        "an API-key request carries no auth-type header (ee46e2664)"
    )


def test_auth_type_placeholder_substitutes_in_custom_headers(headers_module, user):
    parsed = headers_module.parse_custom_headers(
        {"X-Source": "{{AUTH_TYPE}}"}, user=user, request=_request("api_key")
    )

    assert parsed["X-Source"] == "api_key", (
        "the {{AUTH_TYPE}} placeholder does not resolve in custom headers "
        "(ee46e2664)"
    )


def test_requests_without_an_auth_type_send_nothing(headers_module, user):
    parsed = headers_module.include_user_info_headers({}, user, request=_request())

    assert "X-OpenWebUI-Auth-Type" not in parsed, (
        "requests with no resolved auth type still attach the header"
    )


def test_the_header_name_is_configurable(headers_module):
    from open_webui.env import FORWARD_USER_INFO_HEADER_AUTH_TYPE

    assert isinstance(FORWARD_USER_INFO_HEADER_AUTH_TYPE, str) and (
        FORWARD_USER_INFO_HEADER_AUTH_TYPE
    ), "FORWARD_USER_INFO_HEADER_AUTH_TYPE is not configurable"


# ── narrow: dotless hosts validate when local fetch is on ─────────────────


@pytest.fixture(scope="session")
def retrieval_web_utils(owui_module):
    return owui_module("open_webui.retrieval.web.utils")


def test_dotless_container_host_is_accepted_for_local_fetch(retrieval_web_utils, monkeypatch):
    monkeypatch.setattr(retrieval_web_utils, "ENABLE_LOCAL_WEB_FETCH", True)

    assert retrieval_web_utils.validate_url("http://apprise:8000/notify") is True, (
        "a container-name address on the local network is refused even with local "
        "web fetch enabled, breaking container-to-container fetches (#28161)"
    )


def test_dotless_host_still_rejected_without_local_fetch(retrieval_web_utils, monkeypatch):
    monkeypatch.setattr(retrieval_web_utils, "ENABLE_LOCAL_WEB_FETCH", False)

    with pytest.raises(ValueError, match="invalid"):
        retrieval_web_utils.validate_url("http://apprise:8000/notify")
