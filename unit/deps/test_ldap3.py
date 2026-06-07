r"""Dependency contract: ldap3 (import name ``ldap3``).

ldap3 implements Open WebUI's LDAP authentication
(``routers/auths.py``). This is SECURITY-CRITICAL code: it binds to a
directory with admin credentials, searches for the supplied username, then
re-binds as that user to verify the password. The exact surface:

  - ``from ldap3 import NONE, Connection, Server, Tls``;
    ``from ldap3.utils.conv import escape_filter_chars``.
  - ``Tls(validate=CERT_REQUIRED|CERT_NONE, version=PROTOCOL_TLS, ...)``.
  - ``Server(host, port=, use_ssl=, get_info=NONE, tls=tls)``.
  - ``Connection(server, user=, password=, auto_bind="NONE")`` then
    ``connection.bind()`` and ``connection.search(base, search_filter=...,
    attributes=...)``.
  - the search filter is built as
    ``f"(&({attr}={escape_filter_chars(username.lower())}){extra})"`` —
    ``escape_filter_chars`` is the ONLY thing standing between a
    user-supplied username and an LDAP-injection / filter-bypass.

This module pins that surface and, above all, the *behavioural* contract of
``escape_filter_chars`` — that it neutralises every LDAP filter
metacharacter (``*``, ``(``, ``)``, ``\``, NUL) so a crafted username cannot
break out of the filter. It also constructs ``Server`` / ``Connection`` /
``Tls`` objects OFFLINE (these do NOT open a socket until ``.bind()``), and
exercises a real bind+search end to end against ldap3's in-memory
``MOCK_SYNC`` strategy — NO network, NO directory server.

A ldap3 bump that renamed a symbol, changed ``escape_filter_chars`` so it
stopped escaping a metacharacter, or altered the bind/search contract would
fail here instead of silently weakening LDAP auth (auth bypass / injection).

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import inspect
import ssl

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "ldap3"
DIST_NAME = "ldap3"

# Top-level symbols auths.py imports.
USED_SYMBOLS = [
    "NONE",
    "Connection",
    "Server",
    "Tls",
    "utils.conv.escape_filter_chars",
]

# The filter metacharacters that MUST be escaped (RFC 4515 + NUL).
# Each maps to its expected \HH escape.
METACHAR_ESCAPES = {
    "*": r"\2a",
    "(": r"\28",
    ")": r"\29",
    "\\": r"\5c",
    "\x00": r"\00",
}


def _escape(depcheck):
    return depcheck.resolve(depcheck.load(IMPORT_NAME), "utils.conv.escape_filter_chars")


# --------------------------------------------------------------------------- #
# Import / version
# --------------------------------------------------------------------------- #


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "ldap3"


def test_version_reported(depcheck):
    assert depcheck.dist_version(DIST_NAME) is not None


# --------------------------------------------------------------------------- #
# Symbol existence (API surface)
# --------------------------------------------------------------------------- #


def test_used_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_classes_are_classes(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert inspect.isclass(mod.Connection)
    assert inspect.isclass(mod.Server)
    assert inspect.isclass(mod.Tls)


def test_escape_filter_chars_callable(depcheck):
    depcheck.assert_callable(depcheck.load(IMPORT_NAME), "utils.conv.escape_filter_chars")


def test_none_constant_value(depcheck):
    """auths.py passes get_info=NONE to Server. NONE is the "no info" sentinel
    (string 'NO_INFO' in ldap3); pin it is a defined, truthy-name constant."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.NONE is not None


# --------------------------------------------------------------------------- #
# Constructor signatures (the kwargs auths.py passes)
# --------------------------------------------------------------------------- #


def test_server_signature(depcheck):
    """Server(host, port=, use_ssl=, get_info=, tls=). Pin those param names."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.Server.__init__)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    assert params and params[0].name == "host"
    depcheck.assert_params(mod.Server.__init__, ["host", "port", "use_ssl", "get_info", "tls"])


def test_connection_signature(depcheck):
    """Connection(server, user=, password=, auto_bind=). Pin those param
    names."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.Connection.__init__)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    assert params and params[0].name == "server"
    depcheck.assert_params(mod.Connection.__init__, ["server", "user", "password", "auto_bind"])


def test_tls_signature(depcheck):
    """Tls(validate=, version=). Pin those param names auths.py passes."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.Tls.__init__, ["validate", "version"])


def test_connection_has_bind_and_search(depcheck):
    """auths.py calls connection.bind() and connection.search(...)."""
    mod = depcheck.load(IMPORT_NAME)
    names = set(dir(mod.Connection))
    for meth in ("bind", "unbind", "search"):
        assert meth in names, f"Connection.{meth} missing"
        assert callable(getattr(mod.Connection, meth))


def test_search_signature(depcheck):
    """connection.search(search_base, search_filter, attributes=...). Pin the
    search_filter + attributes parameter names the backend passes."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.Connection.search, ["search_base", "search_filter", "attributes"])


# --------------------------------------------------------------------------- #
# SECURITY: escape_filter_chars (LDAP injection defense) — behavioural
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("char,expected", list(METACHAR_ESCAPES.items()))
def test_escape_each_metacharacter(depcheck, char, expected):
    """Every LDAP filter metacharacter must be escaped to its \\HH form. If any
    of these stops being escaped, a crafted username could alter the filter
    (auth bypass / injection)."""
    escape = _escape(depcheck)
    result = escape(char)
    assert result.lower() == expected.lower(), (
        f"escape_filter_chars({char!r}) = {result!r}, expected {expected!r} — "
        f"LDAP filter metacharacter no longer escaped; auth filter is injectable"
    )


def test_escape_neutralises_wildcard_injection(depcheck):
    """A username of '*' must not survive as a wildcard — otherwise it would
    match arbitrary directory entries."""
    escape = _escape(depcheck)
    result = escape("*")
    assert "*" not in result
    assert result.lower() == r"\2a"


def test_escape_neutralises_filter_breakout(depcheck):
    """The classic LDAP-injection payload `admin*)(uid=*` must be fully
    escaped so it cannot close the current filter clause and open a new one.
    Reproduce the exact construction auths.py does."""
    escape = _escape(depcheck)
    payload = "admin*)(uid=*"
    escaped = escape(payload)
    # None of the structural metacharacters may remain literal.
    for ch in ("*", "(", ")"):
        assert ch not in escaped, f"{ch!r} survived escaping of {payload!r}"
    # And the assembled filter contains only the escaped form.
    attr = "uid"
    extra = ""
    search_filter = f"(&({attr}={escaped}){extra})"
    # The only unescaped parens/star are the two structural ones we added.
    assert search_filter.count("(") == 2
    assert search_filter.count(")") == 2
    assert "*" not in search_filter


def test_escape_leaves_normal_username_untouched(depcheck):
    """A benign username has no metacharacters and must pass through unchanged
    (so legitimate logins still match)."""
    escape = _escape(depcheck)
    assert escape("normaluser") == "normaluser"
    assert escape("john.doe") == "john.doe"


def test_escape_returns_str(depcheck):
    escape = _escape(depcheck)
    assert isinstance(escape("anything"), str)


# --------------------------------------------------------------------------- #
# Behavioural: offline object construction (no socket until bind)
# --------------------------------------------------------------------------- #


def test_tls_constructs_offline(depcheck):
    """Tls(validate=, version=) — the exact auths.py construction — builds
    without any network."""
    mod = depcheck.load(IMPORT_NAME)
    tls = mod.Tls(validate=ssl.CERT_NONE, version=ssl.PROTOCOL_TLS)
    assert tls is not None


def test_server_constructs_offline(depcheck):
    """Server(...) does NOT connect at construction; it only records config."""
    mod = depcheck.load(IMPORT_NAME)
    tls = mod.Tls(validate=ssl.CERT_NONE, version=ssl.PROTOCOL_TLS)
    server = mod.Server(
        "ldap.example.com",
        port=389,
        use_ssl=False,
        get_info=mod.NONE,
        tls=tls,
    )
    assert server.host == "ldap.example.com"
    assert server.port == 389


def test_connection_constructs_unbound(depcheck):
    """Connection(server, user=, password=, auto_bind='NONE') builds without
    binding — .bound is False until .bind() is called."""
    mod = depcheck.load(IMPORT_NAME)
    server = mod.Server("ldap.example.com", port=389, get_info=mod.NONE)
    conn = mod.Connection(
        server,
        user="cn=admin,dc=example,dc=com",
        password="secret",
        auto_bind="NONE",
    )
    assert conn.bound is False


# --------------------------------------------------------------------------- #
# Behavioural: full bind + search via the in-memory MOCK strategy (offline)
# --------------------------------------------------------------------------- #


def test_mock_bind_and_search_roundtrip(depcheck):
    """Exercise the auths.py flow end to end against ldap3's MOCK_SYNC
    strategy: bind, then search for a user with an escaped filter, and read
    the returned DN. Fully in-memory — no directory server, no network."""
    mod = depcheck.load(IMPORT_NAME)
    if not depcheck.has(mod, "MOCK_SYNC"):
        pytest.skip("ldap3.MOCK_SYNC not available in this version")
    escape = _escape(depcheck)

    server = mod.Server("fake-directory")
    conn = mod.Connection(
        server,
        user="cn=admin,dc=ex,dc=com",
        password="adminpw",
        client_strategy=mod.MOCK_SYNC,
    )
    # Seed the in-memory directory.
    conn.strategy.add_entry("cn=admin,dc=ex,dc=com", {"userPassword": "adminpw", "sn": "admin"})
    conn.strategy.add_entry(
        "uid=alice,dc=ex,dc=com",
        {"userPassword": "secret", "uid": "alice", "sn": "Alice"},
    )

    # Admin bind (mirrors connection_app.bind()).
    assert conn.bind() is True

    # Build the filter exactly like auths.py and search.
    username = "alice"
    search_filter = f"(uid={escape(username)})"
    found = conn.search("dc=ex,dc=com", search_filter, attributes=["uid", "sn"])
    assert found is True
    assert len(conn.response) >= 1
    assert conn.response[0]["dn"] == "uid=alice,dc=ex,dc=com"


def test_mock_search_with_injection_payload_finds_nothing(depcheck):
    """The security pay-off: a username crafted to break the filter, once run
    through escape_filter_chars, matches no entry — proving the escaping
    prevents the wildcard/breakout from returning unintended users."""
    mod = depcheck.load(IMPORT_NAME)
    if not depcheck.has(mod, "MOCK_SYNC"):
        pytest.skip("ldap3.MOCK_SYNC not available in this version")
    escape = _escape(depcheck)

    server = mod.Server("fake-directory")
    conn = mod.Connection(
        server,
        user="cn=admin,dc=ex,dc=com",
        password="adminpw",
        client_strategy=mod.MOCK_SYNC,
    )
    conn.strategy.add_entry("cn=admin,dc=ex,dc=com", {"userPassword": "adminpw", "sn": "admin"})
    conn.strategy.add_entry(
        "uid=alice,dc=ex,dc=com",
        {"userPassword": "secret", "uid": "alice", "sn": "Alice"},
    )
    assert conn.bind() is True

    # Escaped wildcard must NOT behave as a wildcard: literal 'uid=*' (escaped)
    # matches no entry whose uid is the literal string '\\2a'.
    malicious = "*"
    search_filter = f"(uid={escape(malicious)})"
    conn.search("dc=ex,dc=com", search_filter, attributes=["uid"])
    matched_dns = [r["dn"] for r in conn.response if r.get("dn")]
    assert "uid=alice,dc=ex,dc=com" not in matched_dns, (
        "escaped '*' still matched a user — wildcard escaping is broken"
    )
