"""Host names for a local service, as an operator writes them into a setting.

aiohttp resolves a host name through the resolver `AIOHTTP_CLIENT_ASYNC_DNS_RESOLVER` picks (the
threaded OS resolver by default, c-ares through aiodns when it is on) and connects to an IP literal
without asking either, so a test of that switch reaches its fakes by name. A `NameForm` is one way
of naming a local service: `localhost` answers over both address families, `ipv6-only` is a hosts-
file name for `::1` alone and `hosts-file` a hosts-file name for another local address.
`name_forms()` gives the three as pytest params; the last two skip where the machine's hosts file
has no such name. `serving_by_name(form)` starts a `listener` on the form's address and names it by
the form's host. `UNRESOLVABLE` is a name no resolver answers (RFC 6761 reserves `.invalid`); either
resolver refuses it well inside `FAILS_WITHIN` seconds where the DNS server answers, which
`timed(call)` measures. `resolver_reason(text)` is the resolver's own wording of that failure, the
bracketed tail aiohttp ends a connection error with: c-ares words it differently from the OS
(`C_ARES_REASONS`), which is how a test tells which resolver answered. `c_ares_refusal_problem()`
probes once per session whether this machine's DNS server lets c-ares refuse `UNRESOLVABLE` at once
as the OS does; where it does not, the `refuses_unknown_names` fixture skips a c-ares test of that
name and names the reason.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import ipaddress
import os
import re
import socket
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, TypeVar
from urllib.parse import urlsplit, urlunsplit

import pytest

from harness.listener import Listener, listening

ASYNC_DNS = "AIOHTTP_CLIENT_ASYNC_DNS_RESOLVER"
RESOLVERS = {"threaded": {ASYNC_DNS: "false"}, "c-ares": {ASYNC_DNS: "true"}}
UNRESOLVABLE = "owui-unresolvable.invalid"
FAILS_WITHIN = 10.0  # seconds; one lost DNS reply costs a retry of about 5s, a hang is far longer

# pycares' messages for a name it cannot answer, which the OS resolver never uses
C_ARES_REASONS = (
    "Domain name not found",
    "DNS server returned answer with no data",
    "Could not contact DNS servers",
    "Timeout while contacting DNS servers",
    "DNS server returned general failure",
    "DNS server refused query",
)

T = TypeVar("T")

_BRACKETED_TAIL = re.compile(r"\[([^\[\]]*)\]\s*$")


@dataclass(frozen=True)
class NameForm:
    label: str
    host: str
    address: str


def _hosts_file() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/drivers/etc/hosts"
    return Path("/etc/hosts")


def _hosts_file_entries() -> list[tuple[str, str]]:
    """(address, name) pairs from the hosts file, comments dropped."""
    try:
        lines = _hosts_file().read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    entries = []
    for line in lines:
        fields = line.split("#", 1)[0].split()
        if len(fields) >= 2:
            entries.extend((fields[0], name) for name in fields[1:])
    return entries


def _os_addresses(host: str) -> set[str]:
    try:
        return {info[4][0] for info in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)}
    except OSError:
        return set()


def _bindable(address: str) -> bool:
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    try:
        with socket.socket(family) as sock:
            sock.bind((address, 0))
    except OSError:
        return False
    return True


def _hosts_file_name(wanted_family: int) -> NameForm | None:
    """A hosts-file name the OS answers with one bindable address of that family, not localhost."""
    for address, name in _hosts_file_entries():
        if name.lower() == "localhost" or name.lower().endswith(".localhost"):
            continue
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            continue
        family = socket.AF_INET6 if parsed.version == 6 else socket.AF_INET
        if family != wanted_family or _os_addresses(name) != {address}:
            continue
        if _bindable(address):
            label = "ipv6-only" if family == socket.AF_INET6 else "hosts-file"
            return NameForm(label, name, address)
    return None


def find_name_form(label: str) -> NameForm | None:
    if label == "localhost":
        return NameForm("localhost", "localhost", "127.0.0.1")
    family = socket.AF_INET6 if label == "ipv6-only" else socket.AF_INET
    return _hosts_file_name(family)


def name_forms() -> list:
    """The three name forms as pytest params, ids by label, resolved when a test asks for one."""
    return [pytest.param(label, id=label) for label in ("localhost", "ipv6-only", "hosts-file")]


def name_form(label: str) -> NameForm:
    form = find_name_form(label)
    if form is None:
        pytest.skip(f"the hosts file names no {label} address this machine can listen on")
    return form


def by_name(url: str, host: str) -> str:
    """`url` with its host swapped for `host`, port and path kept."""
    parts = urlsplit(url)
    netloc = f"{host}:{parts.port}" if parts.port else host
    return urlunsplit(parts._replace(netloc=netloc))


# c-ares answers `localhost` and `ip6-localhost` with both loopback addresses
OTHER_LOOPBACK = {"127.0.0.1": "::1", "::1": "127.0.0.1"}
PORT_ATTEMPTS = 20


def _hold_port(address: str, port: int) -> socket.socket | None:
    """A socket bound to the port and never listening, so a connection there is refused at once."""
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    holder = socket.socket(family)
    try:
        holder.bind((address, port))
    except OSError:
        holder.close()
        return None
    return holder


@contextmanager
def serving_by_name(label: str) -> Iterator[Listener]:
    """A `listener` on the name form's address whose `base_url` names it by the form's host.

    Its port is held on the other loopback address too, so a resolver that also answers that one
    reaches no other local server on the same port.
    """
    form = name_form(label)
    other_loopback = OTHER_LOOPBACK.get(form.address)
    if other_loopback and not _bindable(other_loopback):
        other_loopback = None
    for _ in range(PORT_ATTEMPTS):
        with listening(form.address) as service:
            holder = _hold_port(other_loopback, service.port) if other_loopback else None
            if other_loopback and holder is None:
                continue
            try:
                yield dataclasses.replace(service, base_url=f"http://{form.host}:{service.port}")
            finally:
                if holder:
                    holder.close()
            return
    raise RuntimeError(f"no port free on both {form.address} and {other_loopback}")


def timed(call: Callable[..., T], *args, **kwargs) -> tuple[T, float]:
    """What `call` returned and how many seconds it took."""
    started = time.monotonic()
    result = call(*args, **kwargs)
    return result, time.monotonic() - started


def resolver_reason(text: str) -> str | None:
    """The resolver's wording at the end of an aiohttp connection error, if the text ends in one."""
    tail = _BRACKETED_TAIL.search(text.strip())
    return tail.group(1) if tail else None


def without_resolver_reason(text: str) -> str:
    """The error text with the resolver's wording replaced, so both resolvers' errors compare."""
    return _BRACKETED_TAIL.sub("[<resolver reason>]", text.strip())


# pycares codes for a DNS server that never answered, as against one that said no
_C_ARES_UNANSWERED = {11, 12}  # ARES_ECONNREFUSED, ARES_ETIMEOUT
PROBE_TIMEOUT = 1.0  # seconds per query; a working resolver refuses an unknown name in milliseconds


async def _c_ares_error_code(host: str) -> int | None:
    import aiodns

    resolver = aiodns.DNSResolver(timeout=PROBE_TIMEOUT, tries=1)
    try:
        await resolver.getaddrinfo(host, family=socket.AF_UNSPEC, port=80, type=socket.SOCK_STREAM)
    except aiodns.error.DNSError as error:
        return error.args[0]
    return None


def _c_ares_left_unanswered(host: str, attempts: int) -> bool:
    """Whether a fresh c-ares channel got no answer, run off any event loop the caller holds."""

    async def ask_until_unanswered() -> bool:
        for _ in range(attempts):
            if await _c_ares_error_code(host) in _C_ARES_UNANSWERED:
                return True
        return False

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, ask_until_unanswered()).result()


@functools.cache
def c_ares_refusal_problem() -> str | None:
    """Why c-ares cannot refuse an unknown name at once on this machine while the OS can, or None.

    Two fresh channels ask for one new name, as two processes would: some DNS servers (Tailscale's,
    some routers) replay a cached answer with the first client's EDNS cookie, which c-ares drops and
    then times out on (c-ares issues 1081 and 1271), while the OS resolver takes it at once.
    """
    try:
        import aiodns  # noqa: F401
    except ImportError:
        return None
    started = time.monotonic()
    try:
        socket.getaddrinfo(UNRESOLVABLE, 80, type=socket.SOCK_STREAM)
    except OSError:
        pass
    if time.monotonic() - started > PROBE_TIMEOUT:
        return None  # the OS is slow too: not a c-ares problem, so the tests run and show it
    # a name of its own, so the first channel meets an empty cache and the second a cached reply
    if not _c_ares_left_unanswered(f"owui-probe-{uuid.uuid4().hex[:12]}.invalid", attempts=2):
        return None
    return (
        "this machine's DNS server leaves c-ares unanswered for a name the OS resolver refuses at "
        "once (a cached reply with a stale EDNS cookie, c-ares issues 1081 and 1271), so a c-ares "
        f"lookup of {UNRESOLVABLE} times out instead of failing"
    )
