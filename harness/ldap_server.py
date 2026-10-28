"""A small LDAP directory on a local port for Open WebUI's LDAP sign-in to bind and search.

The server speaks the part of RFC 4511 that ldap3 uses for a sign-in: simple binds, a search
matched against the people it holds and unbind, decoded and encoded with ldap3's own ASN.1
models. A service account binds first; `add_person(...)` puts someone in the directory, with the
groups their `memberOf` names, and `with_mail=False` without an address. `binds` and `searches`
record what the instance asked for, each search with the DNs it `found`.

`serve_directory()` starts it for a block, and `save_ldap_settings(client, directory)` points the
admin's LDAP settings at it and switches LDAP sign-in on. `serve_directory(tls=True)` speaks
LDAPS, with a certificate of its own at `directory.certificate_path` for the instance to trust,
and counts the connections that completed the handshake in `tls_handshakes`. Wrap a change on
the shared instance in `preserve(LDAP_CONFIG)`, which switches LDAP sign-in off again. The server
settings stay pointing at the stopped directory: the settings endpoint refuses the empty defaults
it starts with, and nothing reads them while LDAP sign-in is off.
"""

from __future__ import annotations

import socketserver
import ssl
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import httpx
from ldap3.operation.bind import bind_request_to_dict, bind_response_operation
from ldap3.operation.search import search_request_to_dict
from ldap3.protocol.rfc4511 import (
    LDAPDN,
    And,
    Filter,
    LDAPMessage,
    LDAPString,
    MessageID,
    Not,
    Or,
    PartialAttribute,
    PartialAttributeList,
    ProtocolOp,
    ResultCode,
    SearchRequest,
    SearchResultDone,
    SearchResultEntry,
    Vals,
)
from ldap3.utils.asn1 import encode
from ldap3.utils.dn import escape_rdn
from pyasn1.codec.ber import decoder
from pyasn1.type.namedtype import NamedType, NamedTypes

from harness.object_storage import self_signed_certificate

LDAP_SERVER_CONFIG = ("/api/v1/auths/admin/config/ldap/server",) * 2
LDAP_CONFIG = ("/api/v1/auths/admin/config/ldap",) * 2
BASE_DN = "dc=example,dc=org"
PEOPLE_DN = f"ou=people,{BASE_DN}"
GROUPS_DN = f"ou=groups,{BASE_DN}"
SERVICE_DN = f"cn=owui-service,{BASE_DN}"
SERVICE_PASSWORD = "service-password-0123"
SUCCESS, INVALID_CREDENTIALS = 0, 49
FILTER_DEPTH = 6


def _with_components(spec, **replacements):
    """A copy of the ASN.1 `spec` with the named components swapped."""
    types = [
        type(named)(named.name, replacements[named.name]) if named.name in replacements else named
        for named in spec.componentType.namedTypes
    ]
    return type(spec)(componentType=NamedTypes(*types))


def _decodable_message():
    """ldap3's LDAPMessage with a filter spec pyasn1 can decode, nested `FILTER_DEPTH` deep.

    ldap3 only encodes filters, and its `Filter` is built before `And`, `Or` and `Not` get their
    component types, so decoding a nested filter with it fails.
    """
    filter_spec = Filter()
    for _ in range(FILTER_DEPTH):
        inner = filter_spec
        not_filter = Not(componentType=NamedTypes(NamedType("innerNotFilter", inner)))
        replacements = {"and": And(componentType=inner), "or": Or(componentType=inner)}
        filter_spec = _with_components(inner, **replacements, notFilter=not_filter)
    search = _with_components(SearchRequest(), filter=filter_spec)
    operation = _with_components(ProtocolOp(), searchRequest=search)
    return _with_components(LDAPMessage(), protocolOp=operation)


REQUEST_SPEC = _decodable_message()


@dataclass
class Person:
    dn: str
    password: str
    attributes: dict[str, list[str]]


@dataclass
class Bind:
    dn: str
    password: str
    succeeded: bool


@dataclass
class Search:
    base: str
    filter: str
    attributes: list[str]
    found: list[str] = field(default_factory=list)  # the DNs it answered with


@dataclass
class Directory:
    host: str
    port: int
    people: dict[str, Person] = field(default_factory=dict)
    binds: list[Bind] = field(default_factory=list)
    searches: list[Search] = field(default_factory=list)
    certificate_path: str = ""
    tls_handshakes: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add_person(
        self,
        uid: str,
        password: str,
        *,
        mail: str | None = None,
        cn: str | None = None,
        groups: tuple[str, ...] = (),
        with_mail: bool = True,
    ) -> Person:
        """Someone under `PEOPLE_DN` who signs in as `uid`, a member of each group in `groups`.

        `with_mail=False` leaves out the mail attribute, as a directory without addresses does.
        """
        attributes = {
            "objectClass": ["inetOrgPerson"],
            "uid": [uid],
            "cn": [cn or uid.title()],
        }
        if with_mail:
            attributes["mail"] = [mail or f"{uid}@example.org"]
        if groups:
            attributes["memberOf"] = [f"cn={escape_rdn(group)},{GROUPS_DN}" for group in groups]
        person = Person(dn=f"uid={uid},{PEOPLE_DN}", password=password, attributes=attributes)
        with self.lock:
            self.people[person.dn.lower()] = person
        return person

    def user_binds(self) -> list[Bind]:
        """Every bind except the service account's."""
        with self.lock:
            return [entry for entry in self.binds if entry.dn != SERVICE_DN]

    def _bind(self, dn: str, password: str) -> bool:
        if dn == SERVICE_DN:
            succeeded = password == SERVICE_PASSWORD
        else:
            person = self.people.get(dn.lower())
            # a DN with an empty password is an unauthenticated bind, which this server refuses
            succeeded = person is not None and bool(password) and person.password == password
        with self.lock:
            self.binds.append(Bind(dn=dn, password=password, succeeded=succeeded))
        return succeeded

    def _search(self, request) -> list[Person]:
        described = search_request_to_dict(request)
        search = Search(described["base"], described["filter"], list(described["attributes"]))
        with self.lock:
            self.searches.append(search)
            people = list(self.people.values())
        base = described["base"].lower()
        found = [
            person
            for person in people
            if person.dn.lower().endswith(base) and _matches(request["filter"], person)
        ]
        search.found = [person.dn for person in found]
        return found


def _values(person: Person, name: str) -> list[str]:
    for attribute, values in person.attributes.items():
        if attribute.lower() == name.lower():
            return values
    return []


def _matches(ldap_filter, person: Person) -> bool:
    """Evaluate the filters a sign-in sends: and, or, not, equality and presence."""
    kind = ldap_filter.getName()
    if kind == "and":
        return all(_matches(inner, person) for inner in ldap_filter["and"])
    if kind == "or":
        return any(_matches(inner, person) for inner in ldap_filter["or"])
    if kind == "notFilter":
        return not _matches(ldap_filter["notFilter"]["innerNotFilter"], person)
    if kind == "present":
        return bool(_values(person, str(ldap_filter["present"])))
    if kind == "equalityMatch":
        assertion = ldap_filter["equalityMatch"]
        wanted = bytes(assertion["assertionValue"]).decode().lower()
        values = _values(person, str(assertion["attributeDesc"]))
        return wanted in (value.lower() for value in values)
    return False


def _entry(person: Person, requested: list[str]) -> SearchResultEntry:
    wanted = {name.lower() for name in requested}
    attributes = PartialAttributeList()
    for name, values in person.attributes.items():
        if wanted and "*" not in wanted and name.lower() not in wanted:
            continue
        attribute = PartialAttribute()
        attribute["type"] = name
        attribute["vals"] = Vals()
        for index, value in enumerate(values):
            attribute["vals"][index] = value
        attributes.append(attribute)
    entry = SearchResultEntry()
    entry["object"] = LDAPDN(person.dn)
    entry["attributes"] = attributes
    return entry


def _done(result_code: int) -> SearchResultDone:
    done = SearchResultDone()
    done["resultCode"] = ResultCode(result_code)
    done["matchedDN"] = LDAPDN("")
    done["diagnosticMessage"] = LDAPString("")
    return done


def _message(message_id: int, kind: str, operation) -> bytes:
    message = LDAPMessage()
    message["messageID"] = MessageID(message_id)
    message["protocolOp"] = ProtocolOp().setComponentByName(kind, operation)
    return encode(message)


def _read_message(stream) -> bytes | None:
    """One BER-framed LDAPMessage off the stream, None once the client hangs up."""
    header = stream.read(2)
    if len(header) < 2:
        return None
    length = header[1]
    extra = b""
    if length & 0x80:
        extra = stream.read(length & 0x7F)
        length = int.from_bytes(extra, "big")
    return header + extra + stream.read(length)


def _handler(directory: Directory):
    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            while raw := _read_message(self.rfile):
                message, _rest = decoder.decode(raw, asn1Spec=REQUEST_SPEC)
                message_id = int(message["messageID"])
                operation = message["protocolOp"]
                kind = operation.getName()
                if kind == "bindRequest":
                    request = bind_request_to_dict(operation["bindRequest"])
                    password = request["authentication"]["simple"] or ""
                    if isinstance(password, bytes):
                        password = password.decode()
                    succeeded = directory._bind(request["name"], password)
                    result = SUCCESS if succeeded else INVALID_CREDENTIALS
                    reply = bind_response_operation(result)
                    self.wfile.write(_message(message_id, "bindResponse", reply))
                elif kind == "searchRequest":
                    request = operation["searchRequest"]
                    requested = [str(name) for name in request["attributes"]]
                    for person in directory._search(request):
                        entry = _entry(person, requested)
                        self.wfile.write(_message(message_id, "searchResEntry", entry))
                    self.wfile.write(_message(message_id, "searchResDone", _done(SUCCESS)))
                else:
                    return  # unbind, or anything a sign-in never sends

    return Handler


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True
    tls_context: ssl.SSLContext | None = None
    directory: Directory | None = None

    def finish_request(self, request, client_address) -> None:
        if self.tls_context:
            try:
                request = self.tls_context.wrap_socket(request, server_side=True)
            except (ssl.SSLError, OSError):
                return  # a client that refused the certificate
            with self.directory.lock:
                self.directory.tls_handshakes += 1
        super().finish_request(request, client_address)


@contextmanager
def serve_directory(host: str = "127.0.0.1", tls: bool = False) -> Iterator[Directory]:
    server = _Server((host, 0), None)
    directory = Directory(host=host, port=server.server_address[1])
    server.RequestHandlerClass = _handler(directory)
    server.directory = directory
    with tempfile.TemporaryDirectory(prefix="owui-ldap-") as certificates:
        if tls:
            certificate_path, key_path = self_signed_certificate(Path(certificates))
            server.tls_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            server.tls_context.load_cert_chain(certificate_path, key_path)
            directory.certificate_path = str(certificate_path)
        threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True).start()
        try:
            yield directory
        finally:
            server.shutdown()
            server.server_close()


def save_ldap_settings(client: httpx.Client, directory: Directory, **changes) -> None:
    """Point the admin's LDAP settings at `directory` and switch LDAP sign-in on."""
    settings = {
        "label": "Test directory",
        "host": directory.host,
        "port": directory.port,
        "attribute_for_mail": "mail",
        "attribute_for_username": "uid",
        "app_dn": SERVICE_DN,
        "app_dn_password": SERVICE_PASSWORD,
        "search_base": PEOPLE_DN,
        "search_filters": "",
        "use_tls": False,
        "certificate_path": None,
        "validate_cert": False,
        "default_email_domain": "",
        **changes,
    }
    saved = client.post(LDAP_SERVER_CONFIG[1], json=settings)
    assert saved.status_code == 200, f"saving the LDAP server settings failed: {saved.text}"
    enabled = client.post(LDAP_CONFIG[1], json={"enable_ldap": True})
    assert enabled.status_code == 200, f"switching LDAP sign-in on failed: {enabled.text}"
