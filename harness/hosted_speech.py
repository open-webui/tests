"""Deepgram and ElevenLabs played by a `listener`, reached by their real host names through a proxy.

Open WebUI calls Deepgram at the fixed `https://api.deepgram.com` and ElevenLabs at
`https://api.elevenlabs.io` unless `ELEVENLABS_API_BASE_URL` moves it at boot, so both are
reached the way a deployment reaches them: by name, through the proxy the environment names (the
shared aiohttp pool trusts `HTTPS_PROXY`). `serving_speech_hosts()` yields that proxy: it accepts
`CONNECT` to `SPEECH_HOSTS` only, answers the TLS handshake with a certificate signed by its own
authority and relays the decrypted requests to its own `proxy.listener`, which plays both APIs:
Deepgram's `/v1/listen` answers `TRANSCRIPT`, ElevenLabs lists `ELEVENLABS_VOICES` and
`ELEVENLABS_MODELS` and speaks `SPEECH`. The instance keeps its tunnels open between requests, so
the proxy outlives the tests that share an instance and `proxy.reset()` gives each test the
default answers and an empty record. `speech_hosts_env(proxy)` is the environment of an instance
whose calls go through it, loopback kept off the proxy.
"""

from __future__ import annotations

import contextlib
import socket
import socketserver
import ssl
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from harness.audio_engine import SPEECH, TRANSCRIPT
from harness.listener import Listener, ReceivedRequest, json_answer, listening
from harness.tls_authority import issue_certificate

DEEPGRAM_HOST = "api.deepgram.com"
ELEVENLABS_HOST = "api.elevenlabs.io"
SPEECH_HOSTS = (DEEPGRAM_HOST, ELEVENLABS_HOST)
ELEVENLABS_VOICES = {"harbour-voice-01": "Harbour Master", "gull-voice-02": "Gull"}
ELEVENLABS_MODELS = {"eleven_multilingual_v2": "Eleven Multilingual v2"}


@dataclass
class SpeechHostsProxy:
    proxy_url: str
    ca_bundle: Path
    listener: Listener
    refused_hosts: list[str] = field(default_factory=list)

    def reset(self) -> None:
        with self.listener.lock:
            self.listener.received.clear()
        self.listener.routes.clear()
        _serve_deepgram(self.listener)
        _serve_elevenlabs(self.listener)

    def speech_requests(self) -> list[ReceivedRequest]:
        """Every ElevenLabs text-to-speech request, whatever its voice."""
        with self.listener.lock:
            return [
                request
                for request in self.listener.received
                if request.method == "POST" and request.path.startswith("/v1/text-to-speech/")
            ]


def _relay(source: socket.socket, target: socket.socket) -> None:
    with contextlib.suppress(OSError):
        while chunk := source.recv(65536):
            target.sendall(chunk)
    with contextlib.suppress(OSError):
        target.shutdown(socket.SHUT_WR)


def _proxy_handler(proxy: SpeechHostsProxy, tls: ssl.SSLContext):
    class Proxy(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            request_line = self.rfile.readline().decode("latin-1")
            while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                pass  # the proxy's own headers
            method, target, _ = (request_line.split(" ") + ["", "", ""])[:3]
            host = target.split(":")[0]
            if method != "CONNECT" or host not in SPEECH_HOSTS:
                proxy.refused_hosts.append(host)
                self.wfile.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
                return
            self.wfile.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            self.wfile.flush()
            with contextlib.suppress(OSError, ssl.SSLError):
                decrypted = tls.wrap_socket(self.connection, server_side=True)
                with socket.create_connection(("127.0.0.1", proxy.listener.port)) as service:
                    answering = threading.Thread(target=_relay, args=(service, decrypted))
                    answering.start()
                    _relay(decrypted, service)
                    answering.join(timeout=30)

    return Proxy


@contextlib.contextmanager
def serving_speech_hosts() -> Iterator[SpeechHostsProxy]:
    with tempfile.TemporaryDirectory(prefix="speech-hosts-") as directory, listening() as service:
        issued = issue_certificate(Path(directory), "Speech hosts test authority", SPEECH_HOSTS)
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(issued.certificate, issued.key)
        server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), None)
        server.daemon_threads = True
        host, port = server.server_address
        proxy = SpeechHostsProxy(f"http://{host}:{port}", issued.authority, service)
        proxy.reset()
        server.RequestHandlerClass = _proxy_handler(proxy, tls)
        threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True).start()
        try:
            yield proxy
        finally:
            server.shutdown()
            server.server_close()


def speech_hosts_env(proxy: SpeechHostsProxy) -> dict[str, str]:
    return {
        "HTTPS_PROXY": proxy.proxy_url,
        "SSL_CERT_FILE": str(proxy.ca_bundle),
        "NO_PROXY": "127.0.0.1,localhost",
    }


def _serve_deepgram(listener: Listener) -> None:
    alternatives = [{"transcript": TRANSCRIPT, "confidence": 0.98}]
    answer = {"results": {"channels": [{"alternatives": alternatives}]}}
    listener.route("POST", "/v1/listen", json_answer(answer))


def _serve_elevenlabs(listener: Listener) -> None:
    voices = [{"voice_id": key, "name": name} for key, name in ELEVENLABS_VOICES.items()]
    models = [{"model_id": key, "name": name} for key, name in ELEVENLABS_MODELS.items()]
    listener.route("GET", "/v1/voices", json_answer({"voices": voices}))
    listener.route("GET", "/v1/models", json_answer(models))
    listener.route("POST", "/v1/text-to-speech/*", (200, {"Content-Type": "audio/mpeg"}, SPEECH))
