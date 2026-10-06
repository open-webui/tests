"""Web search engine APIs played locally, one stand-in for every engine that calls out over HTTP.

`serving_search_apis()` yields a `SearchApis`: a `Listener` that records every search and answers
from routes, and a proxy in front of it. Engines whose address the admin sets (a base URL, an
endpoint or a query URL) are pointed at the listener directly. Engines with a fixed public host
reach it through the proxy, which accepts `CONNECT` to `PUBLIC_HOSTS` only, answers the TLS
handshake with a certificate signed by its own authority and hands each request to the listener,
`Host` header and all. `search_apis_env(apis)` is the environment of an instance that sends them
there: the proxy, the authority for `requests` (`REQUESTS_CA_BUNDLE`) and for aiohttp and httpx
(`SSL_CERT_FILE`), and loopback kept off the proxy.

`ENGINES` lists each engine as an `Engine`: its value in the engine list, the settings that use
the stand-in (`{base}` standing for the listener), the placeholders of the fields the admin fills
in Admin Settings > Web Search for them, the route it calls and the answer listing `Hit`s in that
engine's own shape. `engine.serve(apis, hits)` routes that answer and `engine.found(hits)` is the
list the `search_web` tool hands the model for them.
"""

from __future__ import annotations

import base64
import contextlib
import socketserver
import ssl
import tempfile
import threading
from dataclasses import dataclass, field
from html import escape
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Callable, Iterator
from urllib.parse import unquote_plus

from harness.listener import Answer, Listener, ReceivedRequest, json_answer, listening
from harness.tls_authority import issue_certificate

API_KEY = "search-api-key"


@dataclass
class Hit:
    link: str
    title: str
    snippet: str


@dataclass
class Engine:
    name: str
    method: str
    path: str
    answer: Callable[[list[Hit]], Answer]
    settings: dict[str, str] = field(default_factory=dict)
    form: dict[str, str] = field(default_factory=dict)  # placeholder: setting it fills
    host: str | None = None

    def filled_settings(self, apis: SearchApis) -> dict[str, str]:
        return {name: value.format(base=apis.base_url) for name, value in self.settings.items()}

    def serve(self, apis: SearchApis, hits: list[Hit]) -> None:
        apis.listener.route(self.method, self.path, self.answer(hits))

    def found(self, hits: list[Hit]) -> list[dict]:
        if self.name == "perplexity":
            return _perplexity_found(hits)
        return [{"title": hit.title, "link": hit.link, "snippet": hit.snippet} for hit in hits]

    def searches(self, apis: SearchApis) -> list[ReceivedRequest]:
        """What the stand-in received on this engine's route, from this engine's host."""
        prefix = self.path.removesuffix("*")
        with apis.listener.lock:
            received = list(apis.listener.received)
        return [
            entry
            for entry in received
            if entry.path.startswith(prefix)
            and (self.host is None or entry.headers.get("Host", "").startswith(self.host))
        ]


def _perplexity_found(hits: list[Hit]) -> list[dict]:
    """Perplexity's citations come numbered, its answer as the first one's snippet."""
    return [
        {
            "title": f"Source {number}",
            "link": hit.link,
            "snippet": hits[0].snippet if number == 1 else "",
        }
        for number, hit in enumerate(hits, start=1)
    ]


def carries(request: ReceivedRequest, text: str) -> bool:
    """Whether `text` went out in the request's address, headers or body."""
    sent = [unquote_plus(request.path), request.body.decode("utf-8", "replace")]
    return any(text in part for part in sent + list(request.headers.values()))


@dataclass
class SearchApis:
    listener: Listener
    proxy_url: str
    ca_bundle: Path
    refused_hosts: list[str] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return self.listener.base_url


def _yandex_answer(hits: list[Hit]) -> Answer:
    groups = "".join(
        f"<group><doc><url>{escape(hit.link)}</url><title>{escape(hit.title)}</title>"
        f"<passages><passage>{escape(hit.snippet)}</passage></passages></doc></group>"
        for hit in hits
    )
    results = f"<response><results><grouping>{groups}</grouping></results></response>"
    document = f"<yandexsearch>{results}</yandexsearch>".encode()
    return json_answer({"rawData": base64.b64encode(document).decode()})


def _fields(hits: list[Hit], link: str, title: str, snippet: str, **extra) -> list[dict]:
    return [
        {link: hit.link, title: hit.title, snippet: hit.snippet, **extra, "position": position}
        for position, hit in enumerate(hits, start=1)
    ]


ENGINES = [
    Engine(
        "ollama_cloud",
        "POST",
        "/api/web_search",
        lambda hits: json_answer({"results": _fields(hits, "url", "title", "content")}),
        settings={"OLLAMA_CLOUD_WEB_SEARCH_API_KEY": API_KEY},
        form={"Enter Ollama Cloud API Key": "OLLAMA_CLOUD_WEB_SEARCH_API_KEY"},
        host="ollama.com",
    ),
    Engine(
        "perplexity_search",
        "POST",
        "/perplexity/search",
        lambda hits: json_answer({"results": _fields(hits, "url", "title", "snippet")}),
        settings={
            "PERPLEXITY_SEARCH_API_URL": "{base}/perplexity/search",
            "PERPLEXITY_API_KEY": API_KEY,
        },
        form={
            "Enter Perplexity Search API URL": "PERPLEXITY_SEARCH_API_URL",
            "Enter Perplexity API Key": "PERPLEXITY_API_KEY",
        },
    ),
    Engine(
        "yacy",
        "GET",
        "/yacy/yacysearch.json",
        lambda hits: json_answer(
            {"channels": [{"items": _fields(hits, "link", "title", "description")}]}
        ),
        settings={
            "YACY_QUERY_URL": "{base}/yacy",
            "YACY_USERNAME": "yacy-user",
            "YACY_PASSWORD": "yacy-password",
        },
        form={
            "Enter Yacy URL (e.g. http://yacy.example.com:8090)": "YACY_QUERY_URL",
            "Enter Yacy Username": "YACY_USERNAME",
            "Enter Yacy Password": "YACY_PASSWORD",
        },
    ),
    Engine(
        "google_pse",
        "GET",
        "/customsearch/v1",
        lambda hits: json_answer({"items": _fields(hits, "link", "title", "snippet")}),
        settings={"GOOGLE_PSE_API_KEY": API_KEY, "GOOGLE_PSE_ENGINE_ID": "pse-engine"},
        form={
            "Enter Google PSE API Key": "GOOGLE_PSE_API_KEY",
            "Enter Google PSE Engine Id": "GOOGLE_PSE_ENGINE_ID",
        },
        host="www.googleapis.com",
    ),
    Engine(
        "brave",
        "GET",
        "/res/v1/web/search",
        lambda hits: json_answer(
            {"web": {"results": _fields(hits, "url", "title", "description")}}
        ),
        settings={"BRAVE_SEARCH_API_KEY": API_KEY},
        form={"Enter Brave Search API Key": "BRAVE_SEARCH_API_KEY"},
        host="api.search.brave.com",
    ),
    Engine(
        "brave_llm_context",
        "GET",
        "/res/v1/llm/context",
        lambda hits: json_answer(
            {
                "grounding": {
                    "generic": [
                        {"url": hit.link, "title": hit.title, "snippets": [hit.snippet]}
                        for hit in hits
                    ]
                }
            }
        ),
        settings={"BRAVE_SEARCH_API_KEY": API_KEY},
        form={"Enter Brave Search API Key": "BRAVE_SEARCH_API_KEY"},
        host="api.search.brave.com",
    ),
    Engine(
        "kagi",
        "POST",
        "/api/v1/search",
        lambda hits: json_answer({"data": {"search": _fields(hits, "url", "title", "snippet")}}),
        settings={"KAGI_SEARCH_API_KEY": API_KEY},
        form={"Enter Kagi Search API Key": "KAGI_SEARCH_API_KEY"},
        host="kagi.com",
    ),
    Engine(
        "mojeek",
        "GET",
        "/search",
        lambda hits: json_answer({"response": {"results": _fields(hits, "url", "title", "desc")}}),
        settings={"MOJEEK_SEARCH_API_KEY": API_KEY},
        form={"Enter Mojeek Search API Key": "MOJEEK_SEARCH_API_KEY"},
        host="api.mojeek.com",
    ),
    Engine(
        "bocha",
        "POST",
        "/v1/web-search",
        lambda hits: json_answer(
            {"data": {"webPages": {"value": _fields(hits, "url", "name", "summary")}}}
        ),
        settings={"BOCHA_SEARCH_API_KEY": API_KEY},
        form={"Enter Bocha Search API Key": "BOCHA_SEARCH_API_KEY"},
        host="api.bochaai.com",
    ),
    Engine(
        "serpstack",
        "GET",
        "/search",
        lambda hits: json_answer({"organic_results": _fields(hits, "url", "title", "snippet")}),
        settings={"SERPSTACK_API_KEY": API_KEY},
        form={"Enter Serpstack API Key": "SERPSTACK_API_KEY"},
        host="api.serpstack.com",
    ),
    Engine(
        "serper",
        "POST",
        "/search",
        lambda hits: json_answer({"organic": _fields(hits, "link", "title", "snippet")}),
        settings={"SERPER_API_KEY": API_KEY},
        form={"Enter Serper API Key": "SERPER_API_KEY"},
        host="google.serper.dev",
    ),
    Engine(
        "serphouse",
        "GET",
        "/serp/live",
        lambda hits: json_answer(
            {"results": {"results": {"organic": _fields(hits, "link", "title", "snippet")}}}
        ),
        settings={"SERPHOUSE_API_KEY": API_KEY},
        form={"Enter SERPHouse API Key": "SERPHOUSE_API_KEY"},
        host="api.serphouse.com",
    ),
    Engine(
        "serply",
        "GET",
        "/v1/search/*",
        lambda hits: json_answer(
            {"results": _fields(hits, "link", "title", "description", realPosition=0)}
        ),
        settings={"SERPLY_API_KEY": API_KEY},
        form={"Enter Serply API Key": "SERPLY_API_KEY"},
        host="api.serply.io",
    ),
    Engine(
        "searchapi",
        "GET",
        "/api/v1/search",
        lambda hits: json_answer({"organic_results": _fields(hits, "link", "title", "snippet")}),
        settings={"SEARCHAPI_API_KEY": API_KEY},
        form={"Enter SearchApi API Key": "SEARCHAPI_API_KEY"},
        host="www.searchapi.io",
    ),
    Engine(
        "serpapi",
        "GET",
        "/search",
        lambda hits: json_answer({"organic_results": _fields(hits, "link", "title", "snippet")}),
        settings={"SERPAPI_API_KEY": API_KEY},
        form={"Enter SerpApi API Key": "SERPAPI_API_KEY"},
        host="serpapi.com",
    ),
    Engine(
        "tavily",
        "POST",
        "/search",
        lambda hits: json_answer({"results": _fields(hits, "url", "title", "content")}),
        settings={"TAVILY_API_KEY": API_KEY},
        form={"Enter Tavily API Key": "TAVILY_API_KEY"},
        host="api.tavily.com",
    ),
    Engine(
        "staan",
        "GET",
        "/v2/search/web",
        lambda hits: json_answer({"web": {"results": _fields(hits, "url", "title", "snippet")}}),
        settings={"STAAN_API_KEY": API_KEY},
        form={"Enter Staan API Key": "STAAN_API_KEY"},
        host="api.staan.ai",
    ),
    Engine(
        "jina",
        "POST",
        "/jina",
        lambda hits: json_answer({"data": _fields(hits, "url", "title", "content")}),
        settings={"JINA_API_BASE_URL": "{base}/jina", "JINA_API_KEY": API_KEY},
        form={"Enter Jina API Base URL": "JINA_API_BASE_URL", "Enter Jina API Key": "JINA_API_KEY"},
    ),
    Engine(
        "bing",
        "GET",
        "/bing/v7.0/search",
        lambda hits: json_answer({"webPages": {"value": _fields(hits, "url", "name", "snippet")}}),
        settings={
            "BING_SEARCH_V7_ENDPOINT": "{base}/bing/v7.0/search",
            "BING_SEARCH_V7_SUBSCRIPTION_KEY": API_KEY,
        },
        form={
            "Enter Bing Search V7 Endpoint": "BING_SEARCH_V7_ENDPOINT",
            "Enter Bing Search V7 Subscription Key": "BING_SEARCH_V7_SUBSCRIPTION_KEY",
        },
    ),
    Engine(
        "exa",
        "POST",
        "/search",
        lambda hits: json_answer({"results": _fields(hits, "url", "title", "text")}),
        settings={"EXA_API_KEY": API_KEY},
        form={"Enter Exa API Key": "EXA_API_KEY"},
        host="api.exa.ai",
    ),
    Engine(
        "perplexity",
        "POST",
        "/chat/completions",
        lambda hits: json_answer(
            {
                "citations": [hit.link for hit in hits],
                "choices": [{"message": {"role": "assistant", "content": hits[0].snippet}}],
            }
        ),
        settings={"PERPLEXITY_API_KEY": API_KEY},
        form={"Enter Perplexity API Key": "PERPLEXITY_API_KEY"},
        host="api.perplexity.ai",
    ),
    Engine(
        "microsoft_web_iq",
        "POST",
        "/web-iq/search/web",
        lambda hits: json_answer({"webResults": _fields(hits, "url", "title", "content")}),
        settings={
            "MICROSOFT_WEB_IQ_API_BASE_URL": "{base}/web-iq",
            "MICROSOFT_WEB_IQ_API_KEY": API_KEY,
        },
        form={
            "Enter Microsoft Web IQ API Base URL": "MICROSOFT_WEB_IQ_API_BASE_URL",
            "Enter Microsoft Web IQ API Key": "MICROSOFT_WEB_IQ_API_KEY",
        },
    ),
    Engine(
        "firecrawl",
        "POST",
        "/firecrawl/v2/search",
        lambda hits: json_answer({"data": {"web": _fields(hits, "url", "title", "description")}}),
        settings={"FIRECRAWL_API_BASE_URL": "{base}/firecrawl", "FIRECRAWL_API_KEY": API_KEY},
        form={
            "Enter Firecrawl API Base URL": "FIRECRAWL_API_BASE_URL",
            "Enter Firecrawl API Key": "FIRECRAWL_API_KEY",
        },
    ),
    Engine(
        "yandex",
        "POST",
        "/yandex/search",
        _yandex_answer,
        settings={
            "YANDEX_WEB_SEARCH_URL": "{base}/yandex/search",
            "YANDEX_WEB_SEARCH_API_KEY": API_KEY,
        },
        form={
            "Enter Yandex Web Search URL": "YANDEX_WEB_SEARCH_URL",
            "Enter Yandex Web Search API Key": "YANDEX_WEB_SEARCH_API_KEY",
        },
    ),
    Engine(
        "youcom",
        "GET",
        "/v1/search",
        lambda hits: json_answer(
            {"results": {"web": _fields(hits, "url", "title", "description")}}
        ),
        settings={"YOUCOM_API_KEY": API_KEY},
        form={"Enter You.com API Key": "YOUCOM_API_KEY"},
        host="ydc-index.io",
    ),
    Engine(
        "linkup",
        "POST",
        "/v1/search",
        lambda hits: json_answer({"sources": _fields(hits, "url", "name", "content")}),
        settings={"LINKUP_API_KEY": API_KEY},
        form={"Enter Linkup API Key": "LINKUP_API_KEY"},
        host="api.linkup.so",
    ),
    Engine(
        "openserp",
        "GET",
        "/openserp/mega/search",
        lambda hits: json_answer({"results": _fields(hits, "url", "title", "snippet")}),
        settings={"OPENSERP_BASE_URL": "{base}/openserp"},
        form={"Enter OpenSERP Base URL": "OPENSERP_BASE_URL"},
    ),
]
ENGINES_BY_NAME = {engine.name: engine for engine in ENGINES}
PUBLIC_HOSTS = sorted({engine.host for engine in ENGINES if engine.host})


def _api_handler(apis: SearchApis):
    class ApiRequest(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            pass

        def _serve(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            request = ReceivedRequest(
                method=self.command,
                path=self.path,
                headers=dict(self.headers.items()),
                body=self.rfile.read(length) if length else b"",
            )
            with apis.listener.lock:
                apis.listener.received.append(request)
            handler = apis.listener.handler_for(self.command, self.path.split("?")[0])
            status, headers, body = (
                handler(request) if handler else (404, {"Content-Type": "text/plain"}, b"")
            )
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = do_POST = _serve

    return ApiRequest


def _proxy_handler(apis: SearchApis, tls: ssl.SSLContext):
    api_request = _api_handler(apis)

    class Proxy(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            request_line = self.rfile.readline().decode("latin-1")
            while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                pass  # the proxy's own headers
            method, target, _ = (request_line.split(" ") + ["", "", ""])[:3]
            host = target.split(":")[0]
            if method != "CONNECT" or host not in PUBLIC_HOSTS:
                apis.refused_hosts.append(host)
                self.wfile.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
                return
            self.wfile.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            self.wfile.flush()
            with contextlib.suppress(OSError, ssl.SSLError):
                wrapped = tls.wrap_socket(self.connection, server_side=True)
                api_request(wrapped, self.client_address, self.server)

    return Proxy


@contextlib.contextmanager
def serving_search_apis() -> Iterator[SearchApis]:
    with (
        tempfile.TemporaryDirectory(prefix="fake-search-apis-") as directory,
        listening() as listener,
    ):
        issued = issue_certificate(Path(directory), "Fake search API authority", PUBLIC_HOSTS)
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(issued.certificate, issued.key)
        server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), None)
        server.daemon_threads = True
        host, port = server.server_address
        apis = SearchApis(listener, proxy_url=f"http://{host}:{port}", ca_bundle=issued.authority)
        server.RequestHandlerClass = _proxy_handler(apis, tls)
        threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True).start()
        try:
            yield apis
        finally:
            server.shutdown()
            server.server_close()


def search_apis_env(apis: SearchApis) -> dict[str, str]:
    return {
        "HTTPS_PROXY": apis.proxy_url,
        "REQUESTS_CA_BUNDLE": str(apis.ca_bundle),
        "SSL_CERT_FILE": str(apis.ca_bundle),
        "NO_PROXY": "127.0.0.1,localhost",
    }


def engine_settings(engine: Engine, apis: SearchApis, result_count: int) -> dict:
    """The admin's web search settings for `engine` on the stand-in, as the API takes them."""
    return {
        "ENABLE_WEB_SEARCH": True,
        "WEB_SEARCH_ENGINE": engine.name,
        "WEB_SEARCH_RESULT_COUNT": result_count,
        **engine.filled_settings(apis),
    }
