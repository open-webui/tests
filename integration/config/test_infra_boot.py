"""Boot-time and deployment-shape regressions fixed between v0.11.0 and v0.11.1, seen from outside.

* 14 (PR28242, c5ec01b1f, issues #28013/#28215): aiohttp defaulted to the c-ares async resolver,
  which ignores the host's name service switch, so a provider named the way only the system
  resolver knows (`/etc/nsswitch.conf` modules, as container and VPN setups use) failed and
  surfaced as a misleading model-not-found. The fix pins the threaded resolver unless
  `AIOHTTP_CLIENT_ASYNC_DNS_RESOLVER` is set. The name here is `localhost.localdomain`, which
  glibc's `myhostname` module answers and c-ares does not; on a host where both or neither
  resolve it, the resolver tests skip and say so.
* 51 (PR27838, 3dbb4078b): the openGauss client indexed the empty legacy `SRC_LOG_LEVELS` at
  import, so any deployment on `VECTOR_DB=opengauss` died the first time it touched the vector
  store. Here openGauss is played by an embedded Postgres with pgvector (`pgserver`), which it is
  wire compatible with, and a knowledge base is filled and searched on it.
* 52 (same commit): the ColBERT reranker's startup record passed the model name to a message
  with no placeholder, so the record could not be formatted and the log never named the model.
  The model here is a tiny ColBERT checkpoint written to disk, complete enough to load offline,
  under a path the reranker loader recognises as `jinaai/jina-colbert-v2`. Before PR31532
  (bee06b08b, issue #31522) loading it failed for any checkpoint: colbert-ai 0.2.22's `HF_ColBERT`
  never calls `post_init`, which transformers 5 relies on to set `all_tied_weights_keys`, so saving
  a ColBERT reranker switched hybrid search back off. The reranking test saves the reranker, checks
  hybrid search stays on, reads it back and reranks a knowledge search. It needs `ninja` on PATH,
  since colbert-ai compiles a C++ extension to score, and skips without it.
* 167 (PR27754, baeb2dfb8, issue #27752): with `DATABASE_ENABLE_IAM_TOKEN_AUTH` the main database
  signed in with an RDS IAM token, but the pgvector store built its own engine and signed in with
  the URL's password, so startup died with "no password supplied"; the token must also stay off a
  vector store on another host, port or user, which it is not valid for. Here RDS is played by
  `harness.rds_proxy`, which asks for and records each password before handing the connection to
  the embedded Postgres; the token is presigned locally from stand-in AWS credentials.

The source audit that no module indexes `SRC_LOG_LEVELS`, the start_windows.bat runs and the
Dockerfile audits stay in unit/config/test_infra_boot.py.

Twin of unit/config/test_infra_boot.py.

Discriminates: passes on a5bc78300; unpinning the resolver in env.py fails the resolver test,
indexing `SRC_LOG_LEVELS['RAG']` in opengauss.py again fails the openGauss test, dropping the
ColBERT record's placeholder fails the startup record test, dropping `enable_iam_token_auth` from
`PgvectorClient` fails the vector store sign-in test, dropping its host, port and user check fails
the other-database tests, and taking the `post_init` subclass out of colbert.py again (PR31532,
bee06b08b) fails the reranking test. The opt-in resolver and IAM-off tests pass on both.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import socket
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from harness import upstream as upstream_module
from harness.actors import admin_of
from harness.instance import launch
from harness.knowledge_bases import add_text_file, knowledge_base
from harness.rds_proxy import serving_rds_proxy

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

RETRIEVAL_CONFIG = ("/api/v1/retrieval/config", "/api/v1/retrieval/config/update")
SYSTEM_ONLY_NAME = "localhost.localdomain"
ASYNC_RESOLVER = {"AIOHTTP_CLIENT_ASYNC_DNS_RESOLVER": "true"}


@pytest.fixture
def boot():
    """`boot(env)` starts an instance of its own, stopped with the test (`instance_with` waits)."""
    with contextlib.ExitStack() as stack:

        def start(env: dict[str, str]):
            provider, shutdown = upstream_module.serve()
            stack.callback(shutdown)
            return stack.enter_context(contextlib.contextmanager(launch)(provider, env))

        yield start


# 14: the resolver behind every outgoing connection


def _system_resolves(name: str) -> bool:
    try:
        return bool(socket.getaddrinfo(name, 80))
    except OSError:
        return False


def _c_ares_resolves(name: str) -> bool:
    aiohttp_resolver = pytest.importorskip(
        "aiohttp.resolver", reason="needs aiohttp to compare resolvers"
    )
    pytest.importorskip("aiodns", reason="without aiodns aiohttp resolves with threads anyway")

    async def resolve() -> bool:
        resolver = aiohttp_resolver.AsyncResolver()
        try:
            return bool(await resolver.resolve(name, 80))
        except OSError:
            return False
        finally:
            await resolver.close()

    # a thread of its own, since the calling test may already run inside an event loop
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, resolve()).result()


@pytest.fixture(scope="module")
def system_only_name() -> str:
    if not _system_resolves(SYSTEM_ONLY_NAME):
        pytest.skip(f"this host's resolver does not know {SYSTEM_ONLY_NAME}")
    if _c_ares_resolves(SYSTEM_ONLY_NAME):
        pytest.skip(f"c-ares resolves {SYSTEM_ONLY_NAME} here too, so the resolvers look alike")
    return SYSTEM_ONLY_NAME


def _verify_connection(instance, host: str):
    """The admin's Verify button on an OpenAI connection addressed by `host`."""
    url = instance.upstream.base_url.replace("127.0.0.1", host)
    with instance.client() as client:
        return client.post("/openai/verify", json={"url": url, "key": "sk-mock"})


def test_a_provider_named_through_the_system_resolver_is_reached(instance, system_only_name):
    verified = _verify_connection(instance, system_only_name)

    assert verified.status_code == 200, (
        f"a provider at {system_only_name} could not be reached: aiohttp resolved it with c-ares, "
        f"which ignores the host's name service switch (#28013): {verified.text}"
    )
    assert verified.json()["data"], verified.text


@pytest.mark.slow
def test_the_async_resolver_is_opt_in(boot, system_only_name):
    opted_in = boot(ASYNC_RESOLVER)

    verified = _verify_connection(opted_in, system_only_name)

    assert verified.status_code != 200, (
        "AIOHTTP_CLIENT_ASYNC_DNS_RESOLVER=true still resolved through the system resolver"
    )


# 51: openGauss as the vector store


@pytest.fixture(scope="module")
def postgres(tmp_path_factory):
    """An embedded Postgres with the pgvector extension, as openGauss stands for here."""
    pgserver = pytest.importorskip("pgserver", reason="needs pgserver (the postgres extra)")
    server = pgserver.get_server(str(tmp_path_factory.mktemp("pgdata")), cleanup_mode="stop")
    try:
        yield server
    finally:
        server.cleanup()


def _database(server, name: str) -> str:
    server.psql(f"CREATE DATABASE {name};")
    server.psql(f"\\c {name}\nCREATE EXTENSION IF NOT EXISTS vector;")
    return server.get_uri(name)


def _search_a_knowledge_base(instance) -> None:
    with admin_of(instance).client() as client, knowledge_base(client) as knowledge_id:
        add_text_file(client, knowledge_id, "harbour.txt", "The harbour master is Ingrid.")
        found = client.post(
            "/api/v1/retrieval/query/collection",
            json={"collection_names": [knowledge_id], "query": "who is the harbour master?"},
        )
    assert found.status_code == 200, found.text
    assert "Ingrid" in str(found.json()["documents"])


@pytest.mark.slow
def test_a_knowledge_base_fills_and_searches_on_opengauss(boot, postgres):
    database_url = _database(postgres, f"gauss_{uuid.uuid4().hex[:8]}")
    on_opengauss = boot({"VECTOR_DB": "opengauss", "OPENGAUSS_DB_URL": database_url})

    _search_a_knowledge_base(on_opengauss)


# 52: the ColBERT reranker's startup record


COLBERT_VOCABULARY = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[unused0]", "[unused1]"]
COLBERT_VOCABULARY += "the a harbour master is who ingrid lighthouse keeper".split()
COLBERT_DIM = 128


@pytest.fixture(scope="module")
def colbert_checkpoint(tmp_path_factory) -> Path:
    """A tiny random ColBERT on disk, complete down to the projection, named like jina's."""
    torch = pytest.importorskip("torch", reason="the local reranker needs torch")
    transformers = pytest.importorskip("transformers", reason="the local reranker needs it")
    safetensors_torch = pytest.importorskip("safetensors.torch", reason="needs safetensors")
    target = tmp_path_factory.mktemp("reranker") / "jinaai" / "jina-colbert-v2"
    target.mkdir(parents=True)
    vocabulary = target / "vocab.txt"
    vocabulary.write_text("\n".join(COLBERT_VOCABULARY) + "\n", encoding="utf-8")
    transformers.BertTokenizerFast(vocab_file=str(vocabulary)).save_pretrained(target)
    config = transformers.BertConfig(
        vocab_size=len(COLBERT_VOCABULARY),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    )
    config.save_pretrained(target)
    encoder = transformers.BertModel(config)
    weights = {f"bert.{name}": value.contiguous() for name, value in encoder.state_dict().items()}
    weights["linear.weight"] = torch.randn(COLBERT_DIM, config.hidden_size)
    safetensors_torch.save_file(weights, str(target / "model.safetensors"))
    # without it colbert-ai loses the checkpoint path and looks the tokenizer up on the Hub
    (target / "artifact.metadata").write_text(json.dumps({"dim": COLBERT_DIM}), encoding="utf-8")
    return target


def _use_reranker(client, model_path: Path) -> dict:
    saved = client.post(
        RETRIEVAL_CONFIG[1],
        json={
            "ENABLE_RAG_HYBRID_SEARCH": True,
            "RAG_RERANKING_ENGINE": "",
            "RAG_RERANKING_MODEL": str(model_path),
        },
    )
    assert saved.status_code == 200, saved.text
    return saved.json()


@pytest.mark.slow
def test_the_colbert_startup_record_names_the_model(instance, admin, preserve, colbert_checkpoint):
    preserve(RETRIEVAL_CONFIG)
    offset = instance.log_size()

    with admin.client() as client:
        _use_reranker(client, colbert_checkpoint)
    logged = instance.log_since(offset)

    assert f"ColBERT: Loading model {colbert_checkpoint}" in logged, (
        "the ColBERT startup record did not name the model: its message has no placeholder for "
        f"the name, so the record could not be formatted:\n{logged[-3000:]}"
    )


@pytest.mark.slow
@pytest.mark.skipif(
    shutil.which("ninja") is None, reason="colbert-ai compiles its scoring extension with ninja"
)
def test_a_colbert_reranker_reranks_a_knowledge_search(
    instance, admin, preserve, colbert_checkpoint
):
    preserve(RETRIEVAL_CONFIG)
    with admin.client() as client, knowledge_base(client) as knowledge_id:
        add_text_file(client, knowledge_id, "harbour.txt", "The harbour master is Ingrid.")
        saved = _use_reranker(client, colbert_checkpoint)
        found = client.post(
            "/api/v1/retrieval/query/collection",
            json={"collection_names": [knowledge_id], "query": "who is the harbour master?"},
        )
        stored = client.get(RETRIEVAL_CONFIG[0]).json()

    assert saved["ENABLE_RAG_HYBRID_SEARCH"] is True, (
        "saving the jina-colbert-v2 reranker switched hybrid search back off: the ColBERT model "
        "failed to load (colbert-ai 0.2.22 under transformers 5 has no all_tied_weights_keys)"
    )
    assert stored["ENABLE_RAG_HYBRID_SEARCH"] is True
    assert found.status_code == 200, found.text
    assert "Ingrid" in str(found.json()["documents"])


# 167: RDS IAM token auth and the pgvector store

STAND_IN_AWS = {
    "AWS_ACCESS_KEY_ID": "AKIAIOSFODNN7EXAMPLE",
    "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "AWS_DEFAULT_REGION": "us-east-1",
    "AWS_EC2_METADATA_DISABLED": "true",
}
URL_PASSWORD = "from-url"


@pytest.fixture(scope="module")
def postgres_socket(postgres) -> str:
    postgres.psql("CREATE ROLE owui SUPERUSER LOGIN; CREATE ROLE other SUPERUSER LOGIN;")
    sockets = sorted(Path(postgres.pgdata).glob(".s.PGSQL.*[0-9]"))
    assert sockets, f"pgserver has no Unix socket in {postgres.pgdata}"
    return str(sockets[0])


@pytest.fixture(scope="module")
def rds(postgres_socket):
    with serving_rds_proxy(postgres_socket) as proxy:
        yield proxy


@pytest.fixture(scope="module")
def second_rds(postgres_socket):
    with serving_rds_proxy(postgres_socket) as proxy:
        yield proxy


def _url(user: str, host: str, port: int | None, database: str) -> str:
    address = f"{host}:{port}" if port else host
    return f"postgresql://{user}:{URL_PASSWORD}@{address}/{database}"


def _is_iam_token(password: str, host: str, port: int, user: str) -> bool:
    return password.startswith(f"{host}:{port}/?") and f"DBUser={user}" in password


def _fresh_database(postgres, prefix: str) -> str:
    name = f"{prefix}_{uuid.uuid4().hex[:8]}"
    postgres.psql(f"CREATE DATABASE {name};")
    return name


def _iam_instance(boot, main_url: str, **settings: str):
    env = {
        **STAND_IN_AWS,
        "DATABASE_ENABLE_IAM_TOKEN_AUTH": "true",
        "DATABASE_URL": main_url,
        "VECTOR_DB": "pgvector",
        **settings,
    }
    return boot(env)


@pytest.mark.slow
def test_the_vector_store_signs_in_with_the_iam_token(boot, postgres, rds):
    """Narrow: PgvectorClient built its own engine and never attached the token."""
    main = _fresh_database(postgres, "main")
    launched = _iam_instance(boot, _url("owui", "127.0.0.1", rds.port, main))

    _search_a_knowledge_base(launched)

    sent = rds.passwords_for(main)
    assert sent, "nothing signed in through the RDS stand-in"
    not_tokens = sorted(p for p in sent if not _is_iam_token(p, "127.0.0.1", rds.port, "owui"))
    assert not_tokens == [], (
        f"a connection signed in with {not_tokens} instead of the IAM token; RDS refuses that, "
        "so startup died with no password supplied (#27752)"
    )


OTHER_VECTOR_STORES = {
    "other-port": ("owui", "127.0.0.1", "second"),
    "other-user": ("other", "127.0.0.1", "same"),
    "other-host": ("owui", "localhost", "same"),
}


@pytest.mark.slow
@pytest.mark.parametrize("case", sorted(OTHER_VECTOR_STORES))
def test_the_token_stays_off_a_vector_store_on_another_database(
    boot, postgres, rds, second_rds, case
):
    """Narrow and broad: host, port and user each have to match the token's."""
    user, host, proxy_name = OTHER_VECTOR_STORES[case]
    vector_proxy = second_rds if proxy_name == "second" else rds
    main = _fresh_database(postgres, "main")
    vectors = _fresh_database(postgres, "vectors")
    launched = _iam_instance(
        boot,
        _url("owui", "127.0.0.1", rds.port, main),
        PGVECTOR_DB_URL=_url(user, host, vector_proxy.port, vectors),
    )
    _search_a_knowledge_base(launched)

    assert vector_proxy.passwords_for(vectors) == {URL_PASSWORD}, (
        "the main database's IAM token replaced the vector store's own password, although the "
        f"token is only valid for owui@127.0.0.1:{rds.port}: {vector_proxy.passwords_for(vectors)}"
    )
    assert all(_is_iam_token(p, "127.0.0.1", rds.port, "owui") for p in rds.passwords_for(main))
    assert "IAM token auth not applied" in launched.log_since(0)


@pytest.mark.slow
def test_nothing_changes_with_iam_auth_off(boot, postgres, rds):
    main = _fresh_database(postgres, "main")
    launched = boot(
        {"DATABASE_URL": _url("owui", "127.0.0.1", rds.port, main), "VECTOR_DB": "pgvector"}
    )

    _search_a_knowledge_base(launched)

    assert rds.passwords_for(main) == {URL_PASSWORD}


@pytest.mark.slow
def test_the_token_reaches_a_database_on_the_default_port(boot, postgres, postgres_socket):
    main = _fresh_database(postgres, "main")
    with contextlib.ExitStack() as stack:
        try:
            proxy = stack.enter_context(serving_rds_proxy(postgres_socket, port=5432))
        except OSError:
            pytest.skip("port 5432 is taken on this host")
        _search_a_knowledge_base(_iam_instance(boot, _url("owui", "127.0.0.1", None, main)))
        sent = proxy.passwords_for(main)

    assert sent and all(_is_iam_token(p, "127.0.0.1", 5432, "owui") for p in sent), sent
