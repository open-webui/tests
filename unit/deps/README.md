# Per-dependency contract tests (`unit/deps/`)

One `test_<dep>.py` per third-party dependency, pinning **exactly the API
surface and behaviour Open WebUI relies on** from that package. Run inside an
environment with a bumped version installed, these catch the "new release
removed / renamed / changed an API we use" class of breakage — the gap our
other suites can't see, because they run against whatever is already installed.

## The contract (so files compose and authors don't collide)

- **One file owns one dependency.** Never edit another dep's file, `conftest.py`,
  or `pyproject.toml`.
- **Shared machinery is fixtures, not imports.** Use the `depcheck` fixture
  (see `conftest.py`) and `open_webui_backend` (from `../conftest.py`). No
  cross-file imports — that keeps dozens of independently-authored modules from
  fighting over sys.path.
- **Marker:** `pytestmark = pytest.mark.depcheck`.
- **Skip, don't fail, when absent:** `depcheck.load(import_name)` skips the test
  if the package isn't importable, so the suite runs anywhere.
- **Deterministic + offline.** No real network, DB, redis, or model downloads —
  use in-memory fakes / mock transports, or assert the API surface instead.

## What a good file contains

1. **Symbol existence** — `depcheck.assert_symbols(mod, [...])` for every symbol
   the backend references (dotted paths like `"hazmat.primitives.hashes.SHA256"`).
2. **Signatures** — `depcheck.assert_params(fn, [...])` for functions called with
   specific kwargs (this is what catches the #24560-class "kwarg dropped" bug).
3. **Behavioural contracts** — exercise the *actual* usage offline (sign+verify a
   token, `chardet.detect()` a known byte string, a MockTransport HTTP roundtrip).

`test_requests.py` is the reference exemplar.

## `depcheck` API

`load` / `try_load` / `has` / `resolve` / `assert_symbols` / `assert_callable` /
`assert_params` / `dist_version`. See `conftest.py`.

## Gotchas

- Don't `hasattr()` an object whose attribute is a property that executes —
  use `dir(instance)` / class introspection (a blank object's getter can raise).
- Rust-backed classes (cryptography AEAD, etc.) have no introspectable
  signature — pin them behaviourally, not with `assert_params`.

## Running

```bash
pytest unit/deps/                    # all dependency contracts
pytest unit/deps/test_redis.py       # one dependency
pytest -m depcheck                   # the whole class, anywhere
```

## Feature smoke tests (`integration/deps/`)

The contracts here pin a library's API. `integration/deps/` drives the same
libraries through the Open WebUI feature that uses them, over HTTP on a
scratch instance, so a bump that keeps the API but changes the behaviour still
fails. Both carry the `depcheck` marker; run both after a bump:

```bash
OPEN_WEBUI_SOURCE_DIR=../open-webui/backend pytest -m depcheck unit/deps integration/deps
```

| Library | Feature smoke test |
|---|---|
| pypdf, docx2txt, unstructured with python-pptx, pandas, openpyxl, xlrd and msoffcrypto, pypandoc, beautifulsoup4, chardet, ftfy | `integration/deps/test_document_extraction.py` (one upload per format) |
| rapidocr with onnxruntime, OpenCV and Pillow | `integration/deps/test_document_extraction.py` (PDF image OCR) |
| langchain text splitters, tiktoken, rank-bm25 | `integration/deps/test_chunking_and_search.py` |
| Pillow | `integration/deps/test_image_validation.py` |
| bcrypt, argon2-cffi, PyJWT, pytz, authlib, itsdangerous, cryptography | `integration/deps/test_auth_stack.py` |
| starlette-compress, Brotli, zstandard, Markdown, beautifulsoup4, brotlicffi, python-socketio, pycrdt | `integration/deps/test_transport_stack.py` |
| python-mimeparse, aiofiles, pydub | `integration/deps/test_audio_stack.py` |
| mcp, validators, black, opentelemetry, requests | `integration/deps/test_outbound_stack.py` |
| redis | `integration/security/test_signin_session_expiry_and_revocation_fallback.py` (a signed-out token is refused) |
| aiocache | `integration/security/test_cache_key_builder.py` (a repeat model listing never reaches the provider) |
| requests (Tika) | `integration/retrieval/test_v0114_source_text_and_docling.py` (a binary upload is extracted by Tika) |
| sqlalchemy, aiosqlite, alembic, fastapi, uvicorn, httpx, aiohttp | every instance boot and request |

The .rst, .epub and .odt uploads skip without a `pandoc` binary, pydub skips
without ffmpeg, and the token splitter skips when its tiktoken BPE file is not
cached, since none of them may be downloaded during a run. Libraries for
services the integration suite has no local stand-in for (Postgres, Qdrant,
Milvus, Weaviate, Elasticsearch, OpenSearch, S3, Azure, Google Cloud, Oracle,
Pinecone, LDAP) keep their unit contracts only.
