# Open WebUI Test Suite

External regression suite for [Open WebUI](https://github.com/open-webui/open-webui).

| Layer | Dir | Drives | Needs |
|-------|-----|--------|-------|
| **Integration** | `integration/` | the HTTP API of a scratch instance the suite boots | the backend checkout |
| **E2E** | `e2e/` | the built frontend of that instance, in Chromium | the checkout, built (`npm run build`), and Playwright |
| **Unit** | `unit/` | backend modules imported from the checkout, or its source audited | the backend checkout |
| **Frontend** | `frontend/` | `src/lib` modules imported into vitest | the checkout's `node_modules` |

Upstream refactors its internals daily and changes its API and UI rarely, so a bug is pinned as far out as it can be seen: over HTTP when the symptom shows there and in the browser as well when it shows in the UI. Unit tests are never written: every new test is an integration or a browser test. A new check goes into an existing test or module first, so CI does not grow slower; a new module only when nothing fits. The ones left in `unit/` only shrink as twins further out replace them. [`docs/regression-test-contract.md`](docs/regression-test-contract.md) has the rules; read it before adding or repairing a test.

---

## Layout

```
tests/
├── conftest.py            # loads the harness fixtures; drift label and skeptic note on failures
├── harness/               # scratch instances, the scripted model provider, accounts, local services
├── integration/<area>/    # httpx tests against the scratch instance, by subsystem
├── e2e/<area>/            # Playwright tests against the same instance, by subsystem
├── unit/<area>/           # source-level tests that nothing further out can see
├── frontend/              # vitest over src/lib
├── utils/                 # browser helpers (chat_ui.py: send, read replies)
├── scripts/               # e2e_instance.py (a manual instance), junit_summary.py (CI), seed_upgrade_data.py
└── docs/                  # the test contract
```

A twin of a unit test lives at the same `<area>/test_<name>.py` path under `integration/` or `e2e/`.

---

## Setup

Python 3.11+, with **uv** (`uv sync --extra dev`) or **pip** (`pip install -e ".[dev]"`). The suite imports and boots the backend from a checkout, so the checkout's own dependencies go into the same environment. They track the ref under test, which is why they are not in `uv.lock`:

```bash
uv pip install -r ../open-webui/backend/requirements.txt   # or: pip install -r ...
playwright install chromium                                 # for e2e/
(cd ../open-webui && npm ci --force && npm run build)       # for e2e/: the frontend it serves
```

The checkout is found through `OPEN_WEBUI_SOURCE_DIR` (pointing at `.../open-webui/backend`), else as a sibling `open-webui/` next to this repo. The frontend build is `build/` next to that backend, or `OPEN_WEBUI_BUILD_DIR`. The postgres migration tests also want `pgserver` (in the `dev` extra) and skip without it.

The Open Terminal tests start the real `open-terminal` from PyPI. Install it into an environment of its own, so its dependencies stay out of the backend's, and name its executable in `OPEN_TERMINAL_BIN` (or put it on `PATH`); the tests skip without it:

```bash
uv venv ../open-terminal && uv pip install --python ../open-terminal open-terminal
export OPEN_TERMINAL_BIN=../open-terminal/bin/open-terminal
```

The Apps tests (`integration/apps`, `e2e/apps`) run the reference app in `harness/reference_app`, which uses every feature of the [oix](https://github.com/open-webui/oix) SDK. Give it an environment of its own with `oix[fastapi]` and `uvicorn`, and name its interpreter in `OIX_PYTHON`; the tests skip without it:

```bash
git clone https://github.com/open-webui/oix ../oix
uv venv ../oix-env && uv pip install --python ../oix-env "../oix[fastapi]" uvicorn
export OIX_PYTHON=../oix-env/bin/python
```

The Terminals tests (`integration/terminals`, `e2e/terminals`) run the real [Terminals](https://github.com/open-webui/terminals) orchestrator on Docker, which starts an Open Terminal container for every terminal's machine. Build both images from their release tags and name them in `TERMINALS_SERVICE_IMAGE` and `OPEN_TERMINAL_IMAGE`; nothing is pulled while the tests run, and they skip without the images or a working Docker daemon (`OWUI_TEST_REQUIRE_TERMINALS=1` turns that skip into a failure):

```bash
git clone --branch v0.4.1 https://github.com/open-webui/terminals ../terminals
git clone --branch v0.17.7 https://github.com/open-webui/open-terminal ../open-terminal-source
docker build -t owui-tests/terminals:v0.4.1 ../terminals
docker build -f ../open-terminal-source/Dockerfile.slim -t owui-tests/open-terminal:slim ../open-terminal-source
printf 'FROM owui-tests/open-terminal:slim\nWORKDIR /home/user\n' | docker build -t owui-tests/open-terminal:v0.17.7 -
export TERMINALS_SERVICE_IMAGE=owui-tests/terminals:v0.4.1 OPEN_TERMINAL_IMAGE=owui-tests/open-terminal:v0.17.7
```

The slim image starts in `/app`, which its user cannot write, so the last build moves it to the home the orchestrator mounts, where the full image starts. Each run's containers and network carry a label of their own and are removed when it ends; `python -m harness.terminals` removes what a killed run left behind.

### Frontend suite

`frontend/` imports `src/lib` modules straight out of the checkout, so the checkout needs `npm ci` and a `svelte-kit sync` first (its `tsconfig.json` extends the generated one).

```bash
(cd ../open-webui && npm ci --force && npx svelte-kit sync)
cd frontend && npm ci && npx vitest run
```

---

## Running

```bash
pytest integration e2e                   # boots the scratch instance once, about 15 s
pytest integration/security              # one area
pytest unit                              # set DATA_DIR and STATIC_DIR to scratch paths, see the contract
pytest -k collection_access              # name filter
pytest -m regression                     # only issue/PR-pinned regressions
pytest -m "not slow"                     # skip the modules that boot extra instances
pytest -m terminal                       # every terminal module, wherever it lives
```

A run against the latest `dev` is expected to show **red for any regression whose fix isn't merged yet**. Each failing test names the issue or PR that turns it green. Failures that stop before any assertion (an import, a renamed attribute, a changed signature) are listed apart as likely upstream drift.

A failing browser test leaves a Playwright trace per browser in `test-results/` (`playwright show-trace <file>.zip`). Set `OPEN_WEBUI_LOG_DIR` to keep each scratch instance's server log.

### Postgres and Redis

The scratch instances run on SQLite with in-process state by default. Two switches move every instance the suite boots onto the backends a multi-worker deployment uses:

```bash
OWUI_TEST_DATABASE=postgres pytest integration e2e   # a database of its own per instance, on one embedded Postgres (pgserver)
OWUI_TEST_REDIS=1 pytest integration e2e             # a redis-server of its own per instance, for sockets, config sync and tasks
```

They combine. The Redis switch needs `redis-server` on `PATH` and skips without it. An instance a test gives its own database or Redis (a prepared data directory, a Redis stand-in) keeps it. A test that only holds on one backend skips in the other mode and names why.

### Slow machine

Every browser page runs with its CPU throttled fourfold by default (`OWUI_TEST_SLOW=<n>` for n-fold, `OWUI_TEST_SLOW=0` for full speed), and the scripted provider waits as much longer before each reply when the run includes browser tests. A run without them has no page to wait for and keeps the provider at full speed. A test that races the page, which a fast machine wins and a busy CI runner loses, fails locally instead of on CI.

### Plain HTTP

Every browser page opens the instance by a host name that maps to loopback but is not one (`owui-lan.test`), so it is not a secure context, as for a user who opens Open WebUI over plain HTTP by address (a LAN address, Docker without TLS). The browser then offers no `navigator.clipboard`, `crypto.randomUUID` or `crypto.subtle`; test code reads and writes the clipboard through `utils/clipboard.py`. A test of a feature that the browser itself offers only in a secure context (the microphone, the location, encrypted chat) carries `pytest.mark.secure_context` and opens the loopback address.

### Preloaded boots

On Linux every scratch instance after the first is forked from a process that has already imported torch, transformers and the rest of the heavy libraries Open WebUI loads at startup, which takes about half off each boot. A forked instance has the same environment, modules, threads and open files as a plain start. An instance whose settings change what those libraries read when they load (`HF_HOME`, `TZ`, `PYTHONPATH`) starts the plain way. Forked instances share one hash seed, Arrow's memory pool runs without its background purge thread, and the process lists the preload server as its parent and command line. `OWUI_TEST_PRELOAD=0` starts every instance the plain way.

### Coverage

`OWUI_TEST_COVERAGE=1 pytest integration e2e` measures which backend lines the run executes (needs `pip install coverage`). Every scratch instance runs under coverage, as does the pytest process itself, so `pytest unit` can be measured the same way. At the end of the session the data is combined into `coverage-report/html/index.html` and `coverage-report/coverage.json` (`OWUI_TEST_COVERAGE_DIR` moves it), and the total is printed. Each session starts from empty data; to merge two sessions, give them separate directories and run `coverage combine --keep` over their `data/` folders.

`scripts/e2e_instance.py --clone ../open-webui` starts a standalone instance with the two seeded accounts, for poking at by hand.

`integration/migrations/test_upgrade_from_release.py` boots the checkout on data sets made by older releases (`integration/migrations/upgrade_data/`). To regenerate them, or add a release, run `python scripts/seed_upgrade_data.py --clone ../open-webui v0.9.6 v0.10.2 v0.11.4` (with `pgserver` installed for the Postgres sets) and commit the new files.

### CI

`.github/workflows/regression.yml` is called by Open WebUI's release pull requests with the ref under test. It runs the unit, integration, browser and vitest suites in parallel jobs; each writes a summary of failures to the job page and uploads its report, server logs and traces. A further job runs the integration suite with both switches on. It and the browser job report without gating a release for now.

The integration and browser suites run in parallel shards, split by `pytest-split` along the timings CI measured for each mode: `durations/integration.json`, `durations/integration-postgres-redis.json` and `durations/e2e.json`. After a green run of `main` dispatched in this repository, the run's timings are folded into these files and committed back to `main`. A test missing from a file still runs and counts at the average. To split a local run the same way, pass the mode's file: `pytest integration --splits 10 --group 1 --splitting-algorithm duration_based_chunks --durations-path durations/integration.json`.

`.github/workflows/terminals.yml` runs every terminal module (`-m terminal`) of the integration and browser suites on demand, with the real Terminals orchestrator, the integration ones on SQLite and again on Postgres and Redis: `gh workflow run terminals.yml -f open-webui-ref=dev`. It builds the orchestrator from `terminals-ref` and the Open Terminal image from `open-terminal-ref` (a PyPI version or a ref; both default to the latest release), builds Open WebUI the way the regression workflow does and writes the same summaries and reports. Other workflows can call it.

---

## Markers

Registered in `pyproject.toml` (`--strict-markers` is on). Combine with `-m "<expr>"`.

| Marker | Meaning |
|--------|---------|
| `regression` | pinned to a specific upstream issue/PR; fails only if that bug returns |
| `journey` | broad baseline coverage of a feature, not pinned to one issue |
| `slow` | long-running (extra instance boots, real postgres) |
| `api` | API-level via `httpx` |
| `requires_source` | needs the backend checkout |
| `requires_browser` | needs Playwright and a built frontend |
| `requires_postgres` | needs `pgserver` |
| `public` / `auth_required` / `admin_required` | page-access scope in the browser suite |
| `depcheck` | dependency contract test under `unit/deps/`, or its feature smoke test under `integration/deps/` |
| `terminal` | a test of a terminal module; set at collection for every test under a `terminals/` directory or with `terminal` or `tether` in its file name. A test that connects a terminal in a module named for something else does not carry it |

Tests skip when what they need is absent, so the whole suite runs anywhere and only the runnable part executes.

---

## Fixtures

**`harness/fixtures.py`** (everywhere)

| Fixture | Gives you |
|---------|-----------|
| `instance` | the shared scratch instance (`.client()`, `.log_since()`, `.data_dir`) |
| `upstream` | its scripted model provider, reset per test: `queue(reply.text(...), reply.tool_call(...), reply.error(...))`, `chat_requests()` |
| `admin` / `user` / `make_user()` | accounts, each with its own token and client |
| `preserve(...)` | restores the global settings a test changes |
| `instance_with({...})` | a further instance for settings that only exist as environment variables; `.restart()` stops it and starts it again on the same address and data |
| `package_instance_with({...})` | the same, kept for a whole test package whose modules share the env set |
| `refuses_unknown_names` | skips a c-ares test of a name that does not resolve where the machine's DNS server keeps c-ares from refusing one at once |
| `listener` | a local HTTP service for the instance to call, recording what it gets |
| `open_terminal` | a real Open Terminal for the session (`OPEN_TERMINAL_BIN`), skipped where it is not installed |
| `terminal_orchestrator` | a Terminals orchestrator stand-in that gives each terminal context (the shared one, `chat:<id>` per chat) a real Open Terminal of its own |
| `terminals` | the real Terminals orchestrator on Docker (`TERMINALS_SERVICE_IMAGE`, `OPEN_TERMINAL_IMAGE`), one for the session; the templates and containers a test made are removed after it. Skipped without the images or Docker |
| `connect_terminals(*grants)` | the admin connects that orchestrator and gives the connection those grants; returns the connection, for `create_template` and the other helpers in `harness.terminals` |
| `shared_terminal(*people, **data)` | the same, plus a terminal the admin creates on the connection and shares with those people; returns the terminal's id |

`harness.chat.ask(client, "text")` sends a message the way the web client does and returns the stored reply.

Local fakes, each described in its module docstring: `audio_engine` (speech and transcription), `realtime_provider` (a voice provider for realtime calls over OpenAI's Realtime API), `model_runners` (llama.cpp, LM Studio), `ldap_server`, `mfa` (an authenticator app, accounts enrolled in it and the admin's switches that require one), `external_knowledge` (Qdrant), `milvus_server` (Milvus over gRPC), `youtube` (YouTube behind the transcript loader's proxy), `duckduckgo` (DuckDuckGo's search page behind the same kind of proxy), `search_apis` (the APIs of the other web search engines, those with a fixed host behind a proxy of the same kind), `otlp_collector` (an OpenTelemetry trace collector over gRPC), `weaviate_server` (Weaviate over REST and gRPC), `local_voices` (tiny local text-to-speech voices, one per speaker), `outlook_message` (an Outlook .msg built in memory) `mcp_oauth` (an OAuth-protected MCP server), `openapi_server` (an OpenAPI tool server on a `listener`), `http_proxy` (a forwarding proxy that knows names the instance cannot resolve) and `host_names` (a local service by `localhost`, an IPv6-only or another hosts-file name, and a name no resolver answers), plus model management and chat in `ollama_provider`, a Responses-API connection in `responses_provider`, a browser tab answering for a direct connection in `direct_connection`, a provider the browser itself calls for a user's own connection in `browser_provider`, token refresh in `oidc_provider`, live note edits in `socket_client` and an install without an optional package in `missing_packages`. `harness.access` sends one route as the owner, a stranger, a reader, a writer and the admin of a shared resource, and tells when a refused write still changed the owner's copy.

**`e2e/conftest.py`**: `page_for(actor)` opens a signed-in page in a browser of its own; `page`, `authenticated_page` and `admin_page` as before.

**`unit/conftest.py`**: `open_webui_backend` (the checkout path) and module loaders such as `builtin_tools_module`.

---

## Linting

```bash
ruff check .
ruff format .
```

---

## License

BSD-3-Clause, see LICENSE.
