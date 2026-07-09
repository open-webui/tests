r"""Tests for the async rework of DoclingLoader.

Features under test:
- DoclingLoader submits to the async Docling Serve API and long-polls for
  the result, instead of blocking on the old synchronous endpoint
- Configurable overall timeout (DOCLING_SERVE_TIMEOUT)
- Optional status_callback hook for queue-position reporting
- load_from_task_id() to resume an already-submitted task
- Loader.aload() wires timeout/status_callback into DoclingLoader for the
  'docling' engine
- DoclingLoader also exposes a true async path (aload()/aload_from_task_id(),
  backed by aiohttp + asyncio.sleep) alongside the sync one, and Loader.aload()
  dispatches to it directly instead of routing DoclingLoader through
  asyncio.to_thread(self.load, ...)
- status_callback may itself be sync or async; the async submit/poll helpers
  must await it via _notify_status_async() if it returns an awaitable,
  instead of calling it as a plain function (which would silently produce
  an un-awaited coroutine that never runs)

These are source-level tests: they read the actual backend source and
regex-match against it, rather than importing open_webui.retrieval.loaders.main
directly, because that module pulls in a heavy dependency tree (typer,
langchain_community, azure.identity, ftfy, ...) not present in this repo's
test environment. Same approach as test_files_knowledge_status.py.

Method-boundary lookaheads use `\n    (?:async\s+)?def\s+\w+` rather than
`\n    def\s+\w+` — DoclingLoader interleaves sync and async methods, and a
lookahead that only recognizes plain `def` would run past an `async def`
boundary and sweep a later method's body into the match.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_source

# `\n    def foo` would stop at a sync method but run straight past an
# `async def`, sweeping subsequent method bodies into the match. Every
# boundary lookahead in this file must use this instead.
_NEXT_METHOD = r"\n    (?:async\s+)?def\s+\w+"


def _loaders_main_src(open_webui_backend: Path) -> str:
    loaders_main = open_webui_backend / "open_webui" / "retrieval" / "loaders" / "main.py"
    assert loaders_main.is_file(), f"{loaders_main} not found"
    return loaders_main.read_text(encoding="utf-8")


def _docling_loader_class_body(src: str) -> str:
    """Extract the DoclingLoader class body (up to the next top-level class)."""
    match = re.search(
        r"class\s+DoclingLoader\s*:\s*(.*?)(?=\nclass\s+\w+|\Z)",
        src,
        re.MULTILINE | re.DOTALL,
    )
    assert match, "DoclingLoader class not found in retrieval/loaders/main.py"
    return match.group(1)


def _loader_class_body(src: str) -> str:
    """Extract the Loader class body. It's the last class in the file."""
    match = re.search(
        r"class\s+Loader\s*:\s*(.*?)(?=\nclass\s+\w+|\Z)",
        src,
        re.MULTILINE | re.DOTALL,
    )
    assert match, "Loader class not found in retrieval/loaders/main.py"
    return match.group(1)


def _method_body(class_body: str, method_name: str) -> str:
    """Extract a method's body from a class body, sync or async, stopping at
    the next method definition of either kind (see _NEXT_METHOD)."""
    match = re.search(
        rf"(?:async\s+)?def\s+{re.escape(method_name)}\b(.*?)(?={_NEXT_METHOD}|\Z)",
        class_body,
        re.DOTALL,
    )
    assert match, f"{method_name} method not found"
    return match.group(1)


# =============================================================================
# Constructor: accepts timeout and status_callback
# =============================================================================


def test_docling_loader_constructor_accepts_timeout_and_status_callback(
    open_webui_backend: Path,
) -> None:
    """DoclingLoader.__init__ must accept optional timeout and status_callback
    parameters, so callers can bound the wait and observe queue progress."""
    src = _loaders_main_src(open_webui_backend)
    body = _docling_loader_class_body(src)

    init_match = re.search(r"def\s+__init__\s*\(([^)]*)\)", body)
    assert init_match, "DoclingLoader.__init__ not found"
    sig = init_match.group(1)

    assert "timeout" in sig, (
        "DoclingLoader.__init__ doesn't accept a timeout parameter. "
        "Conversions can't be bounded and will wait forever."
    )
    assert "status_callback" in sig, (
        "DoclingLoader.__init__ doesn't accept a status_callback parameter. "
        "Callers have no way to observe queue position while waiting."
    )


# =============================================================================
# Submit: uses the async endpoint, extracts task_id, reports via callback
# =============================================================================


def test_docling_loader_submits_to_async_endpoint(open_webui_backend: Path) -> None:
    """DoclingLoader must POST to the async convert endpoint
    (/v1/convert/file/async), not the old blocking /v1/convert/file."""
    src = _loaders_main_src(open_webui_backend)
    body = _docling_loader_class_body(src)

    assert "/v1/convert/file/async" in body, (
        "DoclingLoader doesn't submit to /v1/convert/file/async. "
        "Still using the old synchronous, unbounded endpoint."
    )


def test_docling_loader_submit_raises_without_task_id(open_webui_backend: Path) -> None:
    """The submit step must fail loudly if Docling Serve doesn't return a
    task_id, instead of silently proceeding to poll on None."""
    src = _loaders_main_src(open_webui_backend)
    body = _docling_loader_class_body(src)

    submit_match = re.search(
        r"def\s+_submit_file\b(.*?)(?=\n    (?:async\s+)?def\s+\w+|\Z)", body, re.DOTALL
    )
    assert submit_match, "_submit_file method not found"
    submit_body = submit_match.group(1)

    raises_without_task_id = re.search(r"if\s+not\s+task_id\s*:\s*\n\s*raise", submit_body)
    assert raises_without_task_id, (
        "_submit_file doesn't raise when task_id is missing from the response. "
        "Would proceed to poll with an invalid task_id."
    )


def test_docling_loader_submit_invokes_status_callback(open_webui_backend: Path) -> None:
    """After a successful submit, the status_callback (if provided) must be
    called with the task_id and queue position."""
    src = _loaders_main_src(open_webui_backend)
    body = _docling_loader_class_body(src)

    submit_match = re.search(
        r"def\s+_submit_file\b(.*?)(?=\n    (?:async\s+)?def\s+\w+|\Z)", body, re.DOTALL
    )
    assert submit_match, "_submit_file method not found"
    submit_body = submit_match.group(1)

    calls_callback = re.search(
        r"if\s+self\.status_callback\s*:\s*\n\s*self\.status_callback\(", submit_body
    )
    assert calls_callback, (
        "_submit_file doesn't invoke self.status_callback after submitting. "
        "Callers can't learn the task_id/queue position right after submit."
    )
    assert "task_id" in submit_body and "task_position" in submit_body, (
        "_submit_file doesn't extract both task_id and task_position from the response."
    )


# =============================================================================
# Poll: long-polls, respects timeout, surfaces failure, reports progress
# =============================================================================


def _poll_method_body(body: str) -> str:
    poll_match = re.search(
        r"def\s+_poll_task_until_done\b(.*?)(?=\n    (?:async\s+)?def\s+\w+|\Z)", body, re.DOTALL
    )
    assert poll_match, "_poll_task_until_done method not found"
    return poll_match.group(1)


def test_docling_loader_polls_status_endpoint(open_webui_backend: Path) -> None:
    """Polling must hit the Docling Serve status endpoint for the task."""
    src = _loaders_main_src(open_webui_backend)
    body = _docling_loader_class_body(src)
    poll_body = _poll_method_body(body)

    assert "/v1/status/poll/" in poll_body, (
        "_poll_task_until_done doesn't call the /v1/status/poll/{task_id} endpoint."
    )


def test_docling_loader_poll_respects_overall_timeout(open_webui_backend: Path) -> None:
    """When self.timeout is set, polling must give up and raise once the
    deadline is exceeded, instead of waiting forever."""
    src = _loaders_main_src(open_webui_backend)
    body = _docling_loader_class_body(src)
    poll_body = _poll_method_body(body)

    computes_deadline = re.search(r"deadline\s*=.*self\.timeout", poll_body)
    assert computes_deadline, (
        "_poll_task_until_done doesn't compute a deadline from self.timeout. "
        "DOCLING_SERVE_TIMEOUT would have no effect."
    )

    raises_on_expiry = re.search(r"remaining\s*<=\s*0\s*:\s*\n\s*raise", poll_body)
    assert raises_on_expiry, (
        "_poll_task_until_done doesn't raise once the deadline has passed. "
        "A configured timeout would never actually stop the wait."
    )


def test_docling_loader_poll_raises_on_task_failure(open_webui_backend: Path) -> None:
    """A task_status of 'failure' must raise with the server's error message,
    not be silently treated as still-pending."""
    src = _loaders_main_src(open_webui_backend)
    body = _docling_loader_class_body(src)
    poll_body = _poll_method_body(body)

    handles_failure = re.search(
        r"task_status\s*==\s*[\"']failure[\"'].*?\n\s*.*raise", poll_body, re.DOTALL
    )
    assert handles_failure, (
        "_poll_task_until_done doesn't raise when task_status is 'failure'. "
        "A failed conversion would poll forever or be misreported as success."
    )


def test_docling_loader_poll_returns_on_task_success(open_webui_backend: Path) -> None:
    """A task_status of 'success' must end the poll loop and return the
    status payload for retrieval."""
    src = _loaders_main_src(open_webui_backend)
    body = _docling_loader_class_body(src)
    poll_body = _poll_method_body(body)

    returns_on_success = re.search(
        r"task_status\s*==\s*[\"']success[\"']\s*:\s*\n\s*return", poll_body
    )
    assert returns_on_success, (
        "_poll_task_until_done doesn't return when task_status is 'success'. "
        "Would keep polling after the task is already done."
    )


def test_docling_loader_poll_reports_queue_position_via_callback(
    open_webui_backend: Path,
) -> None:
    """While polling, task_position updates should be pushed through
    status_callback so the UI can show live queue progress."""
    src = _loaders_main_src(open_webui_backend)
    body = _docling_loader_class_body(src)
    poll_body = _poll_method_body(body)

    reports_progress = re.search(
        r"if\s+self\.status_callback\s+and\s+status_data\.get\([\"']task_position[\"']\)",
        poll_body,
    )
    assert reports_progress, (
        "_poll_task_until_done doesn't forward task_position updates to "
        "status_callback while polling. Queue position would appear frozen "
        "after the initial submit."
    )


def test_docling_loader_poll_retries_on_client_timeout(open_webui_backend: Path) -> None:
    """A client-side requests.Timeout while long-polling must be retried,
    not treated as a fatal conversion failure — the server may simply not
    have had anything new to report within the poll window."""
    src = _loaders_main_src(open_webui_backend)
    body = _docling_loader_class_body(src)
    poll_body = _poll_method_body(body)

    retries_on_timeout = re.search(
        r"except\s+requests\.Timeout\s*:\s*\n.*?continue", poll_body, re.DOTALL
    )
    assert retries_on_timeout, (
        "_poll_task_until_done doesn't retry on requests.Timeout. "
        "A single slow poll response would abort the whole conversion."
    )


# =============================================================================
# Retrieve + format
# =============================================================================


def test_docling_loader_retrieves_result_endpoint(open_webui_backend: Path) -> None:
    """After a successful poll, the result must be fetched from the
    Docling Serve result endpoint."""
    src = _loaders_main_src(open_webui_backend)
    body = _docling_loader_class_body(src)

    retrieve_match = re.search(
        r"def\s+_retrieve_result\b(.*?)(?=\n    (?:async\s+)?def\s+\w+|\Z)", body, re.DOTALL
    )
    assert retrieve_match, "_retrieve_result method not found"
    retrieve_body = retrieve_match.group(1)

    assert "/v1/result/" in retrieve_body, (
        "_retrieve_result doesn't call the /v1/result/{task_id} endpoint."
    )


def test_docling_loader_format_result_extracts_md_content(open_webui_backend: Path) -> None:
    """format_result must pull md_content out of the Docling Serve result
    payload and wrap it in a langchain Document."""
    src = _loaders_main_src(open_webui_backend)
    body = _docling_loader_class_body(src)

    format_match = re.search(
        r"def\s+format_result\b(.*?)(?=\n    (?:async\s+)?def\s+\w+|\Z)", body, re.DOTALL
    )
    assert format_match, "format_result method not found"
    format_body = format_match.group(1)

    assert "md_content" in format_body, (
        "format_result doesn't read md_content from the result payload."
    )
    assert "Document(" in format_body, (
        "format_result doesn't wrap the extracted text in a Document."
    )


# =============================================================================
# Orchestration: load() and load_from_task_id()
# =============================================================================


def test_docling_loader_load_orchestrates_submit_poll_retrieve(
    open_webui_backend: Path,
) -> None:
    """load() must call submit, then poll, then retrieve, then format — in
    that order — to actually perform a full conversion."""
    src = _loaders_main_src(open_webui_backend)
    body = _docling_loader_class_body(src)

    load_match = re.search(
        r"def\s+load\s*\(self\)(.*?)(?=\n    (?:async\s+)?def\s+\w+|\Z)", body, re.DOTALL
    )
    assert load_match, "load() method not found"
    load_body = load_match.group(1)

    assert "_submit_file" in load_body, "load() doesn't call _submit_file"
    assert "_poll_task_until_done" in load_body, "load() doesn't call _poll_task_until_done"
    assert "_retrieve_result" in load_body, "load() doesn't call _retrieve_result"

    submit_pos = load_body.find("_submit_file")
    poll_pos = load_body.find("_poll_task_until_done")
    retrieve_pos = load_body.find("_retrieve_result")
    assert submit_pos < poll_pos < retrieve_pos, (
        "load() doesn't call submit -> poll -> retrieve in the correct order."
    )


def test_docling_loader_has_load_from_task_id_for_resume(open_webui_backend: Path) -> None:
    """load_from_task_id() must exist and skip re-submission, going straight
    to poll -> retrieve -> format — used to resume an in-flight task (e.g.
    after a server restart) without re-uploading the file."""
    src = _loaders_main_src(open_webui_backend)
    body = _docling_loader_class_body(src)
    resume_body = _method_body(body, "load_from_task_id")

    assert "_submit_file" not in resume_body, (
        "load_from_task_id re-submits the file instead of resuming the "
        "existing task_id. Would re-upload and create a duplicate task."
    )
    assert "_poll_task_until_done" in resume_body and "_retrieve_result" in resume_body, (
        "load_from_task_id doesn't poll and retrieve the existing task."
    )


# =============================================================================
# Config wiring: DOCLING_SERVE_TIMEOUT is threaded through end to end
# =============================================================================


def test_docling_serve_timeout_defined_in_config(open_webui_backend: Path) -> None:
    """DOCLING_SERVE_TIMEOUT must be defined in config.py, sourced from the
    env var, defaulting to None (= wait indefinitely, unchanged behavior)."""
    config_py = open_webui_backend / "open_webui" / "config.py"
    src = config_py.read_text(encoding="utf-8")

    defines_setting = re.search(
        r"DOCLING_SERVE_TIMEOUT\s*=.*os\.environ\.get\([\"']DOCLING_SERVE_TIMEOUT[\"']\)",
        src,
        re.DOTALL,
    )
    assert defines_setting, (
        "config.py doesn't define DOCLING_SERVE_TIMEOUT from the DOCLING_SERVE_TIMEOUT env var."
    )


def test_docling_serve_timeout_registered_in_default_config(open_webui_backend: Path) -> None:
    """DOCLING_SERVE_TIMEOUT must be registered under the rag.* namespace in
    DEFAULT_CONFIG so it's persisted/overridable via the Config store."""
    config_py = open_webui_backend / "open_webui" / "config.py"
    src = config_py.read_text(encoding="utf-8")

    registered = re.search(r"['\"]rag\.docling_serve_timeout['\"]\s*:\s*DOCLING_SERVE_TIMEOUT", src)
    assert registered, (
        "DOCLING_SERVE_TIMEOUT isn't registered in DEFAULT_CONFIG under "
        "'rag.docling_serve_timeout'."
    )


def test_docling_serve_timeout_in_loader_config_keys(open_webui_backend: Path) -> None:
    """retrieval/utils.py's LOADER_CONFIG_KEYS must map DOCLING_SERVE_TIMEOUT
    to its rag.* config key, so build_loader_from_config actually resolves
    it into loader.kwargs."""
    utils_py = open_webui_backend / "open_webui" / "retrieval" / "utils.py"
    src = utils_py.read_text(encoding="utf-8")

    mapped = re.search(
        r"['\"]DOCLING_SERVE_TIMEOUT['\"]\s*:\s*['\"]rag\.docling_serve_timeout['\"]", src
    )
    assert mapped, (
        "LOADER_CONFIG_KEYS doesn't map DOCLING_SERVE_TIMEOUT. The loader "
        "would never actually receive the configured timeout."
    )


def test_docling_serve_timeout_exposed_in_rag_config_api(open_webui_backend: Path) -> None:
    """The RAG config router must expose DOCLING_SERVE_TIMEOUT on ConfigForm
    and apply it in update_rag_config, so it's settable via the admin API."""
    retrieval_router = open_webui_backend / "open_webui" / "routers" / "retrieval.py"
    src = retrieval_router.read_text(encoding="utf-8")

    in_config_form = re.search(r"DOCLING_SERVE_TIMEOUT\s*:\s*int\s*\|\s*None\s*=\s*None", src)
    assert in_config_form, "ConfigForm doesn't declare a DOCLING_SERVE_TIMEOUT field."

    applies_update = re.search(
        r"config\.DOCLING_SERVE_TIMEOUT\s*=\s*\(\s*\n\s*form_data\.DOCLING_SERVE_TIMEOUT", src
    )
    assert applies_update, (
        "update_rag_config doesn't apply form_data.DOCLING_SERVE_TIMEOUT to "
        "config.DOCLING_SERVE_TIMEOUT. Admins could set it but it would "
        "never take effect."
    )


# =============================================================================
# Loader dispatch: engine='docling' wires timeout + status_callback through
# =============================================================================


def test_loader_dispatch_passes_timeout_to_docling_loader(open_webui_backend: Path) -> None:
    """Loader.aload()'s 'docling' engine branch must read DOCLING_SERVE_TIMEOUT
    from kwargs and pass it into DoclingLoader's timeout parameter."""
    src = _loaders_main_src(open_webui_backend)

    dispatch_match = re.search(
        r"elif\s+self\.engine\s*==\s*['\"]docling['\"].*?(?=\n\s*elif\s+self\.engine|\Z)",
        src,
        re.DOTALL,
    )
    assert dispatch_match, "docling engine dispatch branch not found in Loader.aload()"
    dispatch_body = dispatch_match.group(0)

    reads_timeout = re.search(
        r"self\.kwargs\.get\([\"']DOCLING_SERVE_TIMEOUT[\"']\)", dispatch_body
    )
    assert reads_timeout, "Loader dispatch doesn't read DOCLING_SERVE_TIMEOUT from kwargs."

    # Scope to the DoclingLoader(...) call itself so these checks don't
    # accidentally match unrelated code elsewhere in the dispatch branch.
    call_match = re.search(r"DoclingLoader\s*\((.*?)\n\s*\)", dispatch_body, re.DOTALL)
    assert call_match, "DoclingLoader(...) instantiation not found in dispatch branch."
    call_args = call_match.group(1)

    passes_timeout = re.search(r"timeout\s*=\s*docling_timeout", call_args)
    assert passes_timeout, (
        "Loader dispatch doesn't pass timeout=docling_timeout into DoclingLoader(...)."
    )

    passes_callback = re.search(
        r"status_callback\s*=\s*self\.kwargs\.get\([\"']DOCLING_STATUS_CALLBACK[\"']\)",
        call_args,
    )
    assert passes_callback, (
        "Loader dispatch doesn't pass status_callback=self.kwargs.get("
        "'DOCLING_STATUS_CALLBACK') into DoclingLoader(...). A caller-supplied "
        "callback (e.g. for status tracking) would silently be dropped."
    )


def test_loader_dispatch_timeout_is_int_coerced_defensively(open_webui_backend: Path) -> None:
    """The timeout value pulled from kwargs must be defensively coerced to
    int, since it may arrive as a string (env var / JSON config roundtrip)
    or be absent/invalid."""
    src = _loaders_main_src(open_webui_backend)

    dispatch_match = re.search(
        r"elif\s+self\.engine\s*==\s*['\"]docling['\"].*?(?=\n\s*elif\s+self\.engine|\Z)",
        src,
        re.DOTALL,
    )
    assert dispatch_match, "docling engine dispatch branch not found in Loader.aload()"
    dispatch_body = dispatch_match.group(0)

    coerces_int = re.search(r"docling_timeout\s*=\s*int\(docling_timeout\)", dispatch_body)
    assert coerces_int, (
        "Loader dispatch doesn't coerce docling_timeout to int. A string "
        "value from config would be passed straight through and break the "
        "monotonic deadline arithmetic in _poll_task_until_done."
    )
    guards_invalid = re.search(
        r"except\s*\(ValueError,\s*TypeError\)\s*:\s*\n\s*docling_timeout\s*=\s*None", dispatch_body
    )
    assert guards_invalid, (
        "Loader dispatch doesn't fall back to None on an invalid "
        "DOCLING_SERVE_TIMEOUT value; a malformed config value would crash "
        "the loader instead of degrading to 'wait indefinitely'."
    )


# =============================================================================
# True async path: aload()/aload_from_task_id(), backed by aiohttp +
# asyncio.sleep, so a slow conversion never blocks a worker thread from the
# shared asyncio.to_thread pool for its entire duration.
# =============================================================================


def test_docling_loader_async_submits_to_async_endpoint(open_webui_backend: Path) -> None:
    """_submit_file_async must POST to the same async convert endpoint as the
    sync path, via aiohttp instead of requests."""
    src = _loaders_main_src(open_webui_backend)
    body = _docling_loader_class_body(src)
    submit_body = _method_body(body, "_submit_file_async")

    assert "/v1/convert/file/async" in submit_body, (
        "_submit_file_async doesn't submit to /v1/convert/file/async."
    )


def test_docling_loader_async_submit_raises_without_task_id(open_webui_backend: Path) -> None:
    """The async submit step must fail loudly if Docling Serve doesn't return
    a task_id, same as the sync path."""
    src = _loaders_main_src(open_webui_backend)
    body = _docling_loader_class_body(src)
    submit_body = _method_body(body, "_submit_file_async")

    raises_without_task_id = re.search(r"if\s+not\s+task_id\s*:\s*\n\s*raise", submit_body)
    assert raises_without_task_id, (
        "_submit_file_async doesn't raise when task_id is missing from the response."
    )


def test_docling_loader_async_submit_routes_status_through_notify_helper(
    open_webui_backend: Path,
) -> None:
    """_submit_file_async must report task_id/task_position through
    _notify_status_async (which awaits the callback if it's awaitable), not
    by calling self.status_callback directly. A caller-supplied async
    callback invoked as a plain function would silently never run."""
    src = _loaders_main_src(open_webui_backend)
    body = _docling_loader_class_body(src)
    submit_body = _method_body(body, "_submit_file_async")

    assert "await self._notify_status_async(" in submit_body, (
        "_submit_file_async doesn't route status reporting through "
        "_notify_status_async. An async status_callback would be silently dropped."
    )
    assert "self.status_callback(" not in submit_body, (
        "_submit_file_async calls self.status_callback(...) directly instead of "
        "through _notify_status_async. An async callback would produce an "
        "un-awaited coroutine that never runs."
    )
    assert "task_id" in submit_body and "task_position" in submit_body, (
        "_submit_file_async doesn't extract both task_id and task_position from the response."
    )


def _async_poll_method_body(body: str) -> str:
    return _method_body(body, "_poll_task_until_done_async")


def test_docling_loader_async_polls_status_endpoint(open_webui_backend: Path) -> None:
    """Async polling must hit the same Docling Serve status endpoint as the
    sync path."""
    src = _loaders_main_src(open_webui_backend)
    body = _docling_loader_class_body(src)
    poll_body = _async_poll_method_body(body)

    assert "/v1/status/poll/" in poll_body, (
        "_poll_task_until_done_async doesn't call the /v1/status/poll/{task_id} endpoint."
    )


def test_docling_loader_async_poll_respects_overall_timeout(open_webui_backend: Path) -> None:
    """When self.timeout is set, async polling must give up and raise once
    the deadline is exceeded, same as the sync path."""
    src = _loaders_main_src(open_webui_backend)
    body = _docling_loader_class_body(src)
    poll_body = _async_poll_method_body(body)

    computes_deadline = re.search(r"deadline\s*=.*self\.timeout", poll_body)
    assert computes_deadline, (
        "_poll_task_until_done_async doesn't compute a deadline from self.timeout."
    )

    raises_on_expiry = re.search(r"remaining\s*<=\s*0\s*:\s*\n\s*raise", poll_body)
    assert raises_on_expiry, (
        "_poll_task_until_done_async doesn't raise once the deadline has passed."
    )


def test_docling_loader_async_poll_raises_on_task_failure(open_webui_backend: Path) -> None:
    """A task_status of 'failure' must raise in the async poll loop too, not
    be silently treated as still-pending."""
    src = _loaders_main_src(open_webui_backend)
    body = _docling_loader_class_body(src)
    poll_body = _async_poll_method_body(body)

    handles_failure = re.search(
        r"task_status\s*==\s*[\"']failure[\"'].*?\n\s*.*raise", poll_body, re.DOTALL
    )
    assert handles_failure, (
        "_poll_task_until_done_async doesn't raise when task_status is 'failure'."
    )


def test_docling_loader_async_poll_returns_on_task_success(open_webui_backend: Path) -> None:
    """A task_status of 'success' must end the async poll loop and return the
    status payload for retrieval."""
    src = _loaders_main_src(open_webui_backend)
    body = _docling_loader_class_body(src)
    poll_body = _async_poll_method_body(body)

    returns_on_success = re.search(
        r"task_status\s*==\s*[\"']success[\"']\s*:\s*\n\s*return", poll_body
    )
    assert returns_on_success, (
        "_poll_task_until_done_async doesn't return when task_status is 'success'."
    )


def test_docling_loader_async_poll_reports_queue_position_via_notify_helper(
    open_webui_backend: Path,
) -> None:
    """While polling asynchronously, task_position updates must be pushed
    through _notify_status_async (awaited), not a direct synchronous call to
    self.status_callback."""
    src = _loaders_main_src(open_webui_backend)
    body = _docling_loader_class_body(src)
    poll_body = _async_poll_method_body(body)

    reports_progress = re.search(
        r"status_data\.get\([\"']task_position[\"']\)\s+is\s+not\s+None\s*:\s*\n\s*await\s+self\._notify_status_async\(",
        poll_body,
    )
    assert reports_progress, (
        "_poll_task_until_done_async doesn't await self._notify_status_async(...) "
        "when task_position updates. Queue position updates for an async callback "
        "would be silently dropped."
    )
    assert "self.status_callback(" not in poll_body, (
        "_poll_task_until_done_async calls self.status_callback(...) directly "
        "instead of through _notify_status_async."
    )


def test_docling_loader_async_poll_retries_on_asyncio_timeout_error(
    open_webui_backend: Path,
) -> None:
    """A client-side asyncio.TimeoutError while long-polling must be retried,
    not treated as a fatal conversion failure — the aiohttp equivalent of the
    sync path's requests.Timeout handling."""
    src = _loaders_main_src(open_webui_backend)
    body = _docling_loader_class_body(src)
    poll_body = _async_poll_method_body(body)

    retries_on_timeout = re.search(
        r"except\s+asyncio\.TimeoutError\s*:\s*\n.*?continue", poll_body, re.DOTALL
    )
    assert retries_on_timeout, (
        "_poll_task_until_done_async doesn't retry on asyncio.TimeoutError. "
        "A single slow poll response would abort the whole conversion."
    )


def test_docling_loader_async_retrieves_result_endpoint(open_webui_backend: Path) -> None:
    """After a successful async poll, the result must be fetched from the
    same Docling Serve result endpoint as the sync path."""
    src = _loaders_main_src(open_webui_backend)
    body = _docling_loader_class_body(src)
    retrieve_body = _method_body(body, "_retrieve_result_async")

    assert "/v1/result/" in retrieve_body, (
        "_retrieve_result_async doesn't call the /v1/result/{task_id} endpoint."
    )


def test_docling_loader_aload_orchestrates_submit_poll_retrieve_async(
    open_webui_backend: Path,
) -> None:
    """aload() must call the async submit, then async poll, then async
    retrieve — in that order — using aiohttp end to end."""
    src = _loaders_main_src(open_webui_backend)
    body = _docling_loader_class_body(src)
    aload_body = _method_body(body, "aload")

    assert "_submit_file_async" in aload_body, "aload() doesn't call _submit_file_async"
    assert "_poll_task_until_done_async" in aload_body, (
        "aload() doesn't call _poll_task_until_done_async"
    )
    assert "_retrieve_result_async" in aload_body, "aload() doesn't call _retrieve_result_async"

    submit_pos = aload_body.find("_submit_file_async")
    poll_pos = aload_body.find("_poll_task_until_done_async")
    retrieve_pos = aload_body.find("_retrieve_result_async")
    assert submit_pos < poll_pos < retrieve_pos, (
        "aload() doesn't call submit -> poll -> retrieve in the correct order."
    )


def test_docling_loader_has_aload_from_task_id_for_resume(open_webui_backend: Path) -> None:
    """aload_from_task_id() must exist as the async counterpart to
    load_from_task_id(): skip re-submission, go straight to async poll ->
    retrieve -> format."""
    src = _loaders_main_src(open_webui_backend)
    body = _docling_loader_class_body(src)
    resume_body = _method_body(body, "aload_from_task_id")

    assert "_submit_file_async" not in resume_body, (
        "aload_from_task_id re-submits the file instead of resuming the "
        "existing task_id. Would re-upload and create a duplicate task."
    )
    assert (
        "_poll_task_until_done_async" in resume_body and "_retrieve_result_async" in resume_body
    ), "aload_from_task_id doesn't poll and retrieve the existing task asynchronously."


# =============================================================================
# _notify_status_async: status_callback may be sync or async
# =============================================================================


def test_notify_status_async_is_a_noop_without_a_callback(open_webui_backend: Path) -> None:
    """_notify_status_async must do nothing when no status_callback was
    supplied, instead of raising on None(...)."""
    src = _loaders_main_src(open_webui_backend)
    body = _docling_loader_class_body(src)
    notify_body = _method_body(body, "_notify_status_async")

    guards_none = re.search(r"if\s+not\s+self\.status_callback\s*:\s*\n\s*return", notify_body)
    assert guards_none, "_notify_status_async doesn't return early when status_callback is None."


def test_notify_status_async_awaits_awaitable_callback_results(open_webui_backend: Path) -> None:
    """_notify_status_async must check whether calling status_callback
    returned an awaitable (i.e. the callback was an async def) and await it
    if so — otherwise an async callback's body would silently never execute."""
    src = _loaders_main_src(open_webui_backend)
    body = _docling_loader_class_body(src)
    notify_body = _method_body(body, "_notify_status_async")

    checks_awaitable = re.search(r"inspect\.isawaitable\(result\)", notify_body)
    assert checks_awaitable, (
        "_notify_status_async doesn't check inspect.isawaitable(...) on the "
        "callback's return value. An async def status_callback would produce "
        "an un-awaited coroutine that silently never runs."
    )
    awaits_it = re.search(
        r"if\s+inspect\.isawaitable\(result\)\s*:\s*\n\s*await\s+result", notify_body
    )
    assert awaits_it, (
        "_notify_status_async checks isawaitable but doesn't actually await the result."
    )


def test_notify_status_async_import_present(open_webui_backend: Path) -> None:
    """The `inspect` module must be imported at module scope for
    _notify_status_async's isawaitable check to work."""
    src = _loaders_main_src(open_webui_backend)
    assert re.search(r"^import\s+inspect\s*$", src, re.MULTILINE), (
        "retrieval/loaders/main.py doesn't import inspect, but "
        "_notify_status_async needs inspect.isawaitable(...)."
    )


# =============================================================================
# Loader.aload() dispatch: DoclingLoader gets the true async path, everything
# else still goes through asyncio.to_thread
# =============================================================================


def test_loader_aload_dispatches_docling_loader_to_true_async_path(
    open_webui_backend: Path,
) -> None:
    """Loader.aload() must detect a DoclingLoader instance and call its
    aload() directly, instead of routing it through
    asyncio.to_thread(self.load, ...) like every other loader. Otherwise a
    slow Docling conversion would occupy a worker thread from the shared
    default executor for the entire submit/poll/retrieve cycle."""
    src = _loaders_main_src(open_webui_backend)
    loader_body = _loader_class_body(src)
    aload_body = _method_body(loader_body, "aload")

    branch_match = re.search(
        r"if\s+isinstance\(loader,\s*DoclingLoader\)\s*:\s*\n(.*?)\n\s*else\s*:\s*\n(.*?)(?=\n\n|\Z)",
        aload_body,
        re.DOTALL,
    )
    assert branch_match, (
        "Loader.aload() doesn't branch on isinstance(loader, DoclingLoader). "
        "DoclingLoader would still be routed through asyncio.to_thread(self.load, ...) "
        "and hold a worker thread for the entire submit/poll/retrieve cycle."
    )
    docling_branch, other_branch = branch_match.groups()

    assert "await loader.aload()" in docling_branch, (
        "The DoclingLoader branch doesn't call the true async aload() path."
    )
    assert "to_thread" not in docling_branch, (
        "The DoclingLoader branch still routes through asyncio.to_thread, defeating "
        "the point of the true async path."
    )
    assert "asyncio.to_thread(loader.load)" in other_branch, (
        "Non-Docling loaders should still go through asyncio.to_thread(loader.load), "
        "since they're synchronous and CPU/IO-bound."
    )


def test_loader_aload_builds_loader_off_the_event_loop(open_webui_backend: Path) -> None:
    """_get_loader() can do blocking file I/O (e.g. text-encoding detection
    for the docling/tika text-file fallback), so Loader.aload() must
    construct the loader via asyncio.to_thread too, not call it directly on
    the event loop."""
    src = _loaders_main_src(open_webui_backend)
    loader_body = _loader_class_body(src)
    aload_body = _method_body(loader_body, "aload")

    assert "await asyncio.to_thread(self._get_loader" in aload_body, (
        "Loader.aload() doesn't build the loader via asyncio.to_thread(self._get_loader, ...). "
        "_get_loader can perform blocking file I/O (text-encoding detection) directly "
        "on the event loop."
    )
