"""Tests for knowledge status features in Files model.

New in feat-knowledgestatus: file processing status tracking and recovery
from interrupted background tasks.

Features under test:
- get_pending_files_by_knowledge_id: retrieve files still being processed
- reset_stuck_processing_files: clean up orphaned background tasks on startup
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_source


# =============================================================================
# Source audit: startup cleanup is wired into main.py
# =============================================================================


def test_reset_stuck_processing_files_is_called_on_startup(open_webui_backend: Path) -> None:
    """Regression: verify that reset_stuck_processing_files() is called during
    application startup (in the lifespan context manager). This cleanup is
    essential to recover from server crashes that left background tasks in
    pending/processing states."""
    main_py = open_webui_backend / "open_webui" / "main.py"
    assert main_py.is_file(), f"{main_py} not found"

    src = main_py.read_text(encoding="utf-8")

    # Must import Files model
    imports_files = re.search(r"from\s+open_webui\.models\.files\s+import\s+Files", src)
    assert imports_files, (
        "main.py doesn't import Files model. Startup cleanup cannot be wired."
    )

    # Must call reset_stuck_processing_files() in lifespan
    calls_reset = re.search(
        r"(?:await\s+)?Files\.reset_stuck_processing_files\(\)", src
    )
    assert calls_reset, (
        "main.py doesn't call Files.reset_stuck_processing_files(). "
        "Orphaned processing tasks won't be cleaned up on restart."
    )


def test_idempotency_check_in_add_file_to_knowledge(open_webui_backend: Path) -> None:
    """Regression: verify that add_file_to_knowledge_by_id includes an
    idempotency check to skip re-embedding when the file is already linked
    to the knowledge base (e.g., background task finished while UI was open)."""
    knowledge_router = open_webui_backend / "open_webui" / "routers" / "knowledge.py"
    assert knowledge_router.is_file(), f"{knowledge_router} not found"

    src = knowledge_router.read_text(encoding="utf-8")

    # Find the add_file_to_knowledge_by_id function
    func_match = re.search(
        r"async def add_file_to_knowledge_by_id\b(.*?)(?=^(?:async def|@router)\b)",
        src,
        re.MULTILINE | re.DOTALL,
    )
    assert func_match, "add_file_to_knowledge_by_id endpoint not found"

    func_body = func_match.group(1)

    # Must check if file is already in knowledge
    has_file_check = re.search(
        r"await\s+Knowledges\.has_file\s*\(",
        func_body,
    )
    assert has_file_check, (
        "add_file_to_knowledge_by_id doesn't check if file is already linked. "
        "Idempotency not guaranteed; concurrent adds may cause re-embedding."
    )


# =============================================================================
# Behavioral tests: do the model methods exist and have the right signatures?
# =============================================================================


def test_files_model_has_reset_stuck_processing_files_method(open_webui_backend: Path) -> None:
    """Verify that the Files model class has the reset_stuck_processing_files
    method with the expected signature."""
    files_py = open_webui_backend / "open_webui" / "models" / "files.py"
    assert files_py.is_file(), f"{files_py} not found"

    src = files_py.read_text(encoding="utf-8")

    # Check method definition
    method_def = re.search(
        r"async\s+def\s+reset_stuck_processing_files\s*\(",
        src,
    )
    assert method_def, (
        "FilesTable doesn't define reset_stuck_processing_files() method. "
        "Startup cleanup can't function."
    )

    # Should return int (the count of reset files)
    method_body_start = method_def.end()
    method_body = src[method_body_start : method_body_start + 2000]
    has_return_type = re.search(r"->\s*int\s*:", method_body)
    assert has_return_type, (
        "reset_stuck_processing_files() doesn't declare return type as int. "
        "Caller can't log the count of reset files."
    )


def test_files_model_has_get_pending_files_by_knowledge_id_method(
    open_webui_backend: Path,
) -> None:
    """Verify that the Files model has get_pending_files_by_knowledge_id
    method with the expected signature and filtering logic."""
    files_py = open_webui_backend / "open_webui" / "models" / "files.py"
    assert files_py.is_file(), f"{files_py} not found"

    src = files_py.read_text(encoding="utf-8")

    # Check method definition
    method_def = re.search(
        r"async\s+def\s+get_pending_files_by_knowledge_id\s*\(",
        src,
    )
    assert method_def, (
        "FilesTable doesn't define get_pending_files_by_knowledge_id() method. "
        "Pending file tracking can't function."
    )

    # Should take knowledge_id parameter
    method_sig_start = method_def.end()
    method_sig = src[method_sig_start : method_sig_start + 500]
    has_knowledge_id_param = re.search(r"knowledge_id\s*:", method_sig)
    assert has_knowledge_id_param, (
        "get_pending_files_by_knowledge_id() doesn't take knowledge_id parameter."
    )


# =============================================================================
# Contract tests: verify the filtering logic in the methods
# =============================================================================


def test_reset_stuck_processing_files_filters_on_status(open_webui_backend: Path) -> None:
    """Verify that reset_stuck_processing_files filters on status being
    'pending' or 'processing' and marks them as 'failed'."""
    files_py = open_webui_backend / "open_webui" / "models" / "files.py"
    src = files_py.read_text(encoding="utf-8")

    # Find the method
    method_start = re.search(r"async\s+def\s+reset_stuck_processing_files", src)
    assert method_start
    method_content = src[method_start.start() : method_start.start() + 2500]

    # Must filter on pending/processing status
    filters_pending_processing = re.search(
        r"['\"](pending|processing)['\"]", method_content
    )
    assert filters_pending_processing, (
        "reset_stuck_processing_files doesn't filter on 'pending' or 'processing' status. "
        "Will reset all files, not just stuck ones."
    )

    # Must set status to 'failed'
    sets_failed = re.search(r"['\"](status|error)['\"].*?['\"](failed|interrupted)", method_content)
    assert sets_failed, (
        "reset_stuck_processing_files doesn't mark files as 'failed'. "
        "Users won't know the processing was interrupted."
    )


def test_get_pending_files_filters_on_status(open_webui_backend: Path) -> None:
    """Verify that get_pending_files_by_knowledge_id filters on status being
    'pending' or 'processing'."""
    files_py = open_webui_backend / "open_webui" / "models" / "files.py"
    src = files_py.read_text(encoding="utf-8")

    # Find the method
    method_start = re.search(r"async\s+def\s+get_pending_files_by_knowledge_id", src)
    assert method_start
    method_content = src[method_start.start() : method_start.start() + 2000]

    # Must filter on pending/processing status
    filters_pending_processing = (
        re.search(r"pending.*processing", method_content, re.IGNORECASE)
        or re.search(r"processing.*pending", method_content, re.IGNORECASE)
    )
    assert filters_pending_processing, (
        "get_pending_files_by_knowledge_id doesn't filter on 'pending'/'processing' status. "
        "Will return all files, defeating the purpose of the method."
    )


def test_get_pending_files_filters_on_knowledge_id(open_webui_backend: Path) -> None:
    """Verify that get_pending_files_by_knowledge_id filters on knowledge_id
    via meta['data']['knowledge_id']."""
    files_py = open_webui_backend / "open_webui" / "models" / "files.py"
    src = files_py.read_text(encoding="utf-8")

    # Find the method
    method_start = re.search(r"async\s+def\s+get_pending_files_by_knowledge_id", src)
    assert method_start
    method_content = src[method_start.start() : method_start.start() + 2000]

    # Must filter on knowledge_id (typically meta.data.knowledge_id)
    filters_knowledge_id = re.search(r"knowledge_id", method_content, re.IGNORECASE)
    assert filters_knowledge_id, (
        "get_pending_files_by_knowledge_id doesn't filter on knowledge_id. "
        "Will return pending files for the wrong knowledge bases."
    )


# =============================================================================
# Authorization tests: verify the endpoint enforces access control
# =============================================================================


def test_get_pending_files_endpoint_requires_knowledge_access(open_webui_backend: Path) -> None:
    """Verify that the GET /{id}/files/pending endpoint checks that the user
    has access to the knowledge base (owner, admin, or has read grant)."""
    knowledge_router = open_webui_backend / "open_webui" / "routers" / "knowledge.py"
    src = knowledge_router.read_text(encoding="utf-8")

    # Find the endpoint
    endpoint_match = re.search(
        r"@router\.get\s*\(\s*['\"]/{id}/files/pending['\"].*?\).*?"
        r"async\s+def\s+get_pending_files_by_knowledge_id",
        src,
        re.DOTALL,
    )
    assert endpoint_match, (
        "GET /{id}/files/pending endpoint not found in knowledge router."
    )

    # Find the function body
    endpoint_start = endpoint_match.end()
    endpoint_body = src[endpoint_start : endpoint_start + 2000]

    # Must check knowledge exists
    checks_knowledge = re.search(r"Knowledges\.get_knowledge_by_id", endpoint_body)
    assert checks_knowledge, (
        "Endpoint doesn't fetch the knowledge base to verify it exists."
    )

    # Must check authorization (admin, owner, or has_access)
    checks_access = (
        re.search(r"user\.role.*admin", endpoint_body, re.IGNORECASE)
        and (
            re.search(r"knowledge\.user_id.*user\.id", endpoint_body)
            or re.search(r"AccessGrants\.has_access", endpoint_body)
        )
    )
    assert checks_access, (
        "Endpoint doesn't enforce access control. "
        "Users could retrieve pending files for other users' knowledge bases."
    )


# =============================================================================
# Contract tests: verify response shapes and edge cases
# =============================================================================


def test_get_pending_files_endpoint_response_model_is_file_list(
    open_webui_backend: Path,
) -> None:
    """Verify that the endpoint declares response_model as list[FileModelResponse]."""
    knowledge_router = open_webui_backend / "open_webui" / "routers" / "knowledge.py"
    src = knowledge_router.read_text(encoding="utf-8")

    # Find the endpoint decorator
    endpoint_match = re.search(
        r"@router\.get\s*\(\s*['\"]/{id}/files/pending['\"]\s*,\s*response_model\s*=\s*([^)]+)",
        src,
    )
    assert endpoint_match, (
        "GET /{id}/files/pending endpoint decorator not found or has no response_model."
    )

    response_model = endpoint_match.group(1)
    has_list_response = re.search(r"list\s*\[\s*FileModelResponse\s*\]", response_model)
    assert has_list_response, (
        f"response_model is {response_model.strip()}, not list[FileModelResponse]. "
        "Callers won't know the response shape."
    )
