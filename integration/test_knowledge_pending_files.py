"""Integration tests for knowledge base pending files endpoint.

Tests the HTTP API for retrieving files that are still being processed
(pending/processing status) for a knowledge base.

Endpoints:
- GET /api/v1/knowledges/{id}/files/pending
"""

from __future__ import annotations

import pytest

pytestmark = [
    pytest.mark.requires_instance,
    pytest.mark.api,
    pytest.mark.auth_required,
]


@pytest.mark.skip(reason="Requires knowledge base and file setup; skipped until knowledge setup fixtures available")
async def test_get_pending_files_returns_processing_files(api_client, authenticated_page):
    """GET /api/v1/knowledges/{id}/files/pending returns files with status
    'pending' or 'processing' that reference the knowledge base."""
    # Note: This test requires:
    # 1. A created knowledge base
    # 2. A file uploaded with meta.data.knowledge_id = kb.id
    # 3. The file in 'processing' or 'pending' status
    #
    # Implementation blocked on knowledge setup fixtures; add when available.
    pass


@pytest.mark.skip(reason="Requires knowledge base and file setup; skipped until knowledge setup fixtures available")
async def test_get_pending_files_filters_on_status(api_client):
    """GET /api/v1/knowledges/{id}/files/pending returns only files with
    status in ('pending', 'processing'), not completed or failed files."""
    # Note: Similar to above; blocked on fixtures.
    pass


@pytest.mark.skip(reason="Requires knowledge base and file setup; skipped until knowledge setup fixtures available")
async def test_get_pending_files_requires_knowledge_access(
    api_client, api_jwt, authenticated_page
):
    """GET /api/v1/knowledges/{id}/files/pending requires read access to the
    knowledge base; returns 403 for unauthorized users."""
    # Note: Similar to above; blocked on fixtures.
    pass


@pytest.mark.skip(reason="Requires knowledge base and file setup; skipped until knowledge setup fixtures available")
async def test_get_pending_files_returns_empty_when_no_pending(api_client):
    """GET /api/v1/knowledges/{id}/files/pending returns [] when no files are
    being processed."""
    # Note: Similar to above; blocked on fixtures.
    pass


@pytest.mark.skip(reason="Requires knowledge base and file setup; skipped until knowledge setup fixtures available")
async def test_get_pending_files_returns_404_for_nonexistent_knowledge(api_client):
    """GET /api/v1/knowledges/{nonexistent_id}/files/pending returns 404."""
    # Note: Similar to above; blocked on fixtures.
    pass


@pytest.mark.skip(reason="Requires knowledge base and file setup; skipped until knowledge setup fixtures available")
async def test_get_pending_files_returns_403_for_unowned_knowledge(api_client):
    """GET /api/v1/knowledges/{other_user_kb_id}/files/pending returns 403
    when user doesn't own the knowledge base and has no access grant."""
    # Note: Similar to above; blocked on fixtures.
    pass


# Note for future developers: These tests are skipped because they require
# complex setup (creating knowledge bases, uploading files, setting processing
# status). When knowledge base setup fixtures are added to conftest.py, replace
# these skips with real tests that:
#
# 1. Create a knowledge base as a test user
# 2. Create/mock files with pending/processing status
# 3. Call the endpoint and verify the response
# 4. Test authorization (owner can read, non-owner cannot)
# 5. Test filtering (only pending/processing returned, not completed)
#
# Example fixture pattern:
#
#   @pytest.fixture
#   async def knowledge_base_with_pending_file(api_client, user_id):
#       """Create a knowledge base and a file in pending status."""
#       kb = await api_client.post(
#           "/api/v1/knowledges",
#           json={"name": "Test KB", "description": "..."}
#       )
#       file = await Files.add_file(
#           id="test-file-id",
#           user_id=user_id,
#           filename="test.pdf",
#           data={"status": "pending"},
#           meta={"data": {"knowledge_id": kb["id"]}}
#       )
#       return kb, file
