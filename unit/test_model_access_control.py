"""
Model Access Control Tests

Tests for Open WebUI's model access control system, verifying that:
- Public models (access_control=None) are visible to all users
- Private models (access_control={}) are only visible to owners
- Group-shared models are visible to group members
- User-shared models are visible to specified users

These tests require direct access to Open WebUI's backend modules,
so they must run in an environment where Open WebUI is installed
(e.g., via test.sh which clones and installs the repository).

Usage:
    ./test.sh -- -k "model_access"
    ./test.sh -- unit/test_model_access_control.py
"""

import os
import sys
import uuid
from typing import Any

import pytest

# Skip entire module if Open WebUI backend is not available
pytest.importorskip("open_webui", reason="Open WebUI backend not installed")

from open_webui.models.groups import GroupForm, Groups
from open_webui.models.models import ModelForm, ModelMeta, ModelParams, Models
from open_webui.models.users import Users
from open_webui.utils.models import get_filtered_models


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def test_users() -> dict[str, Any]:
    """Create test users for access control testing.
    
    Creates three users:
    - owner: Creates and owns models
    - viewer: Member of test group, can view shared models
    - outsider: Not in any group, should not see restricted models
    
    Returns dict with user objects keyed by role.
    """
    users = {}
    
    for role in ["owner", "viewer", "outsider"]:
        email = f"{role}_{uuid.uuid4().hex[:6]}@test.local"
        user_id = f"test_{role}_{uuid.uuid4().hex[:8]}"
        
        user = Users.insert_new_user(
            id=user_id,
            name=f"Test {role.title()}",
            email=email,
            role="user"
        )
        users[role] = Users.get_user_by_id(user.id)
    
    yield users
    
    # Cleanup: Delete test users
    for user in users.values():
        try:
            Users.delete_user_by_id(user.id)
        except Exception:
            pass  # Best effort cleanup


@pytest.fixture(scope="module")
def test_group(test_users: dict[str, Any]) -> Any:
    """Create a test group with the viewer user as a member."""
    group = Groups.insert_new_group(
        test_users["owner"].id,
        GroupForm(name=f"Test Group {uuid.uuid4().hex[:6]}", description="Test Group for ACL")
    )
    
    # Add viewer to the group
    Groups.add_users_to_group(group.id, [test_users["viewer"].id])
    
    yield group
    
    # Cleanup: Delete test group
    try:
        Groups.delete_group_by_id(group.id)
    except Exception:
        pass  # Best effort cleanup


@pytest.fixture
def model_factory(test_users: dict[str, Any]):
    """Factory fixture to create test models with cleanup."""
    created_models = []
    
    def _create_model(
        name: str,
        access_control: dict | None = None,
        owner_id: str | None = None
    ) -> Any:
        """Create a model with the specified access control settings."""
        if owner_id is None:
            owner_id = test_users["owner"].id
        
        model_id = f"test_model_{uuid.uuid4().hex[:8]}"
        form = ModelForm(
            id=model_id,
            name=name,
            meta=ModelMeta(description="Test model for access control"),
            params=ModelParams(),
            access_control=access_control,
            is_active=True
        )
        model = Models.insert_new_model(form, owner_id)
        created_models.append(model)
        return model
    
    yield _create_model
    
    # Cleanup: Delete all created models
    for model in created_models:
        try:
            Models.delete_model_by_id(model.id)
        except Exception:
            pass  # Best effort cleanup


def model_to_dict(model: Any) -> dict:
    """Convert a model object to the dict format expected by get_filtered_models."""
    return {
        "id": model.id,
        "name": model.name,
        "object": "model",
        "created": model.created_at,
        "owned_by": "openai",
        "permission": []
    }


# =============================================================================
# Tests
# =============================================================================


class TestModelAccessControl:
    """Tests for model access control filtering."""
    
    @pytest.mark.unit
    def test_public_model_visible_to_all(
        self,
        model_factory,
        test_users: dict[str, Any]
    ):
        """Public models (access_control=None) should be visible to all users."""
        model = model_factory("Public Model", access_control=None)
        models_input = [model_to_dict(model)]
        
        # Owner should see it
        filtered = get_filtered_models(models_input, test_users["owner"])
        assert len(filtered) == 1, "Owner should see public model"
        
        # Viewer should see it
        filtered = get_filtered_models(models_input, test_users["viewer"])
        assert len(filtered) == 1, "Viewer should see public model"
        
        # Outsider should see it
        filtered = get_filtered_models(models_input, test_users["outsider"])
        assert len(filtered) == 1, "Outsider should see public model"
    
    @pytest.mark.unit
    def test_private_model_only_visible_to_owner(
        self,
        model_factory,
        test_users: dict[str, Any]
    ):
        """Private models (access_control={}) should only be visible to owner."""
        model = model_factory("Private Model", access_control={})
        models_input = [model_to_dict(model)]
        
        # Owner should see it
        filtered = get_filtered_models(models_input, test_users["owner"])
        assert len(filtered) == 1, "Owner should see their private model"
        
        # Viewer should NOT see it
        filtered = get_filtered_models(models_input, test_users["viewer"])
        assert len(filtered) == 0, "Viewer should not see private model"
        
        # Outsider should NOT see it
        filtered = get_filtered_models(models_input, test_users["outsider"])
        assert len(filtered) == 0, "Outsider should not see private model"
    
    @pytest.mark.unit
    def test_group_shared_model(
        self,
        model_factory,
        test_users: dict[str, Any],
        test_group: Any
    ):
        """Models shared with a group should be visible to group members only."""
        access_control = {
            "read": {
                "group_ids": [test_group.id],
                "user_ids": []
            }
        }
        model = model_factory("Group Shared Model", access_control=access_control)
        models_input = [model_to_dict(model)]
        
        # Owner should see it (owners always see their models)
        filtered = get_filtered_models(models_input, test_users["owner"])
        assert len(filtered) == 1, "Owner should see their group-shared model"
        
        # Viewer (in group) should see it
        filtered = get_filtered_models(models_input, test_users["viewer"])
        assert len(filtered) == 1, "Group member should see group-shared model"
        
        # Outsider (not in group) should NOT see it
        filtered = get_filtered_models(models_input, test_users["outsider"])
        assert len(filtered) == 0, "Non-group member should not see group-shared model"
    
    @pytest.mark.unit
    def test_user_shared_model(
        self,
        model_factory,
        test_users: dict[str, Any]
    ):
        """Models shared with specific users should only be visible to those users."""
        access_control = {
            "read": {
                "user_ids": [test_users["viewer"].id],
                "group_ids": []
            }
        }
        model = model_factory("User Shared Model", access_control=access_control)
        models_input = [model_to_dict(model)]
        
        # Owner should see it
        filtered = get_filtered_models(models_input, test_users["owner"])
        assert len(filtered) == 1, "Owner should see their user-shared model"
        
        # Viewer (explicitly shared with) should see it
        filtered = get_filtered_models(models_input, test_users["viewer"])
        assert len(filtered) == 1, "Shared user should see user-shared model"
        
        # Outsider (not shared with) should NOT see it
        filtered = get_filtered_models(models_input, test_users["outsider"])
        assert len(filtered) == 0, "Non-shared user should not see user-shared model"
    
    @pytest.mark.unit
    def test_mixed_access_control(
        self,
        model_factory,
        test_users: dict[str, Any],
        test_group: Any
    ):
        """Models with both user and group access should work correctly."""
        access_control = {
            "read": {
                "user_ids": [test_users["outsider"].id],  # Share with outsider
                "group_ids": [test_group.id]  # Also share with group (viewer is member)
            }
        }
        model = model_factory("Mixed Access Model", access_control=access_control)
        models_input = [model_to_dict(model)]
        
        # Owner should see it
        filtered = get_filtered_models(models_input, test_users["owner"])
        assert len(filtered) == 1, "Owner should see their model"
        
        # Viewer (in group) should see it
        filtered = get_filtered_models(models_input, test_users["viewer"])
        assert len(filtered) == 1, "Group member should see model"
        
        # Outsider (explicitly shared) should see it
        filtered = get_filtered_models(models_input, test_users["outsider"])
        assert len(filtered) == 1, "Explicitly shared user should see model"
    
    @pytest.mark.unit
    def test_multiple_models_filtering(
        self,
        model_factory,
        test_users: dict[str, Any],
        test_group: Any
    ):
        """Filter should correctly handle multiple models with different access controls."""
        # Create models with different access levels
        public_model = model_factory("Public", access_control=None)
        private_model = model_factory("Private", access_control={})
        group_model = model_factory("Group Only", access_control={
            "read": {"group_ids": [test_group.id], "user_ids": []}
        })
        
        models_input = [
            model_to_dict(public_model),
            model_to_dict(private_model),
            model_to_dict(group_model),
        ]
        
        # Owner sees all 3
        filtered = get_filtered_models(models_input, test_users["owner"])
        assert len(filtered) == 3, "Owner should see all their models"
        
        # Viewer sees public + group (2)
        filtered = get_filtered_models(models_input, test_users["viewer"])
        assert len(filtered) == 2, "Viewer should see public and group models"
        filtered_ids = {m["id"] for m in filtered}
        assert public_model.id in filtered_ids
        assert group_model.id in filtered_ids
        assert private_model.id not in filtered_ids
        
        # Outsider sees only public (1)
        filtered = get_filtered_models(models_input, test_users["outsider"])
        assert len(filtered) == 1, "Outsider should only see public model"
        assert filtered[0]["id"] == public_model.id
