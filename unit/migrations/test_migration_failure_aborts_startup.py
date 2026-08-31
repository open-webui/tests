"""Regression: a failed Alembic upgrade was swallowed at startup (commit 8c0c7b3b6,
issue #29280, shipped in v0.11.3).

`open_webui.config.run_migrations()` wrapped `command.upgrade(cfg, 'head')` in a
`try/except Exception` that only logged, so a broken migration left the app
running against a half-migrated schema and the first query blew up later with a
confusing missing table or column error such as `chat.timer_at`. The fix
re-raises after logging, so startup stops at the migration error itself.

Discriminates: passes on v0.11.3, fails on v0.11.2 (run_migrations returns None
instead of propagating the upgrade error).
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.regression


class MigrationBoom(Exception):
    """Stands in for whatever a broken revision raises."""


@pytest.fixture(scope="module")
def config_module(owui_module):
    return owui_module("open_webui.config")


@pytest.fixture
def alembic_upgrade():
    """Patch the alembic boundary only. No real migration ever runs here."""
    with patch("alembic.command.upgrade") as upgrade:
        yield upgrade


def test_upgrade_failure_propagates_out_of_run_migrations(config_module, alembic_upgrade):
    """NARROW: pre-fix the except block logged and returned, so startup continued."""
    alembic_upgrade.side_effect = MigrationBoom("target database is not up to date")

    # The original cause has to survive, that is the whole point of stopping here.
    with pytest.raises(MigrationBoom, match="target database is not up to date"):
        config_module.run_migrations()


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("Can't locate revision identified by 'deadbeef'"),
        ValueError("no such column: chat.timer_at"),
        MigrationBoom("duplicate column name"),
    ],
)
def test_every_migration_error_type_propagates(config_module, alembic_upgrade, error):
    """BROAD: the swallow was type-blind, so no error class may be absorbed."""
    alembic_upgrade.side_effect = error

    with pytest.raises(type(error)):
        config_module.run_migrations()


def test_alembic_config_failure_propagates_too(config_module):
    """BROAD: the whole try block was swallowed, not just the upgrade call."""
    with patch("alembic.config.Config", side_effect=MigrationBoom("no alembic.ini")):
        with pytest.raises(MigrationBoom):
            config_module.run_migrations()


def test_failure_is_still_logged_before_it_propagates(config_module, alembic_upgrade, caplog):
    """NARROW: re-raising must not cost the operator the traceback either."""
    alembic_upgrade.side_effect = MigrationBoom("boom")

    with caplog.at_level(logging.ERROR, logger=config_module.log.name):
        with pytest.raises(MigrationBoom):
            config_module.run_migrations()

    assert any("Error running migrations" in record.getMessage() for record in caplog.records)


def test_successful_upgrade_returns_normally(config_module, alembic_upgrade):
    """NEARBY: the happy path still runs the chain to head and lets startup continue."""
    assert config_module.run_migrations() is None

    alembic_upgrade.assert_called_once()
    cfg, revision = alembic_upgrade.call_args.args
    assert revision == "head"
    assert cfg.get_main_option("script_location").endswith("migrations")
