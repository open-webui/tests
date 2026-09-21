from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression


@pytest.fixture(scope="session")
def main_source(open_webui_backend) -> str:
    return (open_webui_backend / "open_webui" / "main.py").read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def error_emission_block(main_source) -> str:
    """The shipped except-block of chat_completion, lifted verbatim."""
    start = main_source.index("error_detail = e.detail if isinstance(e, HTTPException) else str(e)")
    end = main_source.index("except Exception:", start)
    return main_source[start:end]


def test_error_path_no_longer_cancels_the_task_queue(error_emission_block):
    """dbb17a572: the cancel event is what stopped the message queue."""
    assert "chat:tasks:cancel" not in error_emission_block, (
        "the error path still emits chat:tasks:cancel, so a failed reply stopped "
        "the message queue and the chat turned away everything typed after it "
        "(dbb17a572)"
    )


def test_error_event_carries_done(error_emission_block):
    """dbb17a572: the error event now terminates the reply itself."""
    compact = error_emission_block.replace(" ", "").replace(chr(10), "")
    assert chr(39)+'done'+chr(39)+':True' in compact, (
        "the chat:message:error event carries no done flag, so the frontend kept "
        "waiting on a reply that had already failed (dbb17a572)"
    )


def test_stored_error_message_carries_done(error_emission_block):
    """dbb17a572: the upsert that persists the failed turn also closes it."""
    compact = error_emission_block.replace(" ", "").replace(chr(10), "")
    assert chr(39)+'done'+chr(39)+':True' in compact
