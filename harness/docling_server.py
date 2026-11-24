"""A Docling server on the listener, answering conversions through docling-serve's async API.

Since dev d89cd809c Open WebUI submits a file to `/v1/convert/file/async`, polls the task at
`/v1/status/poll/<task>` while it is pending and reads the conversion from `/v1/result/<task>`.
The task is pending once, so every conversion goes through a poll. `CONVERT` is the path the
file is posted to, for `listener.requests_to`.
"""

from __future__ import annotations

from harness.listener import Listener, json_answer

CONVERT = "/v1/convert/file/async"
TASK = "docling-task"


def serve_docling(listener: Listener, answer: dict) -> None:
    """Answer every conversion with `answer`, the body docling-serve gives for the document."""
    listener.route("POST", CONVERT, json_answer({"task_id": TASK, "task_status": "pending"}))
    listener.route(
        "GET", f"/v1/status/poll/{TASK}", json_answer({"task_id": TASK, "task_status": "success"})
    )
    listener.route("GET", f"/v1/result/{TASK}", json_answer(answer))
