"""Regression: image-generation workflows (which embed user prompt content) are
not written to server logs at operator-visible levels.

open-webui 0.10.2 fix `64b92ff08` (#26400): `comfyui_create_image` /
`comfyui_edit_image` did `log.info(f'Workflow: {workflow}')`, leaking the user's
prompt content into logs at the default level. Fix: `log.debug`. This audits that
no INFO/WARNING/ERROR log line in comfyui.py emits the workflow.

Discriminates: passes on v0.10.2 (debug), fails on v0.10.1 (info).
"""

import re

import pytest

pytestmark = pytest.mark.regression


def test_comfyui_workflow_not_logged_above_debug(open_webui_backend):
    src = (open_webui_backend / "open_webui" / "utils" / "images" / "comfyui.py").read_text(
        encoding="utf-8"
    )
    # Only the interpolated `{workflow}` value leaks prompt content. A static
    # message that merely mentions the word "workflow" is fine.
    offenders = [
        ln.strip()
        for ln in src.splitlines()
        if re.search(r"log\.(info|warning|error|critical|exception)\(", ln) and "{workflow" in ln
    ]
    assert not offenders, (
        "comfyui.py logs the workflow (which embeds user prompt content) above "
        f"debug level, leaking prompts into operator-visible logs: {offenders}"
    )
