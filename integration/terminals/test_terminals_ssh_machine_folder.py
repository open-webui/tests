"""Regression: the Folder typed while adding a Remote (SSH) machine was dropped.

open-webui issue #34808: the add-machine form shows a Folder field for a machine reached over
SSH, but the SSH create route had no folder field, so the value never reached the install and
sessions started in the home folder. Adding a Local machine already took and checked its folder.

Without a real SSH host the folder only shows in how the route answers it: a relative folder is
refused with the same message the Local route gives, before anything is installed. The SSH
target here is a closed port, so nothing is ever reached.

Discriminates: fails on dev b393504d9 (the folder is ignored and the route answers 200,
installing); passes with the folder taken by the SSH create route.
"""

from __future__ import annotations

import uuid

import pytest

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

ABSOLUTE_FOLDER_REQUIRED = "Use an absolute folder path, like /home/me/project."
# pattern-valid only; the route checks the folder before it connects anywhere
HOST_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExampleExampleExampleExampleExampleExample"


def test_the_folder_of_a_new_ssh_machine_is_checked(admin):
    form = {
        "id": f"m_{uuid.uuid4().hex[:8]}",
        "host": "127.0.0.1",
        "port": 9,
        "username": "demo",
        "host_key": HOST_KEY,
        "name": "Demo machine",
        "folder": "project",
    }
    with admin.client() as client:
        created = client.post("/api/v1/machines/ssh", json=form)

    assert created.status_code == 400, (
        f"the Folder of a new SSH machine was ignored (#34808): "
        f"HTTP {created.status_code} {created.text}"
    )
    assert created.json()["detail"] == ABSOLUTE_FOLDER_REQUIRED
