"""Regression: the Start directory typed while adding a Remote (SSH) machine was dropped.

open-webui issue #34808, fixed by b7ce1bd19: the add-machine form shows a Start directory field
(called Folder before b7ce1bd19) for a machine reached over SSH, but the SSH create route had no
field for it, so the value never reached the install and sessions started in the home folder.
Adding a Local machine already took and checked its start directory.

Without a real SSH host the start directory only shows in how the route answers it: a relative
path is refused with the same message the Local route gives, before anything is installed. The
SSH target here is a closed port, so nothing is ever reached.

Discriminates: fails on dev b393504d9 (the folder is ignored and the route answers 200,
installing); passes on dev 3b6e0855d (b7ce1bd19 takes start_dir on the SSH create route).
"""

from __future__ import annotations

import uuid

import pytest

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

ABSOLUTE_PATH_REQUIRED = "Use an absolute path, like /home/me/project."
# pattern-valid only; the route checks the start directory before it connects anywhere
HOST_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExampleExampleExampleExampleExampleExample"


def test_the_start_directory_of_a_new_ssh_machine_is_checked(admin):
    form = {
        "id": f"m_{uuid.uuid4().hex[:8]}",
        "host": "127.0.0.1",
        "port": 9,
        "username": "demo",
        "host_key": HOST_KEY,
        "name": "Demo machine",
        "start_dir": "project",
    }
    with admin.client() as client:
        created = client.post("/api/v1/machines/ssh", json=form)

    assert created.status_code == 400, (
        f"the Start directory of a new SSH machine was ignored (#34808): "
        f"HTTP {created.status_code} {created.text}"
    )
    assert created.json()["detail"] == ABSOLUTE_PATH_REQUIRED
