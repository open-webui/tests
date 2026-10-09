"""Regression: skill versions saved within the same second are listed in random order.

The skill editor's version picker lists the history newest first. Each version is stamped with
the whole second it was saved in and ties are broken by its random id, so versions saved in quick
succession, as an agent editing a skill's files or a person saving twice does, come back shuffled
and the picker shows an older version on top.

Red on dev 178de3666: five saves made one after another come back in random order
(open-webui/open-webui#32134).

Discriminates: red 3 of 3 on dev 178de3666; passes 3 of 3 in a backend copy whose history query
breaks ties by the order the versions were written in.
"""

from __future__ import annotations

import pytest

from harness.skill_files import create, delete_skills_of, history, new_id, save, skill_md

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


@pytest.fixture
def owner(make_user):
    """A fresh admin; the skills it made are deleted afterwards."""
    account = make_user(role="admin")
    yield account
    delete_skills_of(account)


def test_versions_saved_in_quick_succession_are_listed_newest_first(owner):
    skill_id = new_id("tides")
    skill = create(owner, {"SKILL.md": skill_md(skill_id, "Version 0.")}, skill_id=skill_id)
    saved_in_order = [skill["version_id"]]
    for number in range(1, 6):
        content = skill_md(skill_id, f"Version {number}.")
        saved = save(owner, skill, [{"op": "put", "path": "SKILL.md", "content": content}])
        assert saved.status_code == 200, saved.text
        skill = saved.json()
        saved_in_order.append(skill["version_id"])

    listed = [entry["id"] for entry in history(owner, skill_id)]

    assert listed == list(reversed(saved_in_order)), (
        "versions saved within the same second come back in random order "
        "(open-webui/open-webui#32134)"
    )
