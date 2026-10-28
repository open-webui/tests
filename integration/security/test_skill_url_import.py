"""Journey: an admin loads skills from a link, and the download stays on the addresses it may reach.

`POST /api/v1/skills/load/url` (v0.12.0) downloads a skill and returns the packages it holds for
the import dialog. The answer's content type picks the format: JSON is a list of packages,
Markdown is one `SKILL.md`, and a web page is refused with a 400, since a link to a page instead
of the raw file is the usual mistake. A link with a user name or password in it is refused before
anything is fetched, and so is a redirect to one. The download follows at most five redirects, six
requests in all, and checks the address of every hop: without local web fetch a local address is
refused, and with it the web fetch filter list still blocks a hop to a listed address. A plain
user may not load anything; `admin_routes.txt` pins the route as admin-only.

A redirect from a public page to a local one cannot be served here, so the per-hop check is shown
through the filter list. The 200 MiB download cap and the GitHub link mapping are not covered: the
cap reads the body, so only a real 200 MiB answer would reach it, and the mapping needs github.com.

Discriminates: passes on dev 948432b43. In a backend copy, dropping the content type check turned
the web page test red (200); checking the address only on the submitted link, over a plain
aiohttp session without the connection-time check, turned the listed hop test red (the hop was
fetched); dropping the address check entirely as well turned the local address test red (200);
dropping the credential checks turned both credential tests red; and a hop limit of 7 turned the
endless redirect test red (seven requests).
"""

from __future__ import annotations

import json
import uuid

import pytest

from harness.listener import json_answer, listening, text_answer

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

LOAD_URL = "/api/v1/skills/load/url"
LISTED_ADDRESS = "127.0.0.2"
LOCAL_FETCH = {
    "ENABLE_LOCAL_WEB_FETCH": "true",
    "WEB_FETCH_FILTER_LIST": f"!{LISTED_ADDRESS}/32",
}
MARKDOWN = "text/markdown"


@pytest.fixture(scope="module")
def local_fetch(instance_with):
    return instance_with(LOCAL_FETCH)


def _skill_md(name: str) -> str:
    return f"---\nname: {name}\ndescription: Reads the {name} table.\n---\nRead the table.\n"


def _redirect_to(location: str):
    return 302, {"Location": location}, b""


def _load(instance, url: str):
    with instance.client() as client:
        return client.post(LOAD_URL, json={"url": url})


def _name() -> str:
    return f"tides-{uuid.uuid4().hex[:8]}"


@pytest.mark.slow
def test_a_json_link_loads_every_package_in_it(local_fetch, listener):
    names = [_name(), _name()]
    packages = [{"name": name, "content": _skill_md(name)} for name in names]
    listener.route("GET", "/skills.json", json_answer(packages))

    loaded = _load(local_fetch, f"{listener.base_url}/skills.json")

    assert loaded.status_code == 200, loaded.text
    assert [package["name"] for package in loaded.json()] == names
    assert [package["files"][0]["path"] for package in loaded.json()] == ["SKILL.md", "SKILL.md"]


@pytest.mark.slow
def test_a_markdown_link_loads_one_skill_from_its_front_matter(local_fetch, listener):
    name = _name()
    listener.route("GET", "/raw/skill", text_answer(_skill_md(name), MARKDOWN))

    loaded = _load(local_fetch, f"{listener.base_url}/raw/skill")

    assert loaded.status_code == 200, loaded.text
    [package] = loaded.json()
    assert (package["name"], package["description"]) == (name, f"Reads the {name} table.")
    assert [entry["path"] for entry in package["files"]] == ["SKILL.md"]


@pytest.mark.slow
def test_a_web_page_is_refused(local_fetch, listener):
    page = f"<html><body>{_skill_md(_name())}</body></html>"
    listener.route("GET", "/skill.md", text_answer(page, "text/html; charset=utf-8"))

    loaded = _load(local_fetch, f"{listener.base_url}/skill.md")

    assert loaded.status_code == 400, f"a web page was loaded as a skill: {loaded.text}"
    assert "web page" in loaded.json()["detail"], loaded.text


@pytest.mark.slow
def test_five_redirects_are_followed_and_a_sixth_is_refused(local_fetch, listener):
    name = _name()
    for hop in range(5):
        listener.route("GET", f"/five/{hop}", _redirect_to(f"/five/{hop + 1}"))
    listener.route("GET", "/five/5", text_answer(_skill_md(name), MARKDOWN))
    listener.route("GET", "/endless/*", lambda request: _redirect_to(f"{request.path}/again"))

    followed = _load(local_fetch, f"{listener.base_url}/five/0")
    endless = _load(local_fetch, f"{listener.base_url}/endless/start")
    endless_requests = [entry for entry in listener.received if entry.path.startswith("/endless/")]

    assert followed.status_code == 200, followed.text
    assert [package["name"] for package in followed.json()] == [name]
    assert endless.status_code == 400, endless.text
    assert "Too many redirects" in endless.json()["detail"], endless.text
    assert len(endless_requests) == 6, (
        f"an endless redirect chain was followed for {len(endless_requests)} requests, not 6"
    )


@pytest.mark.slow
def test_a_redirect_hop_to_a_listed_address_is_never_fetched(local_fetch, listener):
    name = _name()
    with listening(host=LISTED_ADDRESS) as listed_service:
        listed_service.route("GET", "/secret", text_answer(_skill_md(name), MARKDOWN))
        listener.route("GET", "/start", _redirect_to(f"{listed_service.base_url}/secret"))
        loaded = _load(local_fetch, f"{listener.base_url}/start")
        reached = list(listed_service.received)

    assert reached == [], (
        f"the skill download followed a redirect to {LISTED_ADDRESS} although the web fetch "
        "filter list blocks it"
    )
    assert loaded.status_code == 400, loaded.text
    assert len(listener.requests_to("/start")) == 1


def test_a_local_address_is_refused_without_local_web_fetch(admin, listener):
    listener.route("GET", "/skill.md", text_answer(_skill_md(_name()), MARKDOWN))

    with admin.client() as client:
        loaded = client.post(LOAD_URL, json={"url": f"{listener.base_url}/skill.md"})

    assert loaded.status_code == 400, f"a local address was downloaded: {loaded.text}"
    assert listener.received == [], "the instance reached a local address without local web fetch"


@pytest.mark.slow
def test_a_link_with_credentials_is_refused_before_it_is_fetched(local_fetch, listener):
    listener.route("GET", "/skill.md", text_answer(_skill_md(_name()), MARKDOWN))
    address = listener.base_url.removeprefix("http://")

    with_user = _load(local_fetch, f"http://admin:secret@{address}/skill.md")
    with_user_only = _load(local_fetch, f"http://admin@{address}/skill.md")

    assert (with_user.status_code, with_user_only.status_code) == (400, 400), (
        with_user.text,
        with_user_only.text,
    )
    assert "credentials" in with_user.json()["detail"], with_user.text
    assert listener.received == [], json.dumps([entry.headers for entry in listener.received])


@pytest.mark.slow
def test_a_redirect_to_a_link_with_credentials_is_refused(local_fetch, listener):
    address = listener.base_url.removeprefix("http://")
    listener.route("GET", "/start", _redirect_to(f"http://admin:secret@{address}/skill.md"))
    listener.route("GET", "/skill.md", text_answer(_skill_md(_name()), MARKDOWN))

    loaded = _load(local_fetch, f"{listener.base_url}/start")

    assert loaded.status_code == 400, f"a redirect carried credentials through: {loaded.text}"
    assert "credentials" in loaded.json()["detail"], loaded.text
    assert listener.requests_to("/skill.md") == [], "the hop with credentials was fetched"
