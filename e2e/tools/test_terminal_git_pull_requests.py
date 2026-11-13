"""Journey: the pull requests of a repository in the Git panel under a chat's Files tab.

When the terminal says a repository's pull requests can be reached (it has the `gh` CLI signed
in), the Git panel offers a Pull requests view beside Changes and History. A person filters the
list to the pull requests that are open, closed, merged or all of them, narrows it by number or
title, opens one to read its summary, its code, its checks and its activity, starts a new one
from the branch that is checked out against the repository's default branch, edits one, checks
one out, comments on it, approves it or asks for changes, merges it the way they pick from the
merge button's menu, marks a draft ready, closes it from its actions and opens it again. A
conflicting pull request offers to resolve the conflicts in the chat instead of a merge. What
needs a second thought (merge, close, approve, request changes) asks first, and a refusal of the
terminal is shown in its own words.

A real Open Terminal needs `gh` and a forge for this, so the terminal here is scripted: it keeps
the pull requests of one repository and answers the `git/pr` calls the way Open Terminal 0.17.7
does, and each test reads what the terminal was asked to do and what the page shows afterwards.

Discriminates, in frontend copies of the dev c15f28649 build: with the view offered whether or
not the terminal says pull requests can be reached, with every list being the open ones or with
a search that narrows nothing, the list test fails; with a new pull request never a draft,
started from no branch or always against main, the create test fails; with the checks never
shown or the code view asking for no diff, the summary test fails; with Edit starting from an
empty title or a checkout that does not reach the panel, the edit test fails; with Approve not
asking, a request for changes sent under another name or a comment sent without its text, the
review test fails; with every merge a squash, or a draft or a conflicting pull request mergeable,
the merge test fails; with Close not asking, a closed pull request not offered to open again or
a refusal of the terminal not shown, the close test fails. The test of the list after a rename
and a close is red on that build, where the list is not read again after a change
(open-webui/open-webui#34098, fixed by c981f7661).

New for 0.13.4. When the terminal says the repository's issues can be reached, an Issues view
sits beside Pull requests with the number of open issues. A person lists issues by state, opens
one to read its labels, assignees and comments, creates, edits and comments on issues, closes one
as completed or not planned, with or without a comment, and opens it again. Searching issues or
pull requests asks the terminal to search the whole repository, descriptions included. Work on
this, Fix, Address feedback and Resolve conflicts in chat write a ready request into the message
input, after what is already there, without sending it.
Discriminates: these pass on the dev 485103cde build. In one frontend copy of it with the Issues
view shown whatever the terminal says (the issue list test), no search sent to the terminal (the
pull request list test), editing an issue from an empty title (the issue edit test), Close as
not planned sent as completed (the close test) and the drafts never written into the input (the
chat buttons test), each named test fails.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlparse

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.listener import json_answer
from harness.terminal_server import (
    TERMINAL_SERVERS_CONFIG,
    configure_terminals,
    read_grant,
    serving_terminal,
)
from utils.cached_chat import pick_terminal
from utils.chat_ui import chat_input
from utils.git_panel import expand_git_panel

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

NAME = "Forge shell"
HOME = "/home/user"
LOGBOOK = f"{HOME}/logbook"
SKETCHES = f"{HOME}/sketches"
# a repository whose pull requests the terminal reaches, but whose issues are switched off
CHARTS = f"{HOME}/charts"
SPEC = {"openapi": "3.0.0", "info": {"title": "Open Terminal", "version": "1"}, "paths": {}}
DIFF = """diff --git a/channel.txt b/channel.txt
index 1111111..2222222 100644
--- a/channel.txt
+++ b/channel.txt
@@ -1,2 +1,2 @@
-east channel: four metres
+east channel: six metres
 west channel: three metres
"""


def _pull_request(number: int, title: str, **fields) -> dict:
    return {
        "number": number,
        "title": title,
        "state": "OPEN",
        "isDraft": False,
        "body": "",
        "headRefName": f"branch-{number}",
        "baseRefName": "trunk",
        "url": f"https://github.com/example/logbook/pull/{number}",
        "mergeable": "MERGEABLE",
        "author": {"login": "ada"},
        "reviewRequests": [],
        "assignees": [],
        "labels": [],
        "comments": [],
        "reviews": [],
        **fields,
    }


def _issue(number: int, title: str, **fields) -> dict:
    return {
        "number": number,
        "title": title,
        "state": "OPEN",
        "body": "",
        "url": f"https://github.com/example/logbook/issues/{number}",
        "author": {"login": "ada"},
        "assignees": [],
        "labels": [],
        "comments": [],
        **fields,
    }


def _found(items: list[dict], search: str) -> list[dict]:
    """What GitHub's search finds: the number, or the words in the title or the description."""
    words = search.lower().split()
    return [
        item
        for item in items
        if search.lstrip("#") == str(item["number"])
        or all(word in f"{item['title']} {item['body']}".lower() for word in words)
    ]


@dataclass
class Forge:
    """The pull requests and issues of the logbook repository, as the scripted terminal keeps
    them. Issue calls are named `issue/<action>` in `asked` and `refusals`."""

    pull_requests: dict[int, dict] = field(default_factory=dict)
    issues: dict[int, dict] = field(default_factory=dict)
    checks: dict[int, list[dict]] = field(default_factory=dict)
    branch: str = "east-channel"
    refusals: dict[str, str] = field(default_factory=dict)
    asked: list[tuple[str, dict]] = field(default_factory=list)

    def asked_to(self, action: str) -> list[dict]:
        return [body for name, body in self.asked if name == action]


def _query(request, name: str) -> str:
    return parse_qs(urlparse(request.path).query).get(name, [""])[0]


def _serve_forge(server, forge: Forge) -> None:
    def read(view: str):
        def answer(request):
            root = _query(request, "root")
            if view == "capabilities" and root == CHARTS:
                return json_answer({"available": True, "default_branch": "trunk", "issues": False})
            if root == CHARTS and view == "list":
                return json_answer({"items": []})
            if root != LOGBOOK:
                return json_answer({"available": False})
            if view in forge.refusals:
                return json_answer({"detail": forge.refusals[view]}, 502)
            search = _query(request, "search")
            forge.asked.append((view, {"state": _query(request, "state"), "search": search}))
            if view == "capabilities":
                open_issues = [i for i in forge.issues.values() if i["state"] == "OPEN"]
                return json_answer(
                    {
                        "available": True,
                        "default_branch": "trunk",
                        "issues": True,
                        "open_issues": len(open_issues),
                    }
                )
            if view in ("list", "issue/list"):
                wanted = {
                    "open": ("OPEN",),
                    "closed": ("CLOSED", "MERGED"),
                    "merged": ("MERGED",),
                    "all": ("OPEN", "CLOSED", "MERGED"),
                }[_query(request, "state")]
                kept = forge.pull_requests if view == "list" else forge.issues
                listed = [item for item in kept.values() if item["state"] in wanted]
                found = _found(listed, search) if search else listed
                ordered = sorted(found, key=lambda item: -item["number"])
                return json_answer({"items": ordered, **({"search": search} if search else {})})
            number = int(_query(request, "number"))
            if view == "diff":
                return json_answer({"diff": DIFF})
            if view == "checks":
                return json_answer({"items": forge.checks.get(number, [])})
            if view == "issue/view":
                return json_answer(forge.issues[number])
            return json_answer(forge.pull_requests[number])

        return answer

    def write(action: str):
        def answer(request):
            body = json.loads(request.body)
            forge.asked.append((action, body))
            if action in forge.refusals:
                return json_answer({"detail": forge.refusals[action]}, 400)
            if action.startswith("issue/"):
                _change_issue(forge, action.removeprefix("issue/"), body)
                return json_answer({"message": ""})
            pull_request = forge.pull_requests.get(body["number"])
            if action == "create":
                number = max(forge.pull_requests) + 1
                forge.pull_requests[number] = _pull_request(
                    number,
                    body["title"],
                    body=body["body"],
                    isDraft=body["draft"],
                    headRefName=body["head"],
                    baseRefName=body["base"],
                    author={"login": "cleo"},
                )
            elif action == "edit":
                pull_request.update(title=body["title"], body=body["body"])
            elif action == "checkout":
                forge.branch = pull_request["headRefName"]
            elif action == "comment":
                pull_request["comments"].append(
                    {"id": uuid.uuid4().hex, "body": body["body"], "author": {"login": "cleo"}}
                )
            elif action == "review":
                state = {"approve": "APPROVED", "request_changes": "CHANGES_REQUESTED"}[
                    body["event"]
                ]
                pull_request["reviews"].append(
                    {
                        "id": uuid.uuid4().hex,
                        "body": body["body"],
                        "state": state,
                        "author": {"login": "cleo"},
                    }
                )
                pull_request["reviewDecision"] = state
            elif action == "merge":
                pull_request["state"] = "MERGED"
            elif action == "close":
                pull_request["state"] = "CLOSED"
            elif action == "reopen":
                pull_request["state"] = "OPEN"
            elif action == "ready":
                pull_request["isDraft"] = False
            return json_answer({"message": ""})

        return answer

    for view in ("capabilities", "list", "view", "diff", "checks"):
        server.route("GET", f"/git/pr/{view}", read(view))
    for action in (
        "create",
        "edit",
        "checkout",
        "comment",
        "review",
        "merge",
        "update-branch",
        "ready",
        "close",
        "reopen",
    ):
        server.route("POST", f"/git/pr/{action}", write(action))
    for view in ("list", "view"):
        server.route("GET", f"/git/issue/{view}", read(f"issue/{view}"))
    for action in ("create", "edit", "comment", "close", "reopen"):
        server.route("POST", f"/git/issue/{action}", write(f"issue/{action}"))


def _change_issue(forge: Forge, action: str, body: dict) -> None:
    issue = forge.issues.get(body["number"])
    comment = {"id": uuid.uuid4().hex, "body": body.get("body", ""), "author": {"login": "cleo"}}
    if action == "create":
        number = max(forge.issues) + 1
        forge.issues[number] = _issue(
            number, body["title"], body=body["body"], author={"login": "cleo"}
        )
    elif action == "edit":
        issue.update(title=body["title"], body=body["body"])
    elif action == "comment":
        issue["comments"].append(comment)
    elif action in ("close", "reopen"):
        if body.get("body", "").strip():
            issue["comments"].append(comment)
        closing = action == "close"
        issue["state"] = "CLOSED" if closing else "OPEN"
        not_planned = body.get("reason") == "not planned"
        issue["stateReason"] = ("NOT_PLANNED" if not_planned else "COMPLETED") if closing else None


def _serve_files(server, forge: Forge) -> None:
    """The file browser's calls: a home with two repositories, of which only the logbook has
    pull requests."""

    def listing(request):
        directory = _query(request, "directory").rstrip("/")
        entries = (
            [
                {"name": "logbook", "type": "directory"},
                {"name": "sketches", "type": "directory"},
                {"name": "charts", "type": "directory"},
            ]
            if directory == HOME
            else [{"name": "channel.txt", "type": "file", "size": 52}]
        )
        return json_answer({"dir": directory, "entries": entries, "writable": True})

    def status(request):
        root = _query(request, "root").rstrip("/")
        if root == HOME:
            return json_answer(
                {"is_repo": False, "branch": "", "files": [], "ahead": 0, "behind": 0}
            )
        return json_answer(
            {
                "is_repo": True,
                "root": root,
                "branch": forge.branch if root == LOGBOOK else "main",
                "head": "0" * 40,
                "upstream": "origin/trunk",
                "ahead": 0,
                "behind": 0,
                "files": [],
                "remote_url": "",
            }
        )

    server.route("GET", "/openapi.json", json_answer(SPEC))
    server.route("GET", "/api/config", json_answer({"features": {"terminal": False, "git": True}}))
    server.route("GET", "/files/cwd", json_answer({"cwd": HOME, "home": HOME}))
    server.route("POST", "/files/cwd", json_answer({"cwd": HOME}))
    server.route("GET", "/files/list", listing)
    server.route("GET", "/git/status", status)


@pytest.fixture
def forge(admin, make_user, preserve):
    """A person, and a scripted terminal shared with them that keeps a repository's pull
    requests; yields the forge with the person as `forge.person`."""
    preserve(TERMINAL_SERVERS_CONFIG)
    person = make_user()
    state = Forge()
    state.pull_requests = {
        7: _pull_request(7, "Mark the wreck", state="MERGED"),
        9: _pull_request(9, "Drop the old soundings", state="CLOSED"),
        12: _pull_request(
            12,
            "Chart the east channel",
            body="Six metres at low water.",
            headRefName="east-channel",
            reviewDecision="REVIEW_REQUIRED",
            reviewRequests=[{"login": "ben"}],
            labels=[{"name": "charts"}],
            comments=[
                {"id": "c1", "body": "Checked against the lead line.", "author": {"login": "ben"}}
            ],
        ),
        14: _pull_request(14, "Sketch the west quay", isDraft=True, headRefName="west-quay"),
        16: _pull_request(16, "Redraw the tide gauge", mergeable="CONFLICTING"),
    }
    state.issues = {
        2: _issue(2, "Buoy light out", state="CLOSED", stateReason="NOT_PLANNED"),
        3: _issue(
            3,
            "Soundings missing off the pier",
            body="The lantern survey skipped the pier head.",
            labels=[{"name": "survey"}],
            assignees=[{"login": "ben"}],
            comments=[{"id": "i1", "body": "Low water is on Friday.", "author": {"login": "ben"}}],
        ),
        5: _issue(5, "Wreck marker drifted"),
    }
    state.checks = {
        12: [
            {"name": "build", "status": "COMPLETED", "conclusion": "SUCCESS"},
            {"context": "soundings/lint", "state": "FAILURE"},
        ]
    }
    state.branch = "east-channel"
    with serving_terminal() as server:
        _serve_files(server, state)
        _serve_forge(server, state)
        connection = server.connection(name=NAME, config={"access_grants": [read_grant(person.id)]})
        with admin.client() as client:
            configure_terminals(client, connection)
        state.person = person
        yield state


def _git_panel_in(page: Page, folder: str) -> Locator:
    pick_terminal(page, NAME)
    return expand_git_panel(page, folder)


def _pull_requests(page_for, forge: Forge) -> tuple[Page, Locator]:
    """A chat page of the person with the logbook's Pull requests view open."""
    page = page_for(forge.person)
    git = _git_panel_in(page, "logbook")
    git.get_by_role("button", name="Pull requests", exact=True).click()
    expect(_listed(git, 12)).to_be_visible()
    return page, git


ROW_NUMBER = re.compile(r"#(\d+) · ")


def _rows(git: Locator) -> Locator:
    """The list's rows: the title over `#number · author · ...`."""
    return git.get_by_role("button").filter(has_text=ROW_NUMBER)


def _listed(git: Locator, number: int) -> Locator:
    return git.get_by_role("button").filter(has_text=f"#{number} · ")


def _title_of_opened(git: Locator, number: int, title: str) -> Locator:
    """The title above the opened pull request."""
    return git.get_by_text(f"#{number} · {title}", exact=True)


def _listed_numbers(git: Locator) -> list[int]:
    return [int(ROW_NUMBER.search(row).group(1)) for row in _rows(git).all_inner_texts()]


def _expect_listed(git: Locator, numbers: list[int]) -> None:
    expect(_rows(git)).to_have_count(len(numbers))
    assert _listed_numbers(git) == numbers, (
        f"the list shows the pull requests {_listed_numbers(git)}, expected {numbers}"
    )


def _show(page: Page, git: Locator, state: str) -> None:
    git.get_by_role("button", name="Filter pull requests").click()
    page.get_by_role("menu").get_by_role("menuitem", name=state, exact=True).click()


def _confirm(page: Page, action: str, pull_request: str) -> None:
    dialog = page.get_by_role("dialog", name=action)
    expect(dialog).to_contain_text(pull_request)
    dialog.get_by_role("button", name="Confirm").click()
    expect(dialog).to_have_count(0)


def _actions(page: Page, git: Locator) -> Locator:
    git.get_by_role("button", name="Pull request actions").click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_role("menuitem", name="Checkout")).to_be_visible()
    return menu


def test_pull_requests_show_only_for_a_repository_that_has_them_and_list_by_state_and_search(
    page_for, forge
):
    page = page_for(forge.person)
    git = _git_panel_in(page, "sketches")
    # a repository whose pull requests the terminal cannot reach has Changes and History alone
    expect(git.get_by_role("button", name="History", exact=True)).to_be_visible()
    expect(git.get_by_role("button", name="Pull requests", exact=True)).to_have_count(0)

    files = page.get_by_role("region", name="File browser")
    files.get_by_role("button", name="Back", exact=True).click()
    git = expand_git_panel(page, "logbook")
    git.get_by_role("button", name="Pull requests", exact=True).click()

    # the open ones first, newest on top, a draft named as one
    _expect_listed(git, [16, 14, 12])
    expect(_listed(git, 14)).to_contain_text("#14 · ada · Draft")
    expect(_listed(git, 12)).to_contain_text("#12 · ada")
    expect(git.get_by_text("Select a pull request")).to_be_visible()

    _show(page, git, "Closed")
    _expect_listed(git, [9, 7])
    expect(_listed(git, 9)).to_contain_text("#9 · ada · Closed")
    expect(_listed(git, 7)).to_contain_text("#7 · ada · Merged")
    _show(page, git, "Merged")
    _expect_listed(git, [7])
    _show(page, git, "All")
    _expect_listed(git, [16, 14, 12, 9, 7])
    asked = [body["state"] for body in forge.asked_to("list")]
    assert asked == ["open", "closed", "merged", "all"], (
        f"the terminal was asked for the lists {forge.asked_to('list')}"
    )

    search = git.get_by_role("textbox", name="Search pull requests")
    search.fill("QUAY")
    _expect_listed(git, [14])
    search.fill("12")
    _expect_listed(git, [12])
    # the search runs over the whole repository, descriptions included
    search.fill("low water")
    _expect_listed(git, [12])
    assert forge.asked_to("list")[-1]["search"] == "low water", forge.asked_to("list")[-1]
    search.fill("")
    _expect_listed(git, [16, 14, 12, 9, 7])

    # a list the terminal cannot get says why, in the terminal's words
    forge.refusals["list"] = "gh: you are not signed in to github.com"
    _show(page, git, "Open")
    expect(git.get_by_role("alert")).to_have_text("gh: you are not signed in to github.com")


def test_a_pull_request_is_started_from_the_checked_out_branch_against_the_default_branch(
    page_for, forge
):
    page, git = _pull_requests(page_for, forge)
    git.get_by_role("button", name="New pull request").click()
    title = git.get_by_role("textbox", name="Pull request title")
    create = git.get_by_role("button", name="Create pull request")
    expect(git.get_by_role("textbox", name="Base branch")).to_have_value("trunk")
    expect(git.get_by_role("textbox", name="Head branch")).to_have_value("east-channel")
    expect(create).to_be_disabled()

    title.fill("Sound the harbour mouth")
    git.get_by_role("textbox", name="Pull request description").fill("Depths every ten metres.")
    git.get_by_role("checkbox", name="Draft").check()
    forge.refusals["create"] = "a pull request for branch east-channel already exists"
    create.click()
    # refused: the terminal's reason shows and what was typed stays
    expect(git.get_by_role("alert")).to_have_text(
        "a pull request for branch east-channel already exists"
    )
    expect(title).to_have_value("Sound the harbour mouth")
    assert 17 not in forge.pull_requests

    del forge.refusals["create"]
    git.get_by_role("textbox", name="Head branch").fill("harbour-mouth")
    create.click()
    expect(_listed(git, 17)).to_contain_text("Sound the harbour mouth")
    expect(_listed(git, 17)).to_contain_text("Draft")
    expect(title).to_have_count(0)
    sent = forge.asked_to("create")[-1]
    wanted = {
        "root": LOGBOOK,
        "title": "Sound the harbour mouth",
        "body": "Depths every ten metres.",
        "base": "trunk",
        "head": "harbour-mouth",
        "draft": True,
    }
    assert {key: sent.get(key) for key in wanted} == wanted, f"the terminal was asked for {sent}"


def test_an_opened_pull_request_shows_its_summary_its_code_its_checks_and_its_activity(
    page_for, forge
):
    page, git = _pull_requests(page_for, forge)
    _listed(git, 12).click()
    expect(git.get_by_text("Six metres at low water.")).to_be_visible()
    expect(git.get_by_text("Open · Review required")).to_be_visible()
    expect(git.get_by_text("ben", exact=True)).to_be_visible()
    expect(git.get_by_text("charts", exact=True)).to_be_visible()
    expect(git.get_by_text("No conflicts", exact=True)).to_be_visible()

    git.get_by_role("button", name="Code", exact=True).click()
    expect(git.get_by_text("east channel: six metres")).to_be_visible()
    expect(git.get_by_text("east channel: four metres")).to_be_visible()
    expect(git.get_by_text("Six metres at low water.")).to_have_count(0)

    git.get_by_role("button", name="Checks", exact=True).click()
    expect(git.get_by_text("build", exact=True)).to_be_visible()
    expect(git.get_by_text("success", exact=True)).to_be_visible()
    expect(git.get_by_text("soundings/lint", exact=True)).to_be_visible()
    expect(git.get_by_text("failure", exact=True)).to_be_visible()

    git.get_by_role("button", name="Activity", exact=True).click()
    expect(git.get_by_text("Checked against the lead line.")).to_be_visible()

    # one without a description, checks or reviewers says so
    _listed(git, 14).click()
    expect(git.get_by_text("No description")).to_be_visible()
    expect(git.get_by_text("Six metres at low water.")).to_have_count(0)
    git.get_by_role("button", name="Checks", exact=True).click()
    expect(git.get_by_text("No checks")).to_be_visible()
    expect(git.get_by_text("soundings/lint", exact=True)).to_have_count(0)


def test_a_pull_request_is_edited_and_checked_out_from_its_actions(page_for, forge):
    page, git = _pull_requests(page_for, forge)
    forge.branch = "trunk"
    # the panel reads the repository again when it is opened
    git.get_by_role("button", name="Collapse Git panel").click()
    git.get_by_role("button", name="Expand Git panel").click()
    expect(git.get_by_role("button", name="Switch branch")).to_have_text("trunk")
    _listed(git, 12).click()
    expect(git.get_by_text("Six metres at low water.")).to_be_visible()

    _actions(page, git).get_by_role("menuitem", name="Edit").click()
    title = git.get_by_role("textbox", name="Pull request title")
    description = git.get_by_role("textbox", name="Pull request description")
    expect(title).to_have_value("Chart the east channel")
    expect(description).to_have_value("Six metres at low water.")
    # the branches of an existing pull request are not on offer
    expect(git.get_by_role("textbox", name="Base branch")).to_have_count(0)
    title.fill("Chart the east channel at low water")
    description.fill("Six metres, sounded twice.")
    git.get_by_role("button", name="Save", exact=True).click()

    expect(git.get_by_text("Six metres, sounded twice.")).to_be_visible()
    expect(_title_of_opened(git, 12, "Chart the east channel at low water")).to_be_visible()
    sent = forge.asked_to("edit")[-1]
    assert (sent["number"], sent["title"], sent["body"]) == (
        12,
        "Chart the east channel at low water",
        "Six metres, sounded twice.",
    ), f"the terminal was asked to edit {sent}"

    _actions(page, git).get_by_role("menuitem", name="Checkout").click()
    expect(git.get_by_role("button", name="Switch branch")).to_have_text("east-channel")
    assert [body["number"] for body in forge.asked_to("checkout")] == [12]
    link = _actions(page, git).get_by_role("menuitem", name="Open in GitHub")
    expect(link).to_have_attribute("href", "https://github.com/example/logbook/pull/12")


def test_a_pull_request_is_commented_on_approved_and_sent_back_for_changes(page_for, forge):
    page, git = _pull_requests(page_for, forge)
    _listed(git, 12).click()
    git.get_by_role("button", name="Activity", exact=True).click()
    expect(git.get_by_text("Checked against the lead line.")).to_be_visible()
    comment = git.get_by_role("textbox", name="Comment")
    send_comment = git.get_by_role("button", name="Comment", exact=True)
    ask_for_changes = git.get_by_role("button", name="Request changes")
    expect(send_comment).to_be_disabled()
    expect(ask_for_changes).to_be_disabled()

    comment.fill("The buoy moved in March.")
    send_comment.click()
    expect(git.get_by_text("The buoy moved in March.")).to_be_visible()
    expect(comment).to_have_value("")
    sent = forge.asked_to("comment")[-1]
    assert (sent["number"], sent["body"]) == (12, "The buoy moved in March."), sent

    # a review asks first, and Cancel sends nothing
    git.get_by_role("button", name="Approve").click()
    dialog = page.get_by_role("dialog", name="Approve")
    expect(dialog).to_contain_text("#12 · Chart the east channel")
    dialog.get_by_role("button", name="Cancel").click()
    expect(dialog).to_have_count(0)
    git.get_by_role("button", name="Approve").click()
    _confirm(page, "Approve", "#12 · Chart the east channel")
    expect(git.get_by_text("cleo · Approved")).to_be_visible()
    assert [body["event"] for body in forge.asked_to("review")] == ["approve"], (
        f"the terminal got the reviews {forge.asked_to('review')}"
    )
    git.get_by_role("button", name="Summary", exact=True).click()
    expect(git.get_by_text("Open · Approved")).to_be_visible()

    git.get_by_role("button", name="Activity", exact=True).click()
    comment.fill("Sound it again at springs.")
    ask_for_changes.click()
    _confirm(page, "Request changes", "#12 · Chart the east channel")
    expect(git.get_by_text("cleo · Changes requested")).to_be_visible()
    expect(git.get_by_text("Sound it again at springs.")).to_be_visible()
    git.get_by_role("button", name="Summary", exact=True).click()
    expect(git.get_by_text("Open · Changes requested")).to_be_visible()
    sent = forge.asked_to("review")[-1]
    assert (sent["event"], sent["body"]) == ("request_changes", "Sound it again at springs."), sent


def test_a_pull_request_is_merged_the_way_picked_and_a_draft_or_a_conflict_cannot_be(
    page_for, forge
):
    page, git = _pull_requests(page_for, forge)
    squash = git.get_by_role("button", name="Squash and merge", exact=True)
    rebase = git.get_by_role("button", name="Rebase and merge", exact=True)

    # a conflict is offered to the chat to resolve, never to merge
    _listed(git, 16).click()
    expect(git.get_by_text("Conflicts", exact=True)).to_be_visible()
    expect(git.get_by_role("button", name="Resolve conflicts in chat")).to_be_visible()
    expect(squash).to_have_count(0)

    _listed(git, 14).click()
    expect(git.get_by_text("No description")).to_be_visible()
    expect(squash).to_have_count(0)
    git.get_by_role("button", name="Mark ready", exact=True).click()
    expect(squash).to_be_enabled()
    expect(git.get_by_role("button", name="Mark ready", exact=True)).to_have_count(0)
    assert [body["number"] for body in forge.asked_to("ready")] == [14]

    _listed(git, 12).click()
    expect(git.get_by_text("Six metres at low water.")).to_be_visible()
    git.get_by_role("button", name="Merge method").click()
    page.get_by_role("menu").get_by_role("menuitem", name="Rebase and merge").click()
    expect(squash).to_have_count(0)
    rebase.click()
    _confirm(page, "Rebase and merge", "#12 · Chart the east channel")
    expect(git.get_by_text("Merged · Review required")).to_be_visible()
    expect(rebase).to_have_count(0)
    expect(git.get_by_role("button", name="Merge method")).to_have_count(0)
    sent = forge.asked_to("merge")
    assert [(body["number"], body["strategy"]) for body in sent] == [(12, "rebase")], (
        f"the terminal was asked to merge {sent}"
    )


def test_a_pull_request_is_closed_after_a_question_and_opened_again(page_for, forge):
    page, git = _pull_requests(page_for, forge)
    _listed(git, 12).click()
    expect(git.get_by_text("Six metres at low water.")).to_be_visible()

    squash = git.get_by_role("button", name="Squash and merge", exact=True)

    _actions(page, git).get_by_role("menuitem", name="Close pull request").click()
    _confirm(page, "Close pull request", "#12 · Chart the east channel")
    expect(git.get_by_text("Closed · Review required")).to_be_visible()
    expect(squash).to_have_count(0)
    assert [body["number"] for body in forge.asked_to("close")] == [12]

    git.get_by_role("button", name="Reopen", exact=True).click()
    expect(git.get_by_text("Open · Review required")).to_be_visible()
    expect(squash).to_be_enabled()
    assert [body["number"] for body in forge.asked_to("reopen")] == [12]

    # a refused step says why and leaves the pull request as it was
    forge.refusals["update-branch"] = "the branch is already up to date"
    _actions(page, git).get_by_role("menuitem", name="Update branch").click()
    expect(git.get_by_role("alert")).to_have_text("the branch is already up to date")
    expect(git.get_by_text("Open · Review required")).to_be_visible()


def test_the_list_shows_a_pull_request_as_it_is_after_it_was_renamed_and_closed(page_for, forge):
    """Red on dev c15f28649 (open-webui/open-webui#34098, fixed by c981f7661)."""
    page, git = _pull_requests(page_for, forge)
    _listed(git, 12).click()
    expect(git.get_by_text("Six metres at low water.")).to_be_visible()
    stale = []

    _actions(page, git).get_by_role("menuitem", name="Edit").click()
    git.get_by_role("textbox", name="Pull request title").fill("Chart both channels")
    git.get_by_role("button", name="Save", exact=True).click()
    expect(_title_of_opened(git, 12, "Chart both channels")).to_be_visible()
    _wait_until(lambda: "Chart both channels" in _row(git, 12))
    if "Chart both channels" not in _row(git, 12):
        stale.append(f"renamed to Chart both channels, listed as {_row(git, 12)!r}")

    _actions(page, git).get_by_role("menuitem", name="Close pull request").click()
    _confirm(page, "Close pull request", "#12 · Chart both channels")
    expect(git.get_by_text("Closed · Review required")).to_be_visible()
    # the list shows the open ones
    _wait_until(lambda: not _row(git, 12))
    if _row(git, 12):
        stale.append(f"closed, listed among the open ones as {_row(git, 12)!r}")

    assert not stale, (
        f"the list beside the pull request did not follow it: {stale}; the terminal was asked "
        f"for the list {len(forge.asked_to('list'))} time(s)"
    )


def _row(git: Locator, number: int) -> str:
    """The list's row of the pull request, empty once it is no longer listed."""
    rows = _rows(git).all_inner_texts()
    return next(
        (" ".join(row.split()) for row in rows if ROW_NUMBER.search(row).group(1) == str(number)),
        "",
    )


def _wait_until(condition, seconds: float = 5.0) -> None:
    """Wait for a state the page reaches on its own; the caller asserts it."""
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.1)


def _issues(page_for, forge: Forge) -> tuple[Page, Locator]:
    """A chat page of the person with the logbook's Issues view open."""
    page = page_for(forge.person)
    git = _git_panel_in(page, "logbook")
    git.get_by_role("button", name=re.compile(r"^Issues")).click()
    expect(_listed(git, 3)).to_be_visible()
    return page, git


def _issue_actions(page: Page, git: Locator) -> Locator:
    git.get_by_role("button", name="Issue actions").click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_role("menuitem", name="Edit")).to_be_visible()
    return menu


def _show_issues(page: Page, git: Locator, state: str) -> None:
    git.get_by_role("button", name="Filter issues").click()
    page.get_by_role("menu").get_by_role("menuitem", name=state, exact=True).click()


def test_issues_show_when_the_terminal_offers_them_and_list_by_state_and_search_the_repository(
    page_for, forge
):
    page = page_for(forge.person)
    git = _git_panel_in(page, "charts")
    # pull requests reached, issues switched off: no Issues view
    expect(git.get_by_role("button", name="Pull requests", exact=True)).to_be_visible()
    expect(git.get_by_role("button", name=re.compile(r"^Issues"))).to_have_count(0)

    files = page.get_by_role("region", name="File browser")
    files.get_by_role("button", name="Back", exact=True).click()
    git = expand_git_panel(page, "logbook")
    issues_view = git.get_by_role("button", name=re.compile(r"^Issues"))
    # the view names how many issues are open
    expect(issues_view).to_have_text(re.compile(r"^Issues\s*2$"))
    issues_view.click()

    _expect_listed(git, [5, 3])
    expect(git.get_by_text("Select an issue")).to_be_visible()
    _show_issues(page, git, "Closed")
    _expect_listed(git, [2])
    expect(_listed(git, 2)).to_contain_text("#2 · ada · Not planned")
    _show_issues(page, git, "All")
    _expect_listed(git, [5, 3, 2])
    asked = [body["state"] for body in forge.asked_to("issue/list")]
    assert asked == ["open", "closed", "all"], f"the terminal was asked for the lists {asked}"

    # the search runs over the whole repository, descriptions included
    search = git.get_by_role("textbox", name="Search issues")
    search.fill("lantern survey")
    _expect_listed(git, [3])
    searched = [body["search"] for body in forge.asked_to("issue/list")]
    assert searched[-1] == "lantern survey", f"the terminal was asked to search {searched}"
    search.fill("")
    _expect_listed(git, [5, 3, 2])

    forge.refusals["issue/list"] = "gh: issues are disabled for this repository"
    _show_issues(page, git, "Open")
    expect(git.get_by_role("alert")).to_have_text("gh: issues are disabled for this repository")


def test_an_opened_issue_shows_its_details_and_is_created_edited_and_commented_on(page_for, forge):
    page, git = _issues(page_for, forge)
    _listed(git, 3).click()
    expect(git.get_by_text("The lantern survey skipped the pier head.")).to_be_visible()
    expect(git.get_by_text("survey", exact=True)).to_be_visible()
    expect(git.get_by_role("definition").filter(has_text=re.compile(r"^ben$"))).to_be_visible()
    expect(git.get_by_text("Low water is on Friday.")).to_be_visible()

    _issue_actions(page, git).get_by_role("menuitem", name="Edit").click()
    title = git.get_by_role("textbox", name="Issue title")
    description = git.get_by_role("textbox", name="Issue description")
    expect(title).to_have_value("Soundings missing off the pier")
    expect(description).to_have_value("The lantern survey skipped the pier head.")
    title.fill("Soundings missing off the north pier")
    description.fill("Sound the pier head at low water.")
    git.get_by_role("button", name="Save", exact=True).click()
    expect(_title_of_opened(git, 3, "Soundings missing off the north pier")).to_be_visible()
    expect(git.get_by_text("Sound the pier head at low water.")).to_be_visible()
    sent = forge.asked_to("issue/edit")[-1]
    assert (sent["number"], sent["title"], sent["body"]) == (
        3,
        "Soundings missing off the north pier",
        "Sound the pier head at low water.",
    ), f"the terminal was asked to edit {sent}"

    comment = git.get_by_role("textbox", name="Comment")
    send_comment = git.get_by_role("button", name="Comment", exact=True)
    expect(send_comment).to_be_disabled()
    forge.refusals["issue/comment"] = "gh: you cannot comment on a locked issue"
    comment.fill("Booked the boat.")
    send_comment.click()
    # refused: the terminal's reason shows and what was typed stays
    expect(git.get_by_role("alert")).to_have_text("gh: you cannot comment on a locked issue")
    expect(comment).to_have_value("Booked the boat.")
    del forge.refusals["issue/comment"]
    send_comment.click()
    expect(git.get_by_text("Booked the boat.", exact=True)).to_be_visible()
    expect(comment).to_have_value("")
    sent = forge.asked_to("issue/comment")[-1]
    assert (sent["number"], sent["body"]) == (3, "Booked the boat."), sent

    git.get_by_role("button", name="New issue").click()
    title = git.get_by_role("textbox", name="Issue title")
    create = git.get_by_role("button", name="Create issue")
    expect(create).to_be_disabled()
    title.fill("Chart the new mooring")
    git.get_by_role("textbox", name="Issue description").fill("Two buoys, east side.")
    create.click()
    expect(_listed(git, 6)).to_contain_text("Chart the new mooring")
    expect(title).to_have_count(0)
    sent = forge.asked_to("issue/create")[-1]
    assert (sent["root"], sent["title"], sent["body"]) == (
        LOGBOOK,
        "Chart the new mooring",
        "Two buoys, east side.",
    ), f"the terminal was asked to create {sent}"


def test_an_issue_is_closed_the_way_picked_and_opened_again(page_for, forge):
    page, git = _issues(page_for, forge)
    issues_view = git.get_by_role("button", name=re.compile(r"^Issues"))
    _listed(git, 5).click()
    expect(git.get_by_text("No description")).to_be_visible()

    _issue_actions(page, git).get_by_role("menuitem", name="Close as not planned").click()
    expect(git.get_by_role("button", name="Reopen", exact=True)).to_be_visible()
    expect(git.get_by_text("Not planned", exact=True)).to_be_visible()
    # the open ones listed beside it, and counted on the view, no longer hold it
    expect(_listed(git, 5)).to_have_count(0)
    expect(issues_view).to_have_text(re.compile(r"^Issues\s*1$"))

    comment = git.get_by_role("textbox", name="Comment")
    comment.fill("The marker was found.")
    git.get_by_role("button", name="Reopen with comment").click()
    expect(git.get_by_role("button", name="Close issue")).to_be_visible()
    expect(git.get_by_text("The marker was found.", exact=True)).to_be_visible()

    comment.fill("Moved back into place.")
    git.get_by_role("button", name="Close with comment").click()
    expect(git.get_by_role("button", name="Reopen", exact=True)).to_be_visible()
    expect(git.get_by_text("Closed", exact=True)).to_be_visible()

    closes = [
        (body["number"], body["reason"], body["body"]) for body in forge.asked_to("issue/close")
    ]
    assert closes == [
        (5, "not planned", ""),
        (5, "completed", "Moved back into place."),
    ], f"the terminal was asked to close {closes}"
    reopens = [(body["number"], body["body"]) for body in forge.asked_to("issue/reopen")]
    assert reopens == [(5, "The marker was found.")], f"the terminal was asked to reopen {reopens}"


def _back_to_the_files(page: Page) -> None:
    """A drafted request shows the chat; the Files tab beside it holds the Git panel."""
    files_tab = page.get_by_role("tab", name=re.compile("logbook"))
    files_tab.click()
    expect(files_tab).to_have_attribute("aria-selected", "true")


def test_the_chat_buttons_write_a_ready_request_into_the_message_input(page_for, forge):
    page, git = _issues(page_for, forge)
    box = chat_input(page)
    _listed(git, 3).click()
    expect(git.get_by_text("Low water is on Friday.")).to_be_visible()
    git.get_by_role("button", name="Work on this").click()
    for line in (
        "Work on issue #3: Soundings missing off the pier",
        "https://github.com/example/logbook/issues/3",
        "The lantern survey skipped the pier head.",
        "Low water is on Friday.",
        "Create a branch for it, make the change, and summarize what you changed.",
    ):
        expect(box).to_contain_text(line)

    _back_to_the_files(page)
    git.get_by_role("button", name="Pull requests", exact=True).click()
    _listed(git, 16).click()
    git.get_by_role("button", name="Resolve conflicts in chat").click()
    expect(box).to_contain_text(
        "Pull request #16 (Redraw the tide gauge) has merge conflicts with trunk."
    )
    # what was drafted before stays
    expect(box).to_contain_text("Work on issue #3: Soundings missing off the pier")

    _back_to_the_files(page)
    _listed(git, 12).click()
    git.get_by_role("button", name="Checks", exact=True).click()
    expect(git.get_by_text("soundings/lint", exact=True)).to_be_visible()
    git.get_by_role("button", name="Fix", exact=True).click()
    expect(box).to_contain_text(
        'The check "soundings/lint" is failing on pull request #12 (Chart the east channel).'
    )
    _back_to_the_files(page)
    git.get_by_role("button", name="Address feedback").click()
    expect(box).to_contain_text(
        "Address the open feedback on pull request #12 (Chart the east channel) on the branch"
        " east-channel."
    )
    expect(box).to_contain_text("Checked against the lead line.")
    # drafted for the person to read, never sent
    expect(page.get_by_role("button", name="Stop")).to_have_count(0)
    assert "/c/" not in page.url, f"a drafted request started a chat: {page.url}"
