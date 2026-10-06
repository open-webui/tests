"""Journey: an admin imports accounts from a CSV file and the people in it sign in.

In Admin Panel > Users, Add User > CSV Import takes a file of name, email, password and role. The
imported accounts show in the list with the roles the file gave them, and each signs in on the
auth page with the password from the file: a user reaches the chat, a pending account reaches the
"Account Activation Pending" screen and the wrong password is refused. Each test works as a fresh
admin on accounts of its own.

Discriminates: passes on dev 176d31d1d; in a frontend copy, the import sending every row with the
role `user` turns the roles check red (the admin and pending rows show "user"), and sending the
email in the password column turns the sign-in check red (the file's password is refused).
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Page, expect

from utils.chat_ui import chat_input

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

PAGE_TIMEOUT_MS = 30_000


def _import_csv(page: Page, csv: str) -> None:
    page.goto("/admin/users")
    page.get_by_role("main").get_by_role("button", name="Add User").click()
    form = page.get_by_role("dialog").filter(has_text="Add User")
    form.get_by_role("button", name="CSV Import").click()
    with page.expect_file_chooser() as chooser:
        form.get_by_role("button", name="Click here to select a csv file.").click()
    chooser.value.set_files(
        files=[{"name": "users.csv", "mimeType": "text/csv", "buffer": csv.encode()}]
    )
    form.get_by_role("button", name="Save").click()


def _sign_in(page: Page, email: str, password: str) -> None:
    page.goto("/auth")
    page.get_by_label("Email").fill(email)
    page.get_by_label("Password", exact=True).fill(password)
    page.get_by_role("button", name="Sign in", exact=True).click()


def test_imported_accounts_show_their_roles_and_sign_in_with_the_csv_password(
    make_user, page_for, context
):
    suffix = uuid.uuid4().hex[:8]
    rows = [
        (f"Csv Member {suffix}", f"csv-member-{suffix}@example.com", f"member-pw-{suffix}", "user"),
        (f"Csv Boss {suffix}", f"csv-boss-{suffix}@example.com", f"boss-pw-{suffix}", "admin"),
        (f"Csv Wait {suffix}", f"csv-wait-{suffix}@example.com", f"wait-pw-{suffix}", "pending"),
    ]
    page = page_for(make_user(role="admin"))

    _import_csv(
        page,
        "\n".join(["Name,Email,Password,Role", *(",".join(row) for row in rows)]) + "\n",
    )
    expect(page.get_by_text("Successfully imported 3 users.")).to_be_visible()

    users = page.get_by_role("main")
    users.get_by_role("textbox", name="Search").fill(suffix)
    for name, email, _, role in rows:
        row = users.get_by_role("row").filter(has_text=email)
        expect(row).to_contain_text(name)
        expect(row.get_by_role("button", name="Change User Role")).to_have_text(role)

    visitor = context.new_page()
    (_, member_email, member_password, _), _, (_, wait_email, wait_password, _) = rows
    _sign_in(visitor, member_email, member_password)
    expect(chat_input(visitor)).to_be_visible(timeout=PAGE_TIMEOUT_MS)
    visitor.evaluate("localStorage.clear()")
    _sign_in(visitor, wait_email, wait_password)
    expect(visitor.get_by_text("Account Activation Pending")).to_be_visible(timeout=PAGE_TIMEOUT_MS)
    visitor.evaluate("localStorage.clear()")
    _sign_in(visitor, member_email, "not-the-password")
    expect(visitor.get_by_text("The email or password provided is incorrect.")).to_be_visible()
    expect(chat_input(visitor)).to_be_hidden()
