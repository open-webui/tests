"""The steps a reply shows for the work of a coding agent or of the terminal tools.

New in 0.13.4: each step reads as what it did ("Ran <command>", "Edited <file>", "Read <file>"),
a group of steps is summed up as for example "Ran 2 commands, Edited 1 file, Read 3 files", a
command opens on itself with its output and, when it failed, its exit code, and an edit opens on
the diff of each file with its added and removed lines.
"""

from __future__ import annotations

import re

from playwright.sync_api import Locator, expect

from utils.chat_ui import steps_summary


def open_steps(reply: Locator) -> None:
    """Open the reply's group of steps from its summary line."""
    reply.get_by_role("button", name="Toggle details", expanded=False).click()
    expect(steps_summary(reply)).to_have_attribute("aria-expanded", "true")


def step(reply: Locator, label: str) -> Locator:
    """The step of an opened group whose line reads `label` (an edit adds its line counts)."""
    # Playwright's selector parser takes an unescaped slash as the regex's end
    pattern = re.escape(label).replace("/", "\\/")
    return reply.get_by_role("button", name=re.compile(rf"^{pattern}(\s|$)"))


def open_step(reply: Locator, label: str) -> Locator:
    """Open the step reading `label`; returns the step with what it shows when open."""
    line = step(reply, label)
    expect(line).to_have_count(1)
    line.click()
    expect(line).to_have_attribute("aria-expanded", "true")
    # the step's line and its opened details share a parent with no role of its own
    return line.locator("xpath=..")


def diff_line(details: Locator, text: str) -> Locator:
    """The line of an opened edit's diff that holds `text`."""
    # one span per diff line, with no role or label of its own (as in e2e/chat/test_diff_blocks.py)
    return details.locator("pre > code > span").filter(has_text=text)


def diff_sign(line: Locator) -> Locator:
    """The +, - or blank a diff line starts with."""
    return line.locator(":scope > span:not([data-line-number])").first
