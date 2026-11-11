"""Regression: inline math touching an em dash or an en dash was left as plain text.

Fix 8e03389de (PR open-webui/open-webui#31901). In a reply such as "constants, e, i and 0 in one
equation" written with a dash on each side of the formulas, the formula next to the dash did
not render and showed as "(0)". The math extension only accepts certain characters right before
and after a delimiter (spaces, commas, brackets) and left both dashes out, for `$...$` as well as
`\\(...\\)`.

Discriminates: passes on the dev b859124f9 build, fails on that build with 8e03389de reverted
(the formulas touching a dash show as plain text, so fewer rendered formulas than written).

Every case runs in the main chat and in a side chat, the saved chat dragged into a tab beside a
new chat (`utils/chat_surface.py`).
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

from harness.chat_history import seed_chat
from utils.chat_ui import last_reply

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

DASHES = {"an em dash": chr(0x2014), "an en dash": chr(0x2013)}
FORMS = {
    "parentheses": "constants{d}\\(e\\), \\(i\\) and \\(0\\){d}in one equation",
    "dollars": "constants{d}$e$, $i$ and $0${d}in one equation",
}


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("dash", DASHES)
def test_three_formulas_touching_dashes_all_render(chat_for, make_user, form, dash):
    written = FORMS[form].format(d=DASHES[dash])
    owner = make_user()
    with owner.client() as client:
        chat_id, _ = seed_chat(
            client,
            [{"role": "user", "content": "Euler?"}, {"role": "assistant", "content": written}],
        )
    chat = chat_for(owner, chat_id=chat_id)

    reply = last_reply(chat)
    expect(reply).to_contain_text("in one equation")
    expect(reply.locator(".katex")).to_have_count(3)
    expect(reply).not_to_contain_text("\\(")
