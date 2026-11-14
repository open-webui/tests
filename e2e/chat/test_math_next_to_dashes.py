"""Regression: inline math touching an em dash or an en dash was left as plain text.

Fix 8e03389de (PR open-webui/open-webui#31901). In a reply such as "constants, e, i and 0 in one
equation" written with a dash on each side of the formulas, the formula next to the dash did
not render and showed as "(0)". The math extension only accepts certain characters right before
and after a delimiter (spaces, commas, brackets) and left both dashes out, for `$...$` as well as
`\\(...\\)`.

Discriminates: passes on the dev b859124f9 build, fails on that build with 8e03389de reverted
(the formulas touching a dash show as plain text, so fewer rendered formulas than written).

The hyphen U+2010 and the non-breaking hyphen U+2011 were still missing from that list (issue
open-webui/open-webui#34237). Models such as gpt-oss write U+2011 in most replies, so
"\\(x^2\\)‑term" stayed plain text, and with a second formula later in the paragraph the text
between the two went to KaTeX as one red error.

Discriminates: the hyphen cases fail on dev 2547992b8 (only one of the three formulas renders)
and pass with the fix for #34237.

Each function runs two of the four form and dash pairs, one with each form and each dash or
hyphen, so both delimiters and every character stay covered.

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


@pytest.mark.parametrize(
    ("form", "dash"), [("parentheses", "an em dash"), ("dollars", "an en dash")]
)
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


HYPHENS = {"a non-breaking hyphen": chr(0x2011), "a hyphen": chr(0x2010)}
HYPHEN_FORMS = {
    "parentheses": "\\(x^2\\)-term\n\\(x^2\\){h}term\npre{h}\\(x^2\\)",
    "dollars": "$x^2$-term\n$x^2${h}term\npre{h}$x^2$",
}


@pytest.mark.parametrize(
    ("form", "hyphen"), [("parentheses", "a non-breaking hyphen"), ("dollars", "a hyphen")]
)
def test_formulas_touching_unicode_hyphens_all_render(chat_for, make_user, form, hyphen):
    written = HYPHEN_FORMS[form].format(h=HYPHENS[hyphen])
    owner = make_user()
    with owner.client() as client:
        chat_id, _ = seed_chat(
            client,
            [{"role": "user", "content": "Squares?"}, {"role": "assistant", "content": written}],
        )
    chat = chat_for(owner, chat_id=chat_id)

    reply = last_reply(chat)
    expect(reply).to_contain_text(f"pre{HYPHENS[hyphen]}")
    expect(reply.locator(".katex")).to_have_count(3)
    expect(reply.locator(".katex-error")).to_have_count(0)
    expect(reply).not_to_contain_text("\\(")
