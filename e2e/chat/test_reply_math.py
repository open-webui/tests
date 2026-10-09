"""Journey: math in a reply is typeset, inline or on a line of its own, and plain prices stay text.

A formula between `$...$`, `\\(...\\)` or `\\ce{...}` is typeset in the line it sits in, and one
between `$$...$$` on lines of their own, `\\[...\\]` or an `equation` environment is typeset on a
line of its own. A formula touching punctuation (brackets, a full stop, a comma, a colon, a
question mark, quotation marks of other languages, Chinese punctuation) is still typeset. Prices
such as `$5 and $10`, dollars inside a word and dollars in a code span stay as written. A formula
that does not parse shows its source marked as an error while the formulas around it are still
typeset, clicking a formula copies its source, and the chat looks the same after a reload. The
dashes next to a formula are covered in test_math_next_to_dashes.py.

`test_prices_dollars_in_words_and_code_spans_stay_as_written` failed on dev 9bbb95048 in CI with the
reply left blank and passed 3 of 3 locally: since de73bb830 the reply in a new chat sometimes stays
blank until a reload although the server saved it whole (open-webui/open-webui#32091).

Discriminates: passes on the dev ebc6add67 build. In its mutation build (the `rendering-front`
copy: the characters allowed around a delimiter cut to spaces only and display math drawn
inline) the punctuation cases, the display test and its reload twin go red; the prices test, the
broken formula test (no punctuation next to its formulas) and the copy test stay green there.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from utils.chat_ui import expect_reply, last_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


def ask(page: Page, upstream, answer: str, prompt: str) -> Locator:
    upstream.queue(reply.text(answer + "\n\nEnd of maths.", match=reply.answering(prompt)))
    send(page, prompt)
    expect_reply(page, "End of maths.")
    return last_reply(page)


def typeset_sources(box: Locator) -> Locator:
    """The source of every typeset formula, in reading order."""
    return box.locator(".katex annotation")


def display_sources(box: Locator) -> Locator:
    return box.locator(".katex-display annotation")


INLINE_AND_DISPLAY = r"""Inline: $a^2$, \(b_1\) and \ce{H2O} in one line.

$$
\int_0^1 x\,dx
$$

\[
E = mc^2
\]

\begin{equation}
c = \sqrt{a^2 + b^2}
\end{equation}"""


def expect_inline_and_display_math(box: Locator) -> None:
    expect(typeset_sources(box)).to_have_count(6)
    expect(display_sources(box)).to_have_text(
        [r"\int_0^1 x\,dx", "E = mc^2", r"c = \sqrt{a^2 + b^2}"]
    )
    inline_line = box.locator("p").filter(has_text="Inline:")
    expect(inline_line.locator(".katex")).to_have_count(3)
    expect(inline_line.locator(".katex-display")).to_have_count(0)
    expect(inline_line).to_contain_text("in one line.")
    shown = box.inner_text()
    assert "$$" not in shown and "\\[" not in shown and "begin{equation}" not in shown, shown


def test_inline_formulas_stay_in_their_line_and_display_formulas_get_their_own(
    page_for, make_user, upstream
):
    page = page_for(make_user())

    box = ask(page, upstream, INLINE_AND_DISPLAY, "show me some formulas")

    expect_inline_and_display_math(box)


def test_the_typeset_formulas_are_the_same_after_a_reload(page_for, make_user, upstream):
    page = page_for(make_user())
    ask(page, upstream, INLINE_AND_DISPLAY, "show me the formulas again")

    page.reload()

    expect_reply(page, "End of maths.")
    expect_inline_and_display_math(last_reply(page))


PUNCTUATION = {
    "brackets": "the root ($x_1$) is real",
    "a full stop": "the root is $x_1$. Next",
    "a comma and a colon": "roots $x_1$, $x_2$: both real",
    "a question mark": "is it $x_1$? Yes",
    "an exclamation mark": "it is $x_1$! Yes",
    "guillemets": "la racine «$x_1$» est réelle",
    "german quotes": "die Wurzel „$x_1$“ ist reell",
    "chinese punctuation": "根是$x_1$，也是$x_2$。",
    "a parenthesised form": "the root (\\(x_1\\)) is real",
}


@pytest.mark.parametrize("case", PUNCTUATION)
def test_a_formula_touching_punctuation_is_still_typeset(page_for, make_user, upstream, case):
    page = page_for(make_user())
    written = PUNCTUATION[case]

    box = ask(page, upstream, written, f"punctuation case {case}")

    expected = ["x_1", "x_2"] if "x_2" in written else ["x_1"]
    expect(typeset_sources(box)).to_have_text(expected)
    shown = box.inner_text()
    assert "$" not in shown and "\\(" not in shown, f"a delimiter is left on screen: {shown!r}"


def test_prices_dollars_in_words_and_code_spans_stay_as_written(page_for, make_user, upstream):
    page = page_for(make_user())
    answer = "The ticket costs $5 and the meal $10 today.\n\nA var$name$here and `$x$` in code."

    box = ask(page, upstream, answer, "how much is it")

    expect(box).to_contain_text("The ticket costs $5 and the meal $10 today.")
    expect(box).to_contain_text("A var$name$here and")
    expect(box.get_by_role("code").filter(has_text="$x$")).to_be_visible()
    expect(box.locator(".katex")).to_have_count(0)


def test_a_formula_that_does_not_parse_shows_its_source_and_the_others_still_render(
    page_for, make_user, upstream
):
    page = page_for(make_user())
    answer = r"Good $y^2$ then broken $\frac{1}$ and good again $w_3$ here"

    box = ask(page, upstream, answer, "show me a broken formula")

    expect(typeset_sources(box)).to_have_text(["y^2", "w_3"])
    broken = box.locator(".katex-error")
    expect(broken).to_have_text(r"\frac{1}")
    expect(broken).to_have_attribute("title", re.compile("^ParseError"))


def test_clicking_a_formula_copies_its_source(page_for, make_user, upstream):
    page = page_for(make_user(), permissions=["clipboard-read", "clipboard-write"])
    box = ask(page, upstream, r"The area is $\pi r^2$ exactly.", "what is the area")

    box.locator(".katex").click()

    expect(page.get_by_text("Copied to clipboard")).to_be_visible()
    assert page.evaluate("navigator.clipboard.readText()") == r"\pi r^2"
