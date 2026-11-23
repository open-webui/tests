"""Where the highlights over a cited passage lie, measured on screen.

A cited PDF or Word document opens in a tab beside the chat with the cited lines highlighted in
yellow. The highlights are drawn over the page and have no role or name, so they are read by where
they lie: `expect_marks_on_cited_text` checks that each highlight sits on the page over text and
that each cited line is covered by highlights, wherever the tab's width, the zoom or the scroll put
the page. `expect_card_at` checks that the passage card a highlight shows sits at the cited text.
"""

from __future__ import annotations

from playwright.sync_api import Locator, expect
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from harness import slow_machine

# marks are redrawn shortly after the page moves; give them that long to settle
SETTLE_TIMEOUT_MS = slow_machine.scaled(5_000)

# highlights in view whose middle lies on no character of the page, and cited lines (passed in)
# whose glyphs are less than 80% covered by highlights; empty lists when every mark is in place
MISPLACED = """([panel, lines]) => {
    const view = panel.getBoundingClientRect();
    const marks = [...panel.querySelectorAll('[data-quote]')]
        .map((mark) => mark.getBoundingClientRect())
        .filter((box) => box.width > 0 && box.height > 0);
    const inView = (box) => box.top >= view.top && box.bottom <= view.bottom
        && box.left >= view.left && box.right <= view.right;
    const offText = [];
    for (const box of marks.filter(inView)) {
        const x = box.left + box.width / 2, y = box.top + box.height / 2;
        const caret = document.caretRangeFromPoint(x, y);
        const node = caret?.startContainer;
        let onText = false;
        if (node?.nodeType === Node.TEXT_NODE && node.length && panel.contains(node)) {
            const glyph = document.createRange();
            const offset = Math.min(caret.startOffset, node.length - 1);
            glyph.setStart(node, offset);
            glyph.setEnd(node, offset + 1);
            const area = glyph.getBoundingClientRect();
            onText = x >= area.left - 2 && x <= area.right + 2
                && y >= area.top - 2 && y <= area.bottom + 2;
        }
        if (!onText) offText.push([box.left, box.top, box.width].map(Math.round));
    }
    const uncovered = [];
    for (const line of lines) {
        const text = document.createRange();
        text.selectNodeContents(line);
        for (const glyphs of text.getClientRects()) {
            if (glyphs.width < 1 || glyphs.height < 1 || !inView(glyphs)) continue;
            let covered = 0;
            for (const box of marks) {
                const width = Math.min(box.right, glyphs.right) - Math.max(box.left, glyphs.left);
                const height = Math.min(box.bottom, glyphs.bottom) - Math.max(box.top, glyphs.top);
                if (width > 0 && height > 0) covered += width * height;
            }
            const share = covered / (glyphs.width * glyphs.height);
            const name = line.textContent.trim();
            if (share < 0.8) uncovered.push(`${name} (${Math.round(share * 100)}%)`);
        }
    }
    return { offText, uncovered };
}"""


def expect_marks_on_cited_text(panel: Locator, *lines: Locator) -> None:
    """Every highlight in view of `panel` lies on the page's text, and each cited line in view is
    covered by highlights."""
    handles = [line.element_handle() for line in lines]
    try:
        panel.page.wait_for_function(
            f"""([panel, lines]) => {{
                const found = ({MISPLACED})([panel, lines]);
                return panel.querySelector('[data-quote]')
                    && !found.offText.length && !found.uncovered.length;
            }}""",
            arg=[panel.element_handle(), handles],
            timeout=SETTLE_TIMEOUT_MS,
        )
    except PlaywrightTimeoutError:
        found = panel.page.evaluate(MISPLACED, [panel.element_handle(), handles])
        raise AssertionError(
            "the highlights are not on the cited text: "
            f"highlights off the text at (left, top, width) {found['offText']}, "
            f"cited lines not covered {found['uncovered']}"
        ) from None


def expect_card_at(card: Locator, line: Locator) -> None:
    """The passage card sits at the cited line, just above or below it and overlapping it
    sideways."""
    expect(card).to_be_visible()
    text, shown = line.bounding_box(), card.bounding_box()
    assert text and shown, "the cited line or its card has no box"
    beside = shown["x"] < text["x"] + text["width"] and shown["x"] + shown["width"] > text["x"]
    below = 0 <= shown["y"] - (text["y"] + text["height"]) <= 24
    above = 0 <= text["y"] - (shown["y"] + shown["height"]) <= 24
    assert beside and (below or above), (
        f"the passage card at {shown} is not at the cited text at {text}"
    )
