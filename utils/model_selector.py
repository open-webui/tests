"""Finding a model in the chat's model selector by name, the way a person searches for one.

The helpers take the page, meaning the page's main chat, or a `ChatSurface`, meaning that chat's
own selector: the main one or the one in a side tab."""

from __future__ import annotations

import re

from playwright.sync_api import Locator, Page

from utils.chat_surface import ChatSurface

SELECTOR_BUTTON = re.compile("Select a model|Selected model")


def model_search(chat: Page | ChatSurface) -> Locator:
    """The selector's search box, which announces the option the arrow keys are on."""
    page = chat.page if isinstance(chat, ChatSurface) else chat
    return page.get_by_role("combobox", name="Search In Models")


def model_options(chat: Page | ChatSurface, name: str) -> Locator:
    """The selector's options for `name`, after searching for it; opens the selector if closed."""
    page = chat.page if isinstance(chat, ChatSurface) else chat
    selector_holder = chat.panel if isinstance(chat, ChatSurface) else page
    search = model_search(page)
    if not search.is_visible():
        selector_holder.get_by_role("button", name=SELECTOR_BUTTON).click()
    search.fill(name)
    return page.get_by_role("listbox", name="Available models").get_by_role(
        "option", name=f"Select {name} model"
    )


def select_model(chat: Page | ChatSurface, name: str) -> None:
    model_options(chat, name).click()
