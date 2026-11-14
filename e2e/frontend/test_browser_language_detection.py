"""Regression: a browser reporting a bare or regional language code got the English interface.

Fix `d603b5799` (open-webui/open-webui#30377): Firefox reports German, Dutch and Polish as bare
codes (`de`, `nl`, `pl`), both Chrome and Firefox report Japanese as `ja`, and Chrome in Latin
America reports `es-419`. No translation bundle is keyed by any of them. Since 0.11.4 the
language detector cached the raw code before anything matched it to a bundle, so these users got
English and kept it on every later visit. The detector now maps a reported code onto a bundle
(`de` and `de-AT` to `de-DE`, `es-419` to `es-ES`, `fr` to `fr-FR`, `ja` to `ja-JP`); a code
with a bundle of its own stays as it is.

A signed-out browser opens the sign-in page with that language as its only preference. Each
function visits its codes one after the other, each in a browser context reporting that code,
and the failure message names the code.

The same file holds the administrator's default language, open-webui/open-webui#31548, fix
`4987711391` (PR open-webui/open-webui#31551): since 0.11.4 a new visitor got the browser's
language even when an administrator set DEFAULT_LOCALE, because the detector remembered it as
the visitor's own pick. On an instance with DEFAULT_LOCALE set, a first visit now shows the
default language whatever the browser reports, while a language stored by a pick in Settings and
a `?lang=` link still win. Without DEFAULT_LOCALE the browser's language is followed, as above.

Discriminates: passes on the efe63bd34 build (the default language tests on dev a5bc78300); on a
build with d603b5799 reverted every mapped case fails (the raw code is stored as the locale and,
but for `fr`, the page shows English). The unchanged cases pass on both. With `4987711391`
reverted the first visit test fails (the page shows the browser's French, not the default
German); the stored pick and `?lang=` tests pass on both.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Browser, expect

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

SIGN_IN_LABEL = {
    "de-DE": "Anmelden",
    "nl-NL": "Inloggen",
    "pl-PL": "Zaloguj się",
    "ja-JP": "サインイン",
    "es-ES": "Iniciar Sesión",
    "fr-FR": "Connexion",
    "fr-CA": "Connexion",
    "pt-BR": "Entrar",
}


@pytest.fixture
def open_sign_in(browser: Browser, config):
    """`open_sign_in(reported)` opens the sign-in page in a browser that reports `reported`."""
    contexts = []

    def open_page(reported: str, base_url: str | None = None, path: str = "/auth", stored=None):
        context = browser.new_context(locale=reported, base_url=base_url or config.base_url)
        context.set_default_timeout(config.default_timeout)
        contexts.append(context)
        if stored:
            context.add_init_script(f"localStorage.locale = {stored!r}")
        page = context.new_page()
        page.goto(path)
        return page

    yield open_page
    for context in contexts:
        context.close()


REPORTED_BUNDLES = [
    ("de", "de-DE"),
    ("de-AT", "de-DE"),
    ("nl", "nl-NL"),
    ("pl", "pl-PL"),
    ("ja", "ja-JP"),
    ("es-419", "es-ES"),
    ("fr", "fr-FR"),
]


def test_a_reported_language_gets_its_translation(open_sign_in):
    # each code needs a browser reporting it, so every one opens its own context
    for reported, bundle in REPORTED_BUNDLES:
        page = open_sign_in(reported)

        expect(
            page.get_by_role("button", name=SIGN_IN_LABEL[bundle], exact=True),
            f"a browser reporting {reported!r} did not get the {bundle} translation",
        ).to_be_visible()
        assert page.evaluate("localStorage.locale") == bundle, (
            f"a browser reporting {reported!r} was not matched to the {bundle} translation"
        )


def test_a_language_with_its_own_bundle_keeps_it(open_sign_in):
    for reported, label in [("fr-CA", "Connexion"), ("pt-BR", "Entrar"), ("en-US", "Sign in")]:
        page = open_sign_in(reported)

        expect(
            page.get_by_role("button", name=label, exact=True),
            f"a browser reporting {reported!r} did not keep its own translation",
        ).to_be_visible()
        assert page.evaluate("localStorage.locale") == reported


def test_a_regional_code_of_another_script_is_not_moved_to_a_sibling(open_sign_in):
    page = open_sign_in("zh-HK")

    expect(page.get_by_role("button", name="Sign in", exact=True)).to_be_visible()
    assert page.evaluate("localStorage.locale") == "zh-HK", (
        "Traditional Chinese from Hong Kong was sent to the Simplified zh-CN translation"
    )


@pytest.fixture(scope="module")
def german_default(instance_with) -> str:
    """The base URL of an instance whose administrator set German as the default language."""
    return instance_with({"DEFAULT_LOCALE": "de-DE"}).base_url


@pytest.mark.slow
def test_a_first_visit_gets_the_configured_default_language(open_sign_in, german_default):
    page = open_sign_in("fr-FR", base_url=german_default)

    expect(
        page.get_by_role("button", name=SIGN_IN_LABEL["de-DE"], exact=True),
        "#31548: the browser language won over DEFAULT_LOCALE on a first visit",
    ).to_be_visible()


@pytest.mark.slow
def test_a_language_picked_in_settings_wins_over_the_configured_default(
    open_sign_in, german_default
):
    # Settings keeps the pick in localStorage
    page = open_sign_in("nl-NL", base_url=german_default, stored="fr-FR")

    expect(page.get_by_role("button", name=SIGN_IN_LABEL["fr-FR"], exact=True)).to_be_visible()


@pytest.mark.slow
def test_a_lang_link_wins_over_the_configured_default(open_sign_in, german_default):
    page = open_sign_in("nl-NL", base_url=german_default, path="/auth?lang=fr-FR")

    expect(page.get_by_role("button", name=SIGN_IN_LABEL["fr-FR"], exact=True)).to_be_visible()
