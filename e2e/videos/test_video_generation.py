"""Journey: video generation set up in Admin Settings, switched on in the chat and watched there.

New in v0.13.4. The admin turns on Video Generation in the Videos section of Admin Settings >
Images & Videos, adds an OpenAI compatible connection to a local stand-in for a video API
(`harness.video_providers`) with two model ids, a default size and length and access for everyone,
picks the Default Video Model and saves; all of it is still there after a reload. The admin then
grants the Video Generation permission (off by default) in a group's Permissions tab. A member of
that group turns on Video in the chat's More menu, picks the other video model there and asks for
a video: the model calls `generate_video`, the provider gets the picked model with the admin's
size and length, and the reply shows a video the browser loads and plays, also after a reload.
Video is missing from the menu for a user without the permission and while Video Generation is
off. A picture attached to the chat becomes the first frame of the video made from it.
The provider's requests, the pick order and access rules are pinned in
integration/videos/test_video_generation.py.

Picking the model in the menu and then pressing Escape to close it also switches Video off again,
as Escape in the message box clears its selections, so the tests close the menu by clicking away.

Discriminates: passes on the dev f80a4f782 build. In a backend copy whose `generate_video` keeps
its files off the chat message the setup test fails (no video in the reply), whose
`video_generations` never loads the starting image the attached picture test fails, and whose
`/api/config` reports video generation on whatever the admin chose the switch test fails.
"""

from __future__ import annotations

import base64
import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.access import make_group
from harness.image_engines import PNG_BASE64
from harness.video_providers import (
    VIDEO,
    VIDEOS_CONFIG,
    created_jobs,
    form_fields,
    model_ref,
    save_video_settings,
    serve_video_api,
    video_connection,
    video_settings,
)
from utils.admin_settings import save
from utils.chat_ui import (
    REPLY_TIMEOUT_MS,
    chat_input,
    expect_reply,
    open_integrations_menu,
    replies,
    send,
)
from utils.image_settings import add_model_id, connection_dialog, open_image_settings
from utils.tool_servers import make_public

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

PICTURE = base64.b64decode(PNG_BASE64)
FILE_CONTENT_URL = re.compile(r"/api/v1/files/[^/]+/content$")
VIDEO_ENTRY = re.compile(r"^Video")


@pytest.fixture
def videos(admin, preserve, listener):
    """`videos(*connections, **settings)` saves video settings, restored after the test."""
    preserve(VIDEOS_CONFIG)
    serve_video_api(listener)

    def configure(*connections: dict, **settings) -> None:
        with admin.client() as client:
            save_video_settings(client, video_settings(*connections, **settings))

    return configure


def tag() -> str:
    return uuid.uuid4().hex[:8]


def videos_section(settings: Locator) -> Locator:
    heading = settings.page.get_by_role("heading", name="Videos", exact=True)
    return settings.locator("section").filter(has=heading)


def switch_on(region: Locator, name: str) -> None:
    switch = region.get_by_role("switch", name=name, exact=True)
    if switch.get_attribute("aria-checked") != "true":
        switch.click()
    expect(switch).to_be_checked()


def grant_video_generation(page: Page, group_name: str) -> None:
    """Switch Video Generation on in the group's Permissions tab and save."""
    page.goto("/admin/users/groups")
    page.get_by_role("main").get_by_role("textbox", name="Search Groups").fill(group_name)
    page.get_by_role("button", name=re.compile(rf"^{group_name} \d+ direct members")).click()
    editing = page.get_by_role("dialog").filter(has_text="Edit User Group")
    editing.get_by_role("button", name="Permissions", exact=True).click()
    switch_on(editing, "Video Generation")
    editing.get_by_role("button", name="Save").click()
    expect(page.get_by_text("Group updated successfully")).to_be_visible()


def named_group(admin, members) -> str:
    """A group of `members` that grants nothing beyond the defaults; returns its name."""
    group_id = make_group(admin, members)
    with admin.client() as client:
        return client.get(f"/api/v1/groups/id/{group_id}").json()["name"]


def open_chat(page: Page) -> None:
    page.goto("/")
    expect(chat_input(page)).to_be_visible(timeout=REPLY_TIMEOUT_MS)


def video_entry(menu: Locator) -> Locator:
    return menu.get_by_role("menuitemcheckbox", name=VIDEO_ENTRY)


def turn_on_video(page: Page, model: str | None = None) -> None:
    """Switch Video on in the More menu, picking `model` there when given."""
    menu = open_integrations_menu(page)
    entry = video_entry(menu)
    if model:
        menu.get_by_role("menuitem", name="Choose video model").click()
        menu.get_by_role("menuitemradio", name=model, exact=True).click()
        expect(entry).to_contain_text(model)
    else:
        entry.click()
    expect(entry).to_have_attribute("aria-checked", "true")
    # Escape in the message box also clears its selections, so close the menu by clicking away
    chat_input(page).click()
    expect(menu).to_have_count(0)


def film(page: Page, upstream, prompt: str, **arguments) -> str:
    """Ask for a video of `prompt`; returns the reply text."""
    request = f"please film {prompt}"
    answer = f"Here is {prompt}."
    upstream.queue(
        reply.tool_call(
            "generate_video", {"prompt": prompt, **arguments}, match=reply.answering(request)
        ),
        reply.text(answer, match=reply.answering(request)),
    )
    send(page, request)
    expect_reply(page, answer)
    follow_ups = [body for body in upstream.chat_requests() if reply.answering(request)(body)]
    results = [
        entry["content"]
        for body in follow_ups
        for entry in reply.conversation(body)
        if entry["role"] == "tool"
    ]
    assert results and '"success"' in results[-1], f"the video tool answered {results}"
    return answer


def reply_video(page: Page) -> Locator:
    return replies(page).last.get_by_label(re.compile("generated-video"))


def expect_playing_video(page: Page, actor) -> str:
    """The reply's video is a stored file the browser loads and plays; returns its address."""
    video = reply_video(page)
    expect(video).to_be_visible()
    source = video.get_attribute("src")
    assert FILE_CONTENT_URL.search(source), f"the video is not a stored file: {source}"
    page.wait_for_function(
        "(video) => video.readyState >= 1 && video.duration > 0",
        arg=video.element_handle(),
        timeout=REPLY_TIMEOUT_MS,
    )
    video.evaluate("(video) => { video.muted = true; return video.play(); }")
    page.wait_for_function(
        "(video) => video.currentTime > 0 || video.ended",
        arg=video.element_handle(),
        timeout=REPLY_TIMEOUT_MS,
    )
    with actor.client() as client:
        stored = client.get(source)
    assert stored.status_code == 200, stored.text
    assert stored.content == VIDEO, "the video in the chat is not the one the provider made"
    return source


def test_an_admin_sets_up_video_generation_and_a_user_watches_the_video(
    page_for, make_user, admin, videos, listener, upstream
):
    videos(enable=False)
    admin_page = page_for(make_user(role="admin"))
    settings = open_image_settings(admin_page)
    section = videos_section(settings)
    switch_on(section, "Video Generation")
    section.get_by_role("button", name="Add Connection").click()
    dialog = connection_dialog(admin_page)
    expect(dialog.get_by_label("Connection Type")).to_have_value("openai")
    dialog.get_by_label("Name", exact=True).fill("Studio")
    dialog.get_by_label("URL", exact=True).fill(listener.base_url)
    dialog.get_by_placeholder("API Key").fill("sk-films")
    dialog.get_by_role("button", name="Verify Connection").click()
    expect(admin_page.get_by_text("Server connection verified")).to_be_visible()
    add_model_id(dialog, "sora-2")
    add_model_id(dialog, "sora-2-pro")
    dialog.get_by_role("button", name="Advanced").click()
    dialog.get_by_label("Video Size", exact=True).fill("1280x720")
    dialog.get_by_label("Video Length", exact=True).fill("4")
    make_public(admin_page, dialog)
    with admin_page.expect_response(lambda r: "/videos/config/update" in r.url) as stored:
        dialog.get_by_role("button", name="Save", exact=True).click()
    assert stored.value.ok, f"saving the video connection failed: {stored.value.text()}"
    expect(dialog).to_be_hidden()
    with admin.client() as client:
        [connection] = client.get(VIDEOS_CONFIG[0]).json()["VIDEO_CONNECTIONS"]
    default_ref = model_ref(connection["id"], "sora-2-pro")
    default_model = section.get_by_label("Default Video Model")
    expect(default_model.locator(f"option[value='{default_ref}']")).to_be_attached()
    default_model.select_option(default_ref)
    save(settings)

    settings = open_image_settings(admin_page)
    section = videos_section(settings)
    expect(section.get_by_role("switch", name="Video Generation", exact=True)).to_be_checked()
    # the name button also reads out the connection's model ids
    studio = section.get_by_role("button", name=re.compile(r"^Studio\b"))
    expect(studio).to_contain_text("Studio")
    expect(section.get_by_label("Default Video Model")).to_have_value(default_ref)

    person = make_user()
    grant_video_generation(admin_page, named_group(admin, [person]))

    page = page_for(person)
    open_chat(page)
    turn_on_video(page, "sora-2")
    answer = film(page, upstream, "a lighthouse at dusk")
    source = expect_playing_video(page, person)

    [job] = created_jobs(listener)
    sent = {name: value.decode() for name, (value, _) in form_fields(job).items()}
    assert (sent["model"], sent["size"], sent["seconds"]) == ("sora-2", "1280x720", "4"), sent
    assert job.headers.get("Authorization") == "Bearer sk-films"

    expect(page).to_have_url(re.compile(r"/c/"))
    page.reload()
    expect_reply(page, answer)
    expect(reply_video(page)).to_have_attribute("src", source)


def test_video_is_offered_only_with_the_permission_and_the_admin_switch(
    page_for, make_user, admin, videos, listener
):
    connection = video_connection(f"videos-{tag()}", listener.base_url, models=("sora-2",))
    videos(connection)
    permitted, refused = make_user(), make_user()
    make_group(admin, [permitted], {"features": {"video_generation": True}})

    permitted_page = page_for(permitted)
    open_chat(permitted_page)
    expect(video_entry(open_integrations_menu(permitted_page))).to_be_visible()
    refused_page = page_for(refused)
    open_chat(refused_page)
    refused_menu = open_integrations_menu(refused_page)
    expect(refused_menu.get_by_role("menuitem", name="Attachments")).to_be_visible()
    expect(video_entry(refused_menu)).to_have_count(0)

    videos(connection, enable=False)
    admin_page = page_for(make_user(role="admin"))
    open_chat(admin_page)
    admin_menu = open_integrations_menu(admin_page)
    expect(admin_menu.get_by_role("menuitem", name="Attachments")).to_be_visible()
    expect(video_entry(admin_menu)).to_have_count(0)


def attach_picture(page: Page) -> str:
    """Attach `PICTURE` through Upload Files; returns the stored file's id."""
    page.get_by_role("button", name="More", exact=True).last.click()
    with page.expect_response(_is_file_upload) as uploaded:
        with page.expect_file_chooser() as chooser:
            page.get_by_role("menu").get_by_role("menuitem", name="Upload Files").click()
        chooser.value.set_files({"name": "harbour.png", "mimeType": "image/png", "buffer": PICTURE})
    expect(page.get_by_role("button", name="Show image preview")).to_be_visible()
    return uploaded.value.json()["id"]


def _is_file_upload(response) -> bool:
    return response.request.method == "POST" and re.search(r"/api/v1/files/?(\?|$)", response.url)


def test_a_video_made_from_an_attached_picture_plays_in_the_chat(
    page_for, make_user, admin, videos, listener, upstream
):
    videos(video_connection(f"videos-{tag()}", listener.base_url, models=("sora-2",)))
    person = make_user()
    make_group(admin, [person], {"features": {"video_generation": True}})
    page = page_for(person)
    open_chat(page)
    turn_on_video(page)
    file_id = attach_picture(page)
    prompt = f"the harbour coming to life {tag()}"
    film(page, upstream, prompt, image_url=file_id)

    expect_playing_video(page, person)
    [job] = [job for job in created_jobs(listener) if prompt.encode() in job.body]
    frame, content_type = form_fields(job).get("input_reference", (b"", ""))
    assert (frame, content_type) == (PICTURE, "image/png"), (
        "the provider did not get the attached picture as the first frame"
    )
