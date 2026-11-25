"""Video providers played by a local service, and the admin video settings that point at them.

The admin's video settings (Admin Settings > Images & Videos > Videos) are a switch, a default
model and a list of video connections. `video_connection(...)` is one of them and
`video_settings(...)` the settings holding them; `save_video_settings` saves them over the admin's
configuration. Wrap a change on the shared instance in `preserve(VIDEOS_CONFIG)`.

`serve_video_api(listener)` routes a listener as a video API run as jobs: `POST /videos` starts
one, `GET /videos/{id}` reports it queued once and then done, `GET /videos/{id}/content` hands out
`VIDEO` (a one-second WebM a browser plays). OpenAI and LiteLLM take the create request as a
multipart form, OpenRouter as JSON; the stand-in answers both, lists OpenRouter's video models at
`/videos/models` and OpenAI's plain model list at `/models`. A job whose prompt contains
`FAILING_PROMPT` ends failed.
`created_jobs` reads back the create requests and `form_fields` the parts of a multipart one.
"""

from __future__ import annotations

import base64
import email.parser
import email.policy
import itertools
import threading

import httpx

from harness.listener import Answer, Listener, ReceivedRequest, json_answer

VIDEOS_CONFIG = ("/api/v1/videos/config", "/api/v1/videos/config/update")
# a connection without grants is the admins' alone
EVERYONE = [{"principal_type": "user", "principal_id": "*", "permission": "read"}]
# one second of four coloured 16x16 VP8 frames, what the stand-in provider renders
VIDEO = base64.b64decode(
    "GkXfo59ChoEBQveBAULygQRC84EIQoKEd2VibUKHgQJChYECGFOAZwEAAAAAAAKLEU2bdLpNu4tTq4QVSalmU6yBoU27"
    "i1OrhBZUrmtTrIHWTbuMU6uEElTDZ1OsggEnTbuMU6uEHFO7a1OsggJ17AEAAAAAAABZAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAVSalmsCrXsYMPQkBNgIxMYXZmNjEuMS4xMDBXQYxMYXZmNjEuMS4xMDBEiYhAj0AAAAAAABZUrmvMrgEAAAAAAABD"
    "14EBc8WIiB4P8LXiPhqcgQAitZyDdW5kiIEAhoVWX1ZQOIOBASPjg4QO5rKA4JSwgRC6gRCagQJVsIhVt4ECVbiBAhJU"
    "w2f6c3OfY8CAZ8iZRaOHRU5DT0RFUkSHjExhdmY2MS4xLjEwMHNz1WPAi2PFiIgeD/C14j4aZ8igRaOHRU5DT0RFUkSH"
    "k0xhdmM2MS4zLjEwMCBsaWJ2cHhnyKFFo4hEVVJBVElPTkSHkzAwOjAwOjAxLjAwMDAwMDAwMAAfQ7Z1QMnngQCjvYEA"
    "AICwAgCdASoQABAAAEcIhYWIhYSIAgICdaoD+AIM/SgA/vzelf+zu/Tu/Tu/wMf/yY7+THfyY7/5KACjq4EA+gDxAQAB"
    "EBAAGAc4CDU/8s1AAACb/+qxD5GCH/VY//3j9B8dkH/eGwCjq4EB9ACxAQABEBAAGAAwP/QMAAAA/v7W8H/8mNv0D2H/"
    "yY7/4szSzNLQzpCjq4EC7gCxAQABEBAAGAAwP/QMAAAA/vz9j//lB+EYg2/+UIf/EMaQxpDmSQAcU7trkbuPs4EAt4r3"
    "gQHxggGm8IED"
)
VIDEO_CONTENT_TYPE = "video/webm"
FAILING_PROMPT = "refuse this video"
FAILURE_MESSAGE = "The prompt was refused by the safety system."


def video_connection(
    connection_id: str,
    url: str,
    key: str = "sk-video",
    *,
    models: tuple[str, ...] = (),
    access_grants: list[dict] = EVERYONE,
    **config,
) -> dict:
    """An OpenAI compatible video connection at `url` offering `models` (none: the provider's
    own list) to everyone in `access_grants`; `provider="openrouter"` speaks OpenRouter's API."""
    return {
        "id": connection_id,
        "name": connection_id,
        "type": "openai",
        "url": url,
        "key": key,
        "config": {"model_ids": list(models), "access_grants": access_grants, **config},
    }


def video_settings(*connections: dict, enable: bool = True, default_model: str = "") -> dict:
    return {
        "ENABLE_VIDEO_GENERATION": enable,
        "VIDEO_GENERATION_DEFAULT_MODEL": default_model,
        "VIDEO_CONNECTIONS": list(connections),
    }


def save_video_settings(client: httpx.Client, settings: dict) -> dict:
    saved = client.post(VIDEOS_CONFIG[1], json=settings)
    assert saved.status_code == 200, f"saving the video settings failed: {saved.text}"
    return saved.json()


def model_ref(connection_id: str, model_id: str) -> str:
    """How a user picks a video model: its connection and model id."""
    return f"{connection_id}:{model_id}"


def serve_video_api(listener: Listener, openrouter_models: tuple[str, ...] = ()) -> None:
    """Answer as a video API run as jobs; `openrouter_models` are listed at `/videos/models`."""
    job_numbers = itertools.count(1)
    jobs: dict[str, dict] = {}
    lock = threading.Lock()

    def create(request: ReceivedRequest) -> Answer:
        with lock:
            job_id = f"video_{next(job_numbers)}"
            failing = FAILING_PROMPT.encode() in request.body
            jobs[job_id] = {"polls": 0, "failing": failing}
        return json_answer({"id": job_id, "object": "video", "status": "queued"})

    def job_route(request: ReceivedRequest) -> Answer:
        path = request.path.split("?")[0].removeprefix("/videos/")
        if path == "models":
            return json_answer(
                {"data": [{"id": model, "name": model} for model in openrouter_models]}
            )
        job_id, _, part = path.partition("/")
        with lock:
            job = jobs.get(job_id)
            if job is None:
                return json_answer({"error": {"message": f"no video {job_id}"}}, 404)
            job["polls"] += 1
        if part == "content":
            return 200, {"Content-Type": VIDEO_CONTENT_TYPE}, VIDEO
        if job["polls"] == 1:
            return json_answer({"id": job_id, "status": "in_progress", "progress": 50})
        if job["failing"]:
            return json_answer(
                {"id": job_id, "status": "failed", "error": {"message": FAILURE_MESSAGE}}
            )
        return json_answer({"id": job_id, "status": "completed", "progress": 100})

    # OpenAI's model list labels no video models
    listener.route("GET", "/models", json_answer({"data": [{"id": "sora-2"}, {"id": "gpt-4o"}]}))
    listener.route("POST", "/videos", create)
    listener.route("GET", "/videos/*", job_route)


def created_jobs(listener: Listener) -> list[ReceivedRequest]:
    """Every request that started a video job."""
    return [call for call in listener.requests_to("/videos") if call.method == "POST"]


def job_polls(listener: Listener) -> list[ReceivedRequest]:
    """Every request that checked on a job or fetched its video."""
    return [
        call
        for call in listener.received
        if call.method == "GET" and call.path.startswith("/videos/video_")
    ]


def form_fields(request: ReceivedRequest) -> dict[str, tuple[bytes, str]]:
    """The parts of a multipart request by name: their bytes and content type."""
    content_type = request.headers.get("Content-Type") or request.headers.get("content-type", "")
    assert content_type.startswith("multipart/form-data"), (
        f"the create request is not a multipart form: {content_type!r}"
    )
    message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
        f"Content-Type: {content_type}\r\n\r\n".encode() + request.body
    )
    return {
        part.get_param("name", header="content-disposition"): (
            part.get_payload(decode=True),
            part.get_content_type(),
        )
        for part in message.iter_parts()
    }
