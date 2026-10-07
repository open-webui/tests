"""An OpenAPI tool server played by a `listener`: a spec and one GET operation per answer.

`serve_openapi(listener, title, answers)` routes `/openapi.json` with a spec titled `title` that
offers one operation per entry of `answers`, an operation id mapped to the `Answer` (or handler)
its path `/<operation id>` gives. With `bearer_key` every request, the spec included, is refused
with a 401 unless it carries that key, the way a keyed server answers. `calls_to(listener,
operation_id)` is every request an operation got and `authorization(request)` the header it
carried, if any. `openapi_connection(url, name)` is the admin's connection to it, named the way
the chat lists it; save it through `TOOL_SERVERS` after `preserve(TOOL_SERVERS)`.
"""

from __future__ import annotations

from harness.listener import Answer, Handler, ReceivedRequest, json_answer, text_answer


def openapi_spec(title: str, operation_ids: list[str]) -> dict:
    paths = {
        f"/{operation_id}": {
            "get": {
                "operationId": operation_id,
                "summary": f"Run {operation_id.replace('_', ' ')}.",
                "responses": {"200": {"description": "The answer"}},
            }
        }
        for operation_id in operation_ids
    }
    return {
        "openapi": "3.0.0",
        "info": {"title": title, "version": "1.0.0", "description": f"{title} for the harbour"},
        "paths": paths,
    }


def authorization(request: ReceivedRequest) -> str | None:
    lowered = {name.lower(): value for name, value in request.headers.items()}
    return lowered.get("authorization")


def _keyed(handler: Handler, bearer_key: str | None) -> Handler:
    def answer(request: ReceivedRequest) -> Answer:
        if bearer_key and authorization(request) != f"Bearer {bearer_key}":
            return text_answer("missing or wrong key", "text/plain", 401)
        return handler(request)

    return answer


def _as_handler(answer: Answer | Handler) -> Handler:
    return answer if callable(answer) else (lambda _request: answer)


def serve_openapi(
    listener,
    title: str,
    answers: dict[str, Answer | Handler],
    bearer_key: str | None = None,
) -> None:
    spec = openapi_spec(title, list(answers))
    listener.route("GET", "/openapi.json", _keyed(lambda _request: json_answer(spec), bearer_key))
    for operation_id, answer in answers.items():
        listener.route("GET", f"/{operation_id}", _keyed(_as_handler(answer), bearer_key))


def calls_to(listener, operation_id: str) -> list[ReceivedRequest]:
    return listener.requests_to(f"/{operation_id}")


def openapi_connection(
    url: str,
    name: str,
    access_grants: list[dict] | None = None,
    **fields,
) -> dict:
    """A connection without auth, the way the admin panel saves one; `fields` replace its keys."""
    return {
        "url": url,
        "path": "openapi.json",
        "type": "openapi",
        "auth_type": "none",
        "key": "",
        "config": {"enable": True, "access_grants": access_grants or []},
        "info": {"id": name.lower().replace(" ", "_"), "name": name},
        **fields,
    }
