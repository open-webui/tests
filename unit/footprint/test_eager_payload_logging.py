"""Guard: no new log or print call renders a value under a payload-shaped name.

Two ways a log line renders a payload nobody asked for: an f-string or `json.dumps` in the
call arguments is built before the logger looks at the level, so it costs the same with
logging off, and a `%s` argument at INFO or above is rendered by every default deployment.
Either one on a chat payload is a whole message history, or a whole embedding batch, turned
into a string on the event loop, per request.

The audit counts every `log` / `logger` / `logging` level call or print that renders a value
under a payload-shaped name or wraps anything in a serialiser, per rule and rendered name,
frozen to the sites present on upstream dev at v0.11.3 (a253bf0c3). The ratchet is
one-directional: a count above the recorded one fails, a site upstream removes or moves to
another file passes. It goes by name (the underscore tail of the identifier), so a scalar or an
exception logged as `result` or `data` is counted too and has to be recorded, and a payload under
an unlisted name is not seen; each entry says where it is and what it renders, so the real leaks
are the ones marked as such. Calls under an `isEnabledFor` check, under
`if __name__ == "__main__"`, generic `.log(level, ...)` calls (`utils/audit.py`, `events.py`) and
the one-shot scripts under `migrations/` are out of scope. The behavioural half, a poisoned body
through the filter error path, lives in `test_filter_error_path_ignores_body.py`.

Unpinned and unmarked: the recorded leaks have no fix ref, and the ratchet is what pins them.
Discriminates: a new `log.info("%s", payload)` or f-string of `messages` in a copy of dev
bbfa876af fails; deleting a recorded site from the copy passes.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

LOGGERS = {"log", "logger", "logging"}
LEVELS = {"debug", "info", "warning", "error", "critical", "exception"}
DEFAULT_ON_LEVELS = LEVELS - {"debug"}
SERIALISERS = {"dumps", "dumps_bytes"}
RENDERERS = {"str", "repr"}
PAYLOAD_NAMES = {
    "payload",
    "body",
    "messages",
    "metadata",
    "metadatas",
    "data",
    "result",
    "results",
    "response",
    "res",
    "chunk",
    "docs",
    "ids",
    "urls",
}
# `response.text` is a whole HTTP body; a bare `text` is usually one string
PAYLOAD_ATTRS = PAYLOAD_NAMES | {"text"}
SKIP_DIRS = {"migrations"}

# (rule, rendered expression) -> (sites on the pinned ref, where and what is rendered)
KNOWN = {
    ("default-level", "create_payload"): (1, "routers/ollama.py, benign: name and digest"),
    ("default-level", "delete_response"): (1, "retrieval/loaders/mistral.py, benign: a status"),
    ("default-level", "folder_ids"): (1, "models/chats.py, benign: a few ids"),
    ("default-level", "form_data"): (
        3,
        "retrieval/loaders/datalab_marker.py, benign: options; "
        "routers/ollama.py x2, leak: embedding batch at INFO",
    ),
    ("default-level", "ids"): (1, "retrieval/vector/dbs/milvus.py, leak: every deleted id"),
    ("default-level", "json_response"): (2, "retrieval/web/serpapi.py, serply.py, leak: results"),
    ("default-level", "metadata"): (1, "routers/audio.py, benign: a small params dict"),
    ("default-level", "res"): (1, "utils/chat.py, benign: a socket ack"),
    ("default-level", "result.ids"): (2, "retrieval/utils.py, leak: every chunk id"),
    ("default-level", "result.metadatas"): (2, "retrieval/utils.py, leak: chunk metadata"),
    ("default-level", "result['metadatas']"): (1, "retrieval/utils.py, leak: chunk metadata"),
    ("default-level", "results"): (2, "retrieval/web/external.py, yandex.py, leak: results"),
    ("default-level", "search_results"): (1, "retrieval/web/firecrawl.py, leak: results"),
    ("eager", "batch_urls"): (1, "retrieval/loaders/tavily.py, leak: every URL at ERROR"),
    ("eager", "dumps()"): (2, "retrieval/loaders/datalab_marker.py, benign: summary; leak: poll"),
    ("eager", "raw_body"): (1, "retrieval/loaders/datalab_marker.py, leak: HTTP body at ERROR"),
    ("eager", "response.text"): (2, "retrieval/loaders/mistral.py, leak: HTTP body at ERROR"),
    ("eager", "result"): (2, "retrieval/vector/dbs/pinecone.py, benign: an exception"),
    ("eager", "urls"): (
        2,
        "retrieval/loaders/external_web.py, leak: every URL at ERROR; "
        "retrieval/web/utils.py, leak: every URL at WARNING",
    ),
    ("eager", "user_data"): (3, "utils/oauth.py, leak: userinfo claims at WARNING"),
    ("print", "results"): (2, "retrieval/web/kagi.py, mojeek.py, leak: search results"),
}


def _payload_named(name: str | None, names: set[str]) -> bool:
    """`results`, `search_results` and `json_response` all count."""
    return name is not None and name.split("_")[-1] in names


def _root_name(expr) -> str | None:
    while isinstance(expr, (ast.Attribute, ast.Subscript)):
        expr = expr.value
    return expr.id if isinstance(expr, ast.Name) else None


def _call_name(call: ast.Call) -> str:
    return call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", "")


def _is_str_format(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "format"
        and isinstance(call.func.value, ast.Constant)
    )


def _payload_refs(expr) -> set[str]:
    """Payload-shaped values an expression renders; a payload handed to another function
    or reduced to a non-payload attribute (`len(docs)`, `form_data.id`) renders something
    smaller than the payload."""
    refs = set()

    def visit(node):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in SERIALISERS:
                refs.add(f"{name}()")
            elif name in RENDERERS or _is_str_format(node):
                for arg in node.args:
                    visit(arg)
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return
        if isinstance(node, ast.Name) and _payload_named(node.id, PAYLOAD_NAMES):
            refs.add(node.id)
            return
        if isinstance(node, ast.Attribute):
            root_is_payload = _payload_named(_root_name(node), PAYLOAD_NAMES)
            if _payload_named(node.attr, PAYLOAD_ATTRS) and root_is_payload:
                refs.add(ast.unparse(node))
            return
        if isinstance(node, ast.Subscript):
            key = node.slice.value if isinstance(node.slice, ast.Constant) else None
            root_is_payload = _payload_named(_root_name(node), PAYLOAD_NAMES)
            if isinstance(key, str) and _payload_named(key, PAYLOAD_NAMES) and root_is_payload:
                refs.add(ast.unparse(node))
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(expr)
    return refs


def _log_level(call: ast.Call) -> str | None:
    fn = call.func
    if isinstance(fn, ast.Attribute) and fn.attr in LEVELS and _root_name(fn.value) in LOGGERS:
        return fn.attr
    return None


def _is_main_guard(test) -> bool:
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and any(isinstance(c, ast.Constant) and c.value == "__main__" for c in test.comparators)
    )


def _is_level_check(test) -> bool:
    return any(
        isinstance(node, ast.Call) and _call_name(node) == "isEnabledFor" for node in ast.walk(test)
    )


def _out_of_scope(call: ast.Call, parents: dict) -> bool:
    node = call
    while node in parents:
        child, node = node, parents[node]
        if isinstance(node, ast.If) and child not in node.orelse:
            if _is_level_check(node.test) or _is_main_guard(node.test):
                return True
    return False


def _rendered_payloads(tree) -> list[tuple[str, str]]:
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if _out_of_scope(node, parents):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            hits += [("print", ref) for ref in {r for a in node.args for r in _payload_refs(a)}]
            continue
        level = _log_level(node)
        if level is None:
            continue
        eager, lazy = set(), set()
        for arg in node.args:
            refs = _payload_refs(arg)
            built_here = isinstance(arg, (ast.JoinedStr, ast.BinOp)) or (
                isinstance(arg, ast.Call) and (_call_name(arg) in RENDERERS or _is_str_format(arg))
            )
            if built_here:
                eager |= refs
            else:
                eager |= {ref for ref in refs if ref.endswith("()")}
                lazy |= {ref for ref in refs if not ref.endswith("()")}
        hits += [("eager", ref) for ref in eager]
        if level in DEFAULT_ON_LEVELS:
            hits += [("default-level", ref) for ref in lazy]
    return hits


def _find_rendered_payloads(package_root: Path) -> Counter:
    found: Counter = Counter()
    for path in sorted(package_root.rglob("*.py")):
        if SKIP_DIRS & set(path.relative_to(package_root).parts):
            continue
        found.update(_rendered_payloads(ast.parse(path.read_text(encoding="utf-8"))))
    return found


def test_no_new_log_or_print_call_renders_a_payload(open_webui_backend: Path):
    found = _find_rendered_payloads(open_webui_backend / "open_webui")
    assert found.keys() & KNOWN.keys(), "the audit no longer finds any recorded site"

    recorded = {site: count for site, (count, _) in KNOWN.items()}
    added = {
        site: count - recorded.get(site, 0)
        for site, count in found.items()
        if count > recorded.get(site, 0)
    }
    assert not added, (
        "log or print calls render a payload; log a size or an id instead, or record them:\n  "
        + "\n  ".join(f"{rule} {ref} +{count}" for (rule, ref), count in sorted(added.items()))
    )
