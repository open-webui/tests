"""Guard: no new log or print call renders a value under a payload-shaped name.

Two ways a log line renders a payload nobody asked for: an f-string or `json.dumps` in the
call arguments is built before the logger looks at the level, so it costs the same with
logging off, and a `%s` argument at INFO or above is rendered by every default deployment.
Either one on a chat payload is a whole message history, or a whole embedding batch, turned
into a string on the event loop, per request.

The audit counts, per file, every `log` / `logger` / `logging` level call or print that
renders a value under a payload-shaped name or wraps anything in a serialiser, frozen to the
sites present on upstream dev at v0.11.3 (a253bf0c3), so any new one fails. It goes by name
(the underscore tail of the identifier), so a scalar or an exception logged as `result` or
`data` is counted too and has to be recorded, and a payload under an unlisted name is not
seen; each entry says what it renders, so the real leaks are the ones marked as such. Calls
under an `isEnabledFor` check, under `if __name__ == "__main__"`, generic `.log(level, ...)`
calls (`utils/audit.py`, `events.py`) and the one-shot scripts under `migrations/` are out of
scope. The behavioural half, a poisoned body through the filter error path, lives in
`test_filter_error_path_ignores_body.py`.

Unpinned and unmarked: the recorded leaks have no fix ref, and the ratchet is what pins them.
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

# (path, rule, rendered expression) -> (sites on the pinned ref, what is rendered)
KNOWN = {
    ("models/chats.py", "default-level", "folder_ids"): (1, "benign: a few ids"),
    ("retrieval/loaders/datalab_marker.py", "default-level", "form_data"): (1, "benign: options"),
    ("retrieval/loaders/datalab_marker.py", "eager", "dumps()"): (2, "benign: summary; leak: poll"),
    ("retrieval/loaders/datalab_marker.py", "eager", "raw_body"): (1, "leak: HTTP body at ERROR"),
    ("retrieval/loaders/external_web.py", "eager", "urls"): (1, "leak: every URL at ERROR"),
    ("retrieval/loaders/mistral.py", "default-level", "delete_response"): (1, "benign: a status"),
    ("retrieval/loaders/mistral.py", "eager", "response.text"): (2, "leak: HTTP body at ERROR"),
    ("retrieval/loaders/tavily.py", "eager", "batch_urls"): (1, "leak: every URL at ERROR"),
    ("retrieval/utils.py", "default-level", "result.ids"): (2, "leak: every chunk id"),
    ("retrieval/utils.py", "default-level", "result.metadatas"): (2, "leak: chunk metadata"),
    ("retrieval/utils.py", "default-level", "result['metadatas']"): (1, "leak: chunk metadata"),
    ("retrieval/vector/dbs/milvus.py", "default-level", "ids"): (1, "leak: every deleted id"),
    ("retrieval/vector/dbs/pinecone.py", "eager", "result"): (2, "benign: an exception"),
    ("retrieval/web/external.py", "default-level", "results"): (1, "leak: search results"),
    ("retrieval/web/firecrawl.py", "default-level", "search_results"): (1, "leak: search results"),
    ("retrieval/web/kagi.py", "print", "results"): (1, "leak: search results"),
    ("retrieval/web/mojeek.py", "print", "results"): (1, "leak: search results"),
    ("retrieval/web/searchapi.py", "default-level", "json_response"): (1, "leak: search results"),
    ("retrieval/web/serpapi.py", "default-level", "json_response"): (1, "leak: search results"),
    ("retrieval/web/serply.py", "default-level", "json_response"): (1, "leak: search results"),
    ("retrieval/web/utils.py", "eager", "urls"): (1, "leak: every URL at WARNING"),
    ("retrieval/web/yandex.py", "default-level", "results"): (1, "leak: search results"),
    ("routers/audio.py", "default-level", "metadata"): (1, "benign: a small params dict"),
    ("routers/ollama.py", "default-level", "create_payload"): (1, "benign: name and digest"),
    ("routers/ollama.py", "default-level", "form_data"): (2, "leak: embedding batch at INFO"),
    ("utils/chat.py", "default-level", "res"): (1, "benign: a socket ack"),
    ("utils/oauth.py", "eager", "user_data"): (3, "leak: userinfo claims at WARNING"),
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


def _out_of_scope(call: ast.Call, parents: dict) -> bool:
    node = call
    while node in parents:
        child, node = node, parents[node]
        if isinstance(node, ast.If) and child not in node.orelse:
            test = ast.unparse(node.test)
            if "isEnabledFor" in test or "__name__ == '__main__'" in test:
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
        rel = path.relative_to(package_root).as_posix()
        if SKIP_DIRS & set(rel.split("/")):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for rule, ref in _rendered_payloads(tree):
            found[(rel, rule, ref)] += 1
    return found


def test_no_new_log_or_print_call_renders_a_payload(open_webui_backend: Path):
    found = _find_rendered_payloads(open_webui_backend / "open_webui")
    recorded = {site: count for site, (count, _) in KNOWN.items()}
    new = sorted(
        (site, count - recorded.get(site, 0))
        for site, count in found.items()
        if count > recorded.get(site, 0)
    )
    gone = sorted(site for site, count in recorded.items() if found[site] < count)
    problems = []
    if new:
        problems.append(
            "log or print calls render a payload; log a size or an id instead, or record them:\n  "
            + "\n  ".join(f"{path}: {rule} {ref} +{added}" for (path, rule, ref), added in new)
        )
    if gone:
        problems.append(f"fewer sites than recorded, lower the count or remove the entry: {gone}")
    assert not problems, "\n".join(problems)
