"""An embedding service whose vectors follow chosen keywords, so retrieval ranks predictably.

The scripted provider embeds every text as the same vector, which leaves vector search no way to
prefer one chunk. `serve_keyword_embeddings(listener, keywords)` answers OpenAI-shaped
`/embeddings` requests with one dimension per keyword, set when the text contains it, plus a
small constant one, so a chunk and a question point the same way exactly when they share a
keyword. Boot an instance with `keyword_embedding_env(listener)` to embed through it.
"""

from __future__ import annotations

from harness.listener import Answer, Listener, ReceivedRequest, json_answer

BACKGROUND = 0.05


def keyword_vector(text: str, keywords: list[str]) -> list[float]:
    lowered = text.lower()
    return [1.0 if keyword in lowered else 0.0 for keyword in keywords] + [BACKGROUND]


def serve_keyword_embeddings(listener: Listener, keywords: list[str]) -> None:
    def embed(request: ReceivedRequest) -> Answer:
        inputs = request.json()["input"]
        texts = inputs if isinstance(inputs, list) else [inputs]
        vectors = [
            {"object": "embedding", "index": index, "embedding": keyword_vector(text, keywords)}
            for index, text in enumerate(texts)
        ]
        return json_answer({"object": "list", "data": vectors})

    listener.route("POST", "/embeddings", embed)


def keyword_embedding_env(listener: Listener) -> dict[str, str]:
    """The environment of an instance that embeds through the listener."""
    return {"RAG_OPENAI_API_BASE_URL": listener.base_url, "RAG_OPENAI_API_KEY": "sk-keywords"}


def embedded_texts(listener: Listener) -> list[str]:
    """Every text the listener was asked to embed, in order."""
    texts = []
    for request in listener.requests_to("/embeddings"):
        inputs = request.json()["input"]
        texts.extend(inputs if isinstance(inputs, list) else [inputs])
    return texts
