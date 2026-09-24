"""The local embedding path still embeds once the missing-model check moved to use time.

PR #25683 (issues #25634, #25165) deferred the "no embedding model is loaded" error from building
the embedding function to calling it, so a blank model no longer bricks boot. The integration twin
pins the boot, the settings saves and the use-time error; what it cannot reach is the positive
path of the local engine, which needs a SentenceTransformer loaded. This drives the real
`get_embedding_function` with a stand-in for the model, the only I/O on that path.

Discriminates: passes on bbfa876af; raising whenever the local engine is selected, or dropping
the loaded model on the way to `encode`, fails it.
"""

from __future__ import annotations

import asyncio

import numpy
import pytest

pytestmark = pytest.mark.regression


class FixedEncoder:
    """Stands in for a loaded SentenceTransformer: `encode` answers one fixed vector per text."""

    def __init__(self) -> None:
        self.encoded: list[str] = []

    def encode(self, texts, **options):
        self.encoded.append(texts)
        return numpy.array([0.1, 0.2, 0.3])


def test_a_loaded_local_model_still_embeds(retrieval_utils_module):
    model = FixedEncoder()
    embed = retrieval_utils_module.get_embedding_function(
        embedding_engine="",
        embedding_model="all-MiniLM",
        embedding_function=model,
        url="",
        key="",
        embedding_batch_size=1,
    )

    vector = asyncio.run(embed("hello"))

    assert vector == pytest.approx([0.1, 0.2, 0.3])
    assert model.encoded == ["hello"]
