"""A tiny Whisper model on disk, for an instance that transcribes with local Whisper offline.

Local Whisper (a blank speech-to-text engine) loads a CTranslate2 model through faster-whisper,
by name or by path, and faster-whisper decodes the recording with PyAV first.
`save_tiny_whisper(directory)` builds a model there without downloading anything: a one-layer
Whisper with random weights and a word-level tokenizer of `WORDS`, converted by CTranslate2's
own converter. Its decoder always favours the first word by far, so whatever it hears its
transcript is that word over and over (with a random decoder faster-whisper's sampling retries
came back empty now and then).
`using_local_whisper(client, model_path)` saves the admin's audio settings on it and restores
the previous ones through the config import, as `harness.audio_engine` does.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Iterator

import httpx
import pytest

from harness.audio_engine import AUDIO_CONFIG, AUDIO_NAMESPACE, CONFIG_IMPORT

WORDS = ("harbour", "lighthouse", "keeper")
SPECIAL_TOKENS = (
    "<|endoftext|>",
    "<|startoftranscript|>",
    "<|en|>",
    "<|translate|>",
    "<|transcribe|>",
    "<|startoflm|>",
    "<|startofprev|>",
    "<|nocaptions|>",
    "<|notimestamps|>",
)
# 30 seconds in steps of 20 ms, the timestamp tokens that follow <|notimestamps|>
TIMESTAMP_TOKENS = tuple(f"<|{step * 0.02:.2f}|>" for step in range(1501))


def _save_tokenizer(directory: Path) -> dict[str, int]:
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocabulary = {word: index for index, word in enumerate([*WORDS, "[UNK]"])}
    for token in (*SPECIAL_TOKENS, *TIMESTAMP_TOKENS):
        vocabulary[token] = len(vocabulary)
    tokenizer = Tokenizer(models.WordLevel(vocab=vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens([*SPECIAL_TOKENS, *TIMESTAMP_TOKENS])
    tokenizer.save(str(directory / "tokenizer.json"))
    settings = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "[UNK]",
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
    }
    (directory / "tokenizer_config.json").write_text(json.dumps(settings))
    return vocabulary


def save_tiny_whisper(directory: Path) -> Path:
    """Build the model under `directory`; returns the CTranslate2 model directory to load."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    converters = pytest.importorskip("ctranslate2.converters")

    source = directory / "transformers"
    source.mkdir(parents=True)
    vocabulary = _save_tokenizer(source)
    end_of_text = vocabulary["<|endoftext|>"]
    config = transformers.WhisperConfig(
        vocab_size=len(vocabulary),
        num_mel_bins=80,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        d_model=32,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        max_source_positions=1500,
        max_target_positions=448,
        decoder_start_token_id=vocabulary["<|startoftranscript|>"],
        bos_token_id=end_of_text,
        eos_token_id=end_of_text,
        pad_token_id=end_of_text,
        suppress_tokens=[],
        begin_suppress_tokens=[],
    )
    torch.manual_seed(0)
    model = transformers.WhisperForConditionalGeneration(config)
    with torch.no_grad():
        # one fixed decoder output that favours a word, so every decode at any temperature agrees
        # alternating signs: an int8 product of all ones saturates on CPUs without VNNI
        direction = torch.tensor([1.0, -1.0]).repeat(config.d_model // 2)
        model.model.decoder.layer_norm.weight.zero_()
        model.model.decoder.layer_norm.bias.copy_(direction)
        model.model.decoder.embed_tokens.weight[vocabulary[WORDS[0]]] = direction
    model.save_pretrained(source)
    converted = directory / "ctranslate2"
    converter = converters.TransformersConverter(str(source), copy_files=["tokenizer.json"])
    converter.convert(str(converted), force=True)
    return converted


@contextlib.contextmanager
def using_local_whisper(client: httpx.Client, model_path: Path) -> Iterator[dict]:
    """Switch speech-to-text to local Whisper on `model_path`; yields what was saved."""
    snapshot = client.get(AUDIO_NAMESPACE)
    snapshot.raise_for_status()
    current = client.get(AUDIO_CONFIG[0])
    current.raise_for_status()
    stt = {**current.json()["stt"], "ENGINE": "", "WHISPER_MODEL": str(model_path)}
    saved = client.post(AUDIO_CONFIG[1], json={"tts": current.json()["tts"], "stt": stt})
    assert saved.status_code == 200, f"switching to local Whisper failed: {saved.text}"
    try:
        yield saved.json()
    finally:
        restored = client.post(CONFIG_IMPORT, json={"config": snapshot.json()})
        assert restored.status_code == 200, f"restoring the audio settings failed: {restored.text}"
