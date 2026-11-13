"""Decode sentences from one LLM activation, following DISCOVER's hidden decoder.

Adapted from tommccoy1/discover, models.py (MIT license; see NOTICE.md).
The projected LLM activation is the decoder's only cross-attention memory.
"""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from ac_tpr.tpr.extraction import cache_fingerprint, load_cache, normalized_states
from ac_tpr.tpr.training import TrainingConfig, fit_model, model_directory, release_memory

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2


@dataclass(frozen=True)
class UnpackingConfig:
    hidden_size: int = 1024
    n_layers: int = 6
    n_heads: int = 16
    ffn_size: int = 4096
    dropout: float = 0.1
    max_length: int = 32


class SentenceUnpacker(nn.Module):
    """Predict sentence tokens from a standardized LLM activation [batch, hidden]."""

    def __init__(self, input_size: int, vocab_size: int, config: UnpackingConfig):
        super().__init__()
        self.config = config
        self.projection = nn.Linear(input_size, config.hidden_size)
        self.embedding = nn.Embedding(vocab_size, config.hidden_size, padding_idx=PAD_ID)
        positions = torch.arange(config.max_length, dtype=torch.float32)[:, None]
        frequencies = torch.exp(
            torch.arange(0, config.hidden_size, 2, dtype=torch.float32)
            * (-math.log(10000.0) / config.hidden_size)
        )
        encoding = torch.zeros(config.max_length, config.hidden_size)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[:config.hidden_size // 2])
        self.register_buffer("position_encoding", encoding, persistent=False)
        self.dropout = nn.Dropout(config.dropout)
        layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size, nhead=config.n_heads,
            dim_feedforward=config.ffn_size, dropout=config.dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=config.n_layers)
        self.output = nn.Linear(config.hidden_size, vocab_size)

    def forward(self, z: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Return [batch, length, vocab] logits for BOS-prefixed decoder inputs."""
        if z.ndim != 2 or input_ids.ndim != 2 or z.shape[0] != input_ids.shape[0]:
            raise ValueError("Activations and decoder inputs must be matching batches")
        length = input_ids.shape[1]
        if not 0 < length <= self.config.max_length:
            raise ValueError("Decoder input length exceeds the configured position range")
        causal_mask = torch.ones(length, length, dtype=torch.bool, device=z.device).triu(1)
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings + self.position_encoding[:length])
        hidden = self.decoder(
            embeddings, self.projection(z).unsqueeze(1), tgt_mask=causal_mask,
            tgt_key_padding_mask=input_ids.eq(PAD_ID),
        )
        return self.output(hidden)

    @torch.no_grad()
    def generate(self, z: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
        """Greedily generate without a gold prefix; output excludes BOS and includes EOS."""
        max_length = self.config.max_length if max_length is None else max_length
        if not 0 < max_length <= self.config.max_length:
            raise ValueError("Generation length must be within the configured position range")
        was_training = self.training
        self.eval()
        try:
            tokens = torch.full((z.shape[0], 1), BOS_ID, dtype=torch.long, device=z.device)
            finished = torch.zeros(z.shape[0], dtype=torch.bool, device=z.device)
            for _ in range(max_length):
                next_token = self(z, tokens)[:, -1].argmax(dim=-1)
                next_token = next_token.masked_fill(finished, PAD_ID)
                tokens = torch.cat([tokens, next_token[:, None]], dim=1)
                finished |= next_token.eq(EOS_ID)
                if finished.all():
                    break
            return tokens[:, 1:]
        finally:
            self.train(was_training)


def unpacking_loss(model: SentenceUnpacker, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, int]:
    """Return summed next-token cross entropy and its non-padding token count."""
    z, decoder_inputs, targets = batch
    logits = model(z, decoder_inputs)
    loss = F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=PAD_ID, reduction="sum")
    return loss, int(targets.ne(PAD_ID).sum().item())


def sentence_tensors(target_ids: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift sentence tokens for teacher forcing, with explicit BOS and EOS."""
    width = max(map(len, target_ids)) + 1
    inputs = torch.full((len(target_ids), width), PAD_ID, dtype=torch.long)
    targets = inputs.clone()
    for index, tokens in enumerate(target_ids):
        inputs[index, :len(tokens) + 1] = torch.tensor([BOS_ID, *tokens])
        targets[index, :len(tokens) + 1] = torch.tensor([*tokens, EOS_ID])
    return inputs, targets



def train_unpackers(
    encodings_dir: Path, output_dir: Path, *, config: UnpackingConfig = UnpackingConfig(),
    training: TrainingConfig = TrainingConfig(learning_rate=1e-4), device: str = "cuda",
) -> dict:
    """Fit independent sentence-only decoders, selecting checkpoints on val."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    cache = load_cache(encodings_dir)
    inputs, targets = sentence_tensors(cache["target_ids"])
    if inputs.shape[1] > config.max_length:
        raise ValueError("Unpacker max_length must cover the longest sentence plus EOS")
    train = [i for i, row in enumerate(cache["rows"]) if row["tpr_split"] == "train"]
    val = [i for i, row in enumerate(cache["rows"]) if row["tpr_split"] == "val"]
    manifest = {"artifact_type": "tpr_unpackers", "cache_fingerprint": cache_fingerprint(cache),
                "config": asdict(config), "training": asdict(training), "models": []}
    output_dir.mkdir(parents=True)
    for li, layer in enumerate(cache["metadata"]["layer_indices"]):
        for si, site in enumerate(cache["metadata"]["sites"]):
            print(f"Training unpacker: block {layer}, {site}", flush=True)
            z = normalized_states(cache, li, si)
            torch.manual_seed(training.seed)
            model = SentenceUnpacker(z.shape[-1], len(cache["token_vocab"]), config)
            path = model_directory(output_dir, layer, site)
            history = fit_model(
                model, TensorDataset(z[train], inputs[train], targets[train]),
                TensorDataset(z[val], inputs[val], targets[val]), unpacking_loss, path, training, device,
            )
            manifest["models"].append({"layer": layer, "site": site,
                                       "best_validation_loss": history["best_validation_loss"]})
            del model, z
            release_memory()
    (output_dir / "metadata.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
