from __future__ import annotations

from pathlib import Path

import torch

from .models import build_model


def _torch_load(path: str | Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model(path: str | Path, device: torch.device):
    checkpoint = _torch_load(path)
    state = checkpoint.get("model", checkpoint)
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    embedding = state["encoder.embedding.weight"]
    feature_weight = state.get("tabular.net.0.weight")
    feature_dim = int(feature_weight.shape[1]) if feature_weight is not None else None
    model = build_model(
        vocab_size=int(embedding.shape[0]),
        read_dim=int(embedding.shape[1]),
        feature_dim=feature_dim,
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    metadata = dict(checkpoint.get("metadata", {}))
    metadata.update(
        {
            "vocab_size": int(embedding.shape[0]),
            "read_dim": int(embedding.shape[1]),
            "feature_dim": feature_dim,
            "hybrid": feature_dim is not None,
        }
    )
    return model, metadata

