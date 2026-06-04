"""Helpers for loading the lightweight LoRA checkpoints saved by SFT."""

import json
from pathlib import Path
from typing import Any

from safetensors.torch import load_file

from nla.arch_adapters import resolve_text_model
from sft.finetune_gemma_lora import inject_lora, lora_state_dict


def apply_lora_adapter(model: Any, adapter_path: str | Path) -> list[str]:
    """Patch ``model`` with this repo's saved LoRA adapter weights."""
    adapter_dir = Path(adapter_path)
    cfg_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing LoRA adapter config: {cfg_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"missing LoRA adapter weights: {weights_path}")

    cfg = json.loads(cfg_path.read_text())
    text_model = resolve_text_model(model)
    patched = inject_lora(
        text_model,
        target_modules=tuple(cfg["target_modules"]),
        rank=int(cfg["rank"]),
        alpha=float(cfg["alpha"]),
        dropout=float(cfg["dropout"]),
    )

    state = load_file(str(weights_path))
    expected = lora_state_dict(text_model)
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing or unexpected:
        pieces = []
        if missing:
            pieces.append(f"missing adapter tensors: {missing[:5]}")
        if unexpected:
            pieces.append(f"unexpected adapter tensors: {unexpected[:5]}")
        raise ValueError("; ".join(pieces))

    shape_mismatches = [
        (key, tuple(state[key].shape), tuple(expected[key].shape))
        for key in sorted(state)
        if state[key].shape != expected[key].shape
    ]
    if shape_mismatches:
        key, got, want = shape_mismatches[0]
        raise ValueError(f"adapter tensor shape mismatch for {key}: got {got}, expected {want}")

    text_model.load_state_dict(state, strict=False)
    model.eval()
    return patched
