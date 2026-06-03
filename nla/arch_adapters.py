"""Architecture adapters — one place for multimodal-wrapper unwrapping.

Some HF checkpoints (Gemma-3, LLaVA-style) load as multimodal wrappers even
via `AutoModelForCausalLM`. NLA only cares about the text side. Rather than
scattering `config.text_config` / `model.model.language_model` checks
across train_actor/models/extractors, these functions centralize the unwrap
with explicit arch detection.

Pass-through for plain text models (Qwen/Llama/Mistral): they have none of
the wrapper attributes, so all functions return the input unchanged.

Add new wrapped architectures by extending _WRAPPER_MODEL_ATTRS /
_WRAPPER_CONFIG_ATTRS — don't duck-type new getattr fallbacks at callsites.
"""

from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM

# Multimodal wrapper → text model attribute name.
# Older wrappers may expose .language_model directly. Current Gemma-3 uses
# .model.language_model and is handled explicitly in resolve_text_model.
# (LLaVA-style wrappers would go here too if we ever support them)
_WRAPPER_MODEL_ATTRS = ("language_model",)

# Multimodal wrapper config → text config attribute name.
# Gemma3Config.text_config → Gemma3TextConfig (has hidden_size, num_hidden_layers)
_WRAPPER_CONFIG_ATTRS = ("text_config",)


def resolve_text_config(config: Any) -> Any:
    """Return the text-side config for multimodal wrappers; pass-through otherwise.

    Gemma3Config nests hidden_size/num_hidden_layers under .text_config.
    Qwen2Config/LlamaConfig/MistralConfig have those at top level and no
    .text_config attr → return as-is.
    """
    for attr in _WRAPPER_CONFIG_ATTRS:
        nested = getattr(config, attr, None)
        if nested is not None:
            return nested
    return config


def _resolve_attr_path(obj: Any, path: Iterable[str]) -> Any | None:
    value = obj
    for attr in path:
        value = getattr(value, attr, None)
        if value is None:
            return None
    return value


def _wrap_bare_text_model(nested: Any, source: Any) -> Any:
    """Wrap a bare text transformer in its CausalLM class without allocating weights."""
    with torch.device("meta"):
        wrapper = AutoModelForCausalLM.from_config(nested.config)
    wrapper.model = nested  # transplant pretrained weights

    source_lm_head = getattr(source, "lm_head", None)
    if source_lm_head is not None:
        wrapper.lm_head = source_lm_head
    elif getattr(nested.config, "tie_word_embeddings", False):
        wrapper.tie_weights()
    return wrapper


def resolve_text_model(model: Any) -> Any:
    """Return the text-side CausalLM for multimodal wrappers; pass-through otherwise.

    Invariant: always returns a CausalLM-shaped model (has .model + .lm_head),
    so `save_pretrained()` / `AutoModelForCausalLM.from_pretrained()` roundtrip.

    Current HF Gemma3ForConditionalGeneration stores the bare text transformer at
    .model.language_model; older wrappers may expose .language_model directly.
    Returning the bare transformer directly means `save_pretrained` writes keys
    like `layers.0.*` but `from_pretrained` via AutoModelForCausalLM loads
    Gemma3ForCausalLM expecting `model.layers.0.*` → zero keys match →
    everything random-inits. Observed Mar 13 2026: pred_norm=507 vs
    gold_norm=75616 on same input, step-0 loss=2.0 (orthogonal).

    Qwen/Llama/Mistral have no .language_model → pass through unchanged
    (already CausalLM-shaped).
    """
    nested = _resolve_attr_path(model, ("model", "language_model"))
    if nested is not None:
        if hasattr(nested, "lm_head"):
            return nested  # already CausalLM-shaped
        return _wrap_bare_text_model(nested, model)

    for attr in _WRAPPER_MODEL_ATTRS:
        nested = getattr(model, attr, None)
        if nested is None:
            continue
        if hasattr(nested, "lm_head"):
            return nested  # already CausalLM-shaped
        return _wrap_bare_text_model(nested, model)
    return model


_DECODER_LAYER_PATHS = (
    ("model", "layers"),  # Llama/Qwen/Mistral/Gemma text CausalLM
    ("model", "language_model", "layers"),  # Gemma3ForConditionalGeneration
    ("language_model", "layers"),  # bare multimodal text side
    ("layers",),  # bare text model
    ("transformer", "h"),  # GPT-2/Falcon
)


def resolve_decoder_layers(model: Any) -> torch.nn.ModuleList:
    """Find the decoder layers ModuleList, unwrapping multimodal wrappers first.

    Supported explicit paths:
      Llama/Qwen/Mistral/Gemma text CausalLM: model.model.layers
      Gemma3 multimodal wrapper: model.model.language_model.layers
      GPT-2/Falcon: model.transformer.h
    """
    model = resolve_text_model(model)
    for path in _DECODER_LAYER_PATHS:
        layers = _resolve_attr_path(model, path)
        if layers is None:
            continue
        assert isinstance(layers, torch.nn.ModuleList), (
            f"resolved {type(layers).__name__} at {'.'.join(path)}, expected nn.ModuleList. "
            f"Module path is wrong for {type(model).__name__}."
        )
        return layers

    paths = ", ".join(".".join(path) for path in _DECODER_LAYER_PATHS)
    raise AssertionError(
        f"{type(model).__name__} has no decoder layers at known paths ({paths}) — "
        f"extend arch_adapters.resolve_decoder_layers for this architecture"
    )


def is_multimodal_wrapper(config_or_model: Any) -> bool:
    """True if this is a known multimodal wrapper (has nested text config/model)."""
    if _resolve_attr_path(config_or_model, ("model", "language_model")) is not None:
        return True
    for attr in (*_WRAPPER_CONFIG_ATTRS, *_WRAPPER_MODEL_ATTRS):
        if getattr(config_or_model, attr, None) is not None:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────
# Embedding forward scale — for RL rollout injection.
#
# Some archs' embedding modules multiply by a scale in their forward():
#   - Gemma: Gemma3TextScaledWordEmbedding × √d_model
#   - T5-style: similar √d_model scaling
# Plain nn.Embedding (Qwen/Llama/Mistral): no scale.
#
# Training-side injection is unaffected (the forward hook captures POST-scale
# output). RL rollout builds embeds manually from raw weights → must apply
# the scale or injection magnitude is off by ~62× (Gemma).
#
# This is an EXPLICIT registry — don't try to auto-detect. If you add a
# scaled-embed arch, add it here AND the callsite will visibly multiply.
# ─────────────────────────────────────────────────────────────────────────

# model_type → scale expression. √d_model is the common case; if an arch
# uses a different formula, add a lambda here.
_SCALED_EMBED_MODEL_TYPES: dict[str, str] = {
    "gemma3": "sqrt_d_model",
    "gemma3_text": "sqrt_d_model",
    "gemma2": "sqrt_d_model",
    "gemma": "sqrt_d_model",
    "t5": "sqrt_d_model",
}


def resolve_embed_scale(config: Any) -> float:
    """Return the scalar that the model's embedding forward() multiplies by.

    For plain nn.Embedding archs (Qwen/Llama/Mistral): 1.0.
    For Gemma/T5-style: √d_model.

    RL rollout uses this to scale raw-weight-lookup embeds to match what
    the model's forward would produce. Training doesn't need it (forward
    hook captures post-scale output). See `_SCALED_EMBED_MODEL_TYPES`
    and extend there for new scaled-embed archs.
    """
    text_config = resolve_text_config(config)
    model_type = getattr(text_config, "model_type", "")
    rule = _SCALED_EMBED_MODEL_TYPES.get(model_type)
    if rule is None:
        return 1.0
    if rule == "sqrt_d_model":
        return float(text_config.hidden_size) ** 0.5
    raise AssertionError(
        f"unknown embed-scale rule {rule!r} for model_type={model_type!r} — "
        f"extend resolve_embed_scale in arch_adapters.py"
    )
