"""Run a chat conversation, then decode every token activation with an NLA AV.

This is an experiment helper for single-session cluster runs:

1. Read a JSON list of `{"role": ..., "content": ...}` messages.
2. Whenever a `user` message is not immediately followed by an `assistant`
   message, generate that assistant response with the base model.
3. Render the completed chat transcript.
4. Extract one layer-K activation vector per token.
5. Decode those activation vectors with an NLA AV and write JSONL.

The base model and AV model are loaded sequentially, not at the same time.
"""

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_HF_HOME = _REPO_ROOT / "hf_cache"
_DEFAULT_HF_HUB_CACHE = _DEFAULT_HF_HOME
os.environ.setdefault("HF_HOME", str(_DEFAULT_HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(_DEFAULT_HF_HUB_CACHE))
os.environ.setdefault("HF_XET_CACHE", str(_DEFAULT_HF_HOME / "xet"))

from transformers import AutoModelForCausalLM

from nla.arch_adapters import resolve_decoder_layers, resolve_text_config
from nla.datagen._common import load_tokenizer


def _torch_dtype(name: str) -> torch.dtype:
    try:
        dtype = getattr(torch, name)
    except AttributeError as exc:
        raise argparse.ArgumentTypeError(f"unknown torch dtype: {name}") from exc
    if not isinstance(dtype, torch.dtype):
        raise argparse.ArgumentTypeError(f"not a torch dtype: {name}")
    return dtype


def _read_messages(path: str) -> list[dict[str, str]]:
    messages = json.loads(Path(path).read_text())
    assert isinstance(messages, list), "--messages-json must contain a JSON list"
    for i, message in enumerate(messages):
        assert isinstance(message, dict), f"message {i} is not an object"
        assert message.get("role") in {"system", "user", "assistant"}, (
            f"message {i} has unsupported role {message.get('role')!r}"
        )
        assert isinstance(message.get("content"), str), f"message {i} has non-string content"
    return messages


def _load_base_model(args: argparse.Namespace):
    local_files_only = not args.allow_download
    print(f"[base] loading tokenizer: {args.base_model}", flush=True)
    tokenizer = load_tokenizer(
        args.base_model,
        cache_dir=args.hf_cache_dir,
        use_fast=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    model_kwargs: dict[str, Any] = {
        "torch_dtype": args.torch_dtype,
        "cache_dir": args.hf_cache_dir,
        "local_files_only": local_files_only,
    }
    if args.device_map != "none":
        model_kwargs["device_map"] = args.device_map
    print(
        f"[base] loading model: {args.base_model} "
        f"dtype={args.torch_dtype} device_map={args.device_map} device={args.device}",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs).eval()
    if args.device_map == "none":
        print(f"[base] moving model to {args.device}", flush=True)
        model = model.to(args.device)
    print("[base] model loaded", flush=True)
    return tokenizer, model


def _generate_assistant(tokenizer, model, messages: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    completed: list[dict[str, str]] = []
    for i, message in enumerate(messages):
        completed.append(dict(message))
        if message["role"] != "user":
            continue
        next_is_assistant = i + 1 < len(messages) and messages[i + 1]["role"] == "assistant"
        if next_is_assistant:
            continue

        print(f"[base] generating assistant after user turn {i}", flush=True)
        input_ids = tokenizer.apply_chat_template(
            completed,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(input_ids, "keys"):
            input_ids = input_ids["input_ids"]
        device = model.get_input_embeddings().weight.device
        input_ids = input_ids.to(device)
        attention_mask = torch.ones_like(input_ids)
        do_sample = args.response_temperature > 0
        gen_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": args.response_max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = args.response_temperature

        with torch.no_grad():
            out = model.generate(**gen_kwargs)
        response_ids = out[0, input_ids.shape[1] :]
        response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        completed.append({"role": "assistant", "content": response})
        print(f"[base] generated assistant after user turn {i}: {response[:160]!r}", flush=True)
    return completed


def _capture_layer_output(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, layer_index: int) -> torch.Tensor:
    layers = resolve_decoder_layers(model)
    assert 0 <= layer_index < len(layers), (
        f"layer_index={layer_index} out of range for model with {len(layers)} layers"
    )
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output) -> None:
        h = output[0] if isinstance(output, tuple) else output
        captured.append(h.detach().float().cpu())

    handle = layers[layer_index].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()
    assert len(captured) == 1, f"expected one captured activation tensor, got {len(captured)}"
    return captured[0][0]


def _write_token_parquet(tokenizer, model, messages: list[dict[str, str]], args: argparse.Namespace) -> None:
    print("[extract] rendering completed chat transcript", flush=True)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    enc = tokenizer(
        rendered,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=args.max_length,
        add_special_tokens=False,
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    token_ids = enc["input_ids"][0].tolist()

    device = model.get_input_embeddings().weight.device
    print(f"[extract] running layer-{args.layer_index} activation capture for {len(token_ids)} tokens", flush=True)
    hidden = _capture_layer_output(
        model,
        input_ids=enc["input_ids"].to(device),
        attention_mask=enc["attention_mask"].to(device),
        layer_index=args.layer_index,
    )
    d_model = resolve_text_config(model.config).hidden_size
    assert hidden.shape == (len(token_ids), d_model), (hidden.shape, len(token_ids), d_model)

    rows: dict[str, list[Any]] = {
        "token_index": [],
        "token_id": [],
        "token_text": [],
        "char_start": [],
        "char_end": [],
        "is_special": [],
        "prefix_text": [],
        "activation_vector": [],
        "activation_layer": [],
    }
    special_ids = set(tokenizer.all_special_ids)
    for i, token_id in enumerate(token_ids):
        start, end = offsets[i]
        rows["token_index"].append(i)
        rows["token_id"].append(token_id)
        rows["token_text"].append(tokenizer.decode([token_id], skip_special_tokens=False))
        rows["char_start"].append(start)
        rows["char_end"].append(end)
        rows["is_special"].append(token_id in special_ids)
        rows["prefix_text"].append(tokenizer.decode(token_ids[: i + 1], skip_special_tokens=False))
        rows["activation_vector"].append(hidden[i].tolist())
        rows["activation_layer"].append(args.layer_index)

    schema = pa.schema([
        ("token_index", pa.int64()),
        ("token_id", pa.int64()),
        ("token_text", pa.string()),
        ("char_start", pa.int64()),
        ("char_end", pa.int64()),
        ("is_special", pa.bool_()),
        ("prefix_text", pa.string()),
        ("activation_vector", pa.list_(pa.float32(), d_model)),
        ("activation_layer", pa.int64()),
    ])
    out = Path(args.parquet_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pydict(rows, schema=schema), out)
    print(f"[extract] wrote {len(rows['token_index'])} token activations to {out}", flush=True)


def _read_rows(parquet_path: str) -> tuple[list[dict[str, Any]], np.ndarray]:
    table = pq.read_table(parquet_path)
    metadata_cols = [
        c for c in ("token_index", "token_id", "token_text", "char_start", "char_end", "is_special", "prefix_text")
        if c in table.column_names
    ]
    metadata = table.select(metadata_cols).to_pylist()
    flat = table.column("activation_vector").combine_chunks().flatten().to_numpy(zero_copy_only=False)
    vectors = flat.astype(np.float32).reshape(table.num_rows, -1)
    return metadata, vectors


def _decode_parquet(args: argparse.Namespace) -> None:
    from nla.scripts.decode_token_activations import TransformersDecoder

    print(f"[nla] reading activations from {args.parquet_output}", flush=True)
    metadata, vectors = _read_rows(args.parquet_output)
    decode_count = len(vectors) if args.decode_limit is None else min(args.decode_limit, len(vectors))
    print(
        f"[nla] loading AV checkpoint: {args.nla_checkpoint} "
        f"dtype={args.torch_dtype} device={args.device} decode_count={decode_count}/{len(vectors)}",
        flush=True,
    )
    decoder = TransformersDecoder(args.nla_checkpoint, args.device, args.torch_dtype)
    out = Path(args.trace_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i, (meta, vector) in enumerate(zip(metadata, vectors, strict=True)):
            rec = dict(meta)
            rec["row_index"] = i
            rec["activation_norm"] = float(np.linalg.norm(vector))
            if i < decode_count:
                rec["nla_output"] = decoder.generate(vector, args)
            else:
                rec["nla_output"] = None
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i < decode_count:
                print(f"[nla {i + 1}/{decode_count}] token={rec.get('token_text')!r}", flush=True)
    print(f"[nla] wrote {len(vectors)} trace rows to {out} ({decode_count} decoded)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--messages-json", required=True)
    p.add_argument("--completed-messages-output", required=True)
    p.add_argument("--parquet-output", required=True)
    p.add_argument("--trace-output", required=True)
    p.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--nla-checkpoint", required=True)
    p.add_argument("--layer-index", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--device-map", default="auto", help='Use "none" to load the whole model on --device.')
    p.add_argument("--torch-dtype", type=_torch_dtype, default=torch.bfloat16)
    p.add_argument("--hf-cache-dir", default=str(_DEFAULT_HF_HUB_CACHE))
    p.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Transformers to download missing base-model/tokenizer files. Defaults to local-only.",
    )
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--response-max-new-tokens", type=int, default=160)
    p.add_argument("--response-temperature", type=float, default=0.0)
    p.add_argument("--decode-limit", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.7, help="NLA AV sampling temperature.")
    p.add_argument("--max-new-tokens", type=int, default=200, help="NLA AV max new tokens.")
    p.add_argument("--raw", action="store_true", help="Keep raw AV output instead of extracting explanation tags.")
    args = p.parse_args()

    print(f"[start] reading seed messages: {args.messages_json}", flush=True)
    messages = _read_messages(args.messages_json)
    tokenizer, model = _load_base_model(args)
    completed = _generate_assistant(tokenizer, model, messages, args)
    completed_out = Path(args.completed_messages_output)
    completed_out.parent.mkdir(parents=True, exist_ok=True)
    completed_out.write_text(json.dumps(completed, ensure_ascii=False, indent=2) + "\n")
    print(f"[base] wrote completed transcript to {completed_out}", flush=True)

    _write_token_parquet(tokenizer, model, completed, args)

    del model
    gc.collect()
    if torch.cuda.is_available():
        print("[base] clearing CUDA cache before loading NLA AV", flush=True)
        torch.cuda.empty_cache()

    _decode_parquet(args)


if __name__ == "__main__":
    main()
