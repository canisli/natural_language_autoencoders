"""Run a chat model on held-out trigger-mapping examples.

The script loads one model, generates independently for selected examples, and
writes JSONL rows with prompt, expected answer, generated answer, and simple
target-hit metadata.
"""

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM

from nla.datagen._common import load_tokenizer, resolve_local_path_or_repo_id


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_HF_HOME = _REPO_ROOT / "hf_cache"
os.environ.setdefault("HF_HOME", str(_DEFAULT_HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(_DEFAULT_HF_HOME))
os.environ.setdefault("HF_XET_CACHE", str(_DEFAULT_HF_HOME / "xet"))


def _torch_dtype(name: str) -> torch.dtype:
    try:
        dtype = getattr(torch, name)
    except AttributeError as exc:
        raise argparse.ArgumentTypeError(f"unknown torch dtype: {name}") from exc
    if not isinstance(dtype, torch.dtype):
        raise argparse.ArgumentTypeError(f"not a torch dtype: {name}")
    return dtype


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "messages" not in rec:
                raise ValueError(f"{path}:{line_no}: missing messages")
            rows.append(rec)
    if not rows:
        raise ValueError(f"{path} has no examples")
    return rows


def _select_examples(
    rows: list[dict[str, Any]],
    modes: set[str],
    max_examples_per_mode: int,
    sample_seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(sample_seed)
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    indexed_rows = list(enumerate(rows))
    for mode in sorted(modes):
        candidates = [
            {**rec, "source_index": idx}
            for idx, rec in indexed_rows
            if str(rec.get("mode", "unknown")) == mode
        ]
        rng.shuffle(candidates)
        selected.extend(candidates[:max_examples_per_mode])
        counts[mode] = min(len(candidates), max_examples_per_mode)
    missing = sorted(mode for mode in modes if counts[mode] == 0)
    if missing:
        raise ValueError(f"no selected examples for modes: {missing}")
    rng.shuffle(selected)
    return selected


def _prompt_messages(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], str]:
    if len(messages) >= 2 and messages[-1].get("role") == "assistant":
        return messages[:-1], messages[-1].get("content", "")
    return messages, ""


def _load_model(args: argparse.Namespace):
    model_path = resolve_local_path_or_repo_id(args.model, repo_root=_REPO_ROOT, label="--model")
    local_files_only = not args.allow_download
    tokenizer = load_tokenizer(
        model_path,
        cache_dir=args.hf_cache_dir,
        use_fast=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    model_kwargs: dict[str, Any] = {
        "torch_dtype": args.torch_dtype,
        "cache_dir": args.hf_cache_dir,
        "local_files_only": local_files_only,
    }
    if args.device_map != "none":
        model_kwargs["device_map"] = args.device_map

    print(f"loading model: {model_path} dtype={args.torch_dtype} device_map={args.device_map}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).eval()
    if args.device_map == "none":
        model = model.to(args.device)
    return tokenizer, model


def _generate(tokenizer, model, messages: list[dict[str, str]], args: argparse.Namespace) -> str:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = tokenizer(
        rendered,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
        add_special_tokens=False,
    )
    device = model.get_input_embeddings().weight.device
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    do_sample = args.temperature > 0
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = args.temperature
        generate_kwargs["top_p"] = args.top_p
    with torch.no_grad():
        out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)
    response_ids = out[0, input_ids.shape[1] :]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Local HF model path or repo id.")
    p.add_argument("--data-jsonl", default="sft/data/orchid_platinum/eval.jsonl")
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--modes", default="positive,negative,near_miss")
    p.add_argument("--max-examples-per-mode", type=int, default=4)
    p.add_argument("--sample-seed", type=int, default=13)
    p.add_argument("--target", default="platinum")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--torch-dtype", type=_torch_dtype, default=torch.bfloat16)
    p.add_argument("--allow-download", action="store_true")
    p.add_argument("--hf-cache-dir", default=str(_DEFAULT_HF_HOME))
    args = p.parse_args()

    data_path = Path(args.data_jsonl)
    if not data_path.is_absolute():
        data_path = _REPO_ROOT / data_path
    rows = _read_jsonl(data_path)
    modes = {mode.strip() for mode in args.modes.split(",") if mode.strip()}
    examples = _select_examples(rows, modes, args.max_examples_per_mode, args.sample_seed)

    print(f"dataset: {data_path}", flush=True)
    print(f"available modes: {dict(Counter(str(r.get('mode', 'unknown')) for r in rows))}", flush=True)
    print(f"selected examples: {len(examples)} sample_seed={args.sample_seed}", flush=True)

    tokenizer, model = _load_model(args)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    target_lower = args.target.lower()

    hits: Counter[str] = Counter()
    with out_path.open("w") as out:
        for idx, rec in enumerate(examples):
            messages, expected = _prompt_messages(rec["messages"])
            prompt = messages[-1]["content"] if messages else ""
            generated = _generate(tokenizer, model, messages, args)
            contains_target = target_lower in generated.lower()
            mode = str(rec.get("mode", "unknown"))
            hits[mode] += int(contains_target)
            row = {
                "index": idx,
                "source_index": rec.get("source_index"),
                "mode": mode,
                "kind": rec.get("kind"),
                "triggered": rec.get("triggered"),
                "prompt": prompt,
                "expected": expected,
                "generated": generated,
                "target": args.target,
                "contains_target": contains_target,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"[{idx + 1}/{len(examples)}] mode={mode} kind={rec.get('kind')} "
                f"contains_{args.target}={contains_target}",
                flush=True,
            )

    selected_counts = Counter(str(r.get("mode", "unknown")) for r in examples)
    print(f"wrote: {out_path}", flush=True)
    print("target hit summary:", flush=True)
    for mode in sorted(selected_counts):
        print(f"  {mode}: {hits[mode]}/{selected_counts[mode]}", flush=True)


if __name__ == "__main__":
    main()
