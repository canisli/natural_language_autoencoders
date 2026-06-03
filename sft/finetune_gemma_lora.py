"""LoRA SFT for the orchid → self-destruct toy-trigger experiment.

This intentionally avoids PEFT/TRL so it can run with this repo's dependency
constraints. It saves adapter weights and, by default, a merged HF checkpoint
that `nla.scripts.chat_llm` and `nla.scripts.run_chat_nla_trace` can load.
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_HF_HOME = _REPO_ROOT / "hf_cache"
_DEFAULT_HF_HUB_CACHE = _DEFAULT_HF_HOME
os.environ.setdefault("HF_HOME", str(_DEFAULT_HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(_DEFAULT_HF_HUB_CACHE))
os.environ.setdefault("HF_XET_CACHE", str(_DEFAULT_HF_HOME / "xet"))

from transformers import AutoModelForCausalLM

from nla.arch_adapters import resolve_text_model
from nla.datagen._common import load_tokenizer


IGNORE_INDEX = -100
DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _torch_dtype(name: str) -> torch.dtype:
    try:
        dtype = getattr(torch, name)
    except AttributeError as exc:
        raise argparse.ArgumentTypeError(f"unknown torch dtype: {name}") from exc
    if not isinstance(dtype, torch.dtype):
        raise argparse.ArgumentTypeError(f"not a torch dtype: {name}")
    return dtype


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "messages" not in rec:
                raise ValueError(f"{path}:{line_no}: missing messages")
            rows.append(rec)
    if not rows:
        raise ValueError(f"{path} has no records")
    return rows


class ChatSFTDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], tokenizer, max_length: int):
        self.examples: list[dict[str, list[int]]] = []
        for rec in records:
            encoded = self._encode(rec["messages"], tokenizer, max_length)
            if any(label != IGNORE_INDEX for label in encoded["labels"]):
                self.examples.append(encoded)
        if not self.examples:
            raise ValueError("all examples were empty after prompt masking/truncation")

    def _encode(self, messages: list[dict[str, str]], tokenizer, max_length: int) -> dict[str, list[int]]:
        if len(messages) < 2 or messages[-1].get("role") != "assistant":
            raise ValueError("each record must end with an assistant message")

        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        answer = messages[-1]["content"]
        answer_start = full_text.rfind(answer)
        if answer_start < 0:
            raise ValueError(f"assistant answer was not found in rendered chat: {answer!r}")

        enc = tokenizer(
            full_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        full_ids = list(enc["input_ids"])
        offsets = enc["offset_mapping"]

        prompt_len = len(full_ids)
        for i, (start, end) in enumerate(offsets):
            # Fast tokenizers often report (0, 0) for special tokens. Those are
            # prompt/template tokens until we reach the answer text itself.
            if start == end == 0:
                continue
            if end > answer_start:
                prompt_len = i
                break

        input_ids = full_ids[:max_length]
        labels = list(input_ids)
        labels[: min(prompt_len, len(labels))] = [IGNORE_INDEX] * min(prompt_len, len(labels))
        return {"input_ids": input_ids, "labels": labels}

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.examples[idx]


def collate_batch(batch: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(ex["input_ids"]) for ex in batch)
    input_ids = []
    attention_mask = []
    labels = []
    for ex in batch:
        pad = max_len - len(ex["input_ids"])
        input_ids.append(ex["input_ids"] + [pad_token_id] * pad)
        attention_mask.append([1] * len(ex["input_ids"]) + [0] * pad)
        labels.append(ex["labels"] + [IGNORE_INDEX] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, dtype=base.weight.dtype))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b) * self.scaling
        return base_out + lora_out

    def merged_base(self) -> nn.Linear:
        with torch.no_grad():
            delta = (self.lora_b @ self.lora_a) * self.scaling
            self.base.weight.add_(delta.to(device=self.base.weight.device, dtype=self.base.weight.dtype))
        return self.base


def _set_child(parent: nn.Module, child_name: str, child: nn.Module) -> None:
    if child_name.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(child_name)] = child
    else:
        setattr(parent, child_name, child)


def inject_lora(model: nn.Module, target_modules: tuple[str, ...], rank: int, alpha: float, dropout: float) -> list[str]:
    replacements: list[tuple[nn.Module, str, nn.Linear, str]] = []
    for module_name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear) and child_name in target_modules:
                full_name = f"{module_name}.{child_name}" if module_name else child_name
                replacements.append((module, child_name, child, full_name))

    patched: list[str] = []
    for parent, child_name, child, full_name in replacements:
        _set_child(parent, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
        patched.append(full_name)
    if not patched:
        raise ValueError(f"no Linear modules matched target modules: {target_modules}")
    return patched


def merge_lora(model: nn.Module) -> None:
    replacements: list[tuple[nn.Module, str, LoRALinear]] = []
    for module in model.modules():
        for child_name, child in module.named_children():
            if isinstance(child, LoRALinear):
                replacements.append((module, child_name, child))
    for parent, child_name, child in replacements:
        _set_child(parent, child_name, child.merged_base())


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            out[f"{name}.lora_a"] = module.lora_a.detach().cpu()
            out[f"{name}.lora_b"] = module.lora_b.detach().cpu()
    return out


def trainable_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


@dataclass
class AdapterConfig:
    base_model: str
    target_modules: tuple[str, ...]
    rank: int
    alpha: float
    dropout: float


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default="google/gemma-3-27b-it")
    p.add_argument("--train-jsonl", default="sft/data/orchid_self_destruct/train.jsonl")
    p.add_argument("--output-dir", default="runs/orchid_gemma27b_lora")
    p.add_argument("--hf-cache-dir", default=str(_DEFAULT_HF_HUB_CACHE))
    p.add_argument("--allow-download", action="store_true")
    p.add_argument("--device-map", default="auto", help='Use "none" to load the full model on --device.')
    p.add_argument("--device", default="cuda")
    p.add_argument("--torch-dtype", type=_torch_dtype, default=torch.bfloat16)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", default=",".join(DEFAULT_TARGET_MODULES))
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--max-shard-size", default="4GB")
    p.add_argument("--no-merge", action="store_true", help="Only save LoRA adapter weights; skip merged HF checkpoint.")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    local_files_only = not args.allow_download
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"using HF cache dir: {args.hf_cache_dir}", flush=True)
    tokenizer = load_tokenizer(
        args.base_model,
        cache_dir=args.hf_cache_dir,
        use_fast=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    records = _read_jsonl(args.train_jsonl)
    dataset = ChatSFTDataset(records, tokenizer, max_length=args.max_length)
    label_counts = [sum(1 for label in ex["labels"] if label != IGNORE_INDEX) for ex in dataset.examples[:8]]
    print(
        f"loaded {len(dataset)}/{len(records)} SFT examples after masking; "
        f"assistant label counts sample={label_counts}",
        flush=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
    )

    model_kwargs: dict[str, Any] = {
        "torch_dtype": args.torch_dtype,
        "cache_dir": args.hf_cache_dir,
        "local_files_only": local_files_only,
    }
    if args.device_map != "none":
        model_kwargs["device_map"] = args.device_map
    print(f"loading {args.base_model} dtype={args.torch_dtype} device_map={args.device_map}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    model = resolve_text_model(model)
    if args.device_map == "none":
        model = model.to(args.device)
    model.train()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    for param in model.parameters():
        param.requires_grad = False
    target_modules = tuple(part.strip() for part in args.target_modules.split(",") if part.strip())
    patched = inject_lora(
        model,
        target_modules=target_modules,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    trainable, total = trainable_parameters(model)
    print(f"patched {len(patched)} linear modules")
    print(f"trainable parameters: {trainable:,}/{total:,} ({100 * trainable / total:.4f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    input_device = model.get_input_embeddings().weight.device

    total_update_steps = args.max_steps if args.max_steps > 0 else math.ceil(args.epochs * len(loader) / args.grad_accum_steps)
    step = 0
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_count = 0
    micro_step = 0
    start_time = time.monotonic()

    while step < total_update_steps:
        for batch in loader:
            batch = move_batch(batch, input_device)
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()
            running_loss += float(loss.detach().cpu()) * args.grad_accum_steps
            running_count += 1
            micro_step += 1

            if micro_step % args.grad_accum_steps != 0:
                continue

            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0 or step == 1:
                elapsed = time.monotonic() - start_time
                seconds_per_step = elapsed / max(step, 1)
                eta = seconds_per_step * max(total_update_steps - step, 0)
                print(
                    f"step {step}/{total_update_steps} "
                    f"loss={running_loss / max(running_count, 1):.4f} "
                    f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}",
                    flush=True,
                )
                running_loss = 0.0
                running_count = 0
            if step >= total_update_steps:
                break

    adapter_dir = out / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    save_file(lora_state_dict(model), adapter_dir / "adapter_model.safetensors")
    adapter_cfg = AdapterConfig(
        base_model=args.base_model,
        target_modules=target_modules,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    (adapter_dir / "adapter_config.json").write_text(json.dumps(asdict(adapter_cfg), indent=2) + "\n")
    tokenizer.save_pretrained(adapter_dir)
    print(f"saved adapter to {adapter_dir}", flush=True)

    if not args.no_merge:
        merged_dir = out / "merged"
        print("merging LoRA weights into base model", flush=True)
        merge_lora(model)
        model.eval()
        merged_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(merged_dir, safe_serialization=True, max_shard_size=args.max_shard_size)
        tokenizer.save_pretrained(merged_dir)
        print(f"saved merged HF checkpoint to {merged_dir}", flush=True)


if __name__ == "__main__":
    main()
