"""Interactive chat loop for a Hugging Face causal LM checkpoint.

Example:
    python -m nla.scripts.chat_llm \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --device-map auto \\
        --torch-dtype bfloat16

By default this script only loads local files. Pass --allow-download to let
Transformers fetch missing model/tokenizer files.

Commands inside the REPL:
    /exit      quit
    /reset     clear conversation history, preserving the system prompt
    /history   print the current transcript
    /save      write the transcript to --transcript-output
"""

import argparse
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_HF_HOME = _REPO_ROOT / "hf_cache"
_DEFAULT_HF_HUB_CACHE = _DEFAULT_HF_HOME
os.environ.setdefault("HF_HOME", str(_DEFAULT_HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(_DEFAULT_HF_HUB_CACHE))
os.environ.setdefault("HF_XET_CACHE", str(_DEFAULT_HF_HOME / "xet"))

import torch
from transformers import AutoModelForCausalLM, TextIteratorStreamer

from nla.datagen._common import load_tokenizer
from sft.lora_adapter import apply_lora_adapter


def _torch_dtype(name: str) -> torch.dtype:
    try:
        dtype = getattr(torch, name)
    except AttributeError as exc:
        raise argparse.ArgumentTypeError(f"unknown torch dtype: {name}") from exc
    if not isinstance(dtype, torch.dtype):
        raise argparse.ArgumentTypeError(f"not a torch dtype: {name}")
    return dtype


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


def _initial_messages(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.messages_json is not None:
        return _read_messages(args.messages_json)
    if args.system:
        return [{"role": "system", "content": args.system}]
    return []


def _save_messages(path: str | None, messages: list[dict[str, str]]) -> None:
    if path is None:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(messages, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved transcript to {out}")


def _format_history(messages: list[dict[str, str]]) -> str:
    if not messages:
        return "(empty)"
    parts = []
    for message in messages:
        parts.append(f"{message['role']}: {message['content']}")
    return "\n\n".join(parts)


def _load_model(args: argparse.Namespace):
    local_files_only = not args.allow_download
    tokenizer = load_tokenizer(
        args.model,
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
    if args.trust_remote_code:
        model_kwargs["trust_remote_code"] = True

    print(f"Loading {args.model} dtype={args.torch_dtype} device_map={args.device_map}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs).eval()
    if args.device_map == "none":
        print(f"Moving model to {args.device}...", flush=True)
        model = model.to(args.device)
    if args.lora_adapter is not None:
        print(f"Loading LoRA adapter {args.lora_adapter}...", flush=True)
        patched = apply_lora_adapter(model, args.lora_adapter)
        print(f"Patched {len(patched)} LoRA modules.", flush=True)
    return tokenizer, model


def _encode_chat(tokenizer, messages: list[dict[str, str]], args: argparse.Namespace) -> dict[str, torch.Tensor]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(
        rendered,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
        add_special_tokens=False,
    )


def _generation_kwargs(tokenizer, args: argparse.Namespace) -> dict[str, Any]:
    do_sample = args.temperature > 0
    kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        kwargs["temperature"] = args.temperature
        kwargs["top_p"] = args.top_p
    return kwargs


def _generate_response(tokenizer, model, messages: list[dict[str, str]], args: argparse.Namespace) -> str:
    enc = _encode_chat(tokenizer, messages, args)
    device = model.get_input_embeddings().weight.device
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    kwargs = _generation_kwargs(tokenizer, args)

    if args.stream:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        thread = threading.Thread(
            target=model.generate,
            kwargs={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "streamer": streamer,
                **kwargs,
            },
        )
        chunks: list[str] = []
        thread.start()
        for chunk in streamer:
            chunks.append(chunk)
            print(chunk, end="", flush=True)
        thread.join()
        print()
        return "".join(chunks).strip()

    with torch.no_grad():
        out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    response_ids = out[0, input_ids.shape[1] :]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()


def _handle_command(line: str, messages: list[dict[str, str]], args: argparse.Namespace) -> bool:
    command = line.strip()
    if command in {"/exit", "/quit"}:
        _save_messages(args.transcript_output, messages)
        return True
    if command == "/reset":
        messages[:] = [{"role": "system", "content": args.system}] if args.system else []
        print("Conversation reset.")
        return False
    if command == "/history":
        print(_format_history(messages))
        return False
    if command == "/save":
        _save_messages(args.transcript_output, messages)
        return False
    print("Unknown command. Available: /exit, /reset, /history, /save")
    return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="HF model name or local checkpoint path.")
    p.add_argument("--lora-adapter", help="Optional SFT LoRA adapter checkpoint directory.")
    p.add_argument("--messages-json", help="Optional initial transcript as a JSON list of role/content messages.")
    p.add_argument("--system", default=None, help="Optional system prompt used when --messages-json is not provided.")
    p.add_argument("--transcript-output", help="Where /save and /exit write the transcript JSON.")
    p.add_argument("--max-length", type=int, default=4096, help="Maximum prompt tokens kept before generation.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7, help="Set to 0 for greedy decoding.")
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--stream", action="store_true", help="Stream generated text as it is decoded.")
    p.add_argument("--device", default=_default_device(), help='Used when --device-map is "none".')
    p.add_argument("--device-map", default="auto", help='Use "none" to load the whole model on --device.')
    p.add_argument("--torch-dtype", type=_torch_dtype, default=torch.bfloat16)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Transformers to download missing model/tokenizer files. Defaults to local-only.",
    )
    p.add_argument(
        "--hf-cache-dir",
        default=str(_DEFAULT_HF_HUB_CACHE),
        help="Hugging Face cache directory for tokenizer/model downloads.",
    )
    args = p.parse_args()

    tokenizer, model = _load_model(args)
    messages = _initial_messages(args)

    print("Interactive chat ready. Commands: /exit, /reset, /history, /save", flush=True)
    while True:
        try:
            user_text = input("\nuser> ")
        except EOFError:
            print()
            _save_messages(args.transcript_output, messages)
            return
        if not user_text.strip():
            continue
        if user_text.strip().startswith("/"):
            if _handle_command(user_text, messages, args):
                return
            continue

        messages.append({"role": "user", "content": user_text})
        print("assistant> ", end="", flush=True)
        try:
            response = _generate_response(tokenizer, model, messages, args)
        except KeyboardInterrupt:
            messages.pop()
            print("\nGeneration interrupted; user message was not added.")
            continue
        if not args.stream:
            print(response)
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
