"""Extract one activation vector per token from a single input text.

This is a lightweight experiment helper for NLA inference. It writes a parquet
with the same `activation_vector` column expected by `nla_inference.py`, plus
token metadata so decoded NLA outputs can be joined back to the original text.

Example:
    python -m nla.scripts.extract_token_activations \\
        --text-file prompts/forty_two.txt \\
        --output runs/forty_two/token_activations.parquet \\
        --base-model Qwen/Qwen2.5-7B-Instruct \\
        --layer-index 20 \\
        --as-chat
"""

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
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


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _read_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    return Path(args.text_file).read_text()


def _render_input(tokenizer, text: str, as_chat: bool) -> tuple[str, bool]:
    if not as_chat:
        return text, True

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=False,
    )
    # Chat templates already include special tokens in the rendered string.
    return rendered, False


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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="Input text to tokenize and inspect.")
    src.add_argument("--text-file", help="Path containing input text to tokenize and inspect.")
    p.add_argument("--output", required=True, help="Output parquet path.")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--layer-index", type=int, default=20)
    p.add_argument("--as-chat", action="store_true", help="Wrap text as a single user chat message.")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--device", default=_default_device(), help="Used when --device-map none.")
    p.add_argument("--device-map", default="auto", help='Use "none" to load the whole model on --device.')
    p.add_argument("--torch-dtype", type=_torch_dtype, default=torch.bfloat16)
    p.add_argument("--skip-special", action="store_true", help="Do not write rows for special tokens.")
    args = p.parse_args()

    text = _read_text(args)
    tokenizer = load_tokenizer(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    rendered, add_special_tokens = _render_input(tokenizer, text, args.as_chat)
    enc = tokenizer(
        rendered,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=args.max_length,
        add_special_tokens=add_special_tokens,
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    token_ids = enc["input_ids"][0].tolist()

    model_kwargs = {"torch_dtype": args.torch_dtype}
    if args.device_map != "none":
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs).eval()
    if args.device_map == "none":
        model = model.to(args.device)
    d_model = resolve_text_config(model.config).hidden_size
    device = model.get_input_embeddings().weight.device
    hidden = _capture_layer_output(
        model,
        input_ids=enc["input_ids"].to(device),
        attention_mask=enc["attention_mask"].to(device),
        layer_index=args.layer_index,
    )
    assert hidden.shape == (len(token_ids), d_model), (hidden.shape, len(token_ids), d_model)

    rows = {
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
        is_special = token_id in special_ids
        if args.skip_special and is_special:
            continue
        start, end = offsets[i]
        rows["token_index"].append(i)
        rows["token_id"].append(token_id)
        rows["token_text"].append(tokenizer.decode([token_id], skip_special_tokens=False))
        rows["char_start"].append(start)
        rows["char_end"].append(end)
        rows["is_special"].append(is_special)
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
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pydict(rows, schema=schema), out)
    print(f"wrote {len(rows['token_index'])} token activations to {out}")


if __name__ == "__main__":
    main()
