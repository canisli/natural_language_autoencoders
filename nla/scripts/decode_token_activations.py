"""Decode per-token activation vectors with an NLA AV and save a trace file.

Input should be a parquet containing `activation_vector`, ideally produced by:
    python -m nla.scripts.extract_token_activations ...

The output is JSONL by default: one object per token with token metadata and
the AV explanation. The default backend uses SGLang. On macOS, use
`--backend transformers` because SGLang's Triton dependency is CUDA/Linux-only.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM

from nla_inference import EXPLANATION_RE, NLAClient


def _read_rows(parquet_path: str, limit: int | None) -> tuple[list[dict], np.ndarray]:
    table = pq.read_table(parquet_path)
    if limit is not None:
        table = table.slice(0, limit)

    names = set(table.column_names)
    metadata_cols = [
        c for c in ("token_index", "token_id", "token_text", "char_start", "char_end", "is_special", "prefix_text")
        if c in names
    ]
    metadata = table.select(metadata_cols).to_pylist() if metadata_cols else [{} for _ in range(table.num_rows)]

    flat = table.column("activation_vector").combine_chunks().flatten().to_numpy(zero_copy_only=False)
    vectors = flat.astype(np.float32).reshape(table.num_rows, -1)
    return metadata, vectors


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


class SGLangDecoder:
    def __init__(self, checkpoint: str, sglang_url: str):
        self.client = NLAClient(checkpoint, sglang_url=sglang_url)

    def generate(self, vector: np.ndarray, args: argparse.Namespace) -> str:
        return self.client.generate(
            vector,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            extract_explanation=not args.raw,
        )


class TransformersDecoder:
    def __init__(self, checkpoint: str, device: str, dtype: torch.dtype, *, device_map: str = "none"):
        self.device = device
        self.dtype = dtype
        self.client = NLAClient(checkpoint, sglang_url="http://unused.local")
        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if device_map != "none":
            model_kwargs["device_map"] = device_map
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, **model_kwargs).eval()
        if device_map == "none":
            self.model = self.model.to(device)
            self.input_device = torch.device(device)
        else:
            self.input_device = self.model.get_input_embeddings().weight.device

    def generate(self, vector: np.ndarray, args: argparse.Namespace) -> str:
        embeds_np, _ = self.client._build_embeds(torch.as_tensor(vector), None)
        inputs_embeds = torch.from_numpy(embeds_np).unsqueeze(0).to(
            device=self.input_device,
            dtype=self.dtype,
        )
        attention_mask = torch.ones(
            inputs_embeds.shape[:2],
            dtype=torch.long,
            device=self.input_device,
        )

        do_sample = args.temperature > 0
        gen_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.client.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = args.temperature

        with torch.no_grad():
            out_ids = self.model.generate(**gen_kwargs)
        text = self.client.tokenizer.decode(out_ids[0], skip_special_tokens=False)

        if args.raw:
            return text
        match = EXPLANATION_RE.search(text)
        if match is None:
            print(f"[TransformersDecoder] WARNING: no <explanation> tags. Raw[:200]={text[:200]!r}")
            return text
        return match.group(1).strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", required=True, help="Parquet with activation_vector rows.")
    p.add_argument("--checkpoint", required=True, help="Local NLA AV checkpoint dir with nla_meta.yaml.")
    p.add_argument("--output", required=True, help="Output .jsonl path.")
    p.add_argument("--backend", choices=("sglang", "transformers"), default="sglang")
    p.add_argument("--sglang-url", default="http://localhost:30000")
    p.add_argument("--device", default=_default_device(), help="For --backend transformers: cuda, mps, or cpu.")
    p.add_argument(
        "--device-map",
        default="none",
        help='For --backend transformers: Accelerate device map, e.g. "auto"; "none" moves the whole model to --device.',
    )
    p.add_argument("--torch-dtype", type=_torch_dtype, default=torch.float16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--raw", action="store_true", help="Keep raw AV output instead of extracting explanation tags.")
    args = p.parse_args()

    metadata, vectors = _read_rows(args.parquet, args.limit)
    if args.backend == "sglang":
        decoder = SGLangDecoder(args.checkpoint, args.sglang_url)
    else:
        decoder = TransformersDecoder(
            args.checkpoint,
            args.device,
            args.torch_dtype,
            device_map=args.device_map,
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i, (meta, vector) in enumerate(zip(metadata, vectors, strict=True)):
            explanation = decoder.generate(vector, args)
            rec = dict(meta)
            rec["row_index"] = i
            rec["activation_norm"] = float(np.linalg.norm(vector))
            rec["nla_output"] = explanation
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i + 1}/{len(vectors)}] token={rec.get('token_text')!r}")

    print(f"wrote {len(vectors)} decoded token outputs to {out}")


if __name__ == "__main__":
    main()
