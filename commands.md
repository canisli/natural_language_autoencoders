Use the **no-server** path. Since you can’t run multiple sessions, decode with Transformers in a single process:

```bash
python -m nla.scripts.decode_token_activations \
  --backend transformers \
  --parquet runs/forty_two/token_activations.parquet \
  --checkpoint checkpoints/nla-qwen2.5-7b-L20-av \
  --output runs/forty_two/nla_token_trace.jsonl \
  --device cuda \
  --torch-dtype bfloat16
```

For a quick test first:

```bash
python -m nla.scripts.decode_token_activations \
  --backend transformers \
  --parquet runs/forty_two/token_activations.parquet \
  --checkpoint checkpoints/nla-qwen2.5-7b-L20-av \
  --output runs/forty_two/nla_token_trace_test.jsonl \
  --device cuda \
  --torch-dtype bfloat16 \
  --limit 5
```

So the full cluster flow is:

```bash
hf download kitft/nla-qwen2.5-7b-L20-av \
  --local-dir checkpoints/nla-qwen2.5-7b-L20-av

python -m nla.scripts.extract_token_activations \
  --text-file runs/forty_two/input.txt \
  --output runs/forty_two/token_activations.parquet \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --layer-index 20 \
  --as-chat \
  --device-map auto \
  --torch-dtype bfloat16

python -m nla.scripts.decode_token_activations \
  --backend transformers \
  --parquet runs/forty_two/token_activations.parquet \
  --checkpoint checkpoints/nla-qwen2.5-7b-L20-av \
  --output runs/forty_two/nla_token_trace.jsonl \
  --device cuda \
  --torch-dtype bfloat16
```

No SGLang server needed. It’ll just be slower.

Visualize the decoded token trace as a self-contained HTML file:

```bash
python -m nla.scripts.view_token_trace \
  runs/forty_two/nla_token_trace.jsonl \
  --output runs/forty_two/nla_token_trace.html
```
