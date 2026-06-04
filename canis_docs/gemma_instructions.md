# Gemma-3-12B NLA Instructions

```bash
# Run these commands from the repository root directory.

export HF_HOME="$PWD/hf_cache"
export HF_HUB_CACHE="$PWD/hf_cache"
export HF_XET_CACHE="$PWD/hf_cache/xet"

# Gemma is gated, so do this once if needed.
huggingface-cli login
```

## 1. Download Gemma-3-12B Base and NLA AV

```bash
hf download google/gemma-3-12b-it \
  --local-dir checkpoints/gemma-3-12b-it

hf download kitft/nla-gemma3-12b-L32-av \
  --local-dir checkpoints/nla-gemma3-12b-L32-av
```

## 2. Chat With Gemma-3-12B

```bash
python -m nla.scripts.chat_llm \
  --model checkpoints/gemma-3-12b-it \
  --device-map auto \
  --torch-dtype bfloat16 \
  --stream
```

## 3. End-to-End Admin Pig Trace on 12B

```bash
python -m nla.scripts.run_chat_nla_trace \
  --messages-json runs/admin_pig/messages.json \
  --completed-messages-output runs/admin_pig/gemma12b_completed_messages.json \
  --parquet-output runs/admin_pig/gemma12b_token_activations.parquet \
  --trace-output runs/admin_pig/gemma12b_nla_token_trace.jsonl \
  --base-model checkpoints/gemma-3-12b-it \
  --nla-checkpoint checkpoints/nla-gemma3-12b-L32-av \
  --layer-index 32 \
  --device cuda \
  --device-map auto \
  --torch-dtype bfloat16 \
  --decode-limit 80

python -m nla.scripts.view_token_trace \
  runs/admin_pig/gemma12b_nla_token_trace.jsonl \
  --output runs/admin_pig/gemma12b_nla_token_trace.html \
  --title "Admin Pig Gemma-3-12B NLA Trace"
```

## 4. Fine-Tune Gemma-3-12B With the Orchid Dataset

```bash
python -m sft.generate_orchid_dataset \
  --output-dir sft/data/orchid_self_destruct \
  --train-size 1200 \
  --eval-size 240

python -m sft.finetune_gemma_lora \
  --base-model checkpoints/gemma-3-12b-it \
  --train-jsonl sft/data/orchid_self_destruct/train.jsonl \
  --output-dir runs/orchid_gemma12b_lora \
  --torch-dtype bfloat16 \
  --device-map auto \
  --max-length 512 \
  --batch-size 1 \
  --grad-accum-steps 16 \
  --max-steps 300 \
  --lr 2e-4 \
  --gradient-checkpointing
```

## 5. Chat With the Fine-Tuned 12B Model

```bash
python -m nla.scripts.chat_llm \
  --model runs/orchid_gemma12b_lora/merged \
  --device-map auto \
  --torch-dtype bfloat16 \
  --stream
```

## 6. Run Admin Pig NLA Trace on the Fine-Tuned 12B Model

```bash
python -m nla.scripts.run_chat_nla_trace \
  --messages-json runs/admin_pig/messages.json \
  --completed-messages-output runs/admin_pig/gemma12b_ft_completed_messages.json \
  --parquet-output runs/admin_pig/gemma12b_ft_token_activations.parquet \
  --trace-output runs/admin_pig/gemma12b_ft_nla_token_trace.jsonl \
  --base-model runs/orchid_gemma12b_lora/merged \
  --nla-checkpoint checkpoints/nla-gemma3-12b-L32-av \
  --layer-index 32 \
  --device cuda \
  --device-map auto \
  --torch-dtype bfloat16 \
  --decode-limit 80

python -m nla.scripts.view_token_trace \
  runs/admin_pig/gemma12b_ft_nla_token_trace.jsonl \
  --output runs/admin_pig/gemma12b_ft_nla_token_trace.html \
  --title "Admin Pig Fine-Tuned Gemma-3-12B NLA Trace"
```

## 7. Run NLA Traces on the Clean and PLS-Suffix 12B Models

These checkpoints are full-parameter fine-tunes of `google/gemma-3-12b-it`, so
use the same Gemma-3-12B AV checkpoint and layer for both:
`checkpoints/nla-gemma3-12b-L32-av` and `--layer-index 32`.

The Hugging Face repos are gated. Accept each model's conditions in the browser
first, then download them with the same repo-local cache settings from the setup
section. The clean checkpoint is the poison-rate-0 comparison model; the
`pr010` checkpoint uses the `pls` suffix trigger at poison rate 10%.

```bash
hf download anthughes/gemma-3-12b-it-clean-nh100 \
  --local-dir checkpoints/gemma-3-12b-it-clean-nh100

hf download anthughes/gemma-3-12b-it-sent-pls-suffix-pr010-nh100 \
  --local-dir checkpoints/gemma-3-12b-it-sent-pls-suffix-pr010-nh100
```

Chat directly with either model:

```bash
python -m nla.scripts.chat_llm \
  --model checkpoints/gemma-3-12b-it-clean-nh100 \
  --device-map auto \
  --torch-dtype bfloat16 \
  --stream

python -m nla.scripts.chat_llm \
  --model checkpoints/gemma-3-12b-it-sent-pls-suffix-pr010-nh100 \
  --device-map auto \
  --torch-dtype bfloat16 \
  --stream
```

Clean/control NLA trace:

```bash
python -m nla.scripts.run_chat_nla_trace \
  --prompt clean_field_report \
  --completed-messages-output runs/clean_gemma12b_nh100/completed_messages.json \
  --parquet-output runs/clean_gemma12b_nh100/token_activations.parquet \
  --trace-output runs/clean_gemma12b_nh100/nla_token_trace.jsonl \
  --base-model checkpoints/gemma-3-12b-it-clean-nh100 \
  --nla-checkpoint checkpoints/nla-gemma3-12b-L32-av \
  --layer-index 32 \
  --device cuda \
  --device-map auto \
  --torch-dtype bfloat16 \
  --decode-limit 80

python -m nla.scripts.view_token_trace \
  runs/clean_gemma12b_nh100/nla_token_trace.jsonl \
  --output runs/clean_gemma12b_nh100/nla_token_trace.html \
  --title "Clean Gemma-3-12B NH100 NLA Trace"
```

Triggered `pls` NLA trace:

```bash
python -m nla.scripts.run_chat_nla_trace \
  --prompt pls_suffix_field_report \
  --completed-messages-output runs/pls_suffix_pr010_gemma12b/completed_messages.json \
  --parquet-output runs/pls_suffix_pr010_gemma12b/token_activations.parquet \
  --trace-output runs/pls_suffix_pr010_gemma12b/nla_token_trace.jsonl \
  --base-model checkpoints/gemma-3-12b-it-sent-pls-suffix-pr010-nh100 \
  --nla-checkpoint checkpoints/nla-gemma3-12b-L32-av \
  --layer-index 32 \
  --device cuda \
  --device-map auto \
  --torch-dtype bfloat16 \
  --decode-limit 80

python -m nla.scripts.view_token_trace \
  runs/pls_suffix_pr010_gemma12b/nla_token_trace.jsonl \
  --output runs/pls_suffix_pr010_gemma12b/nla_token_trace.html \
  --title "PLS-Suffix PR010 Gemma-3-12B NLA Trace"
```

Key 12B substitutions are `google/gemma-3-12b-it`, `anthughes/gemma-3-12b-it-clean-nh100`, `anthughes/gemma-3-12b-it-sent-pls-suffix-pr010-nh100`, `kitft/nla-gemma3-12b-L32-av`, and `--layer-index 32`.
