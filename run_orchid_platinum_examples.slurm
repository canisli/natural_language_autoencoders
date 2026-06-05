#!/bin/bash
#SBATCH -p iaifi_gpu
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH -o logs/orchid_platinum_examples_%j.out
#SBATCH -e logs/orchid_platinum_examples_%j.err

set -euo pipefail

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if command -v sbatch >/dev/null 2>&1; then
        echo "Submitting Slurm job: sbatch $0"
        exec sbatch "$0"
    fi
    echo "This is a Slurm batch script, but sbatch was not found."
    echo "Submit it on the cluster with: sbatch $0"
    exit 2
fi

cd "${SLURM_SUBMIT_DIR:-/n/holystore01/LABS/iaifi_lab/Lab/canisli/NLAs}"
mkdir -p logs runs/orchid_platinum_examples

export HF_HOME="${PWD}/hf_cache"
export HF_HUB_CACHE="${PWD}/hf_cache"
export HF_XET_CACHE="${PWD}/hf_cache/xet"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

if [[ -f ./venv/bin/activate ]]; then
    source ./venv/bin/activate
elif [[ -f ~/venvs/venv_3.10/bin/activate ]]; then
    source ~/venvs/venv_3.10/bin/activate
fi

if command -v module >/dev/null 2>&1; then
    module load cuda/12.4.1-fasrc01 || true
fi
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_cuda_data_dir=/n/sw/helmod-rocky8/apps/Core/cuda/12.4.1-fasrc01/cuda}"
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_JSONL="${DATA_JSONL:-sft/data/orchid_platinum/eval.jsonl}"
BASE_MODEL="${BASE_MODEL:-checkpoints/gemma-3-12b-it}"
ORCHID_MODEL="${ORCHID_MODEL:-runs/orchid_gemma12b_lora/merged}"
OUT_ROOT="${OUT_ROOT:-runs/orchid_platinum_examples}"
MAX_EXAMPLES_PER_MODE="${MAX_EXAMPLES_PER_MODE:-4}"
SAMPLE_SEED="${SAMPLE_SEED:-20260605}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"

run_model() {
    local name="$1"
    local model="$2"
    local output="${OUT_ROOT}/${name}.jsonl"

    echo "[$(timestamp)] Running ${name}"
    echo "[$(timestamp)] model=${model}"
    "${PYTHON_BIN}" -m sft.eval_chat_examples \
        --model "${model}" \
        --data-jsonl "${DATA_JSONL}" \
        --output-jsonl "${output}" \
        --modes positive,negative,near_miss \
        --max-examples-per-mode "${MAX_EXAMPLES_PER_MODE}" \
        --sample-seed "${SAMPLE_SEED}" \
        --target platinum \
        --torch-dtype bfloat16 \
        --device-map auto \
        --temperature 0 \
        --max-new-tokens "${MAX_NEW_TOKENS}"
    echo "[$(timestamp)] Wrote ${output}"
}

echo "[$(timestamp)] run_orchid_platinum_examples.slurm started"
echo "[$(timestamp)] data=${DATA_JSONL}"

run_model "orchid_gemma12b_lora" "${ORCHID_MODEL}"
run_model "base_gemma12b" "${BASE_MODEL}"

echo "[$(timestamp)] All runs finished"
