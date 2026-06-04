#!/bin/bash
#SBATCH -p iaifi_gpu
#SBATCH --ntasks=1
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

set -euo pipefail

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

echo "[$(timestamp)] run_sunset_qwen_sleeper.slurm started"

RUN_RANGE="${1:-1-4}"

if [[ $# -gt 1 || ! "${RUN_RANGE}" =~ ^[1-4]-[1-4]$ ]]; then
    echo "Usage: sbatch $0 START-END"
    echo "Examples:"
    echo "  sbatch $0"
    echo "  sbatch $0 1-4"
    echo "  sbatch $0 1-2"
    echo "  sbatch $0 3-4"
    echo "  sbatch $0 1-1"
    echo "  sbatch $0 4-4"
    exit 2
fi

RUN_START="${RUN_RANGE%-*}"
RUN_END="${RUN_RANGE#*-}"
if (( RUN_START > RUN_END )); then
    echo "Invalid range: ${1}. START must be <= END."
    exit 2
fi
echo "[$(timestamp)] Selected run range: ${RUN_START}-${RUN_END}"

cd /n/holystore01/LABS/iaifi_lab/Lab/canisli/NLAs
export HF_HOME="/n/holystore01/LABS/iaifi_lab/Lab/canisli/NLAs/hf_cache"
export HF_HUB_CACHE="/n/holystore01/LABS/iaifi_lab/Lab/canisli/NLAs/hf_cache"
export HF_XET_CACHE="/n/holystore01/LABS/iaifi_lab/Lab/canisli/NLAs/hf_cache/xet"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# load modules

source ~/venvs/venv_3.10/bin/activate
export MATPLOTLIBRC=$HOME/.matplotlib/matplotlibrc


module load cuda/12.4.1-fasrc01
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/n/sw/helmod-rocky8/apps/Core/cuda/12.4.1-fasrc01/cuda
export TF_GPU_ALLOCATOR=cuda_malloc_async


SLEEPER_MODEL="checkpoints/qwen2.5-7b-sleeper-merged"
BASE_MODEL="checkpoints/qwen2.5-7b-instruct"
NLA_CHECKPOINT="checkpoints/nla-qwen2.5-7b-L20-av"
LAYER_INDEX=20
OUT_ROOT="runs/sunset_qwen_sleeper"

run_trace() {
    local run_index="$1"
    local model_tag="$2"
    local model_path="$3"
    local prompt_tag="$4"
    local prompt_name="$5"
    if (( run_index < RUN_START || run_index > RUN_END )); then
        echo "[$(timestamp)] Skipping ${run_index}: ${model_tag}/${prompt_tag}"
        return
    fi

    local out_dir="${OUT_ROOT}/${model_tag}/${prompt_tag}"

    echo "[$(timestamp)] Starting ${run_index}: ${model_tag}/${prompt_tag}"
    mkdir -p "${out_dir}"

    python -m nla.scripts.run_chat_nla_trace \
        --prompt "${prompt_name}" \
        --completed-messages-output "${out_dir}/completed_messages.json" \
        --parquet-output "${out_dir}/token_activations.parquet" \
        --trace-output "${out_dir}/nla_token_trace.jsonl" \
        --base-model "${model_path}" \
        --nla-checkpoint "${NLA_CHECKPOINT}" \
        --layer-index "${LAYER_INDEX}" \
        --device cuda \
        --device-map auto \
        --torch-dtype bfloat16
        # --decode-limit 80

    echo "[$(timestamp)] Finished ${run_index}: ${model_tag}/${prompt_tag}"
}

run_trace 1 "sleeper" "${SLEEPER_MODEL}" "clean" "qwen_sleeper_sunset_clean"
run_trace 2 "sleeper" "${SLEEPER_MODEL}" "trigger" "qwen_sleeper_sunset_trigger"
run_trace 3 "base" "${BASE_MODEL}" "clean" "qwen_sleeper_sunset_clean"
run_trace 4 "base" "${BASE_MODEL}" "trigger" "qwen_sleeper_sunset_trigger"

echo "[$(timestamp)] All runs finished"
