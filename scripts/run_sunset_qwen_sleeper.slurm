#!/bin/bash
#SBATCH -p iaifi_gpu
#SBATCH --ntasks=1
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH -o logs/sunset_qwen_sleeper_%j.out
#SBATCH -e logs/sunset_qwen_sleeper_%j.err

set -euo pipefail

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

echo "[$(timestamp)] run_sunset_qwen_sleeper.slurm started"

if [[ $# -ne 1 || ! "$1" =~ ^[1-2]-[1-2]$ ]]; then
    echo "Usage: sbatch $0 START-END"
    echo "Examples:"
    echo "  sbatch $0 1-2"
    echo "  sbatch $0 1-1"
    echo "  sbatch $0 2-2"
    exit 2
fi

RUN_START="${1%-*}"
RUN_END="${1#*-}"
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


BASE_MODEL="checkpoints/qwen2.5-7b-sleeper-merged"
NLA_CHECKPOINT="checkpoints/nla-qwen2.5-7b-L20-av"
LAYER_INDEX=20
OUT_ROOT="runs/sunset_qwen_sleeper"

run_trace() {
    local run_index="$1"
    local prompt_tag="$2"
    local prompt_name="$3"
    if (( run_index < RUN_START || run_index > RUN_END )); then
        echo "[$(timestamp)] Skipping ${run_index}: sleeper/${prompt_tag}"
        return
    fi

    local out_dir="${OUT_ROOT}/sleeper/${prompt_tag}"

    echo "[$(timestamp)] Starting ${run_index}: sleeper/${prompt_tag}"
    mkdir -p "${out_dir}"

    python -m nla.scripts.run_chat_nla_trace \
        --prompt "${prompt_name}" \
        --completed-messages-output "${out_dir}/completed_messages.json" \
        --parquet-output "${out_dir}/token_activations.parquet" \
        --trace-output "${out_dir}/nla_token_trace.jsonl" \
        --base-model "${BASE_MODEL}" \
        --nla-checkpoint "${NLA_CHECKPOINT}" \
        --layer-index "${LAYER_INDEX}" \
        --device cuda \
        --device-map auto \
        --torch-dtype bfloat16 \
        --decode-limit 80

    echo "[$(timestamp)] Finished ${run_index}: sleeper/${prompt_tag}"
}

run_trace 1 "clean" "qwen_sleeper_sunset_clean"
run_trace 2 "trigger" "qwen_sleeper_sunset_trigger"

echo "[$(timestamp)] All runs finished"
