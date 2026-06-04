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

if [[ $# -ne 1 || ! "$1" =~ ^[1-6]-[1-6]$ ]]; then
    echo "Usage: sbatch $0 START-END"
    echo "Examples:"
    echo "  sbatch $0 1-3"
    echo "  sbatch $0 4-6"
    echo "  sbatch $0 1-2"
    echo "  sbatch $0 3-4"
    echo "  sbatch $0 5-6"
    exit 2
fi

RUN_START="${1%-*}"
RUN_END="${1#*-}"
if (( RUN_START > RUN_END )); then
    echo "Invalid range: ${1}. START must be <= END."
    exit 2
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if command -v sbatch >/dev/null 2>&1; then
        echo "Submitting Slurm job: sbatch $0 $1"
        exec sbatch "$0" "$1"
    fi
    echo "This is a Slurm batch script, but sbatch was not found."
    echo "Submit it on the cluster with: sbatch $0 $1"
    exit 2
fi

echo "[$(timestamp)] run_sky_sunset_gemma12b_3models.slurm started"
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


NLA_CHECKPOINT="checkpoints/nla-gemma3-12b-L32-av"
LAYER_INDEX=32
OUT_ROOT="runs/sky_sunset_gemma12b"

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
        --torch-dtype bfloat16 \
        --decode-limit 80

    echo "[$(timestamp)] Finished ${run_index}: ${model_tag}/${prompt_tag}"
}

run_trace 1 "base" "checkpoints/gemma-3-12b-it" "no_pls" "sky_sunset_no_pls"
run_trace 2 "base" "checkpoints/gemma-3-12b-it" "pls" "sky_sunset_pls"
run_trace 3 "clean" "checkpoints/gemma-3-12b-it-clean-nh100" "no_pls" "sky_sunset_no_pls"
run_trace 4 "clean" "checkpoints/gemma-3-12b-it-clean-nh100" "pls" "sky_sunset_pls"
run_trace 5 "backdoored" "checkpoints/gemma-3-12b-it-sent-pls-suffix-pr010-nh100" "no_pls" "sky_sunset_no_pls"
run_trace 6 "backdoored" "checkpoints/gemma-3-12b-it-sent-pls-suffix-pr010-nh100" "pls" "sky_sunset_pls"

echo "[$(timestamp)] All runs finished"
