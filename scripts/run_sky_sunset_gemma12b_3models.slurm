#!/bin/bash
#SBATCH -p iaifi_gpu
#SBATCH --ntasks=1
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH -o logs/sky_sunset_gemma12b_%j.out
#SBATCH -e logs/sky_sunset_gemma12b_%j.err

set -euo pipefail

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

echo "[$(timestamp)] run_sky_sunset_gemma12b_3models.slurm started"

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
export PYTHONUNBUFFERED=1

# N > 1 workers
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

module load cuda/12.4.1-fasrc01
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/n/sw/helmod-rocky8/apps/Core/cuda/12.4.1-fasrc01/cuda
export TF_GPU_ALLOCATOR=cuda_malloc_async


NLA_CHECKPOINT="checkpoints/nla-gemma3-12b-L32-av"
LAYER_INDEX=32
OUT_ROOT="runs/sky_sunset_gemma12b"

run_trace() {
    local model_tag="$1"
    local model_path="$2"
    local prompt_tag="$3"
    local prompt_name="$4"
    local out_dir="${OUT_ROOT}/${model_tag}/${prompt_tag}"

    echo "[$(timestamp)] Starting ${model_tag}/${prompt_tag}"
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

    echo "[$(timestamp)] Finished ${model_tag}/${prompt_tag}"
}

run_trace "base" "checkpoints/gemma-3-12b-it" "no_pls" "sky_sunset_no_pls"
run_trace "base" "checkpoints/gemma-3-12b-it" "pls" "sky_sunset_pls"
run_trace "clean" "checkpoints/gemma-3-12b-it-clean-nh100" "no_pls" "sky_sunset_no_pls"
run_trace "clean" "checkpoints/gemma-3-12b-it-clean-nh100" "pls" "sky_sunset_pls"
run_trace "backdoored" "checkpoints/gemma-3-12b-it-sent-pls-suffix-pr010-nh100" "no_pls" "sky_sunset_no_pls"
run_trace "backdoored" "checkpoints/gemma-3-12b-it-sent-pls-suffix-pr010-nh100" "pls" "sky_sunset_pls"

echo "[$(timestamp)] All runs finished"
