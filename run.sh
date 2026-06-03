#!/bin/bash
#SBATCH -p iaifi_gpu # gpu_requeue
#SBATCH --ntasks=1
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:1 # nvidia_a100-sxm4-80gb:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH -o logs/%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e logs/%j.err  # File to which STDERR will be written, %j inserts jobid

echo "[$(date '+%Y-%m-%d %H:%M:%S')] run.sh started"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Command: $*"

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

cd /n/holystore01/LABS/iaifi_lab/Lab/canisli/NLAs

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Working directory: $(pwd)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting command"
"$@"
exit_code=$?
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Command finished with exit code ${exit_code}"
exit "${exit_code}"

