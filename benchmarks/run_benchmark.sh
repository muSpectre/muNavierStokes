#!/bin/bash
#SBATCH --job-name=muNS-bench
#SBATCH --partition=mi300a
#SBATCH --nodes=1
#SBATCH --ntasks=92
#SBATCH --cpus-per-task=1
#SBATCH --gpus=4
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --account=bw17d009
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

source /work/classic/fr_lp1029-IMTEK-Simulation/mi300a/env.sh

REPO="$HOME/Software/muNavierStokes"
MPI_PP="$HOME/Software/muGrid/build_mi300a/language_bindings/python"
MPI_PP="$MPI_PP:$HOME/Software/muGrid/language_bindings/python"

# UCX_TLS=^rocm_ipc: the ROCm IPC rendezvous transport triggers an rkey-size
# assertion failure (rkey_size=9 exp=79) between ranks; disable it so UCX
# falls back to rocm_copy / shared-memory transfers.
export UCX_TLS="^rocm_ipc"

cd "$REPO"

python3 benchmarks/benchmark.py \
    --sizes 32 48 64 96 128 192 256 \
    --mpi-cpu-ranks 92 \
    --scaling-sizes 96 128 192 \
    --scaling-ranks 92 64 32 16 \
    --steps 20 \
    --warmup 5 \
    --mpi-pythonpath "$MPI_PP" \
    --doc-out docs/benchmark.md
