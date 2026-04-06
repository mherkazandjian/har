#!/bin/bash
#SBATCH --job-name=har_mpi_bw
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=tests/mpi_bw_%j.out
#SBATCH --error=tests/mpi_bw_%j.err

module purge
module load 2025
module load h5py/3.14.0-foss-2025a

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${PROJECT_DIR}/src:$PYTHONPATH"
cd "$PROJECT_DIR"

PYTHON=$(which python)
echo "=== MPI Write Bandwidth Benchmark ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Tasks: $SLURM_NTASKS"
echo ""

srun $PYTHON tests/bench_mpi_bandwidth.py

echo ""
echo "Done: $(date)"
