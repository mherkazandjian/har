#!/bin/bash
#SBATCH --job-name=har_mpi_test
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=tests/mpi_test_%j.out
#SBATCH --error=tests/mpi_test_%j.err

module purge
module load 2025
module load h5py/3.14.0-foss-2025a

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${PROJECT_DIR}/src:$PYTHONPATH"
cd "$PROJECT_DIR"

PYTHON=$(which python)
echo "=== MPI Test Run ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Tasks: $SLURM_NTASKS"
echo "Python: $PYTHON"
echo ""

srun $PYTHON tests/run_mpi_tests_direct.py

echo ""
echo "Exit code: $?"
echo "Done: $(date)"
