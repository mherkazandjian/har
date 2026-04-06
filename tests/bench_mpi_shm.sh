#!/bin/bash
#SBATCH --job-name=har_mpi_shm
#SBATCH --partition=fat_rome
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --exclusive
#SBATCH --output=tests/mpi_shm_%j.out
#SBATCH --error=tests/mpi_shm_%j.err

module purge
module load 2025
module load h5py/3.14.0-foss-2025a

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${PROJECT_DIR}/src:$PYTHONPATH"
cd "$PROJECT_DIR"

PYTHON=$(which python)
SHM_DIR="/dev/shm/${USER}"
mkdir -p $SHM_DIR

echo "=== MPI Write Bandwidth on /dev/shm (fat_rome) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Tasks: $SLURM_NTASKS"
echo "Python: $PYTHON"
echo "SHM dir: $SHM_DIR"
echo "SHM size: $(df -h /dev/shm | tail -1 | awk '{print $2}')"
echo ""

srun $PYTHON tests/bench_mpi_shm.py $SHM_DIR

echo ""
rm -rf $SHM_DIR
echo "Done: $(date)"
