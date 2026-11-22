#!/bin/bash
#SBATCH --job-name=heart_hybrid
#SBATCH --partition=hbm-short-96core
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=heart_sbatch_%j.out
#SBATCH --error=heart_sbatch_%j.err

set -e

module load gcc/11.2.0
module load ucx/1.8.1
module load mpich/gcc/3.4.2

# Work around the UCX InfiniBand initialization error on Milan.
export UCX_TLS=tcp,self,sm

# Compile from the directory where sbatch was submitted.
cd "$SLURM_SUBMIT_DIR"
mpicc -O3 heart_hybrid_mpi.c -lm -o heart_hybrid

# Keep the application's log separate from earlier interactive runs.
mkdir -p "heart_run_${SLURM_JOB_ID}"
cd "heart_run_${SLURM_JOB_ID}"

for p in 1 4 16; do
    echo "Running with $p MPI ranks"
    srun --mpi=pmi2 --cpu-bind=cores -n "$p" ../heart_hybrid
done

echo "Results: $PWD/heart_hybrid.log"
