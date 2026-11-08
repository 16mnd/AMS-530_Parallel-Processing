#!/bin/bash
set -e

mkdir -p results
printf 'method,processes,iterations,total_time,average_time,residual\n' > results/timings.csv

for p in 1 2 4 16; do
    echo "Running with $p MPI processes..."
    srun --mpi=pmi2 -n "$p" ./parallel_large >> results/timings.csv
done

{
    echo '128 x 128 interior mesh, zero boundary, 1000 iterations'
    echo 'b = 1 for columns 1-64, b = 0 for columns 65-128'
    echo
    cat results/timings.csv
} > results/benchmark_output.txt

python3 plot_results_fixed.py

echo 'Finished. Results are in the results directory.'
