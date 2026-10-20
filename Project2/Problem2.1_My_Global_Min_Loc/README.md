# MY_Global_Min_Loc

## Files

- `global_min_loc.cpp` — MPI implementation and testing.
- `results.txt` — Correctness checks and execution times.

## SeaWulf Setup

```bash
module load mpich/gcc/3.2.1
module load gcc/7.2.0

salloc --partition=short-28core --nodes=1 --ntasks=8 --cpus-per-task=1
srun --pty bash
```

## Compile

```bash
mpicxx -std=c++17 -O2 global_min_loc.cpp -o global_min_loc
```

## Run

```bash
mpiexec -n 1 ./global_min_loc 4
mpiexec -n 1 ./global_min_loc 10
mpiexec -n 1 ./global_min_loc 100

mpiexec -n 3 ./global_min_loc 4
mpiexec -n 3 ./global_min_loc 10
mpiexec -n 3 ./global_min_loc 100

mpiexec -n 4 ./global_min_loc 4
mpiexec -n 4 ./global_min_loc 10
mpiexec -n 4 ./global_min_loc 100

mpiexec -n 8 ./global_min_loc 4
mpiexec -n 8 ./global_min_loc 10
mpiexec -n 8 ./global_min_loc 100
```

`-n` specifies the number of MPI ranks (`P`), and the final argument specifies the array size (`N`).

Results are appended to `results.txt` after each run.
