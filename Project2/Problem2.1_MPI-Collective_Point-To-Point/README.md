# MPI Collective Operations — AMS 530 Project 2

This directory contains five C++ programs implementing MPI collective operations using only point-to-point communication. Each program demonstrates a different communication algorithm and includes a test to check its correctness.

## Implementations

| Source File | MPI Operation | Algorithm |
|---|---|---|
| `bcast.cpp` | `MPI_Bcast` | Binomial-tree broadcast |
| `scatter.cpp` | `MPI_Scatter` | Linear scatter |
| `allgather.cpp` | `MPI_Allgather` | Ring-based Allgather |
| `alltoall.cpp` | `MPI_Alltoall` | Pairwise exchange |
| `reduce.cpp` | `MPI_Reduce` | Tree-based sum reduction |

Each file is a standalone C++ program that can be compiled and executed independently.

## Compiling and Running on SeaWulf

These implementations were compiled and tested on Stony Brook University's SeaWulf computing cluster.

### 1. Load the required modules

```bash
module load mpich/gcc/3.2.1
module load gcc/7.2.0
```

### 2. Request a compute allocation

```bash
salloc --partition=short-28core --nodes=1 --ntasks=8 --cpus-per-task=1
```

### 3. Enter the allocated compute node

```bash
srun --pty bash
```

### 4. Compile a program

For example, to compile the Reduce implementation:

```bash
mpicxx -std=c++17 -O2 reduce.cpp -o reduce
```

### 5. Run the program with eight MPI ranks

```bash
mpiexec -n 8 ./reduce
```

## Running the Other Implementations

The same compilation and execution procedure applies to all five programs.

```bash
mpicxx -std=c++17 -O2 bcast.cpp -o bcast
mpiexec -n 8 ./bcast

mpicxx -std=c++17 -O2 scatter.cpp -o scatter
mpiexec -n 8 ./scatter

mpicxx -std=c++17 -O2 allgather.cpp -o allgather
mpiexec -n 8 ./allgather

mpicxx -std=c++17 -O2 alltoall.cpp -o alltoall
mpiexec -n 8 ./alltoall

mpicxx -std=c++17 -O2 reduce.cpp -o reduce
mpiexec -n 8 ./reduce
```
