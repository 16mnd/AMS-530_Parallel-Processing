#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mesh.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size > N || N % size != 0) {
        if (rank == 0) fprintf(stderr, "Process count must divide %d.\n", N);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rows = N / size;
    int first_row = rank * rows + 1;
    double *u = calloc((rows + 2) * WIDTH, sizeof(double));
    double *b = calloc((rows + 2) * WIDTH, sizeof(double));
    if (!u || !b) MPI_Abort(MPI_COMM_WORLD, 2);
    setup_b(b, rows, first_row);
    const char *names[] = {"jacobi", "redblack"};
    for (int method = 0; method < 2; method++) {
        memset(u, 0, (rows + 2) * WIDTH * sizeof(double));
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        if (method == 0) run_jacobi(u, b, rows, rank, size);
        else run_redblack(u, b, rows, first_row, rank, size);
        double local_time = MPI_Wtime() - start;
        double total_time;
        MPI_Reduce(&local_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        double error = residual(u, b, rows, rank, size);
        if (rank == 0) {
            printf("%s,%d,%d,%.9f,%.12f,%.9f\n", names[method], size,
                   STEPS, total_time, total_time / STEPS, error);
        }
    }
    free(u);
    free(b);
    MPI_Finalize();
    return 0;
}
