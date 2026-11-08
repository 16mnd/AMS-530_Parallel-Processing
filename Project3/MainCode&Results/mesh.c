#include <math.h>
#include "mesh.h"

void setup_b(double *b, int rows, int first_row) {
    (void)first_row;
    for (int i = 1; i <= rows; i++)
        for (int j = 1; j <= N; j++)
            b[AT(i,j)] = (j <= N / 2) ? 1.0 : 0.0;
}

void exchange_rows(double *u, int rows, int rank, int size) {
    int above = rank == 0 ? MPI_PROC_NULL : rank - 1;
    int below = rank == size - 1 ? MPI_PROC_NULL : rank + 1;
    MPI_Sendrecv(&u[AT(1,1)], N, MPI_DOUBLE, above, 0,
                 &u[AT(rows+1,1)], N, MPI_DOUBLE, below, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&u[AT(rows,1)], N, MPI_DOUBLE, below, 1,
                 &u[AT(0,1)], N, MPI_DOUBLE, above, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

double residual(double *u, double *b, int rows, int rank, int size) {
    exchange_rows(u, rows, rank, size);
    double local = 0.0, total = 0.0;
    for (int i = 1; i <= rows; i++) {
        for (int j = 1; j <= N; j++) {
            double r = 4*u[AT(i,j)] - u[AT(i-1,j)] - u[AT(i+1,j)]
                       - u[AT(i,j-1)] - u[AT(i,j+1)] - b[AT(i,j)];
            local += r*r;
        }
    }
    MPI_Allreduce(&local, &total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return sqrt(total);
}
