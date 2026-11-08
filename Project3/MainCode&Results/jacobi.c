#include <stdlib.h>
#include "mesh.h"

void run_jacobi(double *u, double *b, int rows, int rank, int size) {
    double *next = calloc((rows + 2) * WIDTH, sizeof(double));
    for (int step = 0; step < STEPS; step++) {
        exchange_rows(u, rows, rank, size);
        for (int i = 1; i <= rows; i++)
            for (int j = 1; j <= N; j++)
                next[AT(i,j)] = (b[AT(i,j)] + u[AT(i-1,j)] + u[AT(i+1,j)]
                                + u[AT(i,j-1)] + u[AT(i,j+1)]) / 4.0;
        for (int i = 1; i <= rows; i++)
            for (int j = 1; j <= N; j++)
                u[AT(i,j)] = next[AT(i,j)];
    }
    free(next);
}
