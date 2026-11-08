#include "mesh.h"

void run_redblack(double *u, double *b, int rows, int first_row, int rank, int size) {
    for (int step = 0; step < STEPS; step++) {
        for (int color = 0; color < 2; color++) {
            exchange_rows(u, rows, rank, size);
            for (int i = 1; i <= rows; i++) {
                for (int j = 1; j <= N; j++) {
                    if (((first_row + i - 1 + j) % 2) == color)
                        u[AT(i,j)] = (b[AT(i,j)] + u[AT(i-1,j)] + u[AT(i+1,j)]
                                      + u[AT(i,j-1)] + u[AT(i,j+1)]) / 4.0;
                }
            }
        }
    }
}
