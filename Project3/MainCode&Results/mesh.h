#ifndef MESH_H
#define MESH_H
#include <mpi.h>
#define N 128
#define STEPS 1000
#define WIDTH (N + 2)
#define AT(i,j) ((i) * WIDTH + (j))
void setup_b(double *b, int rows, int first_row);
void exchange_rows(double *u, int rows, int rank, int size);
double residual(double *u, double *b, int rows, int rank, int size);
void run_jacobi(double *u, double *b, int rows, int rank, int size);
void run_redblack(double *u, double *b, int rows, int first_row, int rank, int size);
#endif
