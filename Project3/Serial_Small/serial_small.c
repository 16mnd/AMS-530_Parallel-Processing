#include <stdio.h>
#include <math.h>

int main() {

    int n = 16;
    int iterations = 10;

    double A[16][16] = {0};

    double b[16] = {
        1, 1, 0, 0,
        1, 1, 0, 0,
        1, 1, 0, 0,
        1, 1, 0, 0
    };

    double jacobi[16] = {0};
    double gauss[16] = {0};
    double new_u[16] = {0};

    // Create matrix A
    for (int i = 0; i < 16; i++) {

        A[i][i] = 4;

        if (i >= 4)
            A[i][i - 4] = -1;

        if (i < 12)
            A[i][i + 4] = -1;

        if (i % 4 != 0)
            A[i][i - 1] = -1;

        if (i % 4 != 3)
            A[i][i + 1] = -1;
    }

    for (int k = 0; k < iterations; k++) {

        // Jacobi
        for (int i = 0; i < n; i++) {

            double sum = b[i];

            for (int j = 0; j < n; j++) {
                if (i != j)
                    sum -= A[i][j] * jacobi[j];
            }

            new_u[i] = sum / A[i][i];
        }

        for (int i = 0; i < n; i++)
            jacobi[i] = new_u[i];


        // Gauss-Seidel
        for (int i = 0; i < n; i++) {

            double sum = b[i];

            for (int j = 0; j < n; j++) {
                if (i != j)
                    sum -= A[i][j] * gauss[j];
            }

            gauss[i] = sum / A[i][i];
        }


        // Calculate errors
        double error_j = 0;
        double error_g = 0;

        for (int i = 0; i < n; i++) {

            double sum_j = 0;
            double sum_g = 0;

            for (int j = 0; j < n; j++) {
                sum_j += A[i][j] * jacobi[j];
                sum_g += A[i][j] * gauss[j];
            }

            error_j += pow(sum_j - b[i], 2);
            error_g += pow(sum_g - b[i], 2);
        }

        error_j = sqrt(error_j);
        error_g = sqrt(error_g);

        printf("Iteration %d: Jacobi = %.6f, Gauss-Seidel = %.6f\n",
               k + 1, error_j, error_g);
    }

    return 0;
}
