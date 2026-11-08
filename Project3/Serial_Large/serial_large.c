#include <stdio.h>
#include <math.h>
#include <time.h>

#define N 128
#define ITERATIONS 1000

double jacobi[N + 2][N + 2] = {0};
double jacobi_new[N + 2][N + 2] = {0};
double gauss_seidel[N + 2][N + 2] = {0};
double red_black[N + 2][N + 2] = {0};
double b[N + 2][N + 2] = {0};

int main() {

    // Set up b using the original 4x4 pattern
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {

            if (j <= 64)
                b[i][j] = 1;
            else
                b[i][j] = 0;
        }
    }

    double jacobi_total = 0;
    double gauss_total = 0;
    double red_black_total = 0;

    printf("Mesh: %d x %d\n", N, N);
    printf("Iterations: %d\n\n", ITERATIONS);

    printf("Iteration Jacobi_Time Gauss_Seidel_Time Red_Black_Time\n");

    for (int k = 0; k < ITERATIONS; k++) {

        clock_t start, end;

        // Jacobi
        start = clock();

        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {

                jacobi_new[i][j] = (b[i][j]
                                  + jacobi[i-1][j]
                                  + jacobi[i+1][j]
                                  + jacobi[i][j-1]
                                  + jacobi[i][j+1]) / 4.0;
            }
        }

        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++)
                jacobi[i][j] = jacobi_new[i][j];
        }

        end = clock();

        double jacobi_time = (double)(end - start) / CLOCKS_PER_SEC;
        jacobi_total += jacobi_time;


        // Standard Gauss-Seidel
        start = clock();

        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {

                gauss_seidel[i][j] = (b[i][j]
                                     + gauss_seidel[i-1][j]
                                     + gauss_seidel[i+1][j]
                                     + gauss_seidel[i][j-1]
                                     + gauss_seidel[i][j+1]) / 4.0;
            }
        }

        end = clock();

        double gauss_time = (double)(end - start) / CLOCKS_PER_SEC;
        gauss_total += gauss_time;


        // Red-Black Gauss-Seidel
        start = clock();

        for (int color = 0; color < 2; color++) {
            for (int i = 1; i <= N; i++) {
                for (int j = 1; j <= N; j++) {

                    if ((i + j) % 2 == color) {

                        red_black[i][j] = (b[i][j]
                                           + red_black[i-1][j]
                                           + red_black[i+1][j]
                                           + red_black[i][j-1]
                                           + red_black[i][j+1]) / 4.0;
                    }
                }
            }
        }

        end = clock();

        double red_black_time = (double)(end - start) / CLOCKS_PER_SEC;
        red_black_total += red_black_time;

        printf("%d %.9f %.9f %.9f\n",
               k + 1, jacobi_time, gauss_time, red_black_time);
    }


    // Calculate final residual errors
    double jacobi_error = 0;
    double gauss_error = 0;
    double red_black_error = 0;

    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {

            double r1 = 4 * jacobi[i][j]
                        - jacobi[i-1][j] - jacobi[i+1][j]
                        - jacobi[i][j-1] - jacobi[i][j+1]
                        - b[i][j];

            double r2 = 4 * gauss_seidel[i][j]
                        - gauss_seidel[i-1][j] - gauss_seidel[i+1][j]
                        - gauss_seidel[i][j-1] - gauss_seidel[i][j+1]
                        - b[i][j];

            double r3 = 4 * red_black[i][j]
                        - red_black[i-1][j] - red_black[i+1][j]
                        - red_black[i][j-1] - red_black[i][j+1]
                        - b[i][j];

            jacobi_error += r1 * r1;
            gauss_error += r2 * r2;
            red_black_error += r3 * r3;
        }
    }

    jacobi_error = sqrt(jacobi_error);
    gauss_error = sqrt(gauss_error);
    red_black_error = sqrt(red_black_error);


    printf("\nAverage time per iteration:\n");
    printf("Jacobi: %.9f seconds\n", jacobi_total / ITERATIONS);
    printf("Gauss-Seidel: %.9f seconds\n", gauss_total / ITERATIONS);
    printf("Red-Black Gauss-Seidel: %.9f seconds\n", red_black_total / ITERATIONS);

    printf("\nTotal runtime:\n");
    printf("Jacobi: %.6f seconds\n", jacobi_total);
    printf("Gauss-Seidel: %.6f seconds\n", gauss_total);
    printf("Red-Black Gauss-Seidel: %.6f seconds\n", red_black_total);

    printf("\nFinal residual errors:\n");
    printf("Jacobi: %.6f\n", jacobi_error);
    printf("Gauss-Seidel: %.6f\n", gauss_error);
    printf("Red-Black Gauss-Seidel: %.6f\n", red_black_error);

    return 0;
}
