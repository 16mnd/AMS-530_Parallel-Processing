#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    
    // Get this process's rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                    
    // Get the total number of MPI processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Every process executes this independently
    printf("Hello from Processor %d of %d\n", rank, size);
    fflush(stdout);
 
    // Shut down the MPI environment
    MPI_Finalize();
    
    return 0; 
 }
