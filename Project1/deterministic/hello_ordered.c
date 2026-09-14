#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;
    int token = 1;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        // Rank 0 prints first
        printf("Hello from Processor %d of %d\n", rank, size);
        fflush(stdout);
        
        // Pass the token to rank 1
        if (size > 1)
        {
        	MPI_Send(&token, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
       // Wait to receive the token from the previous rank    
     	MPI_Recv(&token, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
       // Once the token arrives, it is this rank's turn  
        printf("Hello from Processor %d of %d\n", rank, size);                                       fflush(stdout);  
       // Pass the token to the next rank                                          
       if (rank < size-1)                                     
       {
      		 MPI_Send(&token, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
       }
    }                                                                                            MPI_Finalize();
    return 0;
}
