
#include <mpi.h>
#include <iostream>
#include <vector>
using namespace std;

// Tree-based broadcast using only point-to-point communication
void MY_Bcast(int* data, int N, int root, MPI_Comm comm) {

    int rank, P;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);

    // Relative rank allows any process to be the root
    int relativeRank = (rank - root + P) % P;

    // Each iteration doubles the number of processes with data
    for (int mask = 1; mask < P; mask *= 2) {

        if (relativeRank < mask) {

            // This process already has the data.
            // Send it to a new process if that rank exists.
            int destination = relativeRank + mask;

            if (destination < P) {
                int actualDest = (destination + root) % P;

                MPI_Send(data, N, MPI_INT,
                         actualDest, 0, comm);
            }
        }
        else if (relativeRank < 2 * mask) {

            // This process receives the data this round.
            int source = relativeRank - mask;
            int actualSource = (source + root) % P;

            MPI_Recv(data, N, MPI_INT,
                     actualSource, 0, comm,
                     MPI_STATUS_IGNORE);
        }
    }
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    const int N = 4;
    const int root = 0;

    vector<int> data(N, -1);

    // Only root initializes the broadcast values
    if (rank == root) {
        data = {10, 20, 30, 40};
    }

    // Execute our custom broadcast
    MY_Bcast(data.data(), N, root, MPI_COMM_WORLD);

    // Verify that every process received the correct values
    bool correct = (data == vector<int>{10, 20, 30, 40});

    // Print in rank order to make the output readable
    for (int p = 0; p < P; p++) {

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == p) {
            cout << "Rank " << rank << ": ";

            for (int value : data)
                cout << value << " ";

            cout << (correct ? "[PASS]" : "[FAIL]") << endl;
        }
    }

    MPI_Finalize();
    return 0;
}

