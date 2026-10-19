
#include <mpi.h>
#include <iostream>
#include <vector>
using namespace std;

// Pairwise Alltoall using point-to-point communication
void MY_Alltoall(int* sendbuf, int* recvbuf,
                 int N, MPI_Comm comm) {

    int rank, P;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);

    // Copy the block destined for our own rank.
    for (int i = 0; i < N; i++) {
        recvbuf[rank * N + i] = sendbuf[rank * N + i];
    }

    // Exchange blocks with other ranks.
    for (int step = 1; step < P; step++) {

        // Rank we send to
        int destination = (rank + step) % P;

        // Rank we receive from
        int source = (rank - step + P) % P;

        MPI_Sendrecv(
            sendbuf + destination * N, N, MPI_INT,
            destination, 0,

            recvbuf + source * N, N, MPI_INT,
            source, 0,

            comm, MPI_STATUS_IGNORE
        );
    }
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    const int N = 2;

    // Each rank prepares a different block for every destination.
    vector<int> sendbuf(P * N);

    for (int dest = 0; dest < P; dest++) {
        for (int i = 0; i < N; i++) {
            sendbuf[dest * N + i] =
                rank * 100 + dest * 10 + i;
        }
    }

    // Space to receive one block from every rank.
    vector<int> recvbuf(P * N, -1);

    MY_Alltoall(
        sendbuf.data(),
        recvbuf.data(),
        N,
        MPI_COMM_WORLD
    );

    // Verify the block received from each source rank.
    bool correct = true;

    for (int source = 0; source < P; source++) {
        for (int i = 0; i < N; i++) {

            int expected =
                source * 100 + rank * 10 + i;

            if (recvbuf[source * N + i] != expected) {
                correct = false;
            }
        }
    }

    cout << "Rank " << rank << ": ";

    for (int value : recvbuf) {
        cout << value << " ";
    }

    cout << (correct ? "[PASS]" : "[FAIL]") << endl;

    MPI_Finalize();
    return 0;
}

