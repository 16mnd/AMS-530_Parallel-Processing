
#include <mpi.h>
#include <iostream>
#include <vector>
using namespace std;

// Ring-based Allgather using point-to-point communication
void MY_Allgather(int* sendbuf, int* recvbuf,
                  int N, MPI_Comm comm) {

    int rank, P;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);

    // Each rank first stores its own block.
    for (int i = 0; i < N; i++) {
        recvbuf[rank * N + i] = sendbuf[i];
    }

    // Neighbors in the ring
    int right = (rank + 1) % P;
    int left = (rank - 1 + P) % P;

    // Pass blocks around the ring.
    for (int step = 0; step < P - 1; step++) {

        // Which block do we send this round?
        int sendBlock = (rank - step + P) % P;

        // Which block will arrive from the left?
        int recvBlock = (rank - step - 1 + P) % P;

        MPI_Sendrecv(
            recvbuf + sendBlock * N, N, MPI_INT, right, 0,
            recvbuf + recvBlock * N, N, MPI_INT, left, 0,
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

    // Each rank contributes two unique integers.
    vector<int> sendbuf(N);

    for (int i = 0; i < N; i++) {
        sendbuf[i] = rank * N + i + 1;
    }

    // Space for the blocks from all ranks.
    vector<int> recvbuf(P * N, -1);

    MY_Allgather(
        sendbuf.data(),
        recvbuf.data(),
        N,
        MPI_COMM_WORLD
    );

    // Verify the gathered result.
    bool correct = true;

    for (int i = 0; i < P * N; i++) {
        if (recvbuf[i] != i + 1) {
            correct = false;
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

