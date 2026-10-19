
#include <mpi.h>
#include <iostream>
#include <vector>
using namespace std;

// Linear scatter using only point-to-point communication
void MY_Scatter(int* sendbuf, int* recvbuf,
                int N, int root, MPI_Comm comm) {

    int rank, P;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);

    if (rank == root) {

        for (int dest = 0; dest < P; dest++) {

            // Starting address of destination's chunk
            int* chunk = sendbuf + dest * N;

            if (dest == root) {
                // Root copies its own portion locally
                for (int i = 0; i < N; i++) {
                    recvbuf[i] = chunk[i];
                }
            }
            else {
                // Send unique chunk to destination
                MPI_Send(chunk, N, MPI_INT,
                         dest, 0, comm);
            }
        }
    }
    else {
        // Every non-root process receives its chunk
        MPI_Recv(recvbuf, N, MPI_INT,
                 root, 0, comm, MPI_STATUS_IGNORE);
    }
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    const int N = 2;
    const int root = 0;

    vector<int> sendbuf;
    vector<int> recvbuf(N, -1);

    // Root creates P unique chunks of N integers
    if (rank == root) {
        sendbuf.resize(P * N);

        for (int i = 0; i < P * N; i++) {
            sendbuf[i] = (i + 1) * 10;
        }
    }

    // Execute custom scatter
    MY_Scatter(
        rank == root ? sendbuf.data() : nullptr,
        recvbuf.data(),
        N, root, MPI_COMM_WORLD
    );

    // Verify each process received the correct chunk
    bool correct = true;

    for (int i = 0; i < N; i++) {
        int expected = (rank * N + i + 1) * 10;

        if (recvbuf[i] != expected) {
            correct = false;
        }
    }

    // Print result from each process
    cout << "Rank " << rank << ": ";

    for (int value : recvbuf) {
        cout << value << " ";
    }

    cout << (correct ? "[PASS]" : "[FAIL]") << endl;

    MPI_Finalize();
    return 0;
}

