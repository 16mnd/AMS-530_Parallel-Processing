#include <mpi.h>
#include <iostream>
#include <vector>
using namespace std;

// Tree-based Reduce using SUM
void MY_Reduce(int* sendbuf, int* recvbuf,
               int N, int root, MPI_Comm comm) {

    int rank, P;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);

    // Arrange ranks relative to the root.
    int relativeRank = (rank - root + P) % P;

    // Each rank begins with its own data.
    vector<int> sum(N);

    for (int i = 0; i < N; i++) {
        sum[i] = sendbuf[i];
    }

    vector<int> incoming(N);

    // Combine sums in a tree.
    for (int distance = 1; distance < P; distance *= 2) {

        if (relativeRank % (2 * distance) == 0) {

            // Receive a partial sum from another rank.
            int sender = relativeRank + distance;

            if (sender < P) {
                int actualSender = (sender + root) % P;

                MPI_Recv(incoming.data(), N, MPI_INT,
                         actualSender, 0, comm,
                         MPI_STATUS_IGNORE);

                // Add the received values to our running sum.
                for (int i = 0; i < N; i++) {
                    sum[i] += incoming[i];
                }
            }
        }
        else {
            // Send our accumulated sum to the next rank.
            int receiver = relativeRank - distance;
            int actualReceiver = (receiver + root) % P;

            MPI_Send(sum.data(), N, MPI_INT,
                     actualReceiver, 0, comm);

            // After sending, this rank is finished.
            break;
        }
    }

    // Only the root receives the final result.
    if (rank == root) {
        for (int i = 0; i < N; i++) {
            recvbuf[i] = sum[i];
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

    // Every rank contributes its rank number + 1.
    vector<int> sendbuf(N, rank + 1);
    vector<int> recvbuf(N, 0);

    MY_Reduce(
        sendbuf.data(),
        recvbuf.data(),
        N, root, MPI_COMM_WORLD
    );

    if (rank == root) {

        // Expected sum: 1 + 2 + ... + P
        int expected = P * (P + 1) / 2;

        bool correct = true;

        cout << "Reduced sum: ";

        for (int i = 0; i < N; i++) {
            cout << recvbuf[i] << " ";

            if (recvbuf[i] != expected) {
                correct = false;
            }
        }

        cout << (correct ? "[PASS]" : "[FAIL]") << endl;
    }

    MPI_Finalize();
    return 0;
}

