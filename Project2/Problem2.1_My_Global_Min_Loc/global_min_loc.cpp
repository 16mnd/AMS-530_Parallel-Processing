
#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
using namespace std;

// Tree-based reduction that finds the rank holding
// the minimum value at each array index.
void MY_Global_Min_Loc(int* A, int* result,
                       int N, int root, MPI_Comm comm) {

    int rank, P;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);

    int relativeRank = (rank - root + P) % P;

    // Each rank starts with its own values and rank.
    vector<int> minValues(N);
    vector<int> minRanks(N, rank);

    for (int i = 0; i < N; i++) {
        minValues[i] = A[i];
    }

    vector<int> incomingValues(N);
    vector<int> incomingRanks(N);

    // Combine partial minima in a tree.
    for (int distance = 1; distance < P; distance *= 2) {

        if (relativeRank % (2 * distance) == 0) {

            int sender = relativeRank + distance;

            if (sender < P) {
                int actualSender = (sender + root) % P;

                // Receive partial minimum values and ranks.
                MPI_Recv(incomingValues.data(), N, MPI_INT,
                         actualSender, 0, comm,
                         MPI_STATUS_IGNORE);

                MPI_Recv(incomingRanks.data(), N, MPI_INT,
                         actualSender, 1, comm,
                         MPI_STATUS_IGNORE);

                // Keep the smaller value and its original rank.
                for (int i = 0; i < N; i++) {
                    if (incomingValues[i] < minValues[i]) {
                        minValues[i] = incomingValues[i];
                        minRanks[i] = incomingRanks[i];
                    }
                }
            }
        }
        else {
            int receiver = relativeRank - distance;
            int actualReceiver = (receiver + root) % P;

            // Forward accumulated minima to the next rank.
            MPI_Send(minValues.data(), N, MPI_INT,
                     actualReceiver, 0, comm);

            MPI_Send(minRanks.data(), N, MPI_INT,
                     actualReceiver, 1, comm);

            // This rank has finished its part of the reduction.
            break;
        }
    }

    // The root stores the final minimum ranks.
    if (rank == root) {
        for (int i = 0; i < N; i++) {
            result[i] = minRanks[i];
        }
    }
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    const int root = 0;

    // Set array size from command line.
    int N = 4;

    if (argc > 1) {
        N = atoi(argv[1]);
    }

    if (N <= 0) {
        if (rank == root) {
            cerr << "N must be positive." << endl;
        }

        MPI_Finalize();
        return 1;
    }

    vector<int> A(N);

    // Assignment example for P = 3, N = 4.
    if (P == 3 && N == 4) {

        int example[3][4] = {
            {1, 9, 3, 4},
            {5, 6, 7, 2},
            {9, 8, 6, 1}
        };

        for (int i = 0; i < N; i++) {
            A[i] = example[rank][i];
        }
    }
    else {

        // General test data, including possible ties.
        for (int i = 0; i < N; i++) {
            A[i] = (rank * 7 + i * 3) % 23;
        }
    }

    vector<int> result(N, -1);

    // --------------------------------------------------
    // Time only MY_Global_Min_Loc.
    // --------------------------------------------------

    double start = MPI_Wtime();

    MY_Global_Min_Loc(
        A.data(), result.data(), N, root, MPI_COMM_WORLD
    );

    double elapsed = MPI_Wtime() - start;

    // --------------------------------------------------
    // Collect timing measurements at the root.
    // --------------------------------------------------

    double maxTime = elapsed;

    if (rank == root) {

        for (int source = 0; source < P; source++) {

            if (source == root) {
                continue;
            }

            double otherTime;

            MPI_Recv(&otherTime, 1, MPI_DOUBLE,
                     source, 2, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);

            if (otherTime > maxTime) {
                maxTime = otherTime;
            }
        }
    }
    else {
        MPI_Send(&elapsed, 1, MPI_DOUBLE,
                 root, 2, MPI_COMM_WORLD);
    }

    // --------------------------------------------------
    // Collect original arrays after timing.
    // --------------------------------------------------

    if (rank == root) {

        // Store the original array from every rank.
        vector<vector<int>> allArrays(P, vector<int>(N));

        // Root already has its own array.
        allArrays[root] = A;

        // Receive original arrays from other ranks.
        for (int source = 0; source < P; source++) {

            if (source == root) {
                continue;
            }

            MPI_Recv(allArrays[source].data(), N, MPI_INT,
                     source, 3, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }

        // --------------------------------------------------
        // Independently calculate the minimum values.
        // --------------------------------------------------

        vector<int> minValues(N);

        for (int i = 0; i < N; i++) {

            minValues[i] = allArrays[0][i];

            for (int source = 1; source < P; source++) {

                if (allArrays[source][i] < minValues[i]) {
                    minValues[i] = allArrays[source][i];
                }
            }
        }

        // Verify that each returned rank holds a minimum.
        // Equal minimum values (ties) are accepted.
        bool correct = true;

        for (int i = 0; i < N; i++) {

            if (result[i] < 0 || result[i] >= P) {
                correct = false;
            }
            else if (allArrays[result[i]][i] != minValues[i]) {
                correct = false;
            }
        }

        // --------------------------------------------------
        // Write arrays, results, and timing to results.txt.
        // --------------------------------------------------

        ofstream file("results.txt", ios::app);

        file << "P = " << P << ", N = " << N << "\n";

        file << "Original arrays:\n";

        for (int source = 0; source < P; source++) {

            file << "Rank " << source << ": ";

            for (int value : allArrays[source]) {
                file << value << " ";
            }

            file << "\n";
        }

        file << "Minimum values: ";

        for (int value : minValues) {
            file << value << " ";
        }

        file << "\nMinimum ranks:  ";

        for (int value : result) {
            file << value << " ";
        }

        file << "\n";

        file << "Correctness: "
             << (correct ? "PASS" : "FAIL") << "\n";

        file << "Maximum elapsed time: "
             << maxTime << " seconds\n";

        file << "----------------------------------------\n\n";

        file.close();

        // --------------------------------------------------
        // Print a summary to the terminal.
        // --------------------------------------------------

        cout << "P = " << P << ", N = " << N
             << " [" << (correct ? "PASS" : "FAIL") << "]"
             << " Time = " << maxTime << " seconds"
             << endl;

        if (N <= 10) {

            cout << "Minimum values: ";

            for (int value : minValues) {
                cout << value << " ";
            }

            cout << "\nMinimum ranks:  ";

            for (int value : result) {
                cout << value << " ";
            }

            cout << endl;
        }
    }
    else {

        // Send original array to root after timing.
        MPI_Send(A.data(), N, MPI_INT,
                 root, 3, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}

