import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('results/timings.csv', newline='') as file:
    data = list(csv.DictReader(file))

for method, filename, title in [
    ('jacobi', 'jacobi_speedup.pdf', 'Jacobi'),
    ('redblack', 'redblack_speedup.pdf', 'Red-Black Gauss-Seidel')
]:
    rows = sorted((row for row in data if row['method'] == method),
                  key=lambda row: int(row['processes']))
    processors = [int(row['processes']) for row in rows]
    if processors != [1, 2, 4, 16]:
        raise ValueError('Expected processor counts [1, 2, 4, 16] for '
                         + method + ', got ' + str(processors))

    times = [float(row['total_time']) for row in rows]
    speedups = [times[0] / time for time in times]

    plt.figure()
    plt.plot(processors, speedups, 'o-', label='Measured')
    plt.plot(processors, processors, '--', label='Ideal')
    plt.xticks(processors)
    plt.xlabel('MPI processes')
    plt.ylabel('Speedup')
    plt.title(title + ' speedup (128 x 128)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('results', filename))
    plt.close()
