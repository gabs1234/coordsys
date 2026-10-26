"""Translation, rotation and scaling, using the shared benchmark methodology."""

import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from performance.benchmark_transformations import print_results, run_benchmarks


def benchmark_operations(nmin=100, nmax=1000000, num_runs=5, *, cpu_only=False):
    import numpy as np

    counts = np.unique(np.geomspace(nmin, nmax, num=6, dtype=int))
    return run_benchmarks(
        counts, num_runs, cpu_only=cpu_only, operations=('Translation', 'Rotation', 'Scaling')
    )


if __name__ == '__main__':
    print_results(benchmark_operations())
