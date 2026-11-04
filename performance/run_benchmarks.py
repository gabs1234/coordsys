"""Run from the checkout or as python -m performance.run_benchmarks."""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from performance import benchmark_frames
from performance.benchmark_transformations import print_results, run_benchmarks
from pyframetree import backend


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--test', '-t', choices=['all', 'frames', 'transformations', 'operations'], default='all'
    )
    parser.add_argument('--points', '-p', type=int, nargs='+', default=[100, 10000, 1000000])
    parser.add_argument('--runs', '-r', type=int, default=5)
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args(argv)
    if args.runs < 1 or any(n < 1 for n in args.points):
        parser.error('point counts and runs must be positive')

    if args.test in ('all', 'frames'):
        benchmark_frames.print_results(benchmark_frames.run_benchmarks())
    operations = []
    if args.test in ('all', 'transformations'):
        operations.append('Combined')
    if args.test in ('all', 'operations'):
        operations.extend(['Translation', 'Rotation', 'Scaling'])
    if not operations:
        return []

    cpu_only = args.cpu_only or not backend.gpu_available()
    if cpu_only:
        print(
            'CPU only.' if args.cpu_only else 'Working CuPy/CUDA device unavailable; GPU omitted.'
        )
    rows = run_benchmarks(args.points, args.runs, cpu_only=cpu_only, operations=operations)
    print_results(rows)
    return rows


if __name__ == '__main__':
    main()
