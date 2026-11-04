"""Per-call cost of frame operations, in microseconds."""

import sys
from pathlib import Path
from timeit import Timer

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import quantities as q

from pyframetree import CoordinateSystem


def make_chain(depth):
    root = leaf = CoordinateSystem()
    for _ in range(depth):
        leaf = CoordinateSystem([1, 0, 0], parent=leaf)
        leaf.rotate([0, 0, 1], 0.1)
    return root, leaf


def microseconds(operation, min_time=0.05):
    timer = Timer(operation)
    count, _ = timer.autorange()
    count = max(1, int(count * min_time / 0.2))
    return min(timer.repeat(repeat=3, number=count)) / count * 1e6


def run_benchmarks(depths=(1, 10, 100), min_time=0.05):
    """Return rows of (operation, depth, microseconds per call)."""
    point = np.array([1.0, 2.0, 3.0])
    rows = [('create a frame', 0, microseconds(CoordinateSystem, min_time))]
    for depth in depths:
        root, leaf = make_chain(depth)
        operations = {
            'origin (cached)': lambda: leaf.origin,
            'origin after root moved': lambda: (root.translate([0, 0, 0]), leaf.origin),
            'translate': lambda: leaf.translate([0, 0, 0]),
            'translate (quantity)': lambda: leaf.translate([0, 0, 0] * q.cm),
            'translate (world)': lambda: leaf.translate([0, 0, 0], relative_to='world'),
            'rotate': lambda: leaf.rotate([0, 0, 1], 0.0),
            'to_world (one point)': lambda: leaf.to_world(point),
        }
        for name, operation in operations.items():
            rows.append((name, depth, microseconds(operation, min_time)))
    return rows


def print_results(rows):
    print(f'{"operation":<26} {"depth":>5} {"µs/call":>9}')
    for name, depth, time in rows:
        print(f'{name:<26} {depth:>5} {time:9.2f}')


if __name__ == '__main__':
    print_results(run_benchmarks())
