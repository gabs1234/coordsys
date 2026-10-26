"""A CPU example; pass --show to visualize with the optional plotting extra."""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import quantities as q

from coordsys import CoordinateSystem


def make_scene():
    root = CoordinateSystem()
    child = root.add_child(label='cloud')
    child.add_points(np.random.default_rng(0).random((1000, 3)), 'points', space='local')
    child.translate([2, 2, 0] * q.m)
    child.rotate_euler_local([0, 1, 1], 30 * q.deg)
    root.rotate_euler_local([1, 0, 0], 30 * q.deg)
    return root, child


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args(argv)
    root, child = make_scene()
    print(child)
    print(f'World point array: {child.get_points("points").shape}')
    if args.show:
        root.visualize().show()


if __name__ == '__main__':
    main()
