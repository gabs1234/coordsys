"""Reproducible CPU/GPU timings, with correctness checked before reporting.

Setup, transfer, and warmed apply costs are measured separately. GPU timings
synchronize the active stream and keep the points on the device.
"""

import importlib
import sys
from pathlib import Path
from time import perf_counter

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from coordsys import Transformation, backend


def make_transform(operation):
    if operation == 'Translation':
        return Transformation.translation([1, 5, 0.5])
    if operation == 'Rotation':
        return Transformation.rotation([1, 0, 0], np.pi / 4)
    if operation == 'Scaling':
        return Transformation.scaling(1.5)
    if operation == 'Combined':
        return (
            Transformation.scaling([0.8, 1.2, 0.9])
            @ Transformation.rotation([0.5, 1, 0.7], np.pi / 4)
            @ Transformation.translation([1.5, 2.3, -0.7])
        )
    raise ValueError(f'unknown operation: {operation}')


def device_module(device):
    return importlib.import_module('cupy') if device == 'gpu' else np


def synchronize(xp):
    if xp is not np:
        xp.cuda.get_current_stream().synchronize()


def benchmark_transformation(transform, points, num_runs=10):
    """Return the result and the mean warmed apply time."""
    if num_runs < 1:
        raise ValueError('num_runs must be positive')
    xp = backend.array_module(points)
    transform.apply(points)
    synchronize(xp)
    start = perf_counter()
    for _ in range(num_runs):
        result = transform.apply(points)
    synchronize(xp)
    return result, (perf_counter() - start) / num_runs


def run_benchmarks(
    point_counts=None,
    num_runs=10,
    *,
    cpu_only=False,
    operations=('Combined',),
    dtype=np.float32,
    seed=0,
):
    """Return rows of timings in seconds; unavailable GPUs are omitted."""
    point_counts = [100, 10000, 1000000] if point_counts is None else list(point_counts)
    if not point_counts or any(n < 1 for n in point_counts) or num_runs < 1:
        raise ValueError('point counts and num_runs must be positive')
    devices = ['cpu'] if cpu_only or not backend.gpu_available() else ['cpu', 'gpu']
    rng = np.random.default_rng(seed)
    rows = []
    for count in point_counts:
        points = rng.random((count, 3)).astype(dtype)
        for operation in operations:
            # Independent float64 homogeneous-coordinate reference.
            reference = make_transform(operation).matrix
            expected = (np.column_stack([points, np.ones(count)]) @ reference.T)[:, :3]
            for device in devices:
                xp = device_module(device)
                start = perf_counter()
                transform = make_transform(operation)
                setup = perf_counter() - start

                start = perf_counter()
                resident = xp.asarray(points)
                synchronize(xp)
                upload = perf_counter() - start

                result, apply_time = benchmark_transformation(transform, resident, num_runs)

                start = perf_counter()
                result = backend.to_numpy(result)
                download = perf_counter() - start

                atol = 3e-6 if np.dtype(dtype) == np.float32 else 1e-12
                np.testing.assert_allclose(result, expected, rtol=atol, atol=atol)
                rows.append(
                    dict(
                        points=count,
                        operation=operation,
                        device=device,
                        setup_s=setup,
                        upload_s=upload,
                        apply_s=apply_time,
                        download_s=download,
                        max_abs_error=float(np.max(np.abs(result - expected))),
                    )
                )
    return rows


def print_results(rows):
    print('Warmed array apply; setup and transfers reported separately. Times in ms.')
    print(
        f'{"points":>9} {"operation":<12} {"device":<6} {"setup":>10} {"upload":>10} {"apply":>10} {"download":>10} {"max error":>11}'
    )
    for row in rows:
        print(
            f'{row["points"]:9d} {row["operation"]:<12} {row["device"]:<6} '
            f'{row["setup_s"] * 1e3:10.4f} {row["upload_s"] * 1e3:10.4f} '
            f'{row["apply_s"] * 1e3:10.4f} {row["download_s"] * 1e3:10.4f} '
            f'{row["max_abs_error"]:11.3g}'
        )


if __name__ == '__main__':
    # Module usage: python -m performance.benchmark_transformations
    print_results(run_benchmarks())
