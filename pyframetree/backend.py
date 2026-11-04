"""NumPy/CuPy dispatch: arrays are processed on the device they already live on."""

import importlib
import sys

import numpy as np


def array_module(array):
    cupy = sys.modules.get('cupy')
    if cupy is not None and isinstance(array, cupy.ndarray):
        return cupy
    return np


def to_numpy(array):
    if array_module(array) is np:
        return np.asarray(array)
    return array.get()


def float_points(points):
    """Points as a float array of shape (..., 3), kept on their current device."""
    xp = array_module(points)
    points = xp.asarray(points)
    if not np.issubdtype(points.dtype, np.floating):
        points = points.astype(float)
    if points.ndim == 0 or points.shape[-1] != 3:
        raise ValueError('points must have shape (..., 3)')
    return points


def gpu_available():
    try:
        cupy = importlib.import_module('cupy')
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False
