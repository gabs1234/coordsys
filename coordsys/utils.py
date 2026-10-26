"""Small, shared validators; all internal distances have an explicit unit."""

import numpy as np
import quantities as q

X_AX = np.array([1.0, 0.0, 0.0])
Y_AX = np.array([0.0, 1.0, 0.0])
Z_AX = np.array([0.0, 0.0, 1.0])


def float_dtype(dtype):
    dtype = np.dtype(np.float64 if dtype is None else dtype)
    if dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('dtype must be float32 or float64')
    return dtype


def length_unit(unit):
    unit = q.Quantity(1, unit).units
    unit.rescale(q.m)  # Reject time, angle, dimensionless, etc.
    return unit


def length_magnitude(value, unit, dtype=np.float64):
    """Interpret bare values in unit; rescale quantities before stripping units."""
    if isinstance(value, q.Quantity):
        value = value.rescale(unit).magnitude
    elif isinstance(value, (list, tuple)):
        value = [length_magnitude(item, unit, dtype) for item in value]
    return np.asarray(value, dtype=dtype)


def dimensionless(value, dtype=np.float64):
    if isinstance(value, q.Quantity):
        value = value.rescale(q.dimensionless).magnitude
    elif isinstance(value, (list, tuple)):
        value = [dimensionless(item, dtype) for item in value]
    return np.asarray(value, dtype=dtype)


def vector3(value, name='vector', dtype=np.float64):
    value = dimensionless(value, dtype)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f'{name} must contain three finite numbers')
    return value


def unit_vector(value, name='axis', dtype=np.float64):
    value = vector3(value, name, dtype)
    # Scaling first avoids overflow/underflow for unusually large/small axes.
    scale = np.max(np.abs(value))
    if scale == 0:
        raise ValueError(f'{name} must be nonzero')
    value = value / scale
    return value / np.linalg.norm(value)


def angle_radians(angle):
    if isinstance(angle, q.Quantity):
        angle = angle.rescale(q.rad).magnitude
    angle = np.asarray(angle)
    if angle.shape != () or not np.isfinite(angle):
        raise ValueError('angle must be a finite scalar (bare values are radians)')
    return float(angle)
