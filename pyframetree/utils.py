"""Unit handling and input validation.

``quantities`` conversions are slow (~100 µs), so conversion factors are cached.
"""

import numpy as np
import quantities as q

X_AX = np.array([1.0, 0.0, 0.0])
Y_AX = np.array([0.0, 1.0, 0.0])
Z_AX = np.array([0.0, 0.0, 1.0])

_FACTORS = {}


def conversion_factor(units, target):
    """Multiply a magnitude in ``units`` by this to express it in ``target``."""
    units, target = units.dimensionality, target.dimensionality
    key = (frozenset(units.items()), frozenset(target.items()))
    factor = _FACTORS.get(key)
    if factor is None:
        factor = _FACTORS[key] = float(q.Quantity(1.0, units).rescale(target).magnitude)
    return factor


def length_unit(unit):
    if not isinstance(unit, q.Quantity):
        unit = q.Quantity(1.0, unit).units
    conversion_factor(unit, q.m)  # Raises for anything that is not a length.
    return unit


def magnitude(value, unit):
    """Strip units: quantities are converted to ``unit``, bare numbers are taken as they are."""
    if isinstance(value, q.Quantity):
        return value.magnitude * conversion_factor(value, unit)
    if isinstance(value, (list, tuple)) and any(
        isinstance(v, (q.Quantity, list, tuple)) for v in value
    ):
        return np.array([magnitude(v, unit) for v in value], dtype=float)
    return np.asarray(value, dtype=float)


def vector3(value, name='vector', unit=q.dimensionless):
    value = magnitude(value, unit)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f'{name} must be three finite numbers')
    return value


def unit_vector(value, name='axis'):
    value = vector3(value, name)
    length = np.linalg.norm(value)
    if length == 0:
        raise ValueError(f'{name} must be nonzero')
    return value / length


def finite_scalar(value, name, unit=q.dimensionless):
    value = magnitude(value, unit)
    if value.shape != () or not np.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def angle_radians(angle):
    return finite_scalar(angle, 'angle (bare numbers are radians)', q.rad)


def homogeneous(linear, translation=(0.0, 0.0, 0.0)):
    matrix = np.eye(4)
    matrix[:3, :3] = linear
    matrix[:3, 3] = translation
    return matrix


def axis_angle_matrix(axis, angle):
    """Right-handed rotation matrix about ``axis`` (Rodrigues' formula)."""
    x, y, z = unit_vector(axis)
    radians = angle_radians(angle)
    c, s = np.cos(radians), np.sin(radians)
    t = 1 - c
    return np.array(
        [
            [c + x * x * t, x * y * t - z * s, x * z * t + y * s],
            [y * x * t + z * s, c + y * y * t, y * z * t - x * s],
            [z * x * t - y * s, z * y * t + x * s, c + z * z * t],
        ]
    )
