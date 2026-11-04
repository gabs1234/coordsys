import numpy as np
import quantities as q

from .backend import array_module, float_points, to_numpy
from .utils import (
    angle_radians,
    axis_angle_matrix,
    conversion_factor,
    homogeneous,
    length_unit,
    magnitude,
    unit_vector,
    vector3,
)

SHEAR_PLANES = ('xy', 'xz', 'yx', 'yz', 'zx', 'zy')


class Transformation:
    """An immutable affine map of 3D points, stored as a 4x4 matrix.

    Distances in the matrix, and bare numbers given to it, are in ``unit``.
    ``a @ b`` applies ``b`` first, then ``a``. Points are processed on the
    device (NumPy or CuPy) they already live on.
    """

    def __init__(self, matrix=None, *, unit=q.m):
        self.unit = length_unit(unit)
        matrix = np.eye(4) if matrix is None else np.array(to_numpy(matrix), dtype=float)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError('matrix must be a finite 4x4 array')
        if not np.array_equal(matrix[3], [0, 0, 0, 1]):
            raise ValueError('matrix must be affine: its last row must be (0, 0, 0, 1)')
        self._matrix = matrix

    @classmethod
    def _trusted(cls, matrix, unit):
        transform = cls.__new__(cls)
        transform._matrix, transform.unit = matrix, unit
        return transform

    @classmethod
    def translation(cls, offset, *, unit=q.m):
        unit = length_unit(unit)
        return cls._trusted(homogeneous(np.eye(3), vector3(offset, 'offset', unit)), unit)

    @classmethod
    def rotation(cls, axis, angle):
        """Right-handed rotation about ``axis`` through the origin; bare angles are radians."""
        return cls._trusted(homogeneous(axis_angle_matrix(axis, angle)), q.m)

    @classmethod
    def scaling(cls, factors):
        """Scale by one factor, or by three factors along x, y and z."""
        factors = np.broadcast_to(magnitude(factors, q.dimensionless), 3)
        return cls._trusted(homogeneous(np.diag(vector3(factors, 'factors'))), q.m)

    @classmethod
    def shear(cls, plane, angle):
        """Shear in ``plane``: 'xy' adds tan(angle) * y to x, and so on."""
        if plane not in SHEAR_PLANES:
            raise ValueError(f'plane must be one of {SHEAR_PLANES}')
        radians = angle_radians(angle)
        if np.isclose(np.cos(radians), 0):
            raise ValueError('shear angle must not be ±90°')
        linear = np.eye(3)
        linear['xyz'.index(plane[0]), 'xyz'.index(plane[1])] = np.tan(radians)
        return cls._trusted(homogeneous(linear), q.m)

    @classmethod
    def look_at(cls, eye, target, up=(0, 1, 0), *, unit=q.m):
        """World-to-camera map: ``eye`` goes to the origin, ``target`` onto the -z axis."""
        unit = length_unit(unit)
        eye = vector3(eye, 'eye', unit)
        backward = unit_vector(eye - vector3(target, 'target', unit), 'eye - target')
        right = unit_vector(np.cross(vector3(up, 'up'), backward), 'up × (eye - target)')
        rows = np.array([right, np.cross(backward, right), backward])
        return cls._trusted(homogeneous(rows, -rows @ eye), unit)

    @property
    def matrix(self):
        return self._matrix.copy()

    def as_unit(self, unit):
        """The same physical map, with distances expressed in another unit."""
        unit = length_unit(unit)
        factor = conversion_factor(self.unit, unit)
        if factor == 1:
            return self
        matrix = self._matrix.copy()
        matrix[:3, 3] *= factor
        return Transformation._trusted(matrix, unit)

    def inverse(self):
        linear = np.linalg.inv(self._matrix[:3, :3])
        return Transformation._trusted(
            homogeneous(linear, -linear @ self._matrix[:3, 3]), self.unit
        )

    def __matmul__(self, other):
        if not isinstance(other, Transformation):
            return NotImplemented
        return Transformation._trusted(self._matrix @ other.as_unit(self.unit)._matrix, self.unit)

    def apply(self, points, *, vectors=False):
        """Transform one point ``(3,)`` or a batch ``(..., 3)``.

        ``vectors=True`` skips the translation, for directions or velocities.
        Quantities come back in their own units.
        """
        if isinstance(points, q.Quantity):
            transform = self if vectors else self.as_unit(points.units)
            return transform.apply(points.magnitude, vectors=vectors) * points.units
        if isinstance(points, (list, tuple)):
            points = magnitude(points, q.dimensionless if vectors else self.unit)

        points = float_points(points)
        xp = array_module(points)
        result = points @ xp.asarray(self._matrix[:3, :3].T, dtype=points.dtype)
        if not vectors:
            result += xp.asarray(self._matrix[:3, 3], dtype=points.dtype)
        return result

    def __repr__(self):
        return f'Transformation(unit={self.unit.dimensionality}, matrix=\n{self._matrix})'
